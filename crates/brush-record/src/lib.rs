//! Real zero-copy video recording on macOS.
//!
//! The full pipeline keeps pixel bytes on the GPU end-to-end:
//!
//! ```text
//! brush rasterizer (wgpu Buffer, packed RGBA8)
//!   → GPU compute pass (RGBA→BGRA swizzle, into IOSurface-backed BGRA8 texture)
//!   → CVPixelBuffer wrapping the same IOSurface
//!   → AVAssetWriterInputPixelBufferAdaptor.append(buf, pts)
//!   → VideoToolbox hardware encoder (h264_videotoolbox / hevc_videotoolbox)
//!   → mp4 on disk
//! ```
//!
//! CPU never sees a pixel.

#![cfg(target_os = "macos")]

mod encoder;
mod iosurface;
mod swizzle;

use anyhow::Result;
use std::path::Path;

pub use encoder::Codec;

/// Configuration for [`Recorder::new`].
#[derive(Clone, Copy, Debug)]
pub struct RecorderConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub codec: Codec,
}

/// One-shot video recorder. Build with [`Recorder::new`], feed frames
/// via [`Recorder::write_frame`], close with [`Recorder::finish`].
pub struct Recorder {
    encoder: encoder::Encoder,
    iosurface_cache: iosurface::IoSurfaceTextureCache,
    swizzle: swizzle::SwizzlePipeline,
    device: wgpu::Device,
    queue: wgpu::Queue,
    frame_index: i64,
    fps: u32,
    width: u32,
    height: u32,
}

impl Recorder {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        output: &Path,
        config: RecorderConfig,
    ) -> Result<Self> {
        let encoder = encoder::Encoder::new(output, config.width, config.height, config.fps, config.codec)?;
        let swizzle = swizzle::SwizzlePipeline::new(&device, config.width, config.height);
        Ok(Self {
            encoder,
            iosurface_cache: iosurface::IoSurfaceTextureCache::new(),
            swizzle,
            device,
            queue,
            frame_index: 0,
            fps: config.fps,
            width: config.width,
            height: config.height,
        })
    }

    /// Begin a new frame. Returns a handle whose `color_texture()` is
    /// the IOSurface-backed BGRA texture for this frame; the caller is
    /// expected to populate it (swizzle from the splat buffer, then
    /// optionally run additional render passes such as a character
    /// mesh draw), then call [`Frame::finish`] to append to the
    /// encoder.
    pub fn begin_frame(&mut self) -> Result<Frame<'_>> {
        let pixel_buf = self.encoder.dequeue_pixel_buffer()?;
        // Get the cached wgpu texture index for this IOSurface (the
        // pool reuses surfaces across frames, so we cache by surface ID).
        let cache_key = self
            .iosurface_cache
            .ensure_texture(&self.device, &pixel_buf, self.width, self.height)?;
        Ok(Frame {
            recorder: self,
            pixel_buf: Some(pixel_buf),
            cache_key,
            last_submission: None,
        })
    }

    /// Convenience: dequeue, swizzle the packed splat buffer into the
    /// IOSurface, append. Equivalent to `begin_frame` + `swizzle_from`
    /// + `finish`. Useful when there is no per-frame mesh overlay.
    pub fn write_frame(&mut self, src_buffer: &wgpu::Buffer, src_offset: u64) -> Result<()> {
        let mut frame = self.begin_frame()?;
        frame.swizzle_from(src_buffer, src_offset);
        frame.finish()
    }

    /// Flush and close the underlying mp4 file.
    pub async fn finish(self) -> Result<()> {
        self.encoder.finish().await
    }
}

/// Live frame in flight. While a `Frame` exists, the caller has
/// exclusive access to the IOSurface-backed color texture and can
/// render into it. Dropping without calling [`Frame::finish`] discards
/// the frame.
pub struct Frame<'r> {
    recorder: &'r mut Recorder,
    pixel_buf: Option<objc2::rc::Retained<objc2_core_video::CVPixelBuffer>>,
    cache_key: u32,
    /// Tracks the most recent wgpu submission that touched this
    /// frame's texture, so `finish` can fence before handing the
    /// IOSurface to VideoToolbox.
    last_submission: Option<wgpu::SubmissionIndex>,
}

impl<'r> Frame<'r> {
    /// The IOSurface-backed BGRA8 wgpu texture for this frame. Bind as
    /// a render attachment with `LoadOp::Load` to draw additional layers
    /// on top of whatever the swizzle wrote.
    pub fn color_texture(&self) -> &wgpu::Texture {
        self.recorder
            .iosurface_cache
            .texture_by_key(self.cache_key)
            .expect("frame texture cached at begin_frame")
    }

    pub fn width(&self) -> u32 {
        self.recorder.width
    }
    pub fn height(&self) -> u32 {
        self.recorder.height
    }
    pub fn device(&self) -> &wgpu::Device {
        &self.recorder.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.recorder.queue
    }

    /// Swizzle a packed RGBA u32 buffer into the color texture (the
    /// splat compositing step). Records the submission so `finish` can
    /// fence on it.
    pub fn swizzle_from(&mut self, src_buffer: &wgpu::Buffer, src_offset: u64) {
        let dst = self.color_texture();
        let submission = self.recorder.swizzle.dispatch(
            &self.recorder.device,
            &self.recorder.queue,
            src_buffer,
            src_offset,
            dst,
            self.recorder.width,
            self.recorder.height,
        );
        self.last_submission = Some(submission);
    }

    /// Update the fence checkpoint to a later submission (e.g. after a
    /// caller-supplied mesh render pass). `finish` will wait for this.
    pub fn note_submission(&mut self, submission: wgpu::SubmissionIndex) {
        self.last_submission = Some(submission);
    }

    /// Fence the GPU work and append the CVPixelBuffer to the encoder.
    pub fn finish(mut self) -> Result<()> {
        if let Some(sub) = self.last_submission.take() {
            let _ = self.recorder.device.poll(wgpu::PollType::Wait {
                submission_index: Some(sub),
                timeout: None,
            });
        }
        let pixel_buf = self.pixel_buf.take().expect("pixel buf consumed twice");
        let pts = self.recorder.frame_index;
        self.recorder
            .encoder
            .append(pixel_buf, pts, self.recorder.fps as i32)?;
        self.recorder.frame_index += 1;
        Ok(())
    }
}
