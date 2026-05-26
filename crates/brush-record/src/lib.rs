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

    /// Encode one frame. The pixels live in `src_buffer` starting at
    /// `src_offset`, packed as little-endian RGBA u32 per pixel (the
    /// layout brush's rasterizer produces in `TextureMode::Packed`).
    /// The buffer must be `width * height * 4` bytes from `src_offset`.
    pub fn write_frame(
        &mut self,
        src_buffer: &wgpu::Buffer,
        src_offset: u64,
    ) -> Result<()> {
        // Acquire a CVPixelBuffer from the pool; its IOSurface backs the
        // wgpu storage texture we'll write into.
        let pixel_buf = self.encoder.dequeue_pixel_buffer()?;
        let dst_texture = self.iosurface_cache.texture_for(
            &self.device,
            &pixel_buf,
            self.width,
            self.height,
        )?;

        // Run the swizzle compute pass on the GPU. After this submit,
        // the IOSurface contents are the BGRA frame.
        let submission = self.swizzle.dispatch(
            &self.device,
            &self.queue,
            src_buffer,
            src_offset,
            dst_texture,
            self.width,
            self.height,
        );

        // The IOSurface is shared between wgpu (writer) and VideoToolbox
        // (reader). VideoToolbox reads via a separate command stream, so
        // we have to fence wgpu's submission to ensure the swizzle has
        // committed before the encoder consumes the surface. Polling
        // Wait blocks the CPU thread until the submission is complete;
        // for higher throughput we could pipeline via MTLSharedEvent,
        // but the simple fence is fast enough at 1080p.
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        });

        // Append to the AVAssetWriter at the right PTS.
        let pts_value = self.frame_index;
        self.encoder.append(pixel_buf, pts_value, self.fps as i32)?;
        self.frame_index += 1;
        Ok(())
    }

    /// Flush and close the underlying mp4 file.
    pub async fn finish(self) -> Result<()> {
        self.encoder.finish().await
    }
}
