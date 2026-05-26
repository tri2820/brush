//! AVAssetWriter wrapper with a CVPixelBufferPool. Manages the H264/HEVC
//! VideoToolbox encoder sink for the recording pipeline.

use anyhow::{Result, anyhow};
use objc2::rc::Retained;
use objc2_av_foundation::{
    AVAssetWriter, AVAssetWriterInput, AVAssetWriterInputPixelBufferAdaptor,
};
use objc2_core_media::CMTime;
use objc2_core_video::{CVPixelBuffer, kCVPixelFormatType_32BGRA};
use objc2_foundation::{NSDictionary, NSNumber, NSString, NSURL};
use std::path::Path;
use std::ptr::NonNull;

/// Reinterpret a `&CFString` as `&NSString`. Sound because CFStringRef
/// and NSString are toll-free bridged — they're the same Objective-C
/// class (`__NSCFString`) at runtime, so the underlying memory layout
/// and method dispatch are identical.
fn cf_to_ns(s: &objc2_core_foundation::CFString) -> &NSString {
    unsafe { &*(s as *const objc2_core_foundation::CFString as *const NSString) }
}

#[derive(Clone, Copy, Debug, Default)]
pub enum Codec {
    #[default]
    Hevc,
    H264,
}

pub struct Encoder {
    writer: Retained<AVAssetWriter>,
    input: Retained<AVAssetWriterInput>,
    adaptor: Retained<AVAssetWriterInputPixelBufferAdaptor>,
}

impl Encoder {
    pub fn new(output: &Path, width: u32, height: u32, fps: u32, codec: Codec) -> Result<Self> {
        // Remove any prior file at the path; AVAssetWriter refuses if the URL exists.
        let _ = std::fs::remove_file(output);

        let path_str = output
            .to_str()
            .ok_or_else(|| anyhow!("output path is not valid UTF-8: {}", output.display()))?;
        let ns_path = NSString::from_str(path_str);

        let url = NSURL::fileURLWithPath(&ns_path);

        // AVFileTypeMPEG4 is the canonical UTI string for .mp4 containers.
        // The static is loaded lazily by the framework — `.expect` is fine
        // since it can only fail if AVFoundation isn't linked, which won't
        // happen here.
        let file_type = unsafe { objc2_av_foundation::AVFileTypeMPEG4 }
            .expect("AVFileTypeMPEG4 missing — AVFoundation not linked?");

        let writer =
            unsafe { AVAssetWriter::assetWriterWithURL_fileType_error(&url, file_type) }
                .map_err(|e| anyhow!("AVAssetWriter init failed: {e:?}"))?;

        // Video settings dict: { AVVideoCodecKey: codec, AVVideoWidthKey: W, AVVideoHeightKey: H }
        let codec_value = unsafe {
            match codec {
                Codec::Hevc => objc2_av_foundation::AVVideoCodecTypeHEVC,
                Codec::H264 => objc2_av_foundation::AVVideoCodecTypeH264,
            }
        }
        .expect("AVVideoCodecType missing");
        let width_num = NSNumber::new_u32(width);
        let height_num = NSNumber::new_u32(height);

        // SAFETY: keys come from the AVVideoSettings header and are
        // valid for the lifetime of the framework binding.
        let video_settings = unsafe {
            let codec_key = objc2_av_foundation::AVVideoCodecKey
                .expect("AVVideoCodecKey missing");
            let width_key = objc2_av_foundation::AVVideoWidthKey
                .expect("AVVideoWidthKey missing");
            let height_key = objc2_av_foundation::AVVideoHeightKey
                .expect("AVVideoHeightKey missing");
            let keys: [&NSString; 3] = [codec_key, width_key, height_key];
            let values: [&objc2::runtime::AnyObject; 3] = [
                &**codec_value as &objc2::runtime::AnyObject,
                &*width_num as &objc2::runtime::AnyObject,
                &*height_num as &objc2::runtime::AnyObject,
            ];
            NSDictionary::from_slices(&keys, &values)
        };

        let media_type = unsafe { objc2_av_foundation::AVMediaTypeVideo }
            .expect("AVMediaTypeVideo missing");
        let input = unsafe {
            AVAssetWriterInput::assetWriterInputWithMediaType_outputSettings(
                media_type,
                Some(&video_settings),
            )
        };
        // Hint AVFoundation that we'll feed frames in real time-ish;
        // this lets it size internal buffers reasonably.
        unsafe { input.setExpectsMediaDataInRealTime(true) };

        // Pixel-buffer pool attributes: BGRA 8-bit + matching size, so
        // the IOSurface we'll create wgpu textures from has the right
        // format and dimensions. The CV*Key statics are CFStrings but
        // are toll-free bridged with NSString for use in NSDictionary.
        let pix_fmt_key = cf_to_ns(unsafe { objc2_core_video::kCVPixelBufferPixelFormatTypeKey });
        let pix_fmt_value = NSNumber::new_u32(kCVPixelFormatType_32BGRA);
        let w_key = cf_to_ns(unsafe { objc2_core_video::kCVPixelBufferWidthKey });
        let h_key = cf_to_ns(unsafe { objc2_core_video::kCVPixelBufferHeightKey });
        let iosurface_props_key =
            cf_to_ns(unsafe { objc2_core_video::kCVPixelBufferIOSurfacePropertiesKey });
        let empty_dict: Retained<NSDictionary<NSString, objc2::runtime::AnyObject>> =
            NSDictionary::from_slices::<NSString>(&[], &[]);

        let source_attrs = {
            let keys: [&NSString; 4] = [pix_fmt_key, w_key, h_key, iosurface_props_key];
            let values: [&objc2::runtime::AnyObject; 4] = [
                &*pix_fmt_value as &objc2::runtime::AnyObject,
                &*width_num as &objc2::runtime::AnyObject,
                &*height_num as &objc2::runtime::AnyObject,
                &*empty_dict as &objc2::runtime::AnyObject,
            ];
            NSDictionary::from_slices(&keys, &values)
        };

        let adaptor = unsafe {
            AVAssetWriterInputPixelBufferAdaptor::assetWriterInputPixelBufferAdaptorWithAssetWriterInput_sourcePixelBufferAttributes(
                &input,
                Some(&source_attrs),
            )
        };

        if !unsafe { writer.canAddInput(&input) } {
            return Err(anyhow!(
                "AVAssetWriter rejected the video input (codec/size mismatch?)"
            ));
        }
        unsafe { writer.addInput(&input) };

        if !unsafe { writer.startWriting() } {
            let err = unsafe { writer.error() };
            return Err(anyhow!("AVAssetWriter startWriting failed: {err:?}"));
        }
        // CMTime { value: 0, scale: 1 } == 0 seconds.
        unsafe { writer.startSessionAtSourceTime(CMTime::new(0, 1)) };

        log::info!(
            "Encoder opened: {:?} {}x{} @ {}fps → {}",
            codec,
            width,
            height,
            fps,
            output.display()
        );

        let _ = fps; // currently informational; PTS is set per-frame
        Ok(Self { writer, input, adaptor })
    }

    /// Pull a CVPixelBuffer from the adaptor's pool. The buffer's
    /// IOSurface is what the GPU will write into.
    pub fn dequeue_pixel_buffer(&self) -> Result<Retained<CVPixelBuffer>> {
        let pool = unsafe { self.adaptor.pixelBufferPool() }
            .ok_or_else(|| anyhow!("pixel buffer pool not ready (writer not started?)"))?;
        // CVPixelBufferPoolCreatePixelBuffer returns the new buffer
        // through an out-pointer; the resulting CVPixelBuffer is owned
        // (retain count 1) and must be released — Retained::from_raw
        // takes ownership.
        let mut out_ptr: *mut CVPixelBuffer = std::ptr::null_mut();
        let status = unsafe {
            objc2_core_video::CVPixelBufferPool::create_pixel_buffer(
                None,
                &pool,
                NonNull::from(&mut out_ptr),
            )
        };
        if status != 0 {
            return Err(anyhow!("CVPixelBufferPoolCreatePixelBuffer failed: {status}"));
        }
        let ptr = NonNull::new(out_ptr)
            .ok_or_else(|| anyhow!("pool returned a null pixel buffer"))?;
        Ok(unsafe { Retained::from_raw(ptr.as_ptr()) }
            .expect("non-null CVPixelBuffer just returned from CF"))
    }

    /// Append one frame at `pts_num / pts_timescale` seconds. Blocks
    /// if the input is not yet ready for more data (back-pressure).
    pub fn append(
        &self,
        buf: Retained<CVPixelBuffer>,
        pts_num: i64,
        pts_timescale: i32,
    ) -> Result<()> {
        // Spin briefly if the encoder isn't ready (encoder is async on
        // its own thread; this should be rare with realtime hint set).
        while !unsafe { self.input.isReadyForMoreMediaData() } {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let pts = unsafe { CMTime::new(pts_num, pts_timescale) };
        let ok = unsafe {
            self.adaptor
                .appendPixelBuffer_withPresentationTime(&buf, pts)
        };
        if !ok {
            let err = unsafe { self.writer.error() };
            return Err(anyhow!("appendPixelBuffer failed: {err:?}"));
        }
        Ok(())
    }

    pub async fn finish(self) -> Result<()> {
        unsafe { self.input.markAsFinished() };
        // finishWritingWithCompletionHandler runs async; the simpler
        // sync variant is finishWriting (deprecated but still works in
        // current AVFoundation). Use it for clarity.
        #[allow(deprecated)]
        let _ = unsafe { self.writer.finishWriting() };
        let status = unsafe { self.writer.status() };
        if status.0 != 2 {
            // 2 = AVAssetWriterStatusCompleted
            let err = unsafe { self.writer.error() };
            return Err(anyhow!("AVAssetWriter finish bad status {status:?}: {err:?}"));
        }
        Ok(())
    }
}
