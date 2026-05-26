//! IOSurface → wgpu::Texture import via Metal HAL.
//!
//! CVPixelBuffers from an AVAssetWriterInputPixelBufferAdaptor pool are
//! backed by IOSurfaces (when the pool attributes include
//! `kCVPixelBufferIOSurfacePropertiesKey`). We pull each surface out,
//! create an MTLTexture that shares its memory, and wrap that MTLTexture
//! as a wgpu::Texture via wgpu's Metal HAL backdoor. From the GPU's
//! perspective the IOSurface, the MTLTexture, and the wgpu::Texture all
//! point at the same pixel bytes.

use anyhow::{Result, anyhow};
use objc2_core_video::{CVPixelBuffer, CVPixelBufferGetIOSurface};
use objc2_io_surface::IOSurfaceRef;
use std::collections::HashMap;

/// Per-IOSurface wgpu texture cache. Two reasons we cache:
/// 1. Creating an MTLTexture from an IOSurface isn't free.
/// 2. The pool reuses CVPixelBuffers (and their IOSurfaces) — caching
///    lets us identify the same surface across frames.
pub struct IoSurfaceTextureCache {
    by_surface_id: HashMap<u32, wgpu::Texture>,
}

impl IoSurfaceTextureCache {
    pub fn new() -> Self {
        Self {
            by_surface_id: HashMap::new(),
        }
    }

    /// Look up (or create) the wgpu texture for the given pixel buffer's
    /// IOSurface. Returns the surface ID, which the caller can later pass
    /// to [`Self::texture_by_key`] to retrieve the same texture without
    /// re-borrowing `self` mutably.
    pub fn ensure_texture(
        &mut self,
        device: &wgpu::Device,
        pixel_buf: &CVPixelBuffer,
        width: u32,
        height: u32,
    ) -> Result<u32> {
        let surface = CVPixelBufferGetIOSurface(Some(pixel_buf))
            .ok_or_else(|| anyhow!("CVPixelBuffer is not IOSurface-backed"))?;
        let surface_id = IOSurfaceRef::id(&surface);
        if !self.by_surface_id.contains_key(&surface_id) {
            let texture = create_wgpu_texture_from_iosurface(device, &surface, width, height)?;
            self.by_surface_id.insert(surface_id, texture);
        }
        Ok(surface_id)
    }

    pub fn texture_by_key(&self, key: u32) -> Option<&wgpu::Texture> {
        self.by_surface_id.get(&key)
    }
}

fn create_wgpu_texture_from_iosurface(
    device: &wgpu::Device,
    surface: &IOSurfaceRef,
    width: u32,
    height: u32,
) -> Result<wgpu::Texture> {
    use objc2_metal::{
        MTLDevice, MTLPixelFormat, MTLStorageMode, MTLTextureDescriptor, MTLTextureUsage,
    };

    // Build the Metal texture descriptor matching the IOSurface layout.
    let descriptor = unsafe {
        MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
            MTLPixelFormat::BGRA8Unorm,
            width as usize,
            height as usize,
            false,
        )
    };
    // ShaderWrite for the swizzle compute pass; RenderTarget so the
    // brush-character mesh pipeline can use this same IOSurface as a
    // color attachment (LoadOp::Load preserves the splat backdrop).
    descriptor.setUsage(
        MTLTextureUsage::ShaderWrite | MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget,
    );
    descriptor.setStorageMode(MTLStorageMode::Shared);

    // Reach into wgpu's Metal HAL to (a) call newTextureWithDescriptor:
    // iosurface:plane: on the underlying MTLDevice and (b) hand the
    // resulting MTLTexture back to wgpu as a Texture.
    //
    // SAFETY:
    // - We require the caller to have built `device` with the Metal
    //   backend (we do this ourselves in brush-cli before constructing
    //   the Recorder). If the backend isn't Metal, as_hal returns None
    //   and we bail.
    // - The MTLTexture we hand off lives as long as the IOSurface (the
    //   pixel-buffer pool retains both); wgpu takes ownership of the
    //   Retained handle via `texture_from_raw` and will release it when
    //   the wgpu::Texture is dropped.
    let hal_device_guard = unsafe { device.as_hal::<wgpu::hal::api::Metal>() }
        .ok_or_else(|| anyhow!("device.as_hal::<Metal>() returned None — wrong backend?"))?;
    let mtl_device = hal_device_guard.raw_device();
    let mtl_tex = mtl_device
        .newTextureWithDescriptor_iosurface_plane(&descriptor, surface, 0)
        .ok_or_else(|| anyhow!("newTextureWithDescriptor:iosurface:plane: returned nil"))?;
    drop(hal_device_guard);

    let hal_texture = unsafe {
        wgpu::hal::metal::Device::texture_from_raw(
            mtl_tex,
            wgpu::TextureFormat::Bgra8Unorm,
            objc2_metal::MTLTextureType::Type2D,
            1,
            1,
            wgpu::hal::CopyExtent {
                width,
                height,
                depth: 1,
            },
        )
    };

    let desc = wgpu::TextureDescriptor {
        label: Some("iosurface_bgra"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    };
    let texture =
        unsafe { device.create_texture_from_hal::<wgpu::hal::api::Metal>(hal_texture, &desc) };
    Ok(texture)
}
