//! Skinned glTF character rendering and animation, for the NPC system.
//!
//! Layered API:
//!
//! - [`MeshAsset`] / [`load_mesh`] — parse a `.glb` into CPU-side
//!   buffers + skeleton hierarchy + animation tracks. No GPU state.
//! - [`Skeleton::evaluate`] — sample an animation at a time `t`,
//!   produce per-joint skinning matrices ready to upload as a storage
//!   buffer.
//! - [`MeshRenderer`] — wgpu pipeline that draws one or more instances
//!   of the mesh with GPU skinning. Composites on top of an existing
//!   color attachment (the IOSurface holding the splat backdrop).
//! - [`Path`] — produces (position, heading) at time `t` for a single
//!   NPC, driving the per-instance transform.

mod gltf_load;
mod mesh_render;
mod path;
mod skeleton;

pub use gltf_load::{Animation, JointChannel, MeshAsset, Skeleton, Track, load_mesh};
pub use mesh_render::{GpuMesh, MeshRenderer, NpcInstance};
pub use path::{LinearPath, Path};

#[cfg(test)]
mod sanity {
    use std::path::Path;

    /// Load the bundled character.glb and print stats. Run with
    /// `cargo test -p brush-character -- --nocapture` to see the output.
    #[test]
    fn loads_bundled_character() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("assets/character.glb");
        let asset = super::load_mesh(&path).expect("glb load");
        println!(
            "POSITIONS: {} verts; INDICES: {} ({} tris)",
            asset.positions.len(),
            asset.indices.len(),
            asset.indices.len() / 3,
        );
        println!("JOINTS: {}", asset.skeleton.joints.len());
        for (i, j) in asset.skeleton.joints.iter().enumerate().take(8) {
            println!(
                "  [{i:2}] {:?} parent={:?} t={:?} r={:?} s={:?}",
                j.name, j.parent, j.bind_translation, j.bind_rotation, j.bind_scale,
            );
        }
        if asset.skeleton.joints.len() > 8 {
            println!("  ... ({} more)", asset.skeleton.joints.len() - 8);
        }
        println!("ANIMATIONS:");
        for a in &asset.animations {
            println!(
                "  {:?} {:.2}s {} joint-tracks",
                a.name,
                a.duration,
                a.channels.len()
            );
        }
        // Bounding box for sanity.
        let mut lo = glam::Vec3::splat(f32::INFINITY);
        let mut hi = glam::Vec3::splat(f32::NEG_INFINITY);
        for p in &asset.positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        println!("BBOX: min={lo:?} max={hi:?} size={:?}", hi - lo);
    }
}
