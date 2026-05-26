//! Smoke test against the real warehouse voxel asset. Skipped when the file
//! isn't present (e.g. on CI without the asset checked in).

use std::path::PathBuf;

use brush_collision::VoxelCollision;
use glam::Vec3;

fn fixture_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/collision/604b6d93.voxel.json");
    p.exists().then_some(p)
}

#[test]
fn loads_warehouse_voxel() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: 604b6d93.voxel.json not present");
        return;
    };
    let coll = VoxelCollision::load(&path).expect("load voxel");

    // The voxel grid metadata says ~30m × 8m × 37m at 0.05m resolution.
    let size = coll.grid_size();
    assert!(size.x > 20.0 && size.x < 40.0, "unexpected X span: {size:?}");
    assert!(size.y > 6.0 && size.y < 10.0, "unexpected Y span: {size:?}");
    assert!(size.z > 25.0 && size.z < 45.0, "unexpected Z span: {size:?}");

    // Way outside the grid → empty.
    assert!(!coll.is_solid(Vec3::new(1000.0, 1000.0, 1000.0)));

    // Cast a long vertical ray through the center of the grid. In a sealed
    // warehouse there must be solid voxels somewhere along that line — if the
    // octree decoder is broken every voxel reads as empty and the cast
    // returns None.
    let center_xz = coll.grid_min() + Vec3::new(size.x * 0.5, 0.0, size.z * 0.5);
    let origin = Vec3::new(center_xz.x, coll.grid_min().y - 1.0, center_xz.z);
    let hit = coll.ray_cast(origin, Vec3::Y, size.y + 2.0);
    assert!(hit.is_some(), "expected a vertical hit through the warehouse");
}
