//! Exercise the actual floor-snap math that brush-cli applies per frame, at
//! the NPC positions hand-authored in `scene.json`. Run with
//! `cargo test -p brush-collision -- --nocapture` to see the numbers.

use std::path::PathBuf;

use brush_collision::VoxelCollision;
use glam::Vec3;

fn fixture_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/collision/604b6d93.voxel.json");
    p.exists().then_some(p)
}

fn floor_y(coll: &VoxelCollision, p: Vec3) -> Option<f32> {
    let origin = Vec3::new(p.x, coll.grid_min().y - 0.1, p.z);
    let max = coll.grid_size().y + 1.0;
    coll.ray_cast(origin, Vec3::Y, max).map(|h| h.pos.y)
}

#[test]
fn warehouse_npc_offsets() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: 604b6d93.voxel.json not present");
        return;
    };
    let coll = VoxelCollision::load(&path).expect("load voxel");

    // Positions copied straight from scene.json. These should all hit the
    // warehouse floor (the carved navmesh covered all of them).
    let alice_start = Vec3::new(6.0, -0.99, 0.21);
    let alice_end = Vec3::new(1.0, -0.99, 0.21);
    let bob_start = Vec3::new(1.0, -0.99, 1.5);
    let bob_end = Vec3::new(6.0, -0.99, 1.5);

    for (name, p) in [
        ("alice.start", alice_start),
        ("alice.end", alice_end),
        ("bob.start", bob_start),
        ("bob.end", bob_end),
    ] {
        let fy = floor_y(&coll, p);
        let offset = fy.map(|y| p.y - y);
        println!(
            "{name:12} authored_y={:.3}  floor_y={}  offset={}",
            p.y,
            fy.map_or("MISS".into(), |y| format!("{y:.3}")),
            offset.map_or("--".into(), |o| format!("{o:.3}")),
        );
        assert!(fy.is_some(), "{name}: ray missed floor");
    }
}
