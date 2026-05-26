//! Drop-test: starting at the authored NPC Y, run the same gravity +
//! capsule-pushout physics the record/viewer paths use and check where
//! the NPC settles. If `pos.y` after settling is more than a voxel
//! resolution (~0.05 m) above the floor surface, the physics is
//! genuinely leaving the character floating.

use std::path::PathBuf;

use brush_collision::VoxelCollision;
use glam::Vec3;

fn fixture() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/collision/604b6d93.voxel.json");
    p.exists().then_some(p)
}

/// Same constants the record/viewer NPC physics uses.
const GRAVITY: f32 = 9.81;
const HALF: f32 = 0.5;
const RADIUS: f32 = 0.3;
const BODY_HALF: f32 = HALF + RADIUS;

fn floor_y_at(coll: &VoxelCollision, x: f32, z: f32) -> Option<f32> {
    let origin = Vec3::new(x, coll.grid_min().y - 0.1, z);
    let max = coll.grid_size().y + 1.0;
    coll.ray_cast(origin, Vec3::Y, max).map(|h| h.pos.y)
}

fn simulate(coll: &VoxelCollision, start_y: f32, x: f32, z: f32, frames: usize, dt: f32) -> f32 {
    let mut pos = Vec3::new(x, start_y, z);
    let mut vel_y: f32 = 0.0;
    for _ in 0..frames {
        vel_y += GRAVITY * dt;
        pos.y += vel_y * dt;
        let cc = Vec3::new(pos.x, pos.y - BODY_HALF, pos.z);
        if let Some(push) = coll.query_capsule(cc, HALF, RADIUS) {
            pos += push;
            if push.y < 0.0 {
                vel_y = 0.0;
            }
        }
    }
    pos.y
}

#[test]
fn settles_near_floor() {
    let Some(path) = fixture() else {
        eprintln!("skipping: collision fixture not present");
        return;
    };
    let coll = VoxelCollision::load(&path).expect("load");

    // alice's starting XZ (from scene.json) and authored Y.
    let (start_y, x, z) = (-0.99, 6.0, 0.21);
    let floor = floor_y_at(&coll, x, z).expect("floor under alice");
    // 60 frames at 30fps = 2s — enough to settle a 0.4m drop.
    let settled_y = simulate(&coll, start_y, x, z, 60, 1.0 / 30.0);
    let gap = floor - settled_y;
    println!(
        "alice: floor={:.3}  settled={:.3}  gap={:.3} m  (positive = floating)",
        floor, settled_y, gap
    );

    // bob's starting XZ.
    let (start_y, x, z) = (-0.99, 1.0, 1.5);
    let floor = floor_y_at(&coll, x, z).expect("floor under bob");
    let settled_y = simulate(&coll, start_y, x, z, 60, 1.0 / 30.0);
    let gap_bob = floor - settled_y;
    println!(
        "bob:   floor={:.3}  settled={:.3}  gap={:.3} m  (positive = floating)",
        floor, settled_y, gap_bob
    );

    // Voxel resolution is 0.05 m — pos.y should land within one voxel
    // of the ray-hit floor. Anything > 0.1 m is genuinely floating.
    assert!(gap.abs() < 0.1, "alice off floor by {:.3} m", gap);
    assert!(gap_bob.abs() < 0.1, "bob off floor by {:.3} m", gap_bob);
}
