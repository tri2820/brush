//! Dump the model-local Y bounds of the character glb so we can see
//! where the mesh's origin sits relative to the feet/head — diagnoses
//! the "NPC floats above floor" problem.

use std::path::PathBuf;

use brush_character::load_mesh;

#[test]
fn dump_character_bounds() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../assets/character.glb");
    if !p.exists() {
        eprintln!("skipping: {} missing", p.display());
        return;
    }
    let asset = load_mesh(&p).expect("load_mesh");

    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    for p in &asset.positions {
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_z = min_z.min(p.z);
        max_z = max_z.max(p.z);
    }
    println!("character.glb mesh-local bounds:");
    println!(
        "  X: [{:.3}, {:.3}]  width  = {:.3} m",
        min_x,
        max_x,
        max_x - min_x
    );
    println!(
        "  Y: [{:.3}, {:.3}]  height = {:.3} m   (origin offset above lowest = {:.3})",
        min_y,
        max_y,
        max_y - min_y,
        -min_y
    );
    println!(
        "  Z: [{:.3}, {:.3}]  depth  = {:.3} m",
        min_z,
        max_z,
        max_z - min_z
    );
    println!("  joint count: {}", asset.skeleton.joints.len());

    // Where is the world origin of the rendered character in bind pose
    // (no animation, t=0)?  That's what determines whether physics-set
    // `pos.y` actually matches the visual feet position. If the bind
    // pose's root joint or armature shifts the mesh away from local
    // origin, the rendered feet won't be at `pos`.
    let skin_mats = asset.skeleton.evaluate(None, 0.0);
    println!("  armature: translation = {:?}", asset.skeleton.armature.w_axis);
    let root = skin_mats[0];
    println!("  joint[0] skin matrix translation = {:?}", root.w_axis);

    // Apply the skin matrix to the lowest-Y vertex (presumed feet) and
    // see where it ends up — this is the world-space position the
    // bind-pose mesh actually renders the feet at, modulo the per-NPC
    // model matrix.
    let feet_local = asset
        .positions
        .iter()
        .copied()
        .min_by(|a, b| a.y.partial_cmp(&b.y).unwrap())
        .unwrap();
    let feet_after_skin = root.transform_point3(feet_local);
    println!(
        "  feet local {:?} → after skin (using joint[0]) {:?}",
        feet_local, feet_after_skin
    );

    // Find the actual lowest vertex's index, look up its joint indices
    // and weights, and apply the proper LBS combination.
    let mut lowest_idx = 0usize;
    let mut lowest_y = f32::INFINITY;
    for (i, p) in asset.positions.iter().enumerate() {
        if p.y < lowest_y {
            lowest_y = p.y;
            lowest_idx = i;
        }
    }
    let js = asset.joints[lowest_idx];
    let ws = asset.weights[lowest_idx];
    let proper = ws[0] * skin_mats[js[0] as usize]
        + ws[1] * skin_mats[js[1] as usize]
        + ws[2] * skin_mats[js[2] as usize]
        + ws[3] * skin_mats[js[3] as usize];
    let p = asset.positions[lowest_idx];
    let after = proper.transform_point3(p);
    println!(
        "  lowest vertex idx={} pos={:?} joints={:?} weights={:?}",
        lowest_idx, p, js, ws
    );
    println!("  → after proper LBS: {:?}", after);
    println!(
        "  rendered-feet-y vs feet-local-y delta = {:.4} m (positive = mesh rises)",
        after.y - p.y
    );

    // Find global Y bounds AFTER skinning across the whole mesh.
    let mut min_after = f32::INFINITY;
    let mut max_after = f32::NEG_INFINITY;
    for (i, pos) in asset.positions.iter().enumerate() {
        let js = asset.joints[i];
        let ws = asset.weights[i];
        let mat = ws[0] * skin_mats[js[0] as usize]
            + ws[1] * skin_mats[js[1] as usize]
            + ws[2] * skin_mats[js[2] as usize]
            + ws[3] * skin_mats[js[3] as usize];
        let after = mat.transform_point3(*pos);
        if after.y < min_after {
            min_after = after.y;
        }
        if after.y > max_after {
            max_after = after.y;
        }
    }
    println!(
        "  after-skinning bind-pose Y range: [{:.4}, {:.4}]  height={:.4}",
        min_after,
        max_after,
        max_after - min_after
    );

    // Histogram of vertex Y to spot a "ground plane" or other non-body
    // geometry at the bottom. If the LOWEST band has very few vertices
    // and there's a big jump, that band is probably a floor decal, not
    // feet — and the actual feet start higher up.
    let mut bins = [0usize; 18];
    for (i, p) in asset.positions.iter().enumerate() {
        let js = asset.joints[i];
        let ws = asset.weights[i];
        let mat = ws[0] * skin_mats[js[0] as usize]
            + ws[1] * skin_mats[js[1] as usize]
            + ws[2] * skin_mats[js[2] as usize]
            + ws[3] * skin_mats[js[3] as usize];
        let y = mat.transform_point3(*p).y;
        let b = ((y / 0.1) as i32).clamp(0, 17) as usize;
        bins[b] += 1;
    }
    println!("  vertex-Y histogram (bins of 0.1 m):");
    for (i, c) in bins.iter().enumerate() {
        if *c > 0 {
            println!("    Y=[{:.1}, {:.1}): {} vertices", i as f32 * 0.1, (i + 1) as f32 * 0.1, c);
        }
    }
}
