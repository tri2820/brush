//! Voxel-octree collision queries for Brush scenes.
//!
//! Loads the `.voxel.json` + `.voxel.bin` pair produced by PlayCanvas
//! `splat-transform`, then answers point and ray queries against the sparse
//! solid-voxel set. Ported from `supersplat-viewer/src/collision/voxel-collision.ts`.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use glam::Vec3;

/// Sentinel value for a node that represents a fully-solid subtree. The
/// high byte (`childMask` slot) is 0xFF; the low 24 bits are zero, which
/// is impossible for a real interior node because the BFS layout always
/// puts children *after* their parent.
const SOLID_LEAF_MARKER: u32 = 0xFF00_0000;

/// Penetrations shallower than this don't generate a push-out — keeps
/// the iterative resolver from oscillating on float noise.
const PENETRATION_EPSILON: f32 = 1e-4;

/// Maximum corner-resolution passes. Three intersecting walls is the
/// pathological case; one more pass for headroom.
const MAX_RESOLVE_ITERATIONS: usize = 4;

#[derive(Debug, serde::Deserialize)]
struct Bounds {
    min: [f32; 3],
    max: [f32; 3],
}

#[derive(Debug, serde::Deserialize)]
struct VoxelMetadata {
    #[serde(default)]
    version: String,
    #[serde(rename = "gridBounds")]
    grid_bounds: Bounds,
    #[serde(rename = "voxelResolution")]
    voxel_resolution: f32,
    #[serde(rename = "leafSize")]
    leaf_size: u32,
    #[serde(rename = "treeDepth")]
    tree_depth: u32,
    #[serde(rename = "nodeCount")]
    node_count: u32,
    #[serde(rename = "leafDataCount")]
    leaf_data_count: u32,
}

/// A loaded sparse-voxel octree usable for point and ray queries against
/// solid geometry.
pub struct VoxelCollision {
    grid_min: Vec3,
    num_voxels: glam::UVec3,
    voxel_resolution: f32,
    leaf_size: u32,
    tree_depth: u32,
    nodes: Vec<u32>,
    leaf_data: Vec<u32>,
}

/// Result of a successful [`VoxelCollision::ray_cast`].
#[derive(Debug, Clone, Copy)]
pub struct RayHit {
    /// World-space hit point.
    pub pos: Vec3,
    /// Distance along the ray from origin to hit.
    pub t: f32,
}

impl VoxelCollision {
    /// Load a `.voxel.json` and its paired `.voxel.bin` sibling.
    pub fn load(json_path: &Path) -> Result<Self> {
        let json_text = std::fs::read_to_string(json_path)
            .with_context(|| format!("read {}", json_path.display()))?;
        let meta: VoxelMetadata =
            serde_json::from_str(&json_text).context("parse voxel metadata")?;

        // PlayCanvas wrote v1.0 with X/Y axes negated relative to world.
        // We only target v1.1+ for now; reject older files explicitly so a
        // mismatched scene fails loud instead of producing silently-flipped
        // collision.
        let version: f32 = meta.version.parse().unwrap_or(1.1);
        if version < 1.1 {
            return Err(anyhow!(
                "voxel format v{} not supported (need v1.1+, X/Y flip not handled)",
                meta.version
            ));
        }

        let bin_path = json_path
            .to_string_lossy()
            .replace(".voxel.json", ".voxel.bin");
        let bin_bytes = std::fs::read(&bin_path).with_context(|| format!("read {bin_path}"))?;

        let expected = (meta.node_count as usize + meta.leaf_data_count as usize) * 4;
        if bin_bytes.len() < expected {
            return Err(anyhow!(
                "voxel binary truncated: expected {} bytes, got {}",
                expected,
                bin_bytes.len()
            ));
        }
        // Decode word-by-word — `read()` may give 1-byte alignment, which
        // would block `bytemuck::cast_slice`.
        let truncated = (bin_bytes.len() / 4) * 4;
        let words = truncated / 4;
        let mut buf = vec![0u32; words];
        for (i, chunk) in bin_bytes[..truncated].chunks_exact(4).enumerate() {
            buf[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        let leaf_end = meta.node_count as usize + meta.leaf_data_count as usize;
        let nodes = buf[..meta.node_count as usize].to_vec();
        let leaf_data = buf[meta.node_count as usize..leaf_end].to_vec();

        let res = meta.voxel_resolution;
        let span = Vec3::from(meta.grid_bounds.max) - Vec3::from(meta.grid_bounds.min);
        let num_voxels = glam::UVec3::new(
            (span.x / res).round() as u32,
            (span.y / res).round() as u32,
            (span.z / res).round() as u32,
        );

        log::info!(
            "VoxelCollision: {}×{}×{} voxels @ {}m, {} nodes, {} leaves",
            num_voxels.x,
            num_voxels.y,
            num_voxels.z,
            res,
            meta.node_count,
            meta.leaf_data_count
        );

        Ok(Self {
            grid_min: Vec3::from(meta.grid_bounds.min),
            num_voxels,
            voxel_resolution: res,
            leaf_size: meta.leaf_size,
            tree_depth: meta.tree_depth,
            nodes,
            leaf_data,
        })
    }

    /// World-space grid extent (max corner = `grid_min + grid_size`).
    pub fn grid_min(&self) -> Vec3 {
        self.grid_min
    }

    pub fn grid_size(&self) -> Vec3 {
        self.num_voxels.as_vec3() * self.voxel_resolution
    }

    pub fn voxel_resolution(&self) -> f32 {
        self.voxel_resolution
    }

    /// True if the voxel containing world point `p` is solid.
    pub fn is_solid(&self, p: Vec3) -> bool {
        let v = ((p - self.grid_min) / self.voxel_resolution).floor();
        // Reject out-of-grid before casting to unsigned — negative floors must
        // not wrap.
        if v.x < 0.0 || v.y < 0.0 || v.z < 0.0 {
            return false;
        }
        self.is_voxel_solid(v.x as i64, v.y as i64, v.z as i64)
    }

    /// Cast a ray and return the first solid voxel hit, or `None` if the ray
    /// leaves the grid without hitting anything within `max_dist`. Used for
    /// floor snapping (cast downward) and surface picking.
    pub fn ray_cast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RayHit> {
        if self.nodes.is_empty() {
            return None;
        }
        const EPS: f32 = 1e-6;
        let res = self.voxel_resolution;
        let g_min = self.grid_min;
        let g_max = g_min + self.grid_size();

        // Ray-AABB slab intersection to find the [t_near, t_far] interval
        // where the ray is inside the grid box. The DDA below only makes
        // sense once we're inside.
        let mut t_near: f32 = 0.0;
        let mut t_far: f32 = max_dist;
        for axis in 0..3 {
            let o = origin[axis];
            let d = dir[axis];
            let lo = g_min[axis];
            let hi = g_max[axis];
            if d.abs() > EPS {
                let mut t1 = (lo - o) / d;
                let mut t2 = (hi - o) / d;
                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }
                if t1 > t_near {
                    t_near = t1;
                }
                if t2 < t_far {
                    t_far = t2;
                }
                if t_near > t_far {
                    return None;
                }
            } else if o < lo || o >= hi {
                return None;
            }
        }

        let entry = origin + dir * t_near;
        let mut i = glam::IVec3::new(
            (((entry.x - g_min.x) / res).floor() as i32).clamp(0, self.num_voxels.x as i32 - 1),
            (((entry.y - g_min.y) / res).floor() as i32).clamp(0, self.num_voxels.y as i32 - 1),
            (((entry.z - g_min.z) / res).floor() as i32).clamp(0, self.num_voxels.z as i32 - 1),
        );

        let step = glam::IVec3::new(sign_i(dir.x), sign_i(dir.y), sign_i(dir.z));
        let inv = Vec3::new(
            if dir.x.abs() > EPS { 1.0 / dir.x } else { 0.0 },
            if dir.y.abs() > EPS { 1.0 / dir.y } else { 0.0 },
            if dir.z.abs() > EPS { 1.0 / dir.z } else { 0.0 },
        );

        // tMax: parametric distance to next axis crossing per axis
        let next_boundary = |o: f32, d: f32, gmin: f32, idx: i32| -> f32 {
            gmin + (idx + if d > 0.0 { 1 } else { 0 }) as f32 * res - o
        };
        let mut t_max = Vec3::new(
            if dir.x.abs() > EPS {
                next_boundary(origin.x, dir.x, g_min.x, i.x) * inv.x
            } else {
                f32::INFINITY
            },
            if dir.y.abs() > EPS {
                next_boundary(origin.y, dir.y, g_min.y, i.y) * inv.y
            } else {
                f32::INFINITY
            },
            if dir.z.abs() > EPS {
                next_boundary(origin.z, dir.z, g_min.z, i.z) * inv.z
            } else {
                f32::INFINITY
            },
        );
        let t_delta = Vec3::new(
            if dir.x.abs() > EPS {
                res * inv.x.abs()
            } else {
                f32::INFINITY
            },
            if dir.y.abs() > EPS {
                res * inv.y.abs()
            } else {
                f32::INFINITY
            },
            if dir.z.abs() > EPS {
                res * inv.z.abs()
            } else {
                f32::INFINITY
            },
        );

        let mut t = t_near;
        let max_steps = (self.num_voxels.x + self.num_voxels.y + self.num_voxels.z) as usize;
        for _ in 0..max_steps {
            if self.is_voxel_solid(i.x as i64, i.y as i64, i.z as i64) {
                return Some(RayHit {
                    pos: origin + dir * t,
                    t,
                });
            }
            if t_max.x < t_max.y {
                if t_max.x < t_max.z {
                    t = t_max.x;
                    i.x += step.x;
                    t_max.x += t_delta.x;
                } else {
                    t = t_max.z;
                    i.z += step.z;
                    t_max.z += t_delta.z;
                }
            } else if t_max.y < t_max.z {
                t = t_max.y;
                i.y += step.y;
                t_max.y += t_delta.y;
            } else {
                t = t_max.z;
                i.z += step.z;
                t_max.z += t_delta.z;
            }
            if i.x < 0
                || i.y < 0
                || i.z < 0
                || i.x >= self.num_voxels.x as i32
                || i.y >= self.num_voxels.y as i32
                || i.z >= self.num_voxels.z as i32
                || t > max_dist
            {
                return None;
            }
        }
        None
    }

    /// Resolve a vertical capsule out of any solid voxels it overlaps.
    /// The capsule's axis is along Y; its segment runs from
    /// `center - half_height·Ŷ` to `center + half_height·Ŷ`, swept by
    /// `radius`. Returns the displacement to add to `center` so the
    /// capsule no longer penetrates, or `None` if already clear.
    ///
    /// Runs up to [`MAX_RESOLVE_ITERATIONS`] passes so corners (two/three
    /// walls meeting) resolve cleanly instead of oscillating.
    pub fn query_capsule(&self, center: Vec3, half_height: f32, radius: f32) -> Option<Vec3> {
        if self.nodes.is_empty() {
            return None;
        }
        self.resolve_iterative(center, |c| {
            self.deepest_penetration_capsule(c, half_height, radius)
        })
    }

    /// Iteratively apply push-outs from `find` until the body is clear or
    /// [`MAX_RESOLVE_ITERATIONS`] is hit. Each iteration projects the new
    /// push against the normals of previous pushes so we don't undo earlier
    /// constraint resolutions (the standard corner-resolution trick).
    fn resolve_iterative<F>(&self, initial: Vec3, find: F) -> Option<Vec3>
    where
        F: Fn(Vec3) -> Option<Vec3>,
    {
        let mut resolved = initial;
        let mut total = Vec3::ZERO;
        let mut normals: [Vec3; 3] = [Vec3::ZERO; 3];
        let mut num_normals = 0usize;
        let mut had_collision = false;

        for _ in 0..MAX_RESOLVE_ITERATIONS {
            let Some(scratch) = find(resolved) else { break };
            had_collision = true;

            // Project against earlier constraint normals — keeps a wall we
            // already resolved against from being un-resolved by the next
            // push.
            let mut projected = scratch;
            for n in &normals[..num_normals] {
                let dot = projected.dot(*n);
                if dot < 0.0 {
                    projected -= *n * dot;
                }
            }

            // Derive the new constraint normal from the original (un-projected)
            // push so the wall direction stays stable across iterations.
            let scratch_len = scratch.length();
            if scratch_len > PENETRATION_EPSILON && num_normals < 3 {
                normals[num_normals] = scratch / scratch_len;
                num_normals += 1;
            }

            resolved += projected;
            total += projected;
        }

        let significant =
            had_collision && total.length_squared() > PENETRATION_EPSILON * PENETRATION_EPSILON;
        significant.then_some(total)
    }

    /// Find the single deepest-penetrating solid voxel for a vertical
    /// capsule and return the corresponding sphere-vs-AABB push-out vector.
    /// Iterates over every voxel inside the capsule's AABB; with leaf-level
    /// `is_voxel_solid` this is cheap because the sparse octree skips empty
    /// space at descent.
    fn deepest_penetration_capsule(
        &self,
        c: Vec3,
        half_height: f32,
        radius: f32,
    ) -> Option<Vec3> {
        let res = self.voxel_resolution;
        let g_min = self.grid_min;
        let radius_sq = radius * radius;

        let seg_bottom_y = c.y - half_height;
        let seg_top_y = c.y + half_height;

        let ix_min = ((c.x - radius - g_min.x) / res).floor() as i32;
        let iy_min = ((seg_bottom_y - radius - g_min.y) / res).floor() as i32;
        let iz_min = ((c.z - radius - g_min.z) / res).floor() as i32;
        let ix_max = ((c.x + radius - g_min.x) / res).floor() as i32;
        let iy_max = ((seg_top_y + radius - g_min.y) / res).floor() as i32;
        let iz_max = ((c.z + radius - g_min.z) / res).floor() as i32;

        let mut best_push = Vec3::ZERO;
        let mut best_pen = PENETRATION_EPSILON;
        let mut found = false;

        for iz in iz_min..=iz_max {
            for iy in iy_min..=iy_max {
                for ix in ix_min..=ix_max {
                    if !self.is_voxel_solid(ix as i64, iy as i64, iz as i64) {
                        continue;
                    }
                    let v_min = g_min + Vec3::new(ix as f32, iy as f32, iz as f32) * res;
                    let v_max = v_min + Vec3::splat(res);

                    // Closest point on the vertical segment to this AABB. Since
                    // the segment varies only in Y, clamp the AABB's Y center
                    // against the segment's [bottom, top] range.
                    let seg_y = if seg_top_y < v_min.y {
                        seg_top_y
                    } else if seg_bottom_y > v_max.y {
                        seg_bottom_y
                    } else {
                        let aabb_center_y = (v_min.y + v_max.y) * 0.5;
                        aabb_center_y.clamp(seg_bottom_y, seg_top_y)
                    };
                    let sphere = Vec3::new(c.x, seg_y, c.z);

                    // Nearest point on the AABB to the sphere center.
                    let near = sphere.max(v_min).min(v_max);
                    let to_sphere = sphere - near;
                    let dist_sq = to_sphere.length_squared();
                    if dist_sq >= radius_sq {
                        continue;
                    }

                    let (push, pen) = if dist_sq > 1e-12 {
                        // Sphere center outside the voxel → push radially.
                        let dist = dist_sq.sqrt();
                        let pen = radius - dist;
                        (to_sphere * (pen / dist), pen)
                    } else {
                        // Sphere center inside the voxel → push to the
                        // nearest face plus radius.
                        let neg = Vec3::new(sphere.x - v_min.x, seg_y - v_min.y, sphere.z - v_min.z);
                        let pos = Vec3::new(v_max.x - sphere.x, v_max.y - seg_y, v_max.z - sphere.z);
                        let escape = Vec3::new(
                            if neg.x < pos.x { -(neg.x + radius) } else { pos.x + radius },
                            if neg.y < pos.y { -(neg.y + radius) } else { pos.y + radius },
                            if neg.z < pos.z { -(neg.z + radius) } else { pos.z + radius },
                        );
                        let abs = escape.abs();
                        if abs.x <= abs.y && abs.x <= abs.z {
                            (Vec3::new(escape.x, 0.0, 0.0), abs.x)
                        } else if abs.y <= abs.z {
                            (Vec3::new(0.0, escape.y, 0.0), abs.y)
                        } else {
                            (Vec3::new(0.0, 0.0, escape.z), abs.z)
                        }
                    };

                    if pen > best_pen {
                        best_pen = pen;
                        best_push = push;
                        found = true;
                    }
                }
            }
        }

        found.then_some(best_push)
    }

    /// Walk the octree from the root and return whether voxel (ix, iy, iz)
    /// is solid. Out-of-grid indices return false.
    fn is_voxel_solid(&self, ix: i64, iy: i64, iz: i64) -> bool {
        if self.nodes.is_empty()
            || ix < 0
            || iy < 0
            || iz < 0
            || ix >= self.num_voxels.x as i64
            || iy >= self.num_voxels.y as i64
            || iz >= self.num_voxels.z as i64
        {
            return false;
        }

        let leaf_size = self.leaf_size as i64;
        let block_x = ix / leaf_size;
        let block_y = iy / leaf_size;
        let block_z = iz / leaf_size;

        let mut node_index: usize = 0;
        for level in (0..self.tree_depth as i64).rev() {
            let node = self.nodes[node_index];
            if node == SOLID_LEAF_MARKER {
                return true;
            }
            let child_mask = (node >> 24) & 0xFF;
            if child_mask == 0 {
                // Mixed leaf encountered above the leaf level (compacted
                // subtree). The low 24 bits index into leaf_data.
                return self.check_mixed_leaf(node, ix, iy, iz);
            }
            // Octant index from the level-th bit of each block coord
            let bit_x = ((block_x >> level) & 1) as u32;
            let bit_y = ((block_y >> level) & 1) as u32;
            let bit_z = ((block_z >> level) & 1) as u32;
            let octant = (bit_z << 2) | (bit_y << 1) | bit_x;
            if (child_mask & (1 << octant)) == 0 {
                return false;
            }
            let base_offset = (node & 0x00FF_FFFF) as usize;
            // Popcount-of-lower-bits gives the dense child index — siblings
            // are packed contiguously in BFS order.
            let prefix = (1u32 << octant) - 1;
            let child_offset = (child_mask & prefix).count_ones() as usize;
            node_index = base_offset + child_offset;
        }

        // Reached leaf level
        let node = self.nodes[node_index];
        if node == SOLID_LEAF_MARKER {
            return true;
        }
        self.check_mixed_leaf(node, ix, iy, iz)
    }

    fn check_mixed_leaf(&self, node: u32, ix: i64, iy: i64, iz: i64) -> bool {
        let leaf_data_index = (node & 0x00FF_FFFF) as usize;
        // Bit packing: a 4×4×4 block stored as 64 bits, indexed by
        // z*16 + y*4 + x, lo word holds bits 0..32, hi word holds 32..64.
        let vx = (ix & 3) as u32;
        let vy = (iy & 3) as u32;
        let vz = (iz & 3) as u32;
        let bit = vz * 16 + vy * 4 + vx;
        let lo = self.leaf_data[leaf_data_index * 2];
        let hi = self.leaf_data[leaf_data_index * 2 + 1];
        if bit < 32 {
            (lo >> bit) & 1 == 1
        } else {
            (hi >> (bit - 32)) & 1 == 1
        }
    }
}

fn sign_i(v: f32) -> i32 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}
