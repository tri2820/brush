//! Load a `.glb` file into the CPU-side data we need for skinned rendering
//! and animation. We only care about the *first* skinned primitive in the
//! file — these characters typically ship with one mesh, one skin, and
//! one or more animations.

use anyhow::{Result, anyhow};
use glam::{Mat4, Quat, Vec3};
use std::path::Path;

/// One skinned mesh worth of CPU-side data, plus its skeleton and the
/// animation tracks bound to it.
#[derive(Debug)]
pub struct MeshAsset {
    /// Per-vertex position (object-local).
    pub positions: Vec<Vec3>,
    /// Per-vertex normal (object-local).
    pub normals: Vec<Vec3>,
    /// Per-vertex UV coordinates (TEXCOORD_0). Zero-filled if absent.
    pub texcoords: Vec<[f32; 2]>,
    /// Per-vertex tangents. xyz is the tangent direction (object-local),
    /// w is the bitangent sign (+1 or -1). When the glTF doesn't ship
    /// TANGENT, we generate them with the MikkTSpace algorithm so that
    /// normal-map sampling matches the convention DCC tools (Blender,
    /// Substance) and Mixamo expect.
    pub tangents: Vec<[f32; 4]>,
    /// Per-vertex 4 joint indices (into `skeleton.joints`).
    pub joints: Vec<[u16; 4]>,
    /// Per-vertex 4 joint weights, expected to sum to ~1.
    pub weights: Vec<[f32; 4]>,
    /// Triangle list, indexing into `positions`/`normals`/etc.
    pub indices: Vec<u32>,
    /// Joint hierarchy with inverse bind matrices.
    pub skeleton: Skeleton,
    /// Animations keyed by name.
    pub animations: Vec<Animation>,
    /// Material data, ready for GPU upload.
    pub material: Material,
    /// Source filename, for diagnostics.
    pub source: String,
}

/// CPU-side material data for the primitive. Currently extracts the
/// PBR metallic-roughness channels we care about. Texture image data
/// is decoded into row-major RGBA8.
#[derive(Debug, Clone)]
pub struct Material {
    /// sRGB-encoded RGBA. None means "no texture, use base_color_factor".
    pub base_color: Option<TextureImage>,
    /// linear RGBA, normal map in tangent space (xyz). None means flat.
    pub normal: Option<TextureImage>,
    /// Per-material linear-RGBA factor multiplied with the baseColor
    /// sample (or used directly when no texture is set).
    pub base_color_factor: [f32; 4],
    /// Scale factor on the normal map's xy delta (typically 1.0).
    pub normal_scale: f32,
}

#[derive(Debug, Clone)]
pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    /// Row-major, 4 bytes per pixel (RGBA8). For sRGB textures these
    /// bytes are sRGB-encoded; upload as `Rgba8UnormSrgb` to get
    /// hardware linearization on sample. For linear textures (normal,
    /// MR), upload as `Rgba8Unorm`.
    pub rgba: Vec<u8>,
}

/// Joint hierarchy. Joint index 0..N-1 is "joint N"; `parents[i]` is
/// either Some(parent index) or None for root joints.
#[derive(Debug, Clone)]
pub struct Skeleton {
    pub joints: Vec<Joint>,
    /// Cumulative transform of ancestor nodes above the topmost joint
    /// in the glTF scene graph (e.g. the "Armature" node Mixamo emits,
    /// which carries the cm→m scale and a pre-rotation). The skin's
    /// `inverseBindMatrices` are computed against the *global* bind
    /// pose per the glTF spec, so we have to multiply this in when
    /// composing the per-frame global transform of root joints.
    pub armature: Mat4,
}

#[derive(Debug, Clone)]
pub struct Joint {
    /// Human-readable name from the glTF node, for debugging.
    pub name: String,
    /// Index of this joint's parent in `Skeleton.joints`, or None for roots.
    pub parent: Option<u16>,
    /// Local-space TRS at the bind pose (also used as the default
    /// when an animation doesn't drive every channel of this joint).
    pub bind_translation: Vec3,
    pub bind_rotation: Quat,
    pub bind_scale: Vec3,
    /// `inverse(bind_global_transform)`. Used by skinning: a vertex in
    /// joint-local space is `inverse_bind * world_vertex`, after which
    /// applying the *current* joint global transform brings it back into
    /// world animated by that joint's motion.
    pub inverse_bind: Mat4,
}

#[derive(Debug, Clone)]
pub struct Animation {
    pub name: String,
    /// Total duration in seconds.
    pub duration: f32,
    /// Per-target-joint TRS tracks. One entry per affected joint;
    /// joints not listed here keep their bind pose.
    pub channels: Vec<JointChannel>,
}

/// Per-joint animation tracks. Any of T/R/S can be absent (None);
/// missing components fall back to the joint's bind values.
#[derive(Debug, Clone)]
pub struct JointChannel {
    pub joint: u16,
    pub translation: Option<Track<Vec3>>,
    pub rotation: Option<Track<Quat>>,
    pub scale: Option<Track<Vec3>>,
}

/// A keyframe track. Linear interpolation only — STEP / CUBICSPLINE
/// fall back to nearest-key for now (the character glb here uses LINEAR).
#[derive(Debug, Clone)]
pub struct Track<V> {
    pub times: Vec<f32>,
    pub values: Vec<V>,
}

impl<V: Copy> Track<V> {
    pub fn sample<I: Fn(V, V, f32) -> V>(&self, t: f32, interp: I) -> V {
        if self.times.is_empty() {
            panic!("empty track");
        }
        if t <= self.times[0] {
            return self.values[0];
        }
        let last = self.times.len() - 1;
        if t >= self.times[last] {
            return self.values[last];
        }
        // Binary search for the keyframe segment containing `t`.
        let idx = self.times.partition_point(|&k| k <= t) - 1;
        let t0 = self.times[idx];
        let t1 = self.times[idx + 1];
        let alpha = (t - t0) / (t1 - t0);
        interp(self.values[idx], self.values[idx + 1], alpha)
    }
}

pub fn load_mesh(path: &Path) -> Result<MeshAsset> {
    let (doc, buffers, images) = gltf::import(path)?;
    let source = path.display().to_string();

    // Pick the first skinned mesh primitive in the document.
    let (mesh, prim, skin) = doc
        .meshes()
        .flat_map(|m| {
            m.primitives()
                .map(move |p| (m.clone(), p))
        })
        .find_map(|(m, p)| {
            // Find a node that uses this mesh AND has a skin.
            doc.nodes().find_map(|n| {
                if n.mesh().is_some_and(|nm| nm.index() == m.index())
                    && let Some(sk) = n.skin()
                {
                    Some((m.clone(), p.clone(), sk))
                } else {
                    None
                }
            })
        })
        .ok_or_else(|| anyhow!("no skinned primitive found in {}", source))?;

    let reader = prim.reader(|buf| Some(&buffers[buf.index()]));

    let positions: Vec<Vec3> = reader
        .read_positions()
        .ok_or_else(|| anyhow!("primitive missing POSITION attribute"))?
        .map(Vec3::from)
        .collect();

    let normals: Vec<Vec3> = if let Some(it) = reader.read_normals() {
        it.map(Vec3::from).collect()
    } else {
        // Synthesize flat normals from indices if absent.
        vec![Vec3::Y; positions.len()]
    };

    let joints: Vec<[u16; 4]> = reader
        .read_joints(0)
        .ok_or_else(|| anyhow!("primitive missing JOINTS_0"))?
        .into_u16()
        .collect();

    let weights: Vec<[f32; 4]> = reader
        .read_weights(0)
        .ok_or_else(|| anyhow!("primitive missing WEIGHTS_0"))?
        .into_f32()
        .collect();

    let indices: Vec<u32> = reader
        .read_indices()
        .ok_or_else(|| anyhow!("primitive missing indices"))?
        .into_u32()
        .collect();

    // UVs: required for sampling material textures. Zero-fill if the
    // glTF omits them so the rest of the pipeline still works.
    let texcoords: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
        Some(it) => it.into_f32().collect(),
        None => vec![[0.0, 0.0]; positions.len()],
    };

    // Tangents: glTF can provide TANGENT explicitly. Mixamo typically
    // doesn't ship them, so we'll either read them or generate via
    // MikkTSpace below.
    let tangents: Vec<[f32; 4]> = if let Some(it) = reader.read_tangents() {
        it.collect()
    } else {
        generate_tangents(&positions, &normals, &texcoords, &indices)
    };

    // Material: extract the PBR baseColor + normal channels we render.
    let material_data = prim.material();
    let pbr = material_data.pbr_metallic_roughness();
    let base_color_factor = pbr.base_color_factor();
    let base_color = pbr.base_color_texture().and_then(|info| {
        decode_image(&doc, &images, info.texture().source().index())
    });
    let (normal, normal_scale) = match material_data.normal_texture() {
        Some(info) => (
            decode_image(&doc, &images, info.texture().source().index()),
            info.scale(),
        ),
        None => (None, 1.0),
    };

    let material = Material {
        base_color,
        normal,
        base_color_factor,
        normal_scale,
    };

    // Skeleton: enumerate the skin's joints, capture each one's bind
    // TRS and inverse-bind matrix, and resolve parent pointers among
    // them.
    let skin_joints: Vec<gltf::Node> = skin.joints().collect();
    let joint_index_of: std::collections::HashMap<usize, u16> = skin_joints
        .iter()
        .enumerate()
        .map(|(i, n)| (n.index(), i as u16))
        .collect();

    let inverse_binds: Vec<Mat4> = reader_inverse_binds(&skin, &buffers, skin_joints.len());

    let mut skeleton_joints = Vec::with_capacity(skin_joints.len());
    for (i, node) in skin_joints.iter().enumerate() {
        let (t, r, s) = node.transform().decomposed();
        let parent_node = doc
            .nodes()
            .find(|n| n.children().any(|c| c.index() == node.index()));
        let parent = parent_node.and_then(|p| joint_index_of.get(&p.index())).copied();
        skeleton_joints.push(Joint {
            name: node.name().unwrap_or("").to_string(),
            parent,
            bind_translation: Vec3::from(t),
            bind_rotation: Quat::from_array(r),
            bind_scale: Vec3::from(s),
            inverse_bind: inverse_binds[i],
        });
    }

    // Animations: each animation has multiple channels, each channel
    // drives one TRS component of one node. Collapse same-joint
    // channels into JointChannel entries.
    let mut animations = Vec::with_capacity(doc.animations().count());
    for anim in doc.animations() {
        let name = anim.name().unwrap_or("(unnamed)").to_string();
        let mut by_joint: std::collections::HashMap<u16, JointChannel> =
            std::collections::HashMap::new();
        let mut duration: f32 = 0.0;

        for ch in anim.channels() {
            let target_node_index = ch.target().node().index();
            let Some(&joint) = joint_index_of.get(&target_node_index) else {
                // Animation targets a node outside the skin (e.g. root
                // motion node). Ignore for now.
                continue;
            };
            let r = ch.reader(|buf| Some(&buffers[buf.index()]));
            let times: Vec<f32> = r
                .read_inputs()
                .ok_or_else(|| anyhow!("animation channel missing input times"))?
                .collect();
            if let Some(&t) = times.last() {
                duration = duration.max(t);
            }
            let entry = by_joint.entry(joint).or_insert_with(|| JointChannel {
                joint,
                translation: None,
                rotation: None,
                scale: None,
            });
            match r.read_outputs().ok_or_else(|| anyhow!("animation channel missing output"))? {
                gltf::animation::util::ReadOutputs::Translations(it) => {
                    entry.translation = Some(Track {
                        times,
                        values: it.map(Vec3::from).collect(),
                    });
                }
                gltf::animation::util::ReadOutputs::Rotations(rot) => {
                    let values: Vec<Quat> = rot.into_f32().map(Quat::from_array).collect();
                    entry.rotation = Some(Track { times, values });
                }
                gltf::animation::util::ReadOutputs::Scales(it) => {
                    entry.scale = Some(Track {
                        times,
                        values: it.map(Vec3::from).collect(),
                    });
                }
                gltf::animation::util::ReadOutputs::MorphTargetWeights(_) => {
                    // We don't support morph targets here.
                }
            }
        }

        animations.push(Animation {
            name,
            duration,
            channels: by_joint.into_values().collect(),
        });
    }

    // Diagnostic: trace the parent chain above the root joint to expose
    // any ancestor nodes (e.g. the scene's "Armature" group) whose
    // transforms aren't included in our joint hierarchy. Mixamo files
    // commonly bake a cm→m scale into such an ancestor, which our
    // skinning math then needs to be aware of.
    // Walk above the topmost joint to collect the ancestor chain
    // transform (typically a single "Armature" node carrying cm→m
    // scale + pre-rotation in Mixamo exports). The skin's
    // inverseBindMatrices are computed against the *global* bind pose
    // per spec, so this transform has to be re-applied when composing
    // the global per frame.
    let armature = compute_armature_transform(&doc, &skin_joints);

    log::info!(
        "Loaded glb '{}': mesh '{}' prim {}, {} vtx, {} tri, {} joints, {} animations",
        source,
        mesh.name().unwrap_or(""),
        prim.index(),
        positions.len(),
        indices.len() / 3,
        skeleton_joints.len(),
        animations.len(),
    );

    Ok(MeshAsset {
        positions,
        normals,
        texcoords,
        tangents,
        joints,
        weights,
        indices,
        skeleton: Skeleton {
            joints: skeleton_joints,
            armature,
        },
        animations,
        material,
        source,
    })
}

/// Decode a glTF image (which gltf-rs has already extracted as raw
/// pixel data) into row-major RGBA8.
fn decode_image(
    _doc: &gltf::Document,
    images: &[gltf::image::Data],
    index: usize,
) -> Option<TextureImage> {
    let img = images.get(index)?;
    let (width, height) = (img.width, img.height);

    // gltf-rs produces these formats per the spec, depending on the
    // source (PNG/JPEG/...). Expand everything to RGBA8 since the GPU
    // upload path is uniform.
    use gltf::image::Format;
    let rgba: Vec<u8> = match img.format {
        Format::R8G8B8A8 => img.pixels.clone(),
        Format::R8G8B8 => {
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for px in img.pixels.chunks_exact(3) {
                out.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            out
        }
        Format::R8 => {
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for &g in &img.pixels {
                out.extend_from_slice(&[g, g, g, 255]);
            }
            out
        }
        Format::R8G8 => {
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for px in img.pixels.chunks_exact(2) {
                out.extend_from_slice(&[px[0], px[1], 0, 255]);
            }
            out
        }
        other => {
            log::warn!("unsupported glTF image format {other:?}, skipping");
            return None;
        }
    };
    Some(TextureImage { width, height, rgba })
}

/// Generate tangents with MikkTSpace. Required when the glb omits the
/// TANGENT attribute (Mixamo files typically do). The algorithm needs
/// a flat per-triangle-corner view of the mesh; we deinterleave once
/// here, run it, and re-aggregate into per-vertex tangents (matching
/// our other per-vertex attribute layout).
fn generate_tangents(
    positions: &[Vec3],
    normals: &[Vec3],
    texcoords: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    use mikktspace::Geometry;

    struct G<'a> {
        positions: &'a [Vec3],
        normals: &'a [Vec3],
        texcoords: &'a [[f32; 2]],
        indices: &'a [u32],
        tangents: Vec<[f32; 4]>,
    }
    impl<'a> Geometry for G<'a> {
        fn num_faces(&self) -> usize {
            self.indices.len() / 3
        }
        fn num_vertices_of_face(&self, _face: usize) -> usize {
            3
        }
        fn position(&self, face: usize, vert: usize) -> [f32; 3] {
            self.positions[self.indices[face * 3 + vert] as usize].into()
        }
        fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
            self.normals[self.indices[face * 3 + vert] as usize].into()
        }
        fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
            self.texcoords[self.indices[face * 3 + vert] as usize]
        }
        fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
            self.tangents[self.indices[face * 3 + vert] as usize] = tangent;
        }
    }

    let mut geo = G {
        positions,
        normals,
        texcoords,
        indices,
        tangents: vec![[1.0, 0.0, 0.0, 1.0]; positions.len()],
    };
    mikktspace::generate_tangents(&mut geo);
    geo.tangents
}

fn compute_armature_transform(doc: &gltf::Document, skin_joints: &[gltf::Node]) -> Mat4 {
    let Some(root_joint) = skin_joints.first() else {
        return Mat4::IDENTITY;
    };
    let mut ancestor = doc
        .nodes()
        .find(|n| n.children().any(|c| c.index() == root_joint.index()));
    let mut m = Mat4::IDENTITY;
    while let Some(n) = ancestor {
        let (t, r, s) = n.transform().decomposed();
        let local =
            Mat4::from_scale_rotation_translation(Vec3::from(s), Quat::from_array(r), Vec3::from(t));
        m = local * m;
        ancestor = doc
            .nodes()
            .find(|p| p.children().any(|c| c.index() == n.index()));
    }
    m
}

fn reader_inverse_binds(
    skin: &gltf::Skin,
    buffers: &[gltf::buffer::Data],
    n: usize,
) -> Vec<Mat4> {
    if let Some(reader) = Some(skin.reader(|buf| Some(&buffers[buf.index()])))
        && let Some(ibm) = reader.read_inverse_bind_matrices()
    {
        ibm.map(|m| Mat4::from_cols_array_2d(&m)).collect()
    } else {
        // Spec default: identity if absent.
        vec![Mat4::IDENTITY; n]
    }
}
