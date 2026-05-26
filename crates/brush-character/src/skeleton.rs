//! Animation evaluation: sample TRS tracks at time `t`, walk the joint
//! hierarchy to compute global joint transforms, multiply by inverse-bind
//! to get the skinning matrices vertex shaders consume.

use glam::{Mat4, Quat, Vec3};

use crate::gltf_load::{Animation, JointChannel, Skeleton};

impl Skeleton {
    /// Compute one skinning matrix per joint: `global_animated * inverse_bind`.
    /// Pass in the active animation and a time in seconds (it will wrap
    /// modulo the animation's duration).
    pub fn evaluate(&self, animation: Option<&Animation>, t_seconds: f32) -> Vec<Mat4> {
        let n = self.joints.len();
        // 1. Sample local TRS per joint, falling back to bind values
        // where the animation doesn't drive a component (or there's no
        // active animation at all).
        let mut local = vec![Mat4::IDENTITY; n];
        for (i, joint) in self.joints.iter().enumerate() {
            let (mut translation, mut rotation, mut scale) = (
                joint.bind_translation,
                joint.bind_rotation,
                joint.bind_scale,
            );
            if let Some(anim) = animation {
                let t_wrapped = if anim.duration > 0.0 {
                    t_seconds.rem_euclid(anim.duration)
                } else {
                    0.0
                };
                if let Some(ch) = anim.channels.iter().find(|c| c.joint as usize == i) {
                    apply_channel(ch, t_wrapped, &mut translation, &mut rotation, &mut scale);
                }
            }
            local[i] = Mat4::from_scale_rotation_translation(scale, rotation, translation);
        }
        // 2. Compose into global transforms via a single forward pass —
        // joints in glTF are stored in topological order, so a parent
        // index is always < the child's own index. (We assume this; if
        // a file violates it, the global for that joint will pick up
        // last frame's parent matrix, which would manifest as visible
        // drift on that joint. Cheap to detect later if needed.)
        let mut global = vec![Mat4::IDENTITY; n];
        for i in 0..n {
            global[i] = match self.joints[i].parent {
                Some(p) => global[p as usize] * local[i],
                // Root joint: the joint's local transform composes
                // with the Armature transform sitting above the
                // topmost joint in the glTF scene graph (carries
                // cm→m scale + any pre-rotation). The skin's
                // inverseBindMatrices are computed against this
                // global pose, so we have to mirror it here.
                None => self.armature * local[i],
            };
        }
        // 3. Skinning matrix = global_animated * inverse_bind.
        global
            .iter()
            .zip(self.joints.iter())
            .map(|(g, j)| *g * j.inverse_bind)
            .collect()
    }
}

fn apply_channel(
    ch: &JointChannel,
    t: f32,
    translation: &mut Vec3,
    rotation: &mut Quat,
    scale: &mut Vec3,
) {
    if let Some(track) = &ch.translation {
        *translation = track.sample(t, |a, b, alpha| a.lerp(b, alpha));
    }
    if let Some(track) = &ch.rotation {
        *rotation = track.sample(t, |a, b, alpha| a.slerp(b, alpha));
    }
    if let Some(track) = &ch.scale {
        *scale = track.sample(t, |a, b, alpha| a.lerp(b, alpha));
    }
}
