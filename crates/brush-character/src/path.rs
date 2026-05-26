//! Per-NPC scripted paths. Returns `(position, heading)` at time `t`.
//! Headings are yaw in degrees around +Y, matching scene.json's camera
//! `ypr_deg` convention so debugging is consistent.

use glam::Vec3;

/// All NPCs share this trait so the per-frame loop is type-agnostic.
/// At the moment only [`LinearPath`] implements it.
pub trait Path: Send + Sync {
    /// World-space position at time `t` (seconds).
    fn position(&self, t: f32) -> Vec3;
    /// Yaw (degrees, around +Y) the NPC should face at time `t`.
    fn heading_deg(&self, t: f32) -> f32;
    /// Speed magnitude at time `t`; drives the walk-animation playback
    /// rate so faster movement = faster footfalls.
    fn speed(&self, t: f32) -> f32;
    /// Duration after which the path loops. NPCs with `duration == 0`
    /// are static at `position(0)`.
    fn duration(&self) -> f32;
}

/// Straight-line, constant-velocity path from `start` to `end` over
/// `duration_s`. Loops by wrapping `t` modulo duration.
#[derive(Debug, Clone)]
pub struct LinearPath {
    pub start: Vec3,
    pub end: Vec3,
    pub duration_s: f32,
}

impl LinearPath {
    /// Triangle-wave alpha in [0,1] over a period of `2 * duration_s`:
    /// alpha climbs 0→1 in the first half, then 1→0 in the second.
    /// Ping-ponging like this avoids the visual "teleport" you get
    /// when the loop wraps from `end` straight back to `start`.
    fn alpha(&self, t: f32) -> f32 {
        if self.duration_s <= 0.0 {
            return 0.0;
        }
        let phase = t.rem_euclid(2.0 * self.duration_s) / self.duration_s;
        if phase <= 1.0 { phase } else { 2.0 - phase }
    }

    /// True when the character is currently walking start→end, false
    /// when it's on the return leg end→start. Used to flip the
    /// heading mid-cycle so the character always faces its motion.
    fn going_forward(&self, t: f32) -> bool {
        if self.duration_s <= 0.0 {
            return true;
        }
        let phase = t.rem_euclid(2.0 * self.duration_s) / self.duration_s;
        phase <= 1.0
    }
}

impl Path for LinearPath {
    fn position(&self, t: f32) -> Vec3 {
        self.start.lerp(self.end, self.alpha(t))
    }

    fn heading_deg(&self, t: f32) -> f32 {
        // After the model-matrix X-flip, the glTF mesh's local +Z
        // direction lands on world -Z — i.e. at yaw=0 the character
        // faces world -Z (back to the +Z viewer). To make the character
        // walk facing its motion direction:
        //   1. Flip direction on the ping-pong return leg so we keep
        //      facing forward both ways.
        //   2. Rotate by atan2(dx, dz) + 180° so the character's
        //      facing direction (-Z at yaw=0) maps to the motion
        //      vector.
        let dir = self.end - self.start;
        if dir.length_squared() < 1e-12 {
            return 0.0;
        }
        let signed = if self.going_forward(t) { dir } else { -dir };
        signed.x.atan2(signed.z).to_degrees() + 180.0
    }

    fn speed(&self, _t: f32) -> f32 {
        if self.duration_s <= 0.0 {
            0.0
        } else {
            (self.end - self.start).length() / self.duration_s
        }
    }

    fn duration(&self) -> f32 {
        self.duration_s
    }
}
