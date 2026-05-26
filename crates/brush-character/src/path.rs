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
    /// Clamped alpha in [0,1]. Character walks start→end over
    /// `duration_s` seconds, then **holds at end** — no loop, no
    /// ping-pong. The previous ping-pong implementation snapped the
    /// heading 180° at each endpoint while position kept moving,
    /// which read as a "teleport" in the recorded video. Holding at
    /// the end is the simplest behavior that always looks correct;
    /// if you want a loop later, do it with an explicit turn-around
    /// animation rather than an instantaneous yaw flip.
    fn alpha(&self, t: f32) -> f32 {
        if self.duration_s <= 0.0 {
            return 0.0;
        }
        (t / self.duration_s).clamp(0.0, 1.0)
    }
}

impl Path for LinearPath {
    fn position(&self, t: f32) -> Vec3 {
        self.start.lerp(self.end, self.alpha(t))
    }

    fn heading_deg(&self, _t: f32) -> f32 {
        // After the model-matrix X-flip, the glTF mesh's local +Z
        // direction lands on world -Z — i.e. at yaw=0 the character
        // faces world -Z (back to the +Z viewer). To make the
        // character face its motion direction, we rotate by
        // atan2(dx, dz) + 180°: the +180° re-maps the (-Z facing at
        // yaw=0) onto the motion vector.
        let dir = self.end - self.start;
        if dir.length_squared() < 1e-12 {
            return 0.0;
        }
        dir.x.atan2(dir.z).to_degrees() + 180.0
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
