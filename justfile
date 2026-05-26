# Run with `just <recipe>`. `just` is provided by the nix devShell — if
# you don't have it, `direnv allow` in this directory or `nix develop`
# will give you everything.

# Default scene file + record cadence. Override per-invocation by passing
# `key=value` to `just`, e.g. `just frames=200 fps=30 record`.
scene      := "scene.json"
frames     := "80"
fps        := "10"
output_dir := "out/"

# Show available recipes.
default:
    @just --list

# Open the interactive viewer with NPCs walking around the scene.
run scene=scene:
    cargo run -p brush-app --bin brush -- {{scene}}

# Same as `run` but uses the optimized release binary.
run-release scene=scene:
    cargo run --release -p brush-app --bin brush -- {{scene}}

# Open the viewer on a bare splat (no scene config, no NPCs).
view source:
    cargo run -p brush-app --bin brush -- {{source}}

# Record one mp4 per camera in scene.json (defaults: 80 frames @ 10fps → out/).
record *extra:
    cargo run -p brush-app --bin brush -- \
        {{scene}} \
        --record-frames {{frames}} \
        --record-fps {{fps}} \
        --output-dir {{output_dir}} \
        {{extra}}

# Record with the optimized release binary (slower to build, faster to run).
record-release *extra:
    cargo run --release -p brush-app --bin brush -- \
        {{scene}} \
        --record-frames {{frames}} \
        --record-fps {{fps}} \
        --output-dir {{output_dir}} \
        {{extra}}

# Write one PNG per camera in scene.json (no animation).
screenshot:
    cargo run -p brush-app --bin brush -- {{scene}} --screenshot --output-dir {{output_dir}}

# Snapshot: record a short clip through the working record pipeline (so NPCs
# are included) at a realistic fps, then extract a single PNG per camera from
# late in the clip so physics has settled. Use this to verify NPC rendering
# without depending on macOS screencapture.
snapshot:
    cargo run -p brush-app --bin brush -- \
        {{scene}} --record-frames 60 --record-fps 30 --output-dir /tmp/brush-snap
    @for f in /tmp/brush-snap/*.mp4; do \
        ffmpeg -y -loglevel error -sseof -0.1 -i $f -frames:v 1 ${f%.mp4}.png; \
        echo "wrote ${f%.mp4}.png"; \
    done

# Wipe out/ to start the next record clean.
clean:
    rm -rf {{output_dir}}
