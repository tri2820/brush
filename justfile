# Run with `just <recipe>`. `just` is provided by the nix devShell — if
# you don't have it, `direnv allow` in this directory or `nix develop`
# will give you everything.

# Default source / scene / cadence. Override per-invocation by passing
# `key=value` to `just`, e.g. `just frames=200 fps=30 record`.
source     := "/Users/tri/pc-refs/supersplat-viewer/scenes/604b6d93.ply"
scene      := "scene.json"
frames     := "80"
fps        := "10"
output_dir := "out/"

# Show available recipes.
default:
    @just --list

# Record clips for every camera in scene.json (defaults: 80 frames @ 10fps → out/).
record *extra:
    cargo build -p brush-app --bin brush
    ./target/debug/brush \
        {{source}} \
        --scene {{scene}} \
        --record-frames {{frames}} \
        --record-fps {{fps}} \
        --output-dir {{output_dir}} \
        {{extra}}

# Record using the optimized release binary (slower to build, faster to run).
record-release *extra:
    cargo build --release -p brush-app --bin brush
    ./target/release/brush \
        {{source}} \
        --scene {{scene}} \
        --record-frames {{frames}} \
        --record-fps {{fps}} \
        --output-dir {{output_dir}} \
        {{extra}}

# Open the interactive viewer to copy camera values into scene.json.
view source=source:
    cargo run -p brush-app --bin brush -- {{source}}

# Wipe out/ to start the next record clean.
clean:
    rm -rf {{output_dir}}
