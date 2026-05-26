{
  description = "Brush — Gaussian splatting in Rust/wgpu";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, fenix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        rust = fenix.packages.${system}.stable.toolchain;
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            rust
            pkgs.git
            pkgs.pkg-config
            pkgs.cmake
            pkgs.just
          ];

          shellHook = ''
            # libgit2's SSH support trips on macOS keychain; force cargo to
            # shell out to the git CLI for crate fetches.
            export CARGO_NET_GIT_FETCH_WITH_CLI=true
          '';
        };
      });
}
