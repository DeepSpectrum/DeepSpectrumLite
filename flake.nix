{
  description = "Development environment for DS-Lite";

  inputs.nixpkgs.url = "nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.flake-compat = {
    url = "github:edolstra/flake-compat";
    flake = false;
  };
  inputs.pypi-deps-db = {
    url = "github:DavHau/pypi-deps-db";
  };
  inputs.mach-nix = {
    url = "github:DavHau/mach-nix";
    inputs.nixpkgs.follows = "nixpkgs";
    inputs.pypi-deps-db.follows = "pypi-deps-db";
    inputs.flake-utils.follows = "flake-utils";

  };

  outputs = { self, nixpkgs, mach-nix, flake-utils, pypi-deps-db, ... }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          name = "DeepSpectrumLite-Dev";
          pkgs = import nixpkgs {
            config = {
              # CUDA and other "friends" contain unfree licenses. To install them, you need this line:
              allowUnfree = true;
            };
            inherit system;
          };
          machNix38 = import mach-nix {
            python = "python38";
            inherit pkgs;
            pypiData = pypi-deps-db;
          };
          missingLibs = with pkgs; [
            cudatoolkit_11_2
            cudnn_cudatoolkit_11_2
            pkgs.stdenv.cc.cc
          ];
          pythonEnv = machNix38.mkPython rec {
            providers.soundfile = "nixpkgs";
            packagesExtra = [
              ./.
            ];
          };
        in
        rec {
          defaultPackage = pkgs.buildEnv {
            inherit name;
            paths = [
              pythonEnv
            ];
          };
          devShell = pkgs.mkShell {
            inherit name;
            shellHook = ''
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath missingLibs}:${pkgs.cudaPackages.cudatoolkit_11_2}/lib:$LD_LIBRARY_PATH";
              unset SOURCE_DATE_EPOCH
            '';
            buildInputs = [
              defaultPackage
            ];
          };
        }
      );
}
