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
    inputs.nixpkgs.follows = "nixpkgs";
    inputs.mach-nix.follows = "mach-nix";
  };
  inputs.mach-nix = {
    url = "github:DavHau/mach-nix";
    inputs.nixpkgs.follows = "nixpkgs";
    inputs.pypi-deps-db.follows = "pypi-deps-db";
    inputs.flake-utils.follows = "flake-utils";

  };

  outputs = { self, nixpkgs, mach-nix, flake-utils, pypi-deps-db, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ]
      (system:
        let
          name = "DeepSpectrumLite";
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
          developmentEnv = mach-nix.lib.${system}.buildPythonPackage {
            python = "python38";
            providers.soundfile = "nixpkgs";
            src = ./.;
            requirements = builtins.readFile ./requirements-test.txt;
          };
          package = mach-nix.lib.${system}.buildPythonApplication {
            pname = "deepspectrumlite";
            python = "python38";
            src = ./.;
            providers.soundfile = "nixpkgs";
            requirements = builtins.readFile ./requirements.txt;

            postInstall = ''
              wrapProgram $out/bin/deepspectrumlite --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath missingLibs}:${pkgs.cudaPackages.cudatoolkit_11_2}/lib"
            '';
            meta = with pkgs.lib; {
              description = "DeepSpectrumLite";
              homepage = "https://github.com/DeepSpectrum/DeepSpectrumLite";
              #maintainers = [ maintainers.mauricege ];
              license = licenses.gpl3;
              platforms = platforms.linux;
              inherit version;
            };
          };
        in
        rec {
          packages = flake-utils.lib.flattenTree {
            deepspectrumlite = package;
          };
          defaultPackage = packages.deepspectrumlite;
          devShell = pkgs.mkShell {
            name = "${name}Dev";
            shellHook = ''
              export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath missingLibs}:${pkgs.cudaPackages.cudatoolkit_11_2}/lib:$LD_LIBRARY_PATH";
              unset SOURCE_DATE_EPOCH
            '';
            buildInputs = [
              developmentEnv
            ];
          };
          apps.deepspectrumlite = flake-utils.lib.mkApp { drv = packages.deepspectrumlite; };
          defaultApp = apps.deepspectrumlite;
          overlay = final: prev: {
            deepspectrumlite = packages.deepspectrumlite;
          };
        }
      );
}
