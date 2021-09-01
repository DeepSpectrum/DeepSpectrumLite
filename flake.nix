{
  description = "A simple flake for DeepSpectrumLite";

  inputs.nixpkgs.url = "nixpkgs/release-21.05";
  inputs.flake-utils.url = "github:numtide/flake-utils";
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
    let
      name = "deepspectrumlite";
      missingLibs = pkgs: with pkgs; [
        cudatoolkit_11_2
        cudnn_cudatoolkit_11_2
        pkgs.stdenv.cc.cc
      ];
      package = { pkgs ? import <nixpkgs> }: mach-nix.lib.${pkgs.system}.buildPythonApplication
        {
          pname = "deepspectrumlite";
          python = "python38";
          src = ./.;
          providers.soundfile = "nixpkgs";
          requirements = builtins.readFile ./requirements.txt;

          postInstall = ''
            wrapProgram $out/bin/deepspectrumlite --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath (missingLibs pkgs)}:${pkgs.cudaPackages.cudatoolkit_11_2}/lib"
          '';
          meta = with pkgs.lib; {
            description = "DeepSpectrumLite";
            homepage = "https://github.com/DeepSpectrum/DeepSpectrumLite";
            license = licenses.gpl3;
            platforms = platforms.linux;
            inherit version;
          };
        };
      overlay = final: prev: {
        deepspectrumlite = {
          deepspectrumlite = package { pkgs = prev; };
          defaultPackage = package { pkgs = prev; };
        };
      };
      developmentEnv = { pkgs ? import <nixpkgs> }: mach-nix.lib.${pkgs.system}.buildPythonPackage {
        python = "python38";
        pname = name;
        providers.soundfile = "nixpkgs";
        src = ./.;
        requirements = builtins.readFile ./requirements-test.txt;
      };
      shell = { pkgs ? import <nixpkgs> }: pkgs.mkShell {
        name = "${name}-dev";
        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath (missingLibs pkgs)}:${pkgs.cudaPackages.cudatoolkit_11_2}/lib:$LD_LIBRARY_PATH";
          unset SOURCE_DATE_EPOCH
        '';
        buildInputs = [
          (developmentEnv { inherit pkgs; })
        ];
      };
    in
    flake-utils.lib.simpleFlake
      {
        inherit self nixpkgs overlay shell name;
        config = { allowUnfree = true; };
        systems = [ "x86_64-linux" ];
      };
}
