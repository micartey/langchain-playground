{
  pkgs ? import <nixpkgs> { },
}:

let
  python = pkgs.python312;

  # Define the custom langchain-experimental package
  langchain-experimental = python.pkgs.buildPythonPackage rec {
    pname = "langchain_experimental";
    version = "0.3.4";

    src = pkgs.fetchPypi {
      inherit pname version;
      sha256 = "937c4259ee4a639c618d19acf0e2c5c2898ef127050346edc5655259aa281a21";
    };

    # List the dependencies required by langchain-experimental
    # These must already be available in nixpkgs or defined similarly.
    propagatedBuildInputs = with python.pkgs; [
      langchain
      langchain-core
      langchain-community
      numpy
      poetry-core
    ];

    doCheck = false; # Skip tests
    format = "pyproject"; # Use newer build system
  };

in
pkgs.mkShell rec {
  buildInputs = with pkgs; [
    gcc
    python312
    python312Packages.pip
    python312Packages.numpy
    python312Packages.langchain
    python312Packages.langchain-ollama
    python312Packages.langchain-chroma
    python312Packages.langchain-community
    python312Packages.tqdm
    python312Packages.unstructured

    langchain-experimental
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath buildInputs}
  '';
}
