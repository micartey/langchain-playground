{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell rec {
  buildInputs = with pkgs; [
    gcc
    python311
    python311Packages.pip
    python311Packages.numpy
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath buildInputs}


    python -m venv .venv
    source .venv/bin/activate; pip install langchain langchain_community langchain_experimental ollama tqdm unstructured chromadb
  '';
}
