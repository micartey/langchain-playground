{
  pkgs ? import <nixpkgs> { },
}:

let
  langchain-experimental = pkgs.callPackage ../pypi/langchain-experimental.nix {
    python = pkgs.python312;
    pkgs = pkgs;
  };

  deepeval = pkgs.callPackage ../pypi/deepeval.nix {
    python = pkgs.python312;
    pkgs = pkgs;
  };
in
pkgs.mkShell rec {
  buildInputs = with pkgs; [
    gcc
    just

    python312
    python312Packages.pip
    python312Packages.numpy
    python312Packages.langgraph
    python312Packages.langchain
    python312Packages.langchain-ollama
    python312Packages.langchain-chroma
    python312Packages.langchain-community
    python312Packages.tqdm
    python312Packages.unstructured
    python312Packages.tiktoken

    python312Packages.torchWithCuda
    python312Packages.datasets
    python312Packages.transformers
    python312Packages.evaluate
    python312Packages.hf-xet

    langchain-experimental
    deepeval

    python312Packages.sentence-transformers
    python312Packages.scikit-learn
    python312Packages.torch

    # For visualization
    python312Packages.sklearn-compat
    python312Packages.scikit-learn
    python312Packages.umap-learn
    python312Packages.plotly
    python312Packages.pandas
    python312Packages.chromadb
    python312Packages.matplotlib
  ];

  USER_AGENT = "Firefox/11.0.1"; # Probably doesn't even exist

  HF_HUB_OFFLINE = 1;

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath buildInputs}
  '';
}
