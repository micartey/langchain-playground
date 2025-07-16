{ pkgs, ... }:

let
  python = pkgs.python3;
in
python.pkgs.buildPythonPackage rec {
  pname = "deepeval";
  version = "3.2.6";
  format = "wheel";

  src = pkgs.fetchurl {
    url = "https://files.pythonhosted.org/packages/da/4c/cbb6bdba051bd8eecff45455e887eb421d6d0af139aaf197a0586ce3f848/deepeval-3.2.6-py3-none-any.whl";
    sha256 = "d43a0637e926fdaf17ba0e8afc4a6681603215a1ad5281b048b719bad209010d";
  };

  # Runtime dependencies are still required for the Nix environment
  propagatedBuildInputs = with python.pkgs; [
    click
    tqdm
    typer
    ollama
    aiohttp
    anthropic
    pytest
    portalocker
    openai
    google-genai
    sentry-sdk
  ];

  doCheck = false;
}
