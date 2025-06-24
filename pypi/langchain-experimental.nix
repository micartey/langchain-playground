{ pkgs, ... }:

pkgs.python312.pkgs.buildPythonPackage rec {
  pname = "langchain_experimental";
  version = "0.3.4";
  format = "pyproject";

  src = pkgs.fetchPypi {
    inherit pname version;
    sha256 = "937c4259ee4a639c618d19acf0e2c5c2898ef127050346edc5655259aa281a21";
  };

  # Build-time dependencies
  nativeBuildInputs = with pkgs.python312.pkgs; [
    poetry-core
  ];

  # Runtime dependencies
  propagatedBuildInputs = with pkgs.python312Packages; [
    langchain
    langchain-core
    langchain-community
    numpy
  ];

  doCheck = false;
}
