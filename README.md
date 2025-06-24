# langchain rag playground

A simple rag example in langchain using:

- `ollama` as api to use llm modes
- `chroma` for storing vector embeddings
- `DirectoryLoader` load txt files from sub-directory
- `SemanticChunker` to split texts into semantic chunks

And the following ollama models:

- `llama2`
- `mxbai-embed-large`

## Getting started

You should have [nix](https://nixos.org/download/) installed.
Afterwards, you can simply execute:

```shell
nix develop
```

### Creating Embeddings

```shell
just index
```

### Query Content

```shell
just rag "Some prompt to the llm"
```
