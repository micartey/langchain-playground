# langchain playground

A simple langchain example collection using:

- `ollama` as api to use llm modes
- `chroma` for storing vector embeddings
- `DirectoryLoader` load txt files from sub-directory from _/docs_
- `SemanticChunker` to split texts into semantic chunks

And the following ollama models:

- `llama2`
- `mxbai-embed-large`
- `mistral`

## Getting started

You should have [nix](https://nixos.org/download/) installed.
Afterwards, you can simply execute:

```shell
nix develop
```

### Synposis

```
├── docs
│   └── example text files
├── src
│   ├── agent
│   │   └── tool calling, conditional rag
│   ├── evaluation
│   │   └── deepeval, cosine simularity
│   └── rag
│   │   └── storing data to vector db, rag example
│   └── visualize.py
├── pypi
│   └── custom packages
```
