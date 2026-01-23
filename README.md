# The Grimoire Oracle
An Oracle for OSE powered by Langchain and a local LLM

## Prerequisites

### Ollama

Can be installed at [ollama.com](https://ollama.com/)

Make sure to launch the Ollama app manually first. Ollama installs the CLI tools upon first launch.

Run the setup script `./scripts/setup.sh` to install the Ollama models

## Using the Grimoire

After cloning, you need to build the vector index from the vault data. All commands must be run from the project root directory:

```bash
npm install
npx tsx scripts/ingest.ts
```

This creates the `grimoire_index/` folder containing the searchable vector database. You only need to run this once, or again if you update the files in `vault/`.
