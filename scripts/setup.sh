#!/usr/bin/env bash
set -e

echo "==> Checking for Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Error: Ollama is not installed. Install it from https://ollama.com/"
    exit 1
fi

echo "==> Pulling llama3 model (LLM)..."
if ! ollama pull llama3; then
    echo "Error: Failed to pull llama3 model"
    exit 1
fi

echo "==> Pulling nomic-embed-text model (embeddings)..."
if ! ollama pull nomic-embed-text; then
    echo "Error: Failed to pull nomic-embed-text model"
    exit 1
fi

echo "==> Setup complete! Both models are ready."
