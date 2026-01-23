# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This is a learning project for understanding LangChain and RAG concepts. The app itself (a TTRPG rule lookup assistant) is just a vehicle for learning—the primary goal is to understand how LangChain works, not to build a polished product.

## Overview

The Grimoire Oracle is a TTRPG AI assistant that provides rule lookups using local LLMs and RAG (Retrieval Augmented Generation). It uses LangChain with Ollama for AI, and React/Ink for a terminal UI.

## Prerequisites

- Node.js 18+ (project uses 25.4.0)
- Ollama installed and running with models: `llama3` (LLM) and `nomic-embed-text` (embeddings)

## Commands

```bash
./scripts/setup.sh       # Pull required Ollama models (first-time setup)
npx tsx ingest.ts        # Build vector index from vault markdown files
npx tsx src/index.tsx    # Launch terminal UI
```

Run TypeScript files directly with tsx (no build step). No test or lint commands are currently configured.

## Architecture

**Three-layer design:**

1. **Data Layer (Vault)** - `vault/` contains markdown files with OSE TTRPG rules, organized by category (Characters, Classes, Running Adventures)

2. **Ingestion Layer** - `ingest.ts` converts vault markdown into a searchable vector index stored at `./grimoire_index` using HNSWLib and Ollama embeddings

3. **AI Logic Layer** - `oracle-logic.ts` implements history-aware RAG using LangChain chains:
   - `createHistoryAwareRetriever` - rephrases follow-up questions using chat history
   - `createStuffDocumentsChain` - synthesizes retrieved documents into responses
   - `createRetrievalChain` - orchestrates the full pipeline

4. **UI Layer** - `src/index.tsx` is a React/Ink terminal interface with chat history and loading states

## Key Dependencies

- `langchain`, `@langchain/ollama`, `@langchain/community` - AI orchestration
- `hnswlib-node` - Vector store for embeddings
- `ink`, `react` - Terminal UI framework
