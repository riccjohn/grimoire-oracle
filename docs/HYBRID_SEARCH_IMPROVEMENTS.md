# Hybrid Search Implementation: Results & Improvements

## Problem Statement

Our TTRPG rule lookup system (Grimoire Oracle) uses RAG (Retrieval Augmented Generation) to answer questions about game rules. Initially, we used **pure vector search** with semantic embeddings (`nomic-embed-text`).

**The Issue:** Semantic search captures meaning, not exact keywords. When users asked "Tell me about the Thief class", the system returned:

- Halfling monsters (#1)
- Treasure tables (#2-3)
- **Thief class document (#41)** ❌

The Thief class document was mostly stat tables, so it ranked poorly despite being the exact match. Semantic search treated "Thief" as a general concept (someone who steals) rather than a proper noun (the character class name).

## Solution: Hybrid Search

We implemented **hybrid search** using LangChain's `EnsembleRetriever`, which combines:

1. **Vector Search** (semantic similarity) - Finds conceptually similar content
2. **BM25 Keyword Search** - Boosts exact term matches

The ensemble uses **Reciprocal Rank Fusion (RRF)** to merge and re-rank results from both retrievers.

## Implementation

- **Ingestion:** Save processed document chunks to JSON during indexing
- **Runtime:** Load chunks, create BM25 retriever, combine with vector retriever
- **Weights:** 50/50 split (configurable: `VECTOR_RETRIEVER_WEIGHT` / `BM25_RETRIEVER_WEIGHT`)

## Results

### Query: "Tell me about the Thief class"

**Before (Vector-only):**

- Thief class document ranked **#41** (not retrieved with K=3)
- Top results: Halfling monsters, treasure tables, character creation

**After (Hybrid search):**

- Thief class document ranked **#4** ✅
- Appears in top results, correctly retrieved
- System provides accurate information about Thief skills and abilities

### Key Improvements

1. **Exact matches boosted** - Class names, spell names, and specific terms now rank higher
2. **Semantic understanding preserved** - Paraphrased queries still work ("What skills do they have?" after asking about Thief)
3. **Better precision** - Relevant documents appear in top results instead of being buried

## Technical Details

- **Vector Store:** HNSWLib with Ollama embeddings (`nomic-embed-text`)
- **Keyword Search:** BM25 retriever from `@langchain/community`
- **Ensemble:** LangChain `EnsembleRetriever` with weighted RRF
- **History-aware:** Wrapped in `createHistoryAwareRetriever` for conversational context

## Configuration

Current settings (in `src/oracle-logic.ts`):

- `RETRIEVAL_K = 3` - Documents retrieved per query
- `VECTOR_RETRIEVER_WEIGHT = 0.5` - Vector search weight
- `BM25_RETRIEVER_WEIGHT = 0.5` - Keyword search weight

These can be tuned based on use case—more keyword weight helps with exact term matching, more vector weight helps with paraphrased queries.
