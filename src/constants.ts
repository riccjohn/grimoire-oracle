// Model configuration
export const LLM_MODEL = "llama3";
export const EMBEDDINGS_MODEL = "nomic-embed-text";
export const LLM_TEMPERATURE = 0.2;
export const REPHRASE_TEMPERATURE = 0;

// Retrieval tuning
export const RETRIEVAL_K = 8;

// NOTE: VECTOR_RETRIEVER_WEIGHT + BM25_RETRIEVER_WEIGHT should equal 1.0
export const VECTOR_RETRIEVER_WEIGHT = 0.5;
export const BM25_RETRIEVER_WEIGHT = 0.5;

// Paths
export const VAULT_PATH = "./vault";
export const GRIMOIRE_INDEX_PATH = "./grimoire_index";
export const GRIMOIRE_CHUNKS_PATH = "./grimoire_index/grimoire_chunks.json";

// Ingestion
export const CHUNK_SIZE = 1000;
export const CHUNK_OVERLAP = 100;
export const MIN_CHUNK_SIZE = 100;
