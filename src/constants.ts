// Number of documents to retrieve from each retriever before combining via ensemble
export const RETRIEVAL_K = 5;

export const TEMPERATURE = 0.2;
export const CHAT_MODEL = 'llama3';
export const EMBEDDING_MODEL = 'nomic-embed-text';

// Hybrid search weights: EnsembleRetriever combines results from both retrievers.
// These weights determine how much each retriever's results are prioritized.
// 0.5 / 0.5 = equal importance for semantic (vector) and keyword (BM25) matching
// Adjust these if semantic or keyword search is underperforming
export const VECTOR_RETRIEVER_WEIGHT = 0.5;
export const BM25_RETRIEVER_WEIGHT = 0.5;
