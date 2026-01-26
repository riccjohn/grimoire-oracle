import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import {
  RunnablePassthrough,
  RunnableSequence,
} from '@langchain/core/runnables';
import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { createStuffDocumentsChain } from '@langchain/classic/chains/combine_documents';
import { createHistoryAwareRetriever } from '@langchain/classic/chains/history_aware_retriever';
import { EnsembleRetriever } from '@langchain/classic/retrievers/ensemble';
import { BM25Retriever } from '@langchain/community/retrievers/bm25';
import { readFile } from 'node:fs/promises';
import type { BaseMessage } from '@langchain/core/messages';
import { Document } from '@langchain/core/documents';

const RETRIEVAL_K = 3;
const VECTOR_RETRIEVER_WEIGHT = 0.5;
const BM25_RETRIEVER_WEIGHT = 0.5;

type SerializedChunk = {
  pageContent: string;
  metadata: Record<string, unknown>;
};

type OracleOptions = {
  debug?: boolean;
};

/**
 * Sets up the conversational RAG (Retrieval Augmented Generation) chain.
 *
 * @param options.debug - Enable debug logging to see retrieved documents
 *
 * Returns a chain that accepts { input: string, chat_history: BaseMessage[] }
 * and returns { input, chat_history, context: Document[], answer: string }
 */
export const setupOracle = async (options: OracleOptions = {}) => {
  const { debug = false } = options;

  const debugLog = (...args: unknown[]) => {
    if (debug) console.log('[DEBUG]', ...args);
  };

  // --- Setup Core Components ---
  // LLM for chat, embeddings for vector search, vector store for semantic retrieval
  const model = new ChatOllama({ model: 'llama3', temperature: 0.2 });
  const embedder = new OllamaEmbeddings({ model: 'nomic-embed-text' });
  const vectorStore = await HNSWLib.load('./grimoire_index', embedder);

  // --- Setup Hybrid Search (Vector + Keyword) ---
  // Load chunks and create BM25 keyword retriever for exact term matching
  const chunksData: SerializedChunk[] = JSON.parse(
    await readFile('./grimoire_index/grimoire_chunks.json', 'utf-8'),
  );

  const chunks = chunksData.map(
    (chunk) =>
      new Document({
        pageContent: chunk.pageContent,
        metadata: chunk.metadata,
      }),
  );

  const bm25Retriever = BM25Retriever.fromDocuments(chunks, { k: RETRIEVAL_K });
  const vectorRetriever = vectorStore.asRetriever(RETRIEVAL_K);

  // Combine vector (semantic) and BM25 (keyword) retrievers using weighted RRF
  // This boosts exact term matches (like "Thief class") while preserving semantic understanding
  const ensembleRetriever = new EnsembleRetriever({
    retrievers: [vectorRetriever, bm25Retriever],
    weights: [VECTOR_RETRIEVER_WEIGHT, BM25_RETRIEVER_WEIGHT],
  });

  // --- History-Aware Retriever ---
  // Problem: If user asks "What about for elves?" after asking about dwarves,
  // a naive retriever would search for "What about for elves?" which lacks context.
  //
  // Solution: This retriever uses the LLM to rephrase follow-up questions into
  // standalone queries. "What about for elves?" becomes "elf race abilities" or similar.
  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm: model,
    retriever: ensembleRetriever,
    rephrasePrompt: ChatPromptTemplate.fromMessages([
      new MessagesPlaceholder('chat_history'),
      ['human', '{input}'],
      [
        'human',
        'Given the conversation above, generate a search query to find relevant rules',
      ],
    ]),
  });

  // --- Answer Generation Prompt ---
  // This prompt template is used by the answer chain to generate responses.
  // The {context} placeholder will be filled with retrieved documents.
  const prompt = ChatPromptTemplate.fromMessages([
    [
      'system',
      `You are the Grimoire Oracle, a TTRPG rules assistant. Answer questions using ONLY the context provided below.

IMPORTANT: If the context does not contain the answer, say "I couldn't find that information in the rules." Do NOT make up or invent any rules, numbers, or game mechanics.

Context:
{context}`,
    ],
    new MessagesPlaceholder('chat_history'),
    ['human', '{input}'],
  ]);

  // --- Answer Chain ---
  // "Stuff documents chain" = takes an array of Documents, concatenates ("stuffs")
  // their content into the {context} placeholder, then sends to the LLM.
  // (Alternative strategies exist for many documents: map-reduce, refine, etc.)
  const answerChain = await createStuffDocumentsChain({
    llm: model,
    prompt,
  });

  // --- Compose the Full Pipeline ---
  // We manually compose the chain because createRetrievalChain doesn't pass
  // chat_history to the retriever (it predates history-aware patterns).
  //
  // RunnableSequence: runs steps in order, piping output to input
  // RunnablePassthrough.assign: passes input through, adding new keys
  //
  // Flow:
  //   { input, chat_history }
  //     → assign context (via historyAwareRetriever → ensembleRetriever → hybrid search)
  //   { input, chat_history, context }
  //     → assign answer (via answerChain)
  //   { input, chat_history, context, answer }
  return RunnableSequence.from([
    // Retrieve relevant documents using hybrid search (vector + keyword)
    // The historyAwareRetriever rephrases queries, then ensembleRetriever combines both search methods
    RunnablePassthrough.assign({
      context: async (input: {
        input: string;
        chat_history: BaseMessage[];
      }) => {
        debugLog('Input query:', input.input);
        debugLog('Chat history length:', input.chat_history.length);

        const docs: Document[] = await historyAwareRetriever.invoke(input);

        debugLog(`Retrieved ${docs.length} documents:`);
        docs.forEach((doc, i) => {
          debugLog(`  [${i + 1}] ${doc.metadata.source}`);
          debugLog(`      "${doc.pageContent.slice(0, 100)}..."`);
        });

        return docs;
      },
    }),
    // Generate answer using retrieved context
    RunnablePassthrough.assign({
      answer: answerChain,
    }),
  ]);
};
