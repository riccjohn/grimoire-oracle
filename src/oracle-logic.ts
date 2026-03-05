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
import type { BaseRetriever } from '@langchain/core/retrievers';
import {
  CHAT_MODEL,
  TEMPERATURE,
  EMBEDDING_MODEL,
  RETRIEVAL_K,
  VECTOR_RETRIEVER_WEIGHT,
  BM25_RETRIEVER_WEIGHT,
} from '@src/constants.js';

type SerializedChunk = {
  pageContent: string;
  metadata: Record<string, unknown>;
};

type HistoryAwareRetriever = Awaited<
  ReturnType<typeof createHistoryAwareRetriever>
>;
type StuffDocumentsChain = Awaited<
  ReturnType<typeof createStuffDocumentsChain>
>;

type OracleOptions = {
  debug?: boolean;
};

type CoreComponents = {
  model: ChatOllama;
  vectorStore: HNSWLib;
};

/**
 * Creates core AI components: LLM for chat, embeddings for vector search,
 * and loads the vector store for semantic retrieval.
 */
const createCoreComponents = async (): Promise<CoreComponents> => {
  try {
    const model = new ChatOllama({
      model: CHAT_MODEL,
      temperature: TEMPERATURE,
    });
    const embedder = new OllamaEmbeddings({ model: EMBEDDING_MODEL });
    const vectorStore = await HNSWLib.load('./grimoire_index', embedder);

    return { model, vectorStore };
  } catch (error) {
    throw new Error(
      `Failed to create core components: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
};

/**
 * Loads and deserializes document chunks from JSON file for BM25 keyword search.
 * These are the same chunks used during vector store creation.
 */
const loadChunksForBM25 = async (): Promise<Document[]> => {
  try {
    const chunksData: SerializedChunk[] = JSON.parse(
      await readFile('./grimoire_index/grimoire_chunks.json', 'utf-8'),
    );

    return chunksData.map((chunk) => {
      const { pageContent, metadata } = chunk;

      return new Document({
        pageContent,
        metadata,
      });
    });
  } catch (error) {
    throw new Error(
      `Failed to load chunks for BM25: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
};

/**
 * Creates a history-aware retriever that rephrases follow-up questions using
 * chat history before searching.
 *
 * Example: "What about for elves?" (after asking about dwarves) becomes
 * "elf race abilities" or similar.
 */
const wrapRetrieverWithHistoryAwareness = async (
  model: ChatOllama,
  retriever: BaseRetriever,
) => {
  return await createHistoryAwareRetriever({
    llm: model,
    retriever,
    rephrasePrompt: ChatPromptTemplate.fromMessages([
      new MessagesPlaceholder('chat_history'),
      ['human', '{input}'],
      [
        'human',
        'Given the conversation above, generate a search query to find relevant rules',
      ],
    ]),
  });
};

/**
 * Creates the prompt template for answer generation.
 * The {context} placeholder will be filled with retrieved documents.
 */
const createAnswerPrompt = (): ChatPromptTemplate => {
  return ChatPromptTemplate.fromMessages([
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
};

/**
 * Creates the answer chain that synthesizes retrieved documents into responses.
 * Uses "stuff documents" strategy: concatenates all documents into {context},
 * then sends to the LLM.
 */
const createAnswerChain = async (
  model: ChatOllama,
  prompt: ChatPromptTemplate,
) => {
  return createStuffDocumentsChain({
    llm: model,
    prompt,
  });
};

/**
 * Composes the full RAG pipeline using RunnableSequence.
 *
 * Flow:
 *   { input, chat_history }
 *     → assign context (via historyAwareRetriever → ensembleRetriever → hybrid search)
 *   { input, chat_history, context }
 *     → assign answer (via answerChain)
 *   { input, chat_history, context, answer }
 */
const composeRAGPipeline = (
  historyAwareRetriever: HistoryAwareRetriever,
  answerChain: StuffDocumentsChain,
  debugLog: (...args: unknown[]) => void,
) => {
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

  // Setup core AI components
  const { model, vectorStore } = await createCoreComponents();

  // Setup hybrid search (vector + keyword)
  const chunks = await loadChunksForBM25();
  const bm25Retriever = BM25Retriever.fromDocuments(chunks, { k: RETRIEVAL_K });
  const vectorRetriever = vectorStore.asRetriever(RETRIEVAL_K);

  const ensembleRetriever = new EnsembleRetriever({
    retrievers: [vectorRetriever, bm25Retriever],
    weights: [VECTOR_RETRIEVER_WEIGHT, BM25_RETRIEVER_WEIGHT],
  });

  // Wrap ensemble with history awareness
  const historyAwareRetriever = await wrapRetrieverWithHistoryAwareness(
    model,
    ensembleRetriever,
  );

  // Setup answer generation
  const prompt = createAnswerPrompt();
  const answerChain = await createAnswerChain(model, prompt);

  // Compose the full pipeline
  return composeRAGPipeline(historyAwareRetriever, answerChain, debugLog);
};
