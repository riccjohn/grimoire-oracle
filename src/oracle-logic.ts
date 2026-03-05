import { readFile } from "node:fs/promises";
import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { createHistoryAwareRetriever } from "@langchain/classic/chains/history_aware_retriever";
import { EnsembleRetriever } from "@langchain/classic/retrievers/ensemble";
import { BM25Retriever } from "@langchain/community/retrievers/bm25";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { Document } from "@langchain/core/documents";
import type { BaseMessage } from "@langchain/core/messages";
import {
	ChatPromptTemplate,
	MessagesPlaceholder,
} from "@langchain/core/prompts";
import type { BaseRetriever } from "@langchain/core/retrievers";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import {
	BM25_RETRIEVER_WEIGHT,
	EMBEDDINGS_MODEL,
	GRIMOIRE_CHUNKS_PATH,
	GRIMOIRE_INDEX_PATH,
	LLM_MODEL,
	LLM_TEMPERATURE,
	RETRIEVAL_K,
	VECTOR_RETRIEVER_WEIGHT,
} from "@src/constants";

type SerializedChunk = {
	pageContent: string;
	metadata: Record<string, unknown>;
};

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
	const model = new ChatOllama({ model: LLM_MODEL, temperature: LLM_TEMPERATURE });
	const embedder = new OllamaEmbeddings({ model: EMBEDDINGS_MODEL });
	const vectorStore = await HNSWLib.load(GRIMOIRE_INDEX_PATH, embedder);

	return { model, vectorStore };
};

/**
 * Loads and deserializes document chunks from JSON file for BM25 keyword search.
 * These are the same chunks used during vector store creation.
 */
const loadChunksForBM25 = async (): Promise<Document[]> => {
	const chunksData: SerializedChunk[] = JSON.parse(
		await readFile(GRIMOIRE_CHUNKS_PATH, "utf-8"),
	);

	return chunksData.map((chunk) => {
		const { pageContent, metadata } = chunk;

		return new Document({
			pageContent,
			metadata,
		});
	});
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
			new MessagesPlaceholder("chat_history"),
			["human", "{input}"],
			[
				"human",
				"Given the conversation above, generate a search query to find relevant rules",
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
			"system",
			`You are the Grimoire Oracle, a TTRPG rules assistant. Answer questions using ONLY the context provided below.

IMPORTANT: If the context does not contain the answer, say "I couldn't find that information in the rules." Do NOT make up or invent any rules, numbers, or game mechanics.

Context:
{context}`,
		],
		new MessagesPlaceholder("chat_history"),
		["human", "{input}"],
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
	historyAwareRetriever: Awaited<
		ReturnType<typeof createHistoryAwareRetriever>
	>,
	answerChain: Awaited<ReturnType<typeof createStuffDocumentsChain>>,
	debugLog: (...args: unknown[]) => void,
) => {
	return RunnableSequence.from([
		// Step 1: Retrieve relevant documents using hybrid search (vector + keyword)
		// The historyAwareRetriever rephrases queries using chat history,
		// then the ensembleRetriever combines vector + BM25 keyword results
		async (input: { input: string; chat_history: BaseMessage[] }) => {
			debugLog("Input query:", input.input);
			debugLog("Chat history length:", input.chat_history.length);

			const docs: Document[] = await historyAwareRetriever.invoke(input);

			debugLog(`Retrieved ${docs.length} documents:`);
			docs.forEach((doc, i) => {
				debugLog(`  [${i + 1}] ${doc.metadata.source}`);
				debugLog(`      "${doc.pageContent.slice(0, 100)}..."`);
			});

			return { ...input, context: docs };
		},
		// Step 2: Generate answer using the retrieved context
		async (input: {
			input: string;
			chat_history: BaseMessage[];
			context: Document[];
		}) => {
			const answer = (await answerChain.invoke(input)) as string;
			return { ...input, answer };
		},
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
		if (debug) console.log("[DEBUG]", ...args);
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
