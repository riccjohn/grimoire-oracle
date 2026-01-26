import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import {
	ChatPromptTemplate,
	MessagesPlaceholder,
} from '@langchain/core/prompts';
import { RunnablePassthrough, RunnableSequence } from '@langchain/core/runnables';
import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { createStuffDocumentsChain } from '@langchain/classic/chains/combine_documents';
import { createHistoryAwareRetriever } from '@langchain/classic/chains/history_aware_retriever';
import type { BaseMessage } from '@langchain/core/messages';

const RETRIEVAL_K = 3;

/**
 * Sets up the conversational RAG (Retrieval Augmented Generation) chain.
 *
 * Returns a chain that accepts { input: string, chat_history: BaseMessage[] }
 * and returns { input, chat_history, context: Document[], answer: string }
 */
export const setupOracle = async () => {
	// Core components: LLM for chat, embeddings for vector search, vector store for retrieval
	const model = new ChatOllama({ model: 'llama3', temperature: 0.2 });
	const embedder = new OllamaEmbeddings({ model: 'nomic-embed-text' });
	const vectorStore = await HNSWLib.load('./grimoire_index', embedder);

	// --- STEP 1: History-Aware Retriever ---
	// Problem: If user asks "What about for elves?" after asking about dwarves,
	// a naive retriever would search for "What about for elves?" which lacks context.
	//
	// Solution: This retriever uses the LLM to rephrase follow-up questions into
	// standalone queries. "What about for elves?" becomes "elf race abilities" or similar.
	const historyAwareRetriever = await createHistoryAwareRetriever({
		llm: model,
		retriever: vectorStore.asRetriever(RETRIEVAL_K),
		rephrasePrompt: ChatPromptTemplate.fromMessages([
			new MessagesPlaceholder('chat_history'),
			['human', '{input}'],
			[
				'human',
				'Given the conversation above, generate a search query to find relevant rules',
			],
		]),
	});

	// --- STEP 2: Answer Generation Prompt ---
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

	// --- STEP 3: Answer Chain ---
	// "Stuff documents chain" = takes an array of Documents, concatenates ("stuffs")
	// their content into the {context} placeholder, then sends to the LLM.
	// (Alternative strategies exist for many documents: map-reduce, refine, etc.)
	const answerChain = await createStuffDocumentsChain({
		llm: model,
		prompt,
	});

	// --- STEP 4: Compose the Full Pipeline ---
	// We manually compose the chain because createRetrievalChain doesn't pass
	// chat_history to the retriever (it predates history-aware patterns).
	//
	// RunnableSequence: runs steps in order, piping output to input
	// RunnablePassthrough.assign: passes input through, adding new keys
	//
	// Flow:
	//   { input, chat_history }
	//     → assign context (via historyAwareRetriever)
	//   { input, chat_history, context }
	//     → assign answer (via answerChain)
	//   { input, chat_history, context, answer }
	return RunnableSequence.from([
		// First step: retrieve relevant documents based on rephrased query
		RunnablePassthrough.assign({
			context: async (input: { input: string; chat_history: BaseMessage[] }) => {
				return historyAwareRetriever.invoke(input);
			},
		}),
		// Second step: generate answer using retrieved context
		RunnablePassthrough.assign({
			answer: answerChain,
		}),
	]);
};
