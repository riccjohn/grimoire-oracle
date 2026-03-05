import { createStuffDocumentsChain } from "@langchain/classic/chains/combine_documents";
import { createHistoryAwareRetriever } from "@langchain/classic/chains/history_aware_retriever";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import type { BaseMessage } from "@langchain/core/messages";
import {
	ChatPromptTemplate,
	MessagesPlaceholder,
} from "@langchain/core/prompts";
import {
	RunnablePassthrough,
	RunnableSequence,
} from "@langchain/core/runnables";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";

const main = async () => {
	console.log("Setting up...");

	const model = new ChatOllama({ model: "llama3", temperature: 0.2 });
	const embedder = new OllamaEmbeddings({ model: "nomic-embed-text" });
	const vectorStore = await HNSWLib.load("./grimoire_index", embedder);

	const rephrasePrompt = ChatPromptTemplate.fromMessages([
		new MessagesPlaceholder("chat_history"),
		["human", "{input}"],
		[
			"human",
			"Given the conversation above, generate a search query to find relevant rules",
		],
	]);

	console.log("Creating historyAwareRetriever...");

	const historyAwareRetriever = await createHistoryAwareRetriever({
		llm: model,
		retriever: vectorStore.asRetriever(3),
		rephrasePrompt,
	});

	// Test 1: Direct invocation (this works)
	console.log("\n--- Test 1: Direct invocation ---");
	try {
		const result = await historyAwareRetriever.invoke({
			input: "Tell me about the Thief class",
			chat_history: [],
		});
		console.log("Direct invocation works! Got", result.length, "docs");
	} catch (error) {
		console.error("Direct invocation error:", error);
	}

	// Test 2: Custom chain with historyAwareRetriever
	console.log("\n--- Test 3: Custom chain with historyAwareRetriever ---");

	const answerPrompt = ChatPromptTemplate.fromMessages([
		["system", "Answer based on: {context}"],
		new MessagesPlaceholder("chat_history"),
		["human", "{input}"],
	]);

	const answerChain = await createStuffDocumentsChain({
		llm: model,
		prompt: answerPrompt,
	});

	const customChain = RunnableSequence.from([
		RunnablePassthrough.assign({
			context: async (input: {
				input: string;
				chat_history: BaseMessage[];
			}) => {
				const docs = await historyAwareRetriever.invoke(input);
				return docs;
			},
		}),
		answerChain,
	]);

	try {
		console.log("Invoking custom chain...");
		const result = await customChain.invoke({
			input: "Tell me about the Thief class",
			chat_history: [],
		});
		console.log("Custom chain works! Answer:", result?.substring(0, 100));
	} catch (error) {
		console.error("Custom chain error:", error);
	}
};

main();
