import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import {
	ChatPromptTemplate,
	MessagesPlaceholder,
} from '@langchain/core/prompts';
import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { createStuffDocumentsChain } from '@langchain/classic/chains/combine_documents';
import { createRetrievalChain } from '@langchain/classic/chains/retrieval';
import { createHistoryAwareRetriever } from '@langchain/classic/chains/history_aware_retriever';

const RETRIEVAL_K = 3;

export const setupOracle = async () => {
	const model = new ChatOllama({ model: 'llama3', temperature: 0.2 });
	const embedder = new OllamaEmbeddings({ model: 'nomic-embed-text' });
	const vectorStore = await HNSWLib.load('./grimoire_index', embedder);

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

	const prompt = ChatPromptTemplate.fromMessages([
		[
			'system',
			'You are the Grimoire Oracle, a TTRPG rules assistant. Answer based only on these rules:\n\n{context}',
		],
		new MessagesPlaceholder('chat_history'),
		['human', '{input}'],
	]);

	const combineDocsChain = await createStuffDocumentsChain({
		llm: model,
		prompt,
	});

	return createRetrievalChain({
		retriever: vectorStore.asRetriever(RETRIEVAL_K),
		combineDocsChain,
	});
};
