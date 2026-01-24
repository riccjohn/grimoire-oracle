import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { createStuffDocumentsChain } from '@langchain/classic/chains/combine_documents'
import { createRetrievalChain } from '@langchain/classic/chains/retrieval'

export const setupOracle = async () => {
	const model = new ChatOllama({ model: 'llama3', temperature: 0.2 });
	const embedder = new OllamaEmbeddings({ model: 'nomic-embed-text' });
	const vectorStore = await HNSWLib.load('./grimoire_index', embedder);

	const prompt = ChatPromptTemplate.fromMessages([
		[
			'system',
			'You are the Grimoire Oracle, a TTRPG rules assistant. Answer based only on these rules:\n\n{context}',
		],
		['human', '{input}'],
	]);


  const combineDocsChain = await createStuffDocumentsChain({
    llm: model,
    prompt,
  })

  const RETRIEVAL_K= 3

  return createRetrievalChain({
    retriever: vectorStore.asRetriever(RETRIEVAL_K),
    combineDocsChain
  })


};
