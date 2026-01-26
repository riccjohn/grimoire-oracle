import { OllamaEmbeddings } from '@langchain/ollama';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';

const main = async () => {
	console.log('🔍 Loading vector store...');

	const embedder = new OllamaEmbeddings({ model: 'nomic-embed-text' });
	const vectorStore = await HNSWLib.load('./grimoire_index', embedder);

	const query = 'Tell me about the Thief class';

	const results = await vectorStore.similaritySearch(query, 10);

	results.forEach((result, i) => {
		console.log(`### Match ${i} (${result.metadata.source}) ###`);
		console.log(result.pageContent);
		console.log('\n\n')
	});
};

main();
