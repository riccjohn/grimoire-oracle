import { DirectoryLoader } from '@langchain/classic/document_loaders/fs/directory';
import type { Document } from '@langchain/core/documents';
import { TextLoader } from '@langchain/classic/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { OllamaEmbeddings } from '@langchain/ollama';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';

const CHUNK_SIZE = 500;
const CHUNK_OVERLAP = 50;
const GRIMOIRE_INDEX_PATH = './grimoire_index';

const main = async () => {
	const docs = await loadVaultDocs('./vault');
	const chunks = await splitDocsIntoChunks(docs);
	createVectorIndex(chunks, GRIMOIRE_INDEX_PATH);
};

/**
 * Loads all markdown files from a directory into LangChain Document objects.
 * @param docPath - Path to the directory containing markdown files
 * @returns Array of Document objects with pageContent and metadata.source
 */
const loadVaultDocs = async (docPath: string) => {
	console.log('📂 Loading vault...');

	const loader = new DirectoryLoader(docPath, {
		'.md': (path) => new TextLoader(path),
	});

	const docs = await loader.load();

	console.log(`✅ Loaded ${docs.length} documents:\n`);

	return docs;
};

/**
 * Splits documents into smaller chunks for embedding.
 * Uses recursive character splitting with configurable size and overlap.
 * @param docs - Array of Document objects to split
 * @returns Array of smaller Document chunks, each preserving original metadata
 */
const splitDocsIntoChunks = async (docs: Document<Record<string, any>>[]) => {
	console.log('\n✂️ Splitting into chunks...');
	const splitter = new RecursiveCharacterTextSplitter({
		chunkSize: CHUNK_SIZE,
		chunkOverlap: CHUNK_OVERLAP,
	});

	const chunks = await splitter.splitDocuments(docs);

	console.log(`✅ Created ${chunks.length} chunks:\n`);

	return chunks;
};

/**
 * Converts document chunks into vector embeddings and saves the index to disk.
 * Uses Ollama's nomic-embed-text model for embedding generation.
 * @param chunks - Document chunks to vectorize
 * @param pathToStore - Directory path where the vector index will be saved
 */
const createVectorIndex = async (
	chunks: Document<Record<string, any>>[],
	pathToStore: string,
) => {
	console.log('🧠 Creating embeddings (this may take a moment)...');
	const embedder = new OllamaEmbeddings({ model: 'nomic-embed-text' });
	const vectorStore = await HNSWLib.fromDocuments(chunks, embedder);

	await vectorStore.save(pathToStore);
	console.log('✅ Index saved to grimoire_index/');
};

main();
