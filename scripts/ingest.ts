import { DirectoryLoader } from '@langchain/classic/document_loaders/fs/directory';
import type { Document } from '@langchain/core/documents';
import { TextLoader } from '@langchain/classic/document_loaders/fs/text';
import { MarkdownTextSplitter } from '@langchain/textsplitters';
import { OllamaEmbeddings } from '@langchain/ollama';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';

const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 100;
const MIN_CHUNK_SIZE = 100;
const GRIMOIRE_INDEX_PATH = './grimoire_index';

const main = async () => {
	const docs = await loadVaultDocs('./vault');
	const chunks = await splitDocsIntoChunks(docs);
	const mergedChunks = mergeSmallChunks(chunks);
	const enrichedChunks = enrichChunksWithMetadata(mergedChunks);
	createVectorIndex(enrichedChunks, GRIMOIRE_INDEX_PATH);
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
	const splitter = new MarkdownTextSplitter({
		chunkSize: CHUNK_SIZE,
		chunkOverlap: CHUNK_OVERLAP,
	});

	const chunks = await splitter.splitDocuments(docs);

	console.log(`✅ Created ${chunks.length} chunks:\n`);
	console.log('Sample chunk lengths:', chunks.slice(0, 20).map(c => c.pageContent.length));

	return chunks;
};

/**
 * Merges small chunks with the following chunk from the same document.
 * Prevents orphaned headers from becoming standalone chunks.
 * @param chunks - Array of Document chunks
 * @returns Array with small chunks merged into their neighbors
 */
const mergeSmallChunks = (chunks: Document<Record<string, unknown>>[]) => {
	console.log('\n🔗 Merging small chunks...');

	const result = chunks.reduce<{
		merged: Document<Record<string, unknown>>[];
		skipNext: boolean;
	}>(
		(acc, current, index) => {
			if (acc.skipNext) {
				return { merged: acc.merged, skipNext: false };
			}

			const next = chunks[index + 1];
			const shouldMerge =
				current.pageContent.length < MIN_CHUNK_SIZE &&
				next &&
				current.metadata.source === next.metadata.source;

			if (shouldMerge) {
				return {
					merged: [
						...acc.merged,
						{
							pageContent: current.pageContent + '\n\n' + next.pageContent,
							metadata: current.metadata,
						},
					],
					skipNext: true,
				};
			}

			return { merged: [...acc.merged, current], skipNext: false };
		},
		{ merged: [], skipNext: false },
	);

	console.log(`✅ Merged ${chunks.length} → ${result.merged.length} chunks`);
	return result.merged;
};

/**
 * Prepends document title to each chunk's content to improve retrieval.
 * Helps embedding models match user queries like "Light spell" to relevant chunks.
 * @param chunks - Array of Document chunks with metadata.source containing file paths
 * @returns Array of chunks with titles prepended to pageContent
 */
const enrichChunksWithMetadata = (chunks: Document<Record<string, any>>[]) => {
	return chunks.map((chunk) => {
		const filepath = chunk.metadata.source;
		const title = extractTitleFromPath(filepath);
		chunk.pageContent = `[${title}]\n${chunk.pageContent}`;
		return chunk;
	});
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

/**
 * Extracts a breadcrumb-style title from a file path.
 * Keeps the full path after vault/, cleans number prefixes from each segment.
 * @param filepath - Full path to the markdown file
 * @returns Breadcrumb string (e.g., "vault/rules/5. Adventures/4. Hazards.md" -> "rules > Adventures > Hazards")
 */
const extractTitleFromPath = (filepath: string) => {
	const vaultIndex = filepath.indexOf('vault/');
	const relativePath = vaultIndex !== -1
		? filepath.slice(vaultIndex + 'vault/'.length)
		: filepath;

	return relativePath
		.replace(/\.md$/, '')
		.split('/')
		.map((segment) => segment.replace(/^\d+[\.\-]\s*/, ''))
		.join(' > ');
};

main();
