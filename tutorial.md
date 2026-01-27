# Tutorial: Building "The Grimoire Oracle"

A local, TTRPG-focused AI Assistant using LangChain, Ollama, and Ink.

---

## Step 1: Prerequisites

Before writing code, ensure your "local engines" are running.

1. **Install Ollama:** [Download here](https://ollama.com/).
2. **Pull the AI Models:** Run `./scripts/setup.sh` or manually:

   ```bash
   ollama pull llama3             # The "Brain" (LLM)
   ollama pull nomic-embed-text   # The "Eyes" (Embeddings)
   ```

3. **Node.js:** Ensure you are on Node 18 or higher.
4. **Install dependencies:** Run `npm install`

---

## Step 2: The Vault (Data Setup)

RAG (Retrieval Augmented Generation) is only as good as its data. The "vault" is where we store the knowledge the AI will search through.

```
vault/
└── rules/           # TTRPG rules (OSE markdown files)
```

We're using [Old School Essentials rules in markdown format](https://github.com/Obsidian-TTRPG-Community/Old-School-Essentials-Markdown).

**Observe:** Run `ls vault/rules/` to see the rule categories.

---

## Step 3: Load Documents

Before we can search our vault, we need to load the files into memory. This step introduces **Document Loaders**.

**Create `scripts/ingest.ts`:**

```typescript
import { DirectoryLoader } from '@langchain/classic/document_loaders/fs/directory';
import { TextLoader } from '@langchain/classic/document_loaders/fs/text';

const main = async () => {
	console.log('📂 Loading vault...');

	const loader = new DirectoryLoader('./vault', {
		'.md': (path) => new TextLoader(path),
	});

	const docs = await loader.load();

	console.log(`✅ Loaded ${docs.length} documents:\n`);

	for (const doc of docs) {
		console.log(`--- ${doc.metadata.source} ---`);
		console.log(doc.pageContent.slice(0, 200) + '...\n');
	}
};

main();
```

**Run:** `npx tsx scripts/ingest.ts`

**Observe:** Each "document" has `pageContent` (the text) and `metadata` (like the file path).

---

## Step 4: Split Documents into Chunks

LLMs have context limits, and embeddings work better on smaller pieces of text. This step introduces **Text Splitters**.

**Update `scripts/ingest.ts`:**

```typescript
import { DirectoryLoader } from '@langchain/classic/document_loaders/fs/directory';
import { TextLoader } from '@langchain/classic/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 100;

const main = async () => {
	console.log('📂 Loading vault...');
	const loader = new DirectoryLoader('./vault', {
		'.md': (path) => new TextLoader(path),
	});
	const docs = await loader.load();
	console.log(`Loaded ${docs.length} documents`);

	console.log('\n✂️ Splitting into chunks...');
	const splitter = new RecursiveCharacterTextSplitter({
		chunkSize: CHUNK_SIZE,
		chunkOverlap: CHUNK_OVERLAP,
	});
	const chunks = await splitter.splitDocuments(docs);

	console.log(`✅ Created ${chunks.length} chunks:\n`);

	for (const chunk of chunks.slice(0, 5)) {
		console.log(`--- Chunk from ${chunk.metadata.source} ---`);
		console.log(`"${chunk.pageContent.slice(0, 100)}..."\n`);
	}
};

main();
```

**What is RecursiveCharacterTextSplitter?** It splits text by trying a list of separators in order (paragraphs, then lines, then words, then characters) until chunks are small enough. It's a sensible default, but we'll discover its limitations later.

**Run:** `npx tsx scripts/ingest.ts`

**Observe:** Documents are split into chunks. Notice `chunkOverlap` creates redundancy between chunks—this helps preserve context at boundaries.

---

## Step 5: Create Embeddings and Vector Store

Now we turn text into numbers (embeddings) and store them for fast similarity search.

**Update `scripts/ingest.ts`:**

```typescript
import { DirectoryLoader } from '@langchain/classic/document_loaders/fs/directory';
import type { Document } from '@langchain/core/documents';
import { TextLoader } from '@langchain/classic/document_loaders/fs/text';
import { MarkdownTextSplitter } from '@langchain/textsplitters';
import { OllamaEmbeddings } from '@langchain/ollama';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { writeFile } from 'node:fs/promises';

const CHUNK_SIZE = 1000;
const CHUNK_OVERLAP = 100;
const GRIMOIRE_INDEX_PATH = './grimoire_index';
const GRIMOIRE_CHUNKS_PATH = './grimoire_index/grimoire_chunks.json';

const main = async () => {
	const docs = await loadVaultDocs('./vault');
	const chunks = await splitDocsIntoChunks(docs);

	await createVectorIndex(chunks, GRIMOIRE_INDEX_PATH);
	await saveChunksForBM25(chunks, GRIMOIRE_CHUNKS_PATH);
};

const loadVaultDocs = async (docPath: string) => {
	console.log('📂 Loading vault...');
	const loader = new DirectoryLoader(docPath, {
		'.md': (path) => new TextLoader(path),
	});
	const docs = await loader.load();
	console.log(`✅ Loaded ${docs.length} documents:\n`);
	return docs;
};

const splitDocsIntoChunks = async (docs: Document<Record<string, any>>[]) => {
	console.log('\n✂️ Splitting into chunks...');
	const splitter = new MarkdownTextSplitter({
		chunkSize: CHUNK_SIZE,
		chunkOverlap: CHUNK_OVERLAP,
	});
	const chunks = await splitter.splitDocuments(docs);
	console.log(`✅ Created ${chunks.length} chunks:\n`);
	return chunks;
};

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

const saveChunksForBM25 = async (
	chunks: Document<Record<string, any>>[],
	filePath: string,
) => {
	console.log('\n💾 Saving chunks for BM25 retriever...');
	await writeFile(filePath, JSON.stringify(chunks, null, 2));
	console.log(`✅ Chunks saved to ${filePath}`);
};

main();
```

**Run:** `npx tsx scripts/ingest.ts`

**Observe:** A new `grimoire_index/` folder appears containing:
- Vector database files for semantic search
- `grimoire_chunks.json` for BM25 keyword search (used later in hybrid search)

This only needs to run once (or when your vault changes).

**Note:** We use `MarkdownTextSplitter` instead of `RecursiveCharacterTextSplitter` because it respects markdown structure (headers, code blocks, paragraphs) instead of splitting at arbitrary character boundaries.

---

## Step 6: Test Similarity Search

Before adding an LLM, let's verify the vector store can find relevant documents.

**Create `scripts/test-search.ts`:**

```typescript
import { OllamaEmbeddings } from '@langchain/ollama';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';

const main = async () => {
	console.log('🔍 Loading vector store...');

	const embedder = new OllamaEmbeddings({ model: 'nomic-embed-text' });
	const vectorStore = await HNSWLib.load('./grimoire_index', embedder);

	const query = 'How much damage from falling 30 feet';

	const results = await vectorStore.similaritySearch(query, 10);

	results.forEach((result, i) => {
		console.log(`### Match ${i} (${result.metadata.source}) ###`);
		console.log(result.pageContent);
		console.log('\n\n');
	});
};

main();
```

**Run:** `npx tsx scripts/test-search.ts`

**Observe:** The search finds chunks related to your query—no LLM involved yet! This is pure mathematical similarity between your question's embedding and the stored embeddings.

---

## Step 7: Add the LLM (Basic Chain)

Now we connect the retriever to an LLM so it can answer questions in natural language.

**Create `src/oracle-logic.ts`:**

```typescript
import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { createRetrievalChain } from '@langchain/classic/chains/retrieval';
import { createStuffDocumentsChain } from '@langchain/classic/chains/combine_documents';
import { ChatPromptTemplate } from '@langchain/core/prompts';

const RETRIEVAL_K = 3;

export async function setupOracle() {
	const model = new ChatOllama({ model: 'llama3', temperature: 0.2 });
	const embeddings = new OllamaEmbeddings({ model: 'nomic-embed-text' });
	const vectorStore = await HNSWLib.load('./grimoire_index', embeddings);

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
	});

	return createRetrievalChain({
		retriever: vectorStore.asRetriever(RETRIEVAL_K),
		combineDocsChain,
	});
}
```

**Create `scripts/test-oracle.ts`:**

```typescript
import { setupOracle } from '../src/oracle-logic';

async function main() {
	console.log('🔮 Starting oracle...\n');
	const oracle = await setupOracle();

	const question = 'How much damage does a character take from a 30 foot fall?';
	console.log(`Question: ${question}\n`);

	const response = await oracle.invoke({ input: question });

	console.log('Answer:', response.answer);
}

main();
```

**Run:** `npx tsx scripts/test-oracle.ts`

**Observe:** The LLM answers using retrieved context. It should calculate 3d6 damage (30 feet = 3 × 10 feet = 3d6).

---

## Step 8: Add Chat History

The oracle should remember previous questions. This step introduces **History-Aware Retrieval**.

### The Problem with Follow-up Questions

If a user asks "Tell me about the Thief class" and then follows up with "What skills do they have?", a naive retriever would search for "What skills do they have?"—which lacks any context about Thieves. We need the retriever to understand conversation context.

### History-Aware Retriever

The `createHistoryAwareRetriever` solves this by using the LLM to **rephrase** follow-up questions into standalone search queries:

```
User: "Tell me about the Thief class"
User: "What skills do they have?"
  ↓ (LLM rephrases using chat history)
Search query: "Thief class skills abilities"
  ↓ (passed to underlying retriever)
Hybrid search via EnsembleRetriever
```

This is not just bundling messages together—it's an actual LLM call that generates a new query. The `rephrasePrompt` explicitly instructs: "generate a search query to find relevant rules." The rephrased query is then passed to whatever retriever you provide—in our case, the `EnsembleRetriever` that combines vector and BM25 search.

### Two Separate Uses of Chat History

Our implementation uses chat history in **two different places** for different purposes:

| Component | What it does | Why it's needed |
|-----------|--------------|-----------------|
| `rephrasePrompt` in retriever | LLM **rewrites** query before searching | So "What skills do they have?" becomes "Thief class skills" |
| `MessagesPlaceholder` in answer prompt | LLM **sees** previous conversation when answering | So answers can reference prior context ("As I mentioned...") |

Both are necessary. Without the first, retrieval fails. Without the second, the LLM can't reference its own previous answers.

### Why We Can't Use createRetrievalChain

You might expect to combine `createHistoryAwareRetriever` with `createRetrievalChain` like this:

```typescript
// ❌ This doesn't work properly!
return createRetrievalChain({
	retriever: historyAwareRetriever,
	combineDocsChain,
});
```

The problem: `createRetrievalChain` was designed before history-aware patterns existed. It doesn't pass `chat_history` through to the retriever—so the history-aware retriever never receives the conversation context it needs.

### The Solution: Manual Chain Composition with Runnables

LangChain provides lower-level primitives for composing chains manually:

- **`RunnableSequence`** - Runs steps in order, piping output from one step to the next
- **`RunnablePassthrough.assign()`** - Passes input through unchanged while adding new computed keys

**Update `src/oracle-logic.ts`:**

```typescript
import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import { createStuffDocumentsChain } from '@langchain/classic/chains/combine_documents';
import { createHistoryAwareRetriever } from '@langchain/classic/chains/history_aware_retriever';
import { EnsembleRetriever } from '@langchain/classic/retrievers/ensemble';
import { BM25Retriever } from '@langchain/community/retrievers/bm25';
import {
	ChatPromptTemplate,
	MessagesPlaceholder,
} from '@langchain/core/prompts';
import {
	RunnablePassthrough,
	RunnableSequence,
} from '@langchain/core/runnables';
import type { BaseMessage } from '@langchain/core/messages';
import { Document } from '@langchain/core/documents';
import { readFile } from 'node:fs/promises';

const RETRIEVAL_K = 3;

export async function setupOracle() {
	const model = new ChatOllama({ model: 'llama3', temperature: 0.2 });
	const embeddings = new OllamaEmbeddings({ model: 'nomic-embed-text' });
	const vectorStore = await HNSWLib.load('./grimoire_index', embeddings);

	// Step 1: Setup hybrid search (vector + keyword)
	// Load chunks for BM25 keyword search
	const chunksData = JSON.parse(
		await readFile('./grimoire_index/grimoire_chunks.json', 'utf-8'),
	);
	const chunks = chunksData.map(
		(chunk: { pageContent: string; metadata: Record<string, unknown> }) =>
			new Document({ pageContent: chunk.pageContent, metadata: chunk.metadata }),
	);

	const bm25Retriever = BM25Retriever.fromDocuments(chunks, { k: RETRIEVAL_K });
	const vectorRetriever = vectorStore.asRetriever(RETRIEVAL_K);

	const ensembleRetriever = new EnsembleRetriever({
		retrievers: [vectorRetriever, bm25Retriever],
		weights: [0.5, 0.5],
	});

	// Step 2: Create history-aware retriever
	// This rephrases follow-up questions using chat history before searching
	const historyAwareRetriever = await createHistoryAwareRetriever({
		llm: model,
		retriever: ensembleRetriever,
		rephrasePrompt: ChatPromptTemplate.fromMessages([
			new MessagesPlaceholder('chat_history'),
			['human', '{input}'],
			[
				'human',
				'Given the conversation above, generate a search query to find relevant rules.',
			],
		]),
	});

	// Step 3: Create the answer chain
	// "Stuff documents chain" concatenates retrieved docs into the {context} placeholder
	const answerChain = await createStuffDocumentsChain({
		llm: model,
		prompt: ChatPromptTemplate.fromMessages([
			[
				'system',
				'You are the Grimoire Oracle. Answer based on these rules:\n\n{context}',
			],
			new MessagesPlaceholder('chat_history'),
			['human', '{input}'],
		]),
	});

	// Step 4: Compose the full pipeline manually
	// This is what createRetrievalChain does internally, but we need to do it
	// ourselves to properly pass chat_history to the retriever.
	//
	// Data flow:
	//   { input, chat_history }
	//     → assign context (via historyAwareRetriever → ensembleRetriever)
	//   { input, chat_history, context }
	//     → assign answer (via answerChain)
	//   { input, chat_history, context, answer }
	return RunnableSequence.from([
		RunnablePassthrough.assign({
			context: async (input: {
				input: string;
				chat_history: BaseMessage[];
			}) => {
				return historyAwareRetriever.invoke(input);
			},
		}),
		RunnablePassthrough.assign({
			answer: answerChain,
		}),
	]);
}
```

**Update `scripts/test-oracle.ts`:**

```typescript
import { setupOracle } from '../src/oracle-logic';
import { HumanMessage, AIMessage } from '@langchain/core/messages';

async function main() {
	console.log('🔮 Starting oracle...\n');
	const oracle = await setupOracle();

	// First question
	const q1 = 'How much damage from falling 30 feet?';
	console.log(`You: ${q1}`);
	const r1 = await oracle.invoke({ input: q1, chat_history: [] });
	console.log(`Oracle: ${r1.answer}\n`);

	// Follow-up question (references "it" from previous answer)
	const q2 = 'What if I fell 50 feet instead?';
	console.log(`You: ${q2}`);
	const r2 = await oracle.invoke({
		input: q2,
		chat_history: [new HumanMessage(q1), new AIMessage(r1.answer)],
	});
	console.log(`Oracle: ${r2.answer}\n`);
}

main();
```

**Run:** `npx tsx scripts/test-oracle.ts`

**Observe:** The follow-up question understands context from the previous exchange.

---

## Understanding Chat History Cost

An important detail: LLMs are **stateless**. They don't remember previous conversations—every API call must include the full chat history. This has cost implications.

### Linear Growth, Quadratic Cost

The chat history itself grows **linearly**—each turn adds one user message and one assistant response. If each exchange averages 500 tokens, after 20 turns you have ~10,000 tokens of history.

But the **total tokens processed** grows **quadratically** (O(n²)):

| Turn | Tokens Processed |
| ---- | ---------------- |
| 1    | 500              |
| 2    | 1,000            |
| 3    | 1,500            |
| 10   | 5,000            |
| 20   | 10,000           |

Total tokens across all turns = 500 + 1000 + 1500 + ... = O(n²)

This means API costs and latency increase significantly for long conversations.

### Our Implementation's Limitation

The current `setupOracle()` does nothing to manage history length—it will eventually hit context limits or degrade performance. For a learning project, that's fine. Production systems need mitigation strategies.

### Mitigation Strategies (Future Improvement)

1. **Truncation** - Drop oldest messages when approaching the context limit (simplest)

2. **Sliding Window** - Keep only the last N messages (e.g., last 10 exchanges)

3. **Summarization** - Periodically summarize older history into a condensed form, keep recent messages verbatim

4. **Hybrid** - Combine approaches: summary of early conversation + full recent messages

LangChain provides utilities for this:

- `BufferWindowMemory` - Sliding window approach
- `ConversationSummaryMemory` - Automatic summarization
- `ConversationSummaryBufferMemory` - Hybrid approach

---

## Step 9: Debugging Retrieval

You ask the oracle "Tell me about the Thief class" and get "I couldn't find that information." Before assuming the LLM is broken, **debug the retrieval layer**.

### Built-in Debug Mode

Instead of creating standalone debugging scripts for each problem, build debugging into the oracle itself. This way you can investigate **any** issue with the same tool.

**Update `src/oracle-logic.ts` to accept a debug option:**

```typescript
type OracleOptions = {
	debug?: boolean;
};

export const setupOracle = async (options: OracleOptions = {}) => {
	const { debug = false } = options;

	const debugLog = (...args: unknown[]) => {
		if (debug) console.log('[DEBUG]', ...args);
	};

	// ... rest of setup ...

	return RunnableSequence.from([
		RunnablePassthrough.assign({
			context: async (input: { input: string; chat_history: BaseMessage[] }) => {
				debugLog('Input query:', input.input);
				debugLog('Chat history length:', input.chat_history.length);

				const docs = await historyAwareRetriever.invoke(input);

				debugLog(`Retrieved ${docs.length} documents:`);
				docs.forEach((doc, i) => {
					debugLog(`  [${i + 1}] ${doc.metadata.source}`);
					debugLog(`      "${doc.pageContent.slice(0, 100)}..."`);
				});

				return docs;
			},
		}),
		// ... answer chain ...
	]);
};
```

**Update `scripts/test-oracle.ts` to enable debug mode:**

```typescript
const oracle = await setupOracle({ debug: true });
```

**Run:** `npx tsx scripts/test-oracle.ts`

Now you'll see exactly what documents are being retrieved for every query—using the same code path as production.

### What to Look For

1. **Are the right documents in the results?** If "Thief" content isn't in the top 3, retrieval is the problem—not the LLM.

2. **What content is being retrieved?** Sometimes chunks contain the right file but wrong section.

3. **Is chat history being passed?** Check "Chat history length" to verify follow-up questions have context.

### Why Built-in Debugging Beats Standalone Scripts

| Standalone Scripts | Built-in Debug Mode |
|--------------------|---------------------|
| Test retrieval in isolation | Test retrieval in actual context |
| Must recreate embedder, vector store | Uses same instances as production |
| Doesn't test history-aware rephrasing | Shows what historyAwareRetriever actually does |
| One script per problem | One flag for any problem |

The debug flag lets you see inside the real code path, not a simulation of it.

---

## Gotcha: The Retrieval Quality Problem

During development, you'll likely encounter a frustrating issue: **the oracle gives wrong or irrelevant answers**.

### The Symptom

You ask: _"How much damage from falling 30 feet?"_

The oracle mentions "Bombing" rules about flying creatures dropping rocks—completely irrelevant.

### Debugging with Manual Observability

Before assuming the LLM is broken, check what the **retriever** is actually finding:

```typescript
// In test-search.ts, try the same query
const query = 'How much damage from falling 30 feet';
const results = await vectorStore.similaritySearch(query, 10);
```

You might discover the actual falling damage rule isn't in your top results at all.

### The Chunking Dilemma

The problem is often **chunk size**:

| Chunk Size        | Result                                                                                                       |
| ----------------- | ------------------------------------------------------------------------------------------------------------ |
| **500** (default) | Rules get diluted by surrounding content. The 2-line falling damage rule shares a chunk with unrelated text. |
| **250**           | Better precision, but creates fragments that are just headers with no content (`# Cleric`, `## Falling`).    |
| **350**           | Still inconsistent—works for some queries, not others.                                                       |

### The Solution: Markdown-Aware Splitting ✅

Switch from `RecursiveCharacterTextSplitter` to `MarkdownTextSplitter` (already done in Step 5):

```typescript
import { MarkdownTextSplitter } from '@langchain/textsplitters';

const splitter = new MarkdownTextSplitter({
	chunkSize: CHUNK_SIZE,
	chunkOverlap: CHUNK_OVERLAP,
});
```

This respects markdown structure (headers, code blocks, paragraphs) instead of splitting at arbitrary character boundaries.

### The Deeper Problem: Query-Document Alignment

Even with good chunking, some queries fail:

- _"Tell me about the Light spell"_ → returns irrelevant header fragments
- _"conjuring light 15 radius"_ → finds the spell perfectly

Why? The embedding model measures **semantic similarity**. User questions in natural language ("Tell me about X") don't always embed close to structured reference text ("1. **Conjuring light:** In a 15' radius...").

This is a **fundamental RAG challenge**, not a bug in your code. The solution? **Hybrid search** (Strategy 3 below) combines vector similarity with keyword matching, so exact terms like "Light spell" get matched even when the semantic embedding doesn't align perfectly.

---

## Improving Retrieval Quality

When retrieval fails, you have several strategies to try. We've implemented the first three (marked with ✅); the rest are documented for future exploration.

### Strategy 1: Chunk Metadata Enrichment ✅

**Problem:** A chunk from `Thief.md` might contain "**Back-stab:** When attacking an unaware opponent..." but never mentions "Thief." The embedding doesn't know this is Thief content.

**Solution:** Prepend document context to each chunk before embedding:

```typescript
// In scripts/ingest.ts, after splitting
for (const chunk of chunks) {
	const filename = chunk.metadata.source.split('/').pop()?.replace('.md', '');
	chunk.pageContent = `[${filename}]\n${chunk.pageContent}`;
}
```

Now the chunk embeds as `[Thief]\n**Back-stab:** When attacking...` and "Thief" queries match directly.

**Trade-off:** Slightly reduces content space in each chunk, adds noise to embeddings.

### Strategy 2: Query Rewriting ✅

**Problem:** User asks "Tell me about the Light spell" but documents say "**Conjuring light:** In a 15' radius..."

**Solution:** Have the LLM expand/rephrase the query before retrieval:

```typescript
const queryRewriter = ChatPromptTemplate.fromMessages([
	[
		'system',
		'Rewrite this question as a search query. Include synonyms and related terms.',
	],
	['human', '{query}'],
]);

// "Tell me about the Light spell" →
// "Light spell conjuring illumination darkness cleric magic-user duration range"
```

LangChain provides `MultiQueryRetriever` for this pattern.

**Trade-off:** Adds an LLM call (latency + cost) before every retrieval.

### Strategy 3: Hybrid Search (Vector + Keyword) ✅

Combine semantic similarity with BM25 keyword matching:

- **Vector search:** finds conceptually similar content
- **BM25 keyword search:** boosts exact term matches ("Thief" → documents containing "Thief")

LangChain provides `EnsembleRetriever` to combine multiple retrievers with configurable weights:

```typescript
import { EnsembleRetriever } from '@langchain/classic/retrievers/ensemble';
import { BM25Retriever } from '@langchain/community/retrievers/bm25';

// Load the same chunks used during ingestion
const chunksData = JSON.parse(
	await readFile('./grimoire_index/grimoire_chunks.json', 'utf-8'),
);
const chunks = chunksData.map(
	(chunk) => new Document({ pageContent: chunk.pageContent, metadata: chunk.metadata }),
);

// Create both retrievers
const bm25Retriever = BM25Retriever.fromDocuments(chunks, { k: 3 });
const vectorRetriever = vectorStore.asRetriever(3);

// Combine with equal weights
const ensembleRetriever = new EnsembleRetriever({
	retrievers: [vectorRetriever, bm25Retriever],
	weights: [0.5, 0.5],
});
```

The ensemble returns documents that score well on either method, with results weighted and deduplicated.

**Trade-off:** Requires saving chunks during ingestion (already done in Step 5). BM25 loads all chunks into memory.

### Strategy 4: Multi-Query Retrieval

Generate multiple search queries from one question, retrieve for each, combine results:

```
"Tell me about the Thief class" →
  - "Thief class abilities"
  - "Thief skills back-stab"
  - "Rogue OSE pickpocket climb"
```

Retrieves with all three, deduplicates, returns union.

**Trade-off:** 3x retrieval calls, need to handle duplicates.

### Strategy 5: Increase Retrieval K

Simply retrieve more documents (K=10 instead of K=3). More chances to include the right content.

**Trade-off:** More noise, fills context window faster, LLM may get confused.

### Strategy 6: Re-ranking

Retrieve many candidates (K=20), then use a cross-encoder or LLM to re-rank by actual relevance to the query. Return top 3 after re-ranking.

**Trade-off:** Additional model call, but can dramatically improve precision.

### Strategy 7: Parent Document Retriever

Embed small chunks for precise matching, but return the larger parent document (or section) for context.

**Trade-off:** More complex ingestion, larger context per result.

---

## Step 10: Terminal UI

Now that the oracle logic is complete and tested, we can wrap it in an interactive interface. We'll use **Ink**, which lets you build terminal UIs with React. If you know React, you already know Ink—same components, same hooks, same mental model, just rendering to the terminal instead of the browser.

### Why Ink?

Building interactive terminal apps traditionally means managing cursor positions, escape codes, and raw input handling. Ink abstracts all of that:

| Browser React | Ink (Terminal React) |
|---------------|----------------------|
| `<div>` | `<Box>` |
| `<span>` | `<Text>` |
| CSS flexbox | Same flexbox props on `<Box>` |
| `onClick` | `onSubmit`, keyboard handlers |

The mental model is identical: declare your UI as a function of state, and Ink handles re-rendering when state changes.

### Step 10a: Hello Terminal

Let's start with the minimum viable Ink app.

**Create `src/index.tsx`:**

```tsx
import { render, Text } from 'ink';

const App = () => {
	return <Text>Hello from the terminal!</Text>;
};

render(<App />);
```

**Run:** `npx tsx src/index.tsx`

**Observe:** The text appears and the process exits. That's the simplest Ink app—a stateless component that renders once.

### Step 10b: Layout with Box

`<Box>` is Ink's layout primitive, equivalent to a `<div>` with flexbox. Let's create a bordered container.

**Update `src/index.tsx`:**

```tsx
import { render, Text, Box } from 'ink';

const App = () => {
	return (
		<Box
			flexDirection='column'
			padding={1}
			borderStyle='round'
			borderColor='cyan'
		>
			<Text bold color='yellow'>
				⚔️ THE GRIMOIRE ORACLE
			</Text>
			<Text>Ask me about TTRPG rules...</Text>
		</Box>
	);
};

render(<App />);
```

**Run:** `npx tsx src/index.tsx`

**Observe:** A bordered box appears with styled text. The `flexDirection='column'` stacks children vertically (the default is row). Try changing `borderStyle` to `'single'`, `'double'`, or `'classic'`.

### Step 10c: Adding State with useState

Now let's make it interactive. We need to track:
- What the user is currently typing (`query`)
- The current UI state (`status`: idle, loading, or error)

```tsx
import { useState } from 'react';
import { render, Text, Box } from 'ink';
import TextInput from 'ink-text-input';

const App = () => {
	const [query, setQuery] = useState('');
	const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle');

	const handleSubmit = () => {
		if (!query.trim()) return;
		setStatus('loading');
		// We'll add the oracle call here later
		setTimeout(() => setStatus('idle'), 1000); // Fake delay for now
		setQuery('');
	};

	return (
		<Box flexDirection='column' padding={1} borderStyle='round' borderColor='cyan'>
			<Text bold color='yellow'>THE GRIMOIRE ORACLE</Text>

			{status === 'loading' && (
				<Text italic color='gray'>Consulting the grimoire...</Text>
			)}

			{status === 'idle' && (
				<TextInput
					value={query}
					onChange={setQuery}
					onSubmit={handleSubmit}
					placeholder="Ask me about OSE rules..."
				/>
			)}
		</Box>
	);
};

render(<App />);
```

**Key concepts:**

- **`TextInput`** is a controlled component from `ink-text-input`. It doesn't manage its own state—you provide `value` and update it via `onChange`.
- **`onSubmit`** fires when the user presses Enter.
- **Status enum** — Using `'idle' | 'loading' | 'error'` instead of a boolean makes it easier to handle multiple states and extend later.
- **Conditional rendering** (`{status === 'loading' && ...}`) works exactly like React on the web.

**Run:** `npx tsx src/index.tsx`

**Observe:** Type something and press Enter. The loading message appears briefly. The input clears. But nothing actually happens yet—we need to connect the oracle.

### Step 10d: Connecting the Oracle with useEffect

The oracle needs to be initialized before we can use it. This is a side effect that should happen once when the component mounts—the classic use case for `useEffect`.

```tsx
import { useState, useEffect } from 'react';
import { render, Text, Box } from 'ink';
import TextInput from 'ink-text-input';
import { setupOracle } from './oracle-logic';

const App = () => {
	const [query, setQuery] = useState('');
	const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle');
	const [oracle, setOracle] = useState<Awaited<
		ReturnType<typeof setupOracle>
	> | null>(null);

	// Initialize oracle on mount
	useEffect(() => {
		setupOracle().then(setOracle);
	}, []);

	const handleSubmit = async () => {
		if (!oracle || !query.trim()) return;

		setStatus('loading');
		const response = await oracle.invoke({
			input: query,
			chat_history: [], // Empty for now
		});
		console.log(response.answer);
		setStatus('idle');
		setQuery('');
	};

	return (
		<Box flexDirection='column' padding={1} borderStyle='round' borderColor='cyan'>
			<Text bold color='yellow'>THE GRIMOIRE ORACLE</Text>

			{!oracle && <Text color='gray'>Loading oracle...</Text>}

			{status === 'loading' && (
				<Text italic color='gray'>Consulting the grimoire...</Text>
			)}

			{status === 'idle' && (
				<TextInput
					value={query}
					onChange={setQuery}
					onSubmit={handleSubmit}
					placeholder="Ask me about OSE rules..."
				/>
			)}
		</Box>
	);
};

render(<App />);
```

**Key concepts:**

- **`useEffect(() => {...}, [])`** — The empty dependency array means "run once on mount." Without it, the effect would run on every render.
- **TypeScript inference** — `Awaited<ReturnType<typeof setupOracle>>` extracts the return type of the async `setupOracle` function. This is verbose but avoids manually defining the oracle's type.
- **Null check** — `if (!oracle || ...)` prevents calling methods on a null oracle before initialization completes.

**Run:** `npx tsx src/index.tsx`

**Observe:** The oracle initializes, you can ask questions, and answers appear in the console (not the UI yet). We're almost there.

### Step 10e: Chat History Display

A chat interface needs to display the conversation. We'll store messages in state and render them. We'll also use our Catppuccin theme for consistent styling.

First, create a theme file (`src/theme.ts`) using the `@catppuccin/palette` package:

```typescript
import { flavors } from '@catppuccin/palette';

const mocha = flavors.mocha.colors;

export const theme = {
	// Oracle (AI responses)
	oracleTitle: mocha.lavender.hex,
	oracleResponse: mocha.sapphire.hex,

	// User (input & messages)
	userTitle: mocha.green.hex,
	userResponse: mocha.teal.hex,

	// UI states
	dim: mocha.overlay0.hex,
	error: mocha.red.hex,
	success: mocha.green.hex,
	loading: mocha.yellow.hex,
	accent: mocha.peach.hex,

	// Direct palette access (if needed)
	palette: mocha,
};
```

Now the full app (`src/App.tsx`):

```tsx
import { Box, render, Text } from 'ink';
import TextInput from 'ink-text-input';
import { useEffect, useState } from 'react';
import { setupOracle } from './oracle-logic';
import { theme } from './theme';
import { AIMessage, HumanMessage } from '@langchain/core/messages';

type Message = { role: 'human' | 'ai'; content: string };

const App = () => {
	const [oracle, setOracle] = useState<Awaited<
		ReturnType<typeof setupOracle>
	> | null>(null);

	const [query, setQuery] = useState('');
	const [status, setStatus] = useState<'idle' | 'loading' | 'error'>('idle');
	const [messages, setMessages] = useState<Message[]>([]);

	useEffect(() => {
		setupOracle().then(setOracle);
	}, []);

	const handleSubmit = async () => {
		if (!oracle || !query.trim()) {
			setStatus('error');
			return;
		}

		// Add user message immediately (optimistic update)
		const userMessage: Message = { role: 'human', content: query };
		setMessages((prev) => [...prev, userMessage]);
		setQuery('');
		setStatus('loading');

		// Convert our messages to LangChain format for chat_history
		const chatHistory = messages.map((m) =>
			m.role === 'human'
				? new HumanMessage(m.content)
				: new AIMessage(m.content),
		);
		const response = await oracle.invoke({
			input: query,
			chat_history: chatHistory,
		});

		// Add AI response
		const aiMessage: Message = { role: 'ai', content: response.answer };
		setMessages((prev) => [...prev, aiMessage]);
		setStatus('idle');
	};

	return (
		<Box
			flexDirection="column"
			padding={1}
			borderStyle="round"
			borderColor={theme.oracleTitle}
		>
			<Text bold color={theme.oracleTitle}>
				THE GRIMOIRE ORACLE
			</Text>
			<Box flexDirection="column" marginY={1}>
				{messages.slice(-6).map((m, i) => (
					<Text
						key={i}
						color={
							m.role === 'human' ? theme.userResponse : theme.oracleResponse
						}
					>
						{m.role === 'human' ? '> ' : '~ '}
						{m.content}
					</Text>
				))}
			</Box>
			{status === 'loading' && (
				<Text italic color={theme.loading}>
					Consulting the grimoire...
				</Text>
			)}
			{status === 'idle' && (
				<TextInput
					value={query}
					onChange={setQuery}
					onSubmit={handleSubmit}
					placeholder="Ask me about OSE rules..."
				/>
			)}
		</Box>
	);
};

render(<App />);
```

**Key concepts:**

- **Status enum vs boolean** — Using `status: 'idle' | 'loading' | 'error'` instead of a boolean `loading` makes it easier to handle multiple states and add new ones later (like `'error'`).

- **Theme tokens** — Instead of hardcoded colors like `'cyan'` or `'yellow'`, we use semantic tokens from our theme (`theme.oracleTitle`, `theme.userResponse`). This keeps colors consistent and makes it easy to change the palette.

- **Two message formats** — We store messages as simple `{role, content}` objects for our UI, but LangChain needs `HumanMessage`/`AIMessage` instances. We convert between them when calling the oracle.

- **Optimistic updates** — We add the user's message to state immediately (`setMessages((prev) => [...prev, userMessage])`), before waiting for the AI response. This makes the UI feel responsive.

- **Functional state updates** — `setMessages((prev) => [...prev, newMessage])` is safer than `setMessages([...messages, newMessage])` because it uses the latest state value, avoiding race conditions.

- **Windowing** — `messages.slice(-6)` shows only the last 6 messages. Terminals have limited vertical space; without this, long conversations would overflow.

- **Conditional input visibility** — The `TextInput` only appears when `status === 'idle'`, preventing the user from typing while the oracle is thinking.

**Run:** `npm run dev`

**Observe:** A fully functional chat interface with themed colors. Ask questions, see your history, get answers. Press Ctrl+C to exit.

### Understanding the Data Flow

Here's how data moves through the app:

```
User types → query state updates → UI re-renders with current input
     ↓
User presses Enter → handleSubmit fires
     ↓
1. Add user message to messages[] (optimistic)
2. Clear query, set status='loading'
3. Convert messages[] to LangChain format
4. Call oracle.invoke({ input, chat_history })
     ↓
5. Add AI response to messages[]
6. Set status='idle'
     ↓
UI re-renders showing new messages
```

The key insight: **React state is the single source of truth.** The UI is always a pure function of `(query, messages, status, oracle)`. When any of these change, Ink re-renders automatically.

### Exercises

1. **Add error handling** — Wrap the `oracle.invoke()` call in try/catch and set `status='error'` on failure. Display an error message when `status === 'error'`.

2. **Add a startup message** — Show "Oracle ready!" after initialization completes.

3. **Add timestamps** — Extend the `Message` type to include a timestamp, display it next to each message.

4. **Exit gracefully** — Use the `useInput` hook from Ink to detect when the user types "quit" or presses Escape.

---

## Observability

As your system grows, you need consistent ways to see inside it.

### The Debug Option Pattern

We've already added a `debug` option to `setupOracle()`. This pattern scales well:

```typescript
// In your main logic
export const setupOracle = async (options: OracleOptions = {}) => {
	const { debug = false } = options;

	const debugLog = (...args: unknown[]) => {
		if (debug) console.log('[DEBUG]', ...args);
	};

	// Use debugLog at key decision points
	debugLog('Retrieved docs:', docs.map(d => d.metadata.source));
};

// In your test/UI code
const oracle = await setupOracle({ debug: true });
```

You can also support environment variables for production debugging:

```typescript
const debug = options.debug ?? process.env.DEBUG === 'true';
```

Run with: `DEBUG=true npx tsx src/index.tsx`

### Observability Tools

For production systems, add tracing to see inside your chains:

- **LangFuse** (open-source) - Self-hostable, good LangChain integration
- **Phoenix** (open-source, from Arize) - Focused on LLM observability
- **LangSmith** (LangChain's hosted solution) - Tightest integration, but SaaS

These show you exactly which chunks were retrieved, what prompt was sent, token counts, and latency per step.

### Memory Management

For long conversations, implement one of these strategies:

```typescript
import { BufferWindowMemory } from 'langchain/memory';

// Keep only the last 10 exchanges
const memory = new BufferWindowMemory({ k: 10 });
```

Or for automatic summarization:

```typescript
import { ConversationSummaryBufferMemory } from 'langchain/memory';

// Summarize when history exceeds maxTokenLimit
const memory = new ConversationSummaryBufferMemory({
	llm: model,
	maxTokenLimit: 2000,
});
```

This keeps conversations responsive and cost-effective regardless of length.

---

## Summary

You've built a RAG application from scratch:

1. **Vault** → Raw knowledge (markdown files)
2. **Loader** → Read files into documents
3. **Splitter** → Break into searchable chunks (markdown-aware)
4. **Embeddings** → Convert text to vectors
5. **Vector Store** → Index for similarity search
6. **Hybrid Retriever** → Combine vector similarity with BM25 keyword matching
7. **LLM Chain** → Generate answers from context
8. **History-Aware Retriever** → Handle follow-up questions
9. **Runnable Composition** → Manually compose chains when pre-built functions don't fit
10. **UI** → Interactive terminal interface
11. **Debug Retrieval** → Inspect what the retriever actually returns
12. **Improve Retrieval** → Chunk metadata enrichment, hybrid search, and more

Key insights:

- **RAG quality depends on retrieval quality.** The LLM can only work with what the retriever gives it. When answers are wrong, debug the retrieval first—not the LLM.

- **Hybrid search beats pure vector search for domain-specific content.** Vector search finds conceptually similar content but can miss exact terminology. BM25 keyword search catches exact terms but misses paraphrased questions. Combining them via `EnsembleRetriever` covers both cases.

- **LangChain's convenience functions don't always compose well.** When `createRetrievalChain` didn't pass `chat_history` to our retriever, we used lower-level Runnables (`RunnableSequence`, `RunnablePassthrough`) to build exactly what we needed.

- **Chat history has cost implications.** Every API call reprocesses the full history, making total cost O(n²). Plan for memory management in production systems.

- **Build observability early.** A simple `--debug` flag and consistent logging pattern saves hours of guesswork. Graduate to proper tracing tools (LangFuse, Phoenix) as complexity grows.
