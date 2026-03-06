# Plan: EDD + Re-ranking for Grimoire Oracle

**Date:** 2026-03-06
**Branch:** improve-retrieval-consistency

---

## Goal

Build an evaluation framework (golden Q&A dataset + recall@K script) to measure
retrieval quality across the full corpus, then add a cross-encoder re-ranker to
the query pipeline that addresses vocabulary mismatch and table embedding quality
at query time — without changing ingestion.

---

## Acceptance Criteria

- [ ] A golden dataset of ≥15 Q&A pairs covers multiple topic areas (equipment,
      classes, combat, magic, monsters)
- [ ] Eval script runs without starting the full UI and prints recall@K (K=8) to
      stdout
- [ ] Eval script detects the "plate armor" regression: `## Armor` chunk appears
      in top-8 retrieved docs after table normalization changes
- [ ] A re-ranker is integrated into the retrieval pipeline between ensemble
      retrieval and answer generation
- [ ] Re-ranked results demonstrably improve on at least the plate armor query
      (visible in debug log)
- [ ] `npx tsc --noEmit` passes throughout
- [ ] No `any` types, no `!` non-null assertions

---

## Files to Create

```
eval/
  golden.json          # Q&A dataset: [{question, source_file, content_fragment}]
  run-eval.ts          # Recall@K script — runs retrievers, prints results
```

## Files to Modify

```
src/oracle-logic.ts    # Add re-ranker step after ensemble retrieval
src/constants.ts       # Add RERANKER_MODEL, RERANKER_TOP_N constants
```

---

## Implementation Phases

### Phase 1: Golden Dataset

**Goal:** Create a hand-curated set of questions with known correct source
chunks. This is the foundation — without it, all retrieval changes are unverified.

**Tasks:**
- Create `eval/` directory
- Write `eval/golden.json` with ≥15 entries spanning:
  - Equipment costs (plate mail, weapons, mounts)
  - Class abilities (thief skills, cleric spells)
  - Combat rules (initiative, morale)
  - Monster stats
  - Magic research costs
- Each entry: `{ "question": "...", "source_file": "vault/rules/...", "content_fragment": "..." }`
  where `content_fragment` is a short string that must appear in the correct chunk

**Verification:**
- [ ] JSON is valid and parseable
- [ ] At least one entry per major vault category
- [ ] Plate armor question is included

#### Agent Context

```
Files to create: eval/golden.json
Files to modify: (none)
Test command: node -e "JSON.parse(require('fs').readFileSync('eval/golden.json','utf8')); console.log('valid')"
RED gate: (no test phase — content review only)
GREEN gate: JSON parses without error; ≥15 entries; each has question/source_file/content_fragment keys
Constraints:
  - source_file must be a real path relative to project root (verify each one exists)
  - content_fragment must be a substring that would realistically appear in the correct chunk
  - Cover at least: Equipment & Services, Classes, Combat, Magic, Monsters categories
```

---

### Phase 2: Eval Script [no-test]

**Goal:** A script that loads the retriever stack directly (no UI), runs each
golden question through it, and reports recall@K.

**Tasks:**
- Create `eval/run-eval.ts`
- Import `setupOracle` from `@src/oracle-logic` (or extract the retriever
  setup into a shared helper if needed)
- For each golden entry:
  - Invoke the ensemble retriever with the question
  - Check if any retrieved doc's `pageContent` contains `content_fragment`
  - Check if any retrieved doc's `metadata.source` contains `source_file`
- Print per-question pass/fail and a final recall@K score
- Exit with code 1 if recall < 0.8 (useful for CI)

**Verification:**
- [ ] `npx tsx eval/run-eval.ts` runs without error
- [ ] Plate armor question fails (regression detected) before re-ingestion
- [ ] Plate armor question passes after re-ingestion with table normalization
- [ ] Output clearly shows which questions fail and why

#### Agent Context

```
Files to create: eval/run-eval.ts
Files to modify: (none — but may need to extract retriever setup from setupOracle)
Test command: npx tsx eval/run-eval.ts
RED gate: Script runs; plate armor question shows FAIL before re-ingestion
GREEN gate: Script runs; recall@K printed; exits 0 when recall ≥ 0.8
Constraints:
  - Import paths use @src/* alias (tsconfig bundler resolution, no .js extension)
  - No any types, no ! assertions
  - Do NOT spin up the full ink UI — invoke retriever directly
  - Retriever must be initialized the same way as in setupOracle (same embedder,
    same BM25 chunks, same ensemble weights) to ensure eval matches production
  - If setupOracle needs refactoring to expose retriever separately, keep changes minimal
```

---

### Phase 3: Re-ranker Integration [no-test]

**Goal:** Add a cross-encoder re-ranker step after ensemble retrieval that
re-scores candidate docs against the exact query. This addresses vocabulary
mismatch ("plate armor" vs "plate mail") and table embedding issues at query
time.

**Tasks:**
- Research available re-ranker options compatible with local/Ollama setup:
  - `@langchain/community` has `CohereRerank` (requires API key)
  - For fully local: use a simple cross-encoder scoring approach or
    `FlashrankRerank` from `@langchain/community/document_compressors/flashrank`
    (no API key needed, runs locally)
- Add `RERANKER_TOP_N` constant to `src/constants.ts` (suggest: 4)
- Integrate the re-ranker as a `ContextualCompressionRetriever` wrapping the
  ensemble retriever in `setupOracle`
- Update debug logging to show pre- and post-rerank doc order

**Verification:**
- [ ] `npx tsx eval/run-eval.ts` recall@K improves vs baseline
- [ ] Debug log for "plate armor" shows `## Armor` chunk in re-ranked top results
- [ ] `npx tsc --noEmit` passes

#### Agent Context

```
Files to modify:
  src/oracle-logic.ts  — wrap ensembleRetriever with ContextualCompressionRetriever
  src/constants.ts     — add RERANKER_TOP_N = 4
Files to create: (none)
Test command: npx tsx eval/run-eval.ts
GREEN gate: recall@K ≥ previous baseline; plate armor in top-4 re-ranked results
Constraints:
  - Prefer a local re-ranker (no API key) — FlashrankRerank is the first option to try
  - If no local option works, document why and propose CohereRerank with API key as fallback
  - The re-ranker wraps the retriever, it does NOT replace it — ensemble still runs first
  - RERANKER_TOP_N should be ≤ RETRIEVAL_K (re-ranker filters down, never up)
  - No any types, no ! assertions
  - Import paths: @src/* alias
```

---

## Constraints & Considerations

- **No UI required:** Eval script bypasses Ink and invokes retriever directly.
  This may require light refactoring to expose retriever setup as a standalone
  function — keep it minimal.
- **Local-first:** This is a learning project using Ollama. Prefer re-ranker
  options that don't require external API keys.
- **Ingestion must be re-run** before eval is meaningful, since the table
  normalization changes in `scripts/ingest.ts` haven't been applied to the index
  yet.
- **TypeScript rules:** No `any`, no `!`. Bundler module resolution — no `.js`
  extensions on local imports.

---

## Out of Scope

- Query expansion / RAG-Fusion (overlaps with re-ranking; evaluate after)
- Fine-tuning embedding models
- UI changes
- Automated CI integration (eval script exit code is there if needed later)

---

## Approval Checklist

- [ ] Golden dataset questions reviewed for correctness and coverage
- [ ] Eval script design matches how `setupOracle` actually builds the retriever
- [ ] Re-ranker choice confirmed (local vs API-key)
- [ ] Re-ingestion run before final eval numbers are trusted

---

## Inline Task Graph (beads unavailable)

### P1: Golden Dataset [no-test] [no blockers]

Create `eval/golden.json` with ≥15 hand-curated Q&A entries.

**Agent Context:**
- Files to create: `eval/golden.json`
- Test command: `node -e "JSON.parse(require('fs').readFileSync('eval/golden.json','utf8')); console.log('valid')"`
- GREEN gate: JSON parses; ≥15 entries; each has `question`, `source_file`, `content_fragment`
- Constraints: verify each `source_file` path exists; cover Equipment, Classes, Combat, Magic, Monsters

### P2: Eval Script [no-test] [blocked-by: P1]

Create `eval/run-eval.ts` that loads the retriever directly and reports recall@K.

**Agent Context:**
- Files to create: `eval/run-eval.ts`
- Files to modify: `src/oracle-logic.ts` (if retriever needs extracting)
- Test command: `npx tsx eval/run-eval.ts`
- GREEN gate: Runs without error; prints per-question pass/fail and recall@K; exits 1 if recall < 0.8
- Constraints: no UI; use @src/* imports; no `any`, no `!`; retriever must match production setup

### P3: Re-ranker Integration [no-test] [blocked-by: P2]

Wrap ensemble retriever with `ContextualCompressionRetriever` + local cross-encoder.

**Agent Context:**
- Files to modify: `src/oracle-logic.ts`, `src/constants.ts`
- Test command: `npx tsx eval/run-eval.ts`
- GREEN gate: recall@K improves; plate armor appears in top-4 re-ranked results; `npx tsc --noEmit` passes
- Constraints: local re-ranker preferred (FlashrankRerank first); RERANKER_TOP_N ≤ RETRIEVAL_K; no `any`, no `!`

---

## Next Steps

```
Plan file saved to: docs/plans/2026-03-06-edd-and-reranking-plan.md

Before running /craft:
  1. Re-run ingestion:  npx tsx scripts/ingest.ts
  2. Verify plate armor is fixed: npx tsx src/index.tsx
  3. Then run /craft to execute phases in order

Session recovery: phases are listed in order above — if interrupted,
resume from the first incomplete phase.
```
