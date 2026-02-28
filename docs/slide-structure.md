# MetaENCODE DinoCage — Slide-by-Slide Plan

**Total time budget:** ~6 min slides + ~3.5 min demo + ~30s buffer = 10 min
**Presentation date:** March 11, 5:30–8:30 PM, HDSI MPR
**Slide due date:** March 6, 11:59 PM

> Remember: DS3 logo in top right corner of ALL slides.

---

## Slide 1: Title
**Title:** MetaENCODE: A Recommendation Engine for Genomics Datasets
**Time:** ~10 seconds

**Speaker points:**
- "Hi, we're Team MetaENCODE — [names]. We're a DS3 x UBIC collaborative project."
- "We built a recommendation engine that helps genomics researchers find related datasets. Think of it as Netflix recommendations, but for scientific data."

**Visual:** Project name, team member names + roles, DS3 and UBIC logos. Screenshot of the app in the background.

---

## Slide 2: The Problem
**Title:** The Problem: 27,000 Experiments, No Way to Rank Them
**Time:** ~50 seconds

**Speaker points:**
- "Imagine you're a genomics researcher. You ran an experiment measuring which parts of the genome are active in developing mouse brain tissue — and that experiment failed. You need a substitute from a public database."
- "That database is called ENCODE. It has over 27,000 experiments. ENCODE's own search portal lets you filter — show me all mouse brain experiments — but it returns hundreds of unranked results. It can't tell you *which one is the closest match* to your specific context."
- "That ranking is what's missing. Researchers currently solve this by scrolling through hundreds of results and eyeballing metadata. That's the pain point."
- "We built MetaENCODE to solve it: given any experiment, rank the entire catalog by multi-dimensional similarity."

**Visual:** Split screen — LEFT: screenshot of ENCODE portal search returning 200+ flat results. RIGHT: MetaENCODE returning a ranked, scored list. A visual "before/after."

---

## Slide 3: Background — What is ENCODE?
**Title:** Background: The ENCODE Project
**Time:** ~40 seconds

**Speaker points:**
- "Quick domain primer — you don't need to be a biologist to follow this. The human genome is a 3-billion-character instruction manual. Only about 1.5% are genes — the protein recipes. The other 98.5% contains regulatory elements: switches and dimmers that control when, where, and how much each gene is active."
- "ENCODE — the Encyclopedia of DNA Elements — is a massive public project funded by the NIH that catalogs all those regulatory switches. It contains ~27,000 experiments, each measuring a specific type of molecular activity in a specific biological context — a cell line, a tissue, an organism."
- "Each experiment is described by structured metadata: assay type, organism, biosample, lab, developmental stage, and free-text descriptions. That metadata is what we work with."

**Visual:** Simple diagram — DNA strand → zoom into gene (1.5%) vs. regulatory region (98.5%) → ENCODE logo with "27,000+ experiments cataloging these switches." Keep it visual, not text-heavy.

---

## Slide 4: Our Solution
**Title:** MetaENCODE: Similarity-Ranked Dataset Discovery
**Time:** ~30 seconds

**Speaker points:**
- "MetaENCODE is a web application with three core features: faceted search to filter the catalog, similarity ranking to find the most related datasets to any seed experiment, and interactive visualization to explore the dataset landscape."
- "The key differentiator is the similarity engine. It's not keyword matching — it's a multi-modal scoring system that blends semantic text understanding with structured metadata to produce a single relevance score."
- "For a DS audience: this is a large structured catalog. A user picks a reference point. The system ranks the catalog by multidimensional relevance. It's a recommendation system."

**Visual:** App screenshot showing the three tabs. Arrows or callouts labeling: "Search → Select → Rank → Visualize" as a user flow.

---

## Slide 5: Architecture & Data Pipeline
**Title:** System Architecture
**Time:** ~45 seconds

**Speaker points:**
- "Here's how data flows through the system. We ingest experiment metadata from ENCODE's public REST API — rate-limited at 10 requests per second. That raw metadata gets cleaned, enriched with ontology mappings — biosample to organ system, cell type, germ layer — and combined into a unified text field."
- "That text is encoded into 384-dimensional vectors using Sentence-BERT. Those text embeddings are then concatenated with one-hot categorical features and normalized numeric features into a combined ~437-dimensional vector per experiment."
- "All of this is precomputed offline. At runtime, the app loads cached vectors, builds a nearest-neighbors index, and serves similarity queries in under 100 milliseconds."

**Visual:** The architecture pipeline diagram from the README, cleaned up for slides:
`ENCODE API → Metadata Processing → SBERT Embedding → Feature Combination → Cache → Similarity Engine → Streamlit UI`. Use icons/boxes, not code formatting.

---

## Slide 6: How Similarity Scoring Works
**Title:** Multi-Modal Similarity Scoring
**Time:** ~60 seconds ⭐ (This is the technical centerpiece — rubric: Methods/Architecture 6 pts)

**Speaker points:**
- "This is the core of the system. Similarity is computed from a weighted blend of six feature groups." [Point to weight table on slide]
- "The text embedding carries the most weight at 50%. We use SBERT — specifically all-MiniLM-L6-v2 — which converts text descriptions into 384-dimensional vectors where semantically similar texts are geometrically close. So 'Histone modification profiling' and 'ChIP-seq for histone marks' are recognized as similar even though they share almost no words. That's the core advantage over keyword search."
- "The remaining 50% comes from structured features: assay type at 20%, organism at 15%, cell type at 10%, lab at 3%, and numeric features like replicate counts at 2%."
- "One technical detail worth noting: we apply weights as square-root scaling on the sub-vectors, not direct multiplication. This is because cosine similarity contributions are quadratic in magnitude — sqrt(weight) ensures the contribution to the final score is proportional to the intended percentage."
- "The final similarity computation is brute-force cosine over 27,000 combined vectors. At 437 dimensions, that runs in under 100 milliseconds — exact results with no approximation trade-off."

**Visual:** Feature weight table (text 50%, assay 20%, organism 15%, cell type 10%, lab 3%, numeric 2%) with a diagram showing sub-vectors being concatenated. Maybe: `[====text 384d====][assay][org][cell][lab][num]` → cosine similarity → ranked results.

---

## Slide 7: Engineering Quality & Testing
**Title:** Engineering & Evaluation
**Time:** ~35 seconds

**Speaker points:**
- "On the engineering side: the codebase has 708 tests — unit tests across every module and integration tests that verify the full pipeline end-to-end. We verified that weight changes produce different similarity rankings and that same-assay experiments rank higher than cross-assay."
- "Infrastructure-wise: atomic cache writes prevent data corruption, all vocabulary values are fetched from ENCODE's API as a single source of truth — never hardcoded — and the full precomputation pipeline runs on SDSC Expanse for the 27K dataset."
- "The app serves similarity queries in under 100ms and visualization loads are instant from precomputed coordinates."

**Visual:** Key metrics as large numbers: "708 tests" | "27,000 experiments" | "<100ms query time" | "~437-dim vectors". Optionally a small screenshot of test output passing.

---

## Slide 8: Demo Transition
**Title:** Live Demo
**Time:** ~10 seconds

**Speaker points:**
- "Let me show you how this works in practice."
- [Switch to live app]

**Visual:** "Demo" in large text, or a screenshot of the app as a teaser. Keep simple.

---

## ── LIVE DEMO ── (~3–3.5 minutes)

**Demo flow** (see Deliverable 2 for full narration):

1. **Search tab** (~45s): Show faceted search — filter by organism (mouse), assay (ChIP-seq). Show how results populate. Pick a specific experiment.
2. **Similarity tab** (~60s): Click Similar Datasets. Walk through the ranked results, point out similarity scores, highlight that top results share key metadata dimensions. Click through to ENCODE portal link.
3. **Visualization tab** (~60s): Show UMAP scatter plot. Color by assay type — point out clusters. Switch to "Similar Only" view. Show how similar datasets cluster together. Switch color-by to organism — show human/mouse separation.
4. **Return to search** (~15s): Show direct accession lookup. Show spell correction working if time permits.

**Total demo: ~3 minutes**

---

## Slide 9: Limitations & Learnings
**Title:** Limitations & Learnings
**Time:** ~30 seconds

**Speaker points:**
- "Honest limitations: similarity quality depends on what the researcher actually values. Our default weights are a reasonable prior, but different researchers prioritize different dimensions. The faceted search gives users manual control when they need it — similarity is for broader exploration."
- "On the data side: ENCODE is one of several genomic data repositories. We scoped to ENCODE as a proof of concept — the architecture generalizes to other repositories like GEO."
- "On model choice: we use a general-purpose embedding model. Domain-specific models could improve results for biological text — that's a clear next step."

**Visual:** 2-3 concise points on slide (not full sentences — speaker fills in). Keep it honest but brief.

---

## Slide 10: Thank You / Q&A
**Title:** Thank You — Questions?
**Time:** ~10 seconds

**Speaker points:**
- "Thank you. We're happy to take questions."

**Visual:** Team names, GitHub repo link, project logo. "Questions?" prominently displayed.

---

## [SKIP — Q&A RESERVE] Slide 11: Future Implementations
**Title:** Future Implementations

> ⚠️ This slide is HIDDEN during the presentation. Only shown if a judge asks about next steps or limitations. Cue: "Great question — we actually have a slide on that."

**Speaker points (if triggered):**
- "We've identified three high-impact next steps."
- "First: domain-specific embeddings. Our current model — MiniLM — is general-purpose. Models like PubMedBERT or BioSentVec are trained on biomedical text and could significantly improve similarity quality for biological descriptions."
- "Second: user-tunable weights. Right now weights are fixed at precompute time. Allowing researchers to adjust the weight sliders at runtime — 'I care more about organism than assay type' — would make the system more flexible."
- "Third: scaling beyond ENCODE. The architecture is repository-agnostic. Extending to GEO or other databases would increase coverage from 27K to potentially hundreds of thousands of experiments — at which point we'd swap in approximate nearest neighbors via FAISS."

**Visual:** Three boxes: "Domain-Specific Embeddings (PubMedBERT)" | "User-Tunable Weights" | "Scale to GEO + FAISS Indexing"

---

## Timing Summary

| Slide | Content | Time |
|-------|---------|------|
| 1 | Title | 0:10 |
| 2 | The Problem | 0:50 |
| 3 | Background: ENCODE | 0:40 |
| 4 | Our Solution | 0:30 |
| 5 | Architecture | 0:45 |
| 6 | Similarity Scoring | 1:00 |
| 7 | Engineering & Testing | 0:35 |
| 8 | Demo Transition | 0:10 |
| — | **LIVE DEMO** | **3:00–3:30** |
| 9 | Limitations | 0:30 |
| 10 | Thank You / Q&A | 0:10 |
| **Total** | | **~8:30–9:00** |

Buffer of ~1–1.5 min for natural pacing variations. Safe under the 10-min cutoff.
