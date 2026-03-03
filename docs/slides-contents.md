# MetaENCODE DinoCage — Slide-by-Slide Plan

**Total time budget:** ~6 min slides + ~3.5 min demo + ~30s buffer = 10 min
**Presentation date:** March 11, 5:30–8:30 PM, HDSI MPR
**Slide due date:** March 6, 11:59 PM

> DS3 logo in top right corner of ALL slides.
> Design principle: slides show what's HARD to say verbally (diagrams, numbers, structure). Speaker provides narrative. No full sentences on slides.

---

## Slide 1: Title
**Time:** ~10 seconds

**On the slide:**
- "MetaENCODE" (large)
- Subtitle: "A Recommendation Engine for Genomics Datasets"
- Team member names + roles
- DS3 × UBIC logos
- One-liner at bottom: *"Netflix recommendations, but for scientific data."*

**Speaker says:**
- "Hi, we're Team MetaENCODE — [names]. We built a recommendation engine that helps genomics researchers discover related datasets across a catalog of 27,000 experiments."

---

## Slide 2: The Problem
**Title:** 27,000 Experiments. No Way to Rank Them.
**Time:** ~50 seconds

**On the slide:**
- Left column header: **"ENCODE Portal Search"**
  - "Filter by category → flat, unranked list"
  - "Which result is best? ¯\\\_(ツ)\_/¯"
- Right column header: **"MetaENCODE"**
  - "Select a seed → ranked by multi-dimensional similarity"
  - "Score: 0.94, 0.91, 0.87..."
- Bottom: the user story as a one-line scenario: *"Researcher's experiment failed. Needs the closest substitute from 27,000 options."*

**Visual:** This is a conceptual comparison, NOT app screenshots. Use a simple two-column layout with icons or minimal mockups (a generic flat list on the left vs. a scored/ranked list on the right). Think whiteboard-sketch level.

**Speaker says:**
- "Here's the scenario. A researcher ran an experiment measuring genome activity in developing mouse brain — and it failed. They need a substitute from a public database called ENCODE, which has over 27,000 experiments."
- "ENCODE's own portal lets you filter — show me all mouse brain experiments — but it returns a flat, unranked list. It can't tell you which one is the closest match to your specific context."
- "MetaENCODE solves that. Given any experiment, it ranks the entire catalog by multi-dimensional similarity and returns a scored list."

---

## Slide 3: Background — What is ENCODE?
**Title:** The ENCODE Project
**Time:** ~40 seconds

**On the slide:**
- A simple **diagram** (this is the key visual for this slide):
  - DNA strand → zoomed section showing "Genes: 1.5%" and "Regulatory elements: 98.5%"
  - Arrow from regulatory elements → "ENCODE catalogs these"
  - Below the diagram: "27,000+ experiments | 30+ assay types | human & mouse | publicly funded (NIH)"
- That's it. No other text. The diagram does the work.

**Visual:** Hand-drawn or clean diagrammatic style. NOT a stock photo of DNA. The goal is to make the 1.5% vs 98.5% split visually intuitive — maybe a colored bar where the gene portion is tiny and the regulatory portion is huge, with ENCODE's scope highlighted.

**Speaker says:**
- "Quick domain primer. The human genome is a 3-billion-character instruction manual. Only 1.5% are genes — protein recipes. The other 98.5% contains regulatory elements — switches that control when and where each gene is active."
- "ENCODE — the Encyclopedia of DNA Elements — is a public project that catalogs those switches. 27,000 experiments, each measuring a specific molecular signal in a specific biological context. Every experiment is described by structured metadata — assay type, organism, biosample, lab, developmental stage, and free-text descriptions. That metadata is what our system operates on."

---

## Slide 4: Our Solution
**Title:** MetaENCODE: Search, Rank, Visualize
**Time:** ~30 seconds

**On the slide:**
- Three boxes in a row, each with a short label and one-line description:
  - **Search** — "Filter 27K experiments by assay, organism, tissue, lab, keywords"
  - **Rank** — "Multi-modal similarity scoring: text + categorical + numeric features"
  - **Visualize** — "UMAP/PCA/t-SNE projections of the embedding space"
- Below: *"Faceted search for precision. Similarity ranking for exploration."*

**Visual:** Three clean boxes/cards with icons (magnifying glass, ranked list, scatter plot). Simple and structural — communicates the three-feature architecture at a glance.

**Speaker says:**
- "MetaENCODE has three core capabilities. Faceted search to filter the catalog with precise criteria. Similarity ranking — the main feature — which scores and ranks all experiments against a seed. And interactive visualization to explore how experiments cluster in the embedding space."
- "For a DS framing: this is a large structured catalog. A user picks a reference point. The system ranks by multidimensional relevance. It's a recommendation system."

---

## Slide 5: Architecture & Data Pipeline
**Title:** System Architecture
**Time:** ~45 seconds

**On the slide:**
- **Pipeline diagram** (this IS the slide — almost no text besides labels):

```
ENCODE REST API
      ↓
  Rate-Limited Fetch (10 req/s)
      ↓
  Metadata Processing (text cleaning, ontology enrichment)
      ↓
  SBERT Embedding (all-MiniLM-L6-v2 → 384-dim)
      ↓
  Feature Combination (weighted concat → ~437-dim)
      ↓
  Precomputed Cache (pickle, atomic writes)
      ↓
  Cosine Similarity (brute-force k-NN, <100ms)
      ↓
  Streamlit UI
```

- Annotate with: "Offline (precompute once)" bracket on top half, "Online (real-time)" bracket on bottom half
- Small callout: "Full pipeline: ~6 hrs on SDSC Expanse (8 CPU, 40GB RAM)"

**Visual:** Clean flow diagram with boxes and arrows. Use two colors to distinguish offline precomputation from online serving. This should be the most "engineering-looking" slide.

**Speaker says:**
- "Here's the data flow. The top half is offline precomputation — we fetch all metadata from ENCODE's REST API, clean and enrich it, run it through Sentence-BERT to produce text embeddings, then combine those with categorical and numeric features into a single vector per experiment. All cached to disk."
- "The bottom half is online serving. The app loads cached vectors, builds a nearest-neighbors index, and serves similarity queries in under 100 milliseconds. Visualization coordinates are also precomputed so the scatter plots load instantly."

---

## Slide 6: How Similarity Scoring Works
**Title:** Multi-Modal Similarity Scoring
**Time:** ~60 seconds ⭐ (Technical centerpiece)

**On the slide:**
- **Feature weight table** (make this visually prominent — maybe a horizontal stacked bar):

| Feature | Weight | Dimensions |
|---------|--------|------------|
| Text (SBERT) | 50% | 384 |
| Assay type | 20% | ~30 |
| Organism | 15% | ~2 |
| Cell type | 10% | ~50 |
| Lab | 3% | ~15 |
| Numeric | 2% | 4 |

- **Vector diagram** below the table:
  `[========= text 384d =========][assay][org][cell][lab][n]`
  with √weight scaling arrows pointing to each segment

- **Key equation** in a callout box:
  `scale = √(weight)  →  cosine contribution ∝ weight`

- Bottom: "Cosine similarity over 27K vectors → ranked results in <100ms"

**Visual:** The stacked-bar or segmented-vector diagram is the hero visual here. It makes the weighting scheme immediately intuitive — the text segment is visually dominant at 384 dims, with progressively smaller categorical segments. The √weight callout is a small formula box, not a full derivation.

**Speaker says:**
- "This is the core of the system. Each experiment becomes a ~437-dimensional vector built from six weighted feature groups."
- "Text carries 50% weight — Sentence-BERT encodes descriptions into 384-dimensional vectors where semantically similar texts are geometrically close. 'Histone modification profiling' and 'ChIP-seq for histone marks' share almost no words, but SBERT recognizes them as describing similar experiments. That's the differentiator over keyword search."
- "The remaining 50% is structured: assay type at 20%, organism 15%, cell type 10%, lab 3%, numeric features like replicate counts at 2%."
- "One technical detail: weights are applied as square-root scaling, not direct multiplication. Cosine similarity contributions are quadratic in magnitude, so √weight ensures the math matches the intent."
- "Final similarity is brute-force cosine over all 27K vectors. Exact results, no approximation, under 100ms."

---

## Slide 7: Engineering Quality & Testing
**Title:** Engineering & Evaluation
**Time:** ~35 seconds

**On the slide:**
- **Four large metric callouts** (think dashboard-style big numbers):
  - **708** tests (unit + integration)
  - **27,398** experiments indexed
  - **<100ms** query latency
  - **~437** dimensions per vector
- Below the metrics, three short engineering highlights:
  - "Atomic cache writes — no partial-write corruption"
  - "Vocabulary from ENCODE API — never hardcoded"
  - "Integration tests verify: weight changes → different rankings; same-assay → higher similarity"

**Visual:** Big bold numbers dominate the slide. The three text bullets below are small and secondary — the speaker elaborates on them. No screenshots, no code.

**Speaker says:**
- "On engineering quality: 708 tests across every module, including integration tests that verify the full pipeline end-to-end. We confirmed that weight changes produce different rankings and that same-assay experiments rank higher — the system is sensitive to the right signals."
- "Infrastructure: atomic writes prevent cache corruption, all vocabulary is fetched from the ENCODE API as a single source of truth, and the full precomputation pipeline runs on SDSC Expanse."

---

## Slide 8: Demo Transition
**Title:** Live Demo
**Time:** ~10 seconds (then switch to app)

**On the slide:**
- Just the words **"Live Demo"** centered and large
- Maybe a subtle MetaENCODE logo

**Speaker says:**
- "Let me show you how this works."
- [Switch to live app — presenter shares browser window]

---

## ── LIVE DEMO ── (~3–3.5 minutes)

(See Deliverable 2 for full narration script)

**Flow:**
1. **Search tab** (~45s): Filter by organism + assay → show results → select a dataset
2. **Similarity tab** (~60s): Show ranked results with scores → note metadata patterns → click ENCODE link
3. **Visualization tab** (~60s): UMAP colored by assay → show clusters → switch to "Similar Only" → switch color-by
4. **Quick hits** (~15s): Direct accession lookup, spell correction if time allows

---

## Slide 9: Limitations & Learnings
**Title:** Limitations & Learnings
**Time:** ~30 seconds

**On the slide:**
- Three items, each as a short bolded phrase + one-line elaboration:
  - **Similarity is subjective** — "Default weights are a general-purpose prior, not personalized to each researcher's priorities"
  - **ENCODE is one repository** — "Scoped as proof of concept; architecture generalizes to GEO and others"
  - **General-purpose embeddings** — "Domain-specific models (PubMedBERT) are a clear next step"

**Visual:** Clean, three-row layout. No more than ~20 words per row. White space is fine — this slide should breathe.

**Speaker says:**
- "Honest limitations. Similarity quality depends on what the researcher values — our weights are a reasonable prior, not a personalized model. The faceted search gives manual control when precision matters."
- "ENCODE is one of several genomic data repositories. We scoped here as a proof of concept — the architecture generalizes."
- "And our embedding model is general-purpose. Domain-specific models are the top next step."

---

## Slide 10: Thank You / Q&A
**Title:** Questions?
**Time:** ~10 seconds → open Q&A

**On the slide:**
- **"Questions?"** (large, centered)
- Team member names
- GitHub repo URL
- DS3 × UBIC logos

**Speaker says:**
- "Thank you. Happy to take questions."

---

## [SKIP — Q&A RESERVE] Slide 11: Future Implementations

> ⚠️ HIDDEN during presentation. Only shown if a judge asks about next steps.
> Cue: "Great question — we actually have a slide on that."

**On the slide:**
- Three boxes:
  - **Domain-Specific Embeddings** — "PubMedBERT / BioSentVec trained on biomedical literature"
  - **User-Tunable Weights** — "Runtime sliders: 'I care more about organism than assay type'"
  - **Scale Beyond ENCODE** — "GEO integration → 100K+ experiments → FAISS approximate indexing"

**Speaker says (if triggered):**
- "Three high-impact next steps. First, domain-specific embeddings — PubMedBERT is trained on biomedical text and would better capture assay and biosample semantics. Second, letting researchers adjust feature weights at runtime instead of fixed at precompute time. Third, extending to GEO and other repositories, which pushes us into approximate nearest neighbors territory."

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

---

## Slide Design Notes

**What goes on slides vs. what the speaker says:**
- Slides: diagrams, numbers, structure, key terms. Maximum ~20 words of body text per slide (titles don't count).
- Speaker: narrative, explanations, transitions, context. The speaker notes in Deliverable 2 cover this.

**Visuals that actually help (not screenshots):**
- Slide 3: The 1.5% genes vs 98.5% regulatory diagram — makes the domain instantly tangible
- Slide 5: The pipeline diagram with offline/online split — communicates architecture at a glance
- Slide 6: The segmented vector bar + weight table — makes the feature engineering visually intuitive
- Slide 7: Big dashboard-style numbers — instant credibility signal

**Things to avoid:**
- App screenshots on slides (redundant with live demo)
- Full sentences (speaker should not be reading slides)
- Code blocks or raw JSON on slides (save for Q&A if someone asks)
- More than 3-4 elements competing for attention per slide
