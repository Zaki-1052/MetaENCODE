# MetaENCODE — Scripted Explanations & Q&A Prep

These are rehearsable scripts for technically dense sections. Read them aloud, time yourself, and adjust to your natural speaking rhythm. Each is marked with approximate duration.

---

## Script 1: "What is ENCODE and Why Should You Care" (~40 seconds)

> "Quick domain primer so everything that follows makes sense. The human genome is a 3-billion-character instruction manual. Only about 1.5% of it contains genes — the recipes for proteins. The other 98.5% is what scientists used to call 'junk DNA,' but it actually contains regulatory elements — think of them as switches and dimmers that control when, where, and how much each gene is active. A liver cell and a brain cell have the exact same DNA, but different switches are flipped.
>
> ENCODE — the Encyclopedia of DNA Elements — is a massive publicly-funded project that catalogs all of those switches. It contains over 27,000 experiments, each measuring a specific type of molecular activity in a specific biological context — a particular cell type, tissue, or organism. And every experiment is described by structured metadata: assay type, organism, biosample, lab, developmental stage, and free-text descriptions. That metadata is what our system operates on."

---

## Script 2: "How Similarity Scoring Works" (~50 seconds)

> "Here's how we score similarity. Each experiment gets transformed into a ~437-dimensional numerical vector that encodes everything we know about it.
>
> The largest component is a 384-dimensional text embedding from Sentence-BERT — specifically the all-MiniLM-L6-v2 model. This converts the experiment's text description into a vector where semantically similar descriptions are geometrically close — even if they use completely different words.
>
> On top of that, we concatenate one-hot encoded categorical features — assay type, organism, cell type, and lab — plus min-max scaled numeric features like replicate counts.
>
> Each feature group gets a weight: text at 50%, assay at 20%, organism at 15%, cell type at 10%, lab at 3%, numeric at 2%. But here's the key detail: we apply these weights as *square-root* scaling on the sub-vectors. The reason is that cosine similarity contributions are quadratic in vector magnitude. If you want text to contribute 50% of the final similarity score, you scale by sqrt(0.5), not 0.5 directly. This ensures the math matches the intent.
>
> Then we compute cosine similarity between the query vector and all 27,000 vectors. Brute force, exact results, under 100 milliseconds."

---

## Script 3: "Why SBERT Over Keyword Search" (~30 seconds)

> "Let me give you a concrete example of why we use a language model instead of keyword search.
>
> Say you have two experiment descriptions: 'Histone modification profiling' and 'ChIP-seq for histone marks.' A keyword search sees almost no overlap between those strings. But these describe essentially the same kind of experiment. Sentence-BERT understands that, because it was trained on over a billion sentence pairs to learn that similar meanings should map to nearby vectors.
>
> ENCODE's own portal does keyword search. It'll find exact matches. What it can't do is understand that two differently-worded descriptions refer to the same thing. That semantic understanding is what makes our similarity ranking meaningfully different from just filtering."

---

## Script 4: Demo Walkthrough Narration (~3 minutes)

> **[Open app, Search tab visible]**
>
> "Let me walk you through a realistic use case. Say I'm a researcher studying chromatin accessibility in mouse brain tissue. I want to find related experiments.
>
> **[Set filters: Organism → Mouse, Assay Type → ChIP-seq, click Search]**
>
> I can use the faceted search to narrow down. I'll filter to mouse ChIP-seq experiments. The sidebar pulls vocabulary directly from ENCODE's API — these aren't hardcoded lists; they reflect the actual contents of the database. [Pause, results load] We get a filtered table of matching experiments. I can see accession IDs, descriptions, biosamples, and labs.
>
> **[Click on a specific experiment row]**
>
> I'll select this one — [read the accession and brief description]. Now I have my seed dataset.
>
> **[Click 'Similar Datasets' tab]**
>
> Now I click over to Similar Datasets. [Pause for load — should be instant] Here are the top 20 most similar experiments, ranked by our multi-modal similarity score. Notice the scores — the top results are in the high 0.8s and 0.9s. These share the same organism, similar assay types, and related biosamples. As you scroll down, the scores decrease and the metadata diverges.
>
> Each result links directly to the ENCODE portal — [hover over link] — so a researcher can immediately access the raw data files.
>
> **[Click 'Visualize' tab]**
>
> Now let's look at the embedding space. This is a UMAP projection of all 27,000 experiments into 2D. [Pause] Each dot is an experiment. Right now it's colored by assay type — you can see clear clusters forming. ChIP-seq experiments cluster together, RNA-seq over here, ATAC-seq here.
>
> **[Switch color-by to Organism]**
>
> If I color by organism instead — there's a clean separation between human and mouse experiments. The embedding space is capturing real biological structure.
>
> **[Switch to 'Similar Only' view]**
>
> And if I switch to 'Similar Only,' I can see just the datasets that were ranked as most similar to my seed. They cluster tightly — which is a good sanity check that the similarity scores are geometrically meaningful.
>
> **[Return to Search tab briefly]**
>
> One more thing — you can also enter an accession directly if you already know what you're looking for. [Type accession, click Load] And the spell correction handles typos in search terms — it uses a combination of edit-distance and phonetic matching tuned to biological vocabulary.
>
> That's the app."

---

## Script 5: Prepared Q&A Answers

### Q: "How good is your similarity? How do you know the rankings are correct?"

> "That's a great question and one we've thought carefully about. There's no single ground truth for 'similarity' in this domain — what counts as similar depends on what the researcher is looking for. A computational biologist might prioritize assay type; a developmental biologist might prioritize organism and developmental stage.
>
> What we can validate: our integration tests confirm that experiments sharing the same assay type rank higher than experiments with different assays, and that changing the feature weights produces different rankings — the system is sensitive to the signals we're encoding. We also verified that query-time similarity matches batch-computed similarity exactly — no implementation bugs.
>
> That said, we know the default weights are a general-purpose prior, not a personalized model. One of our faculty advisors tested a query where the expected result appeared around rank 20 instead of top 5 — which tells us the weights don't perfectly match every user's intent. The faceted search gives researchers manual control when they need precision; the similarity engine is for broader exploration."

### Q: "Why this embedding model? Why not something domain-specific?"

> "We chose all-MiniLM-L6-v2 because it's the best general-purpose sentence embedding model for its size — 22 million parameters, 384 dimensions, about 2 milliseconds per text on CPU. It's the standard recommendation for semantic similarity tasks.
>
> That said, we know domain-specific models like PubMedBERT or BioSentVec could improve results for biological text. Those models are pre-trained on biomedical literature and would better capture the semantics of assay names, biosample terms, and experiment descriptions. We scoped that as a clear next step — [show hidden slide if appropriate] — but for a proof of concept, the general-purpose model demonstrates that the architecture works and produces meaningful similarity rankings."

### Q: "How does this scale?"

> "Right now we handle 27,000 experiments with brute-force cosine similarity — that's exact nearest neighbors in under 100 milliseconds. At this scale, brute force is fast enough and gives us exact results with no approximation error.
>
> We know exactly where the scalability ceiling is: at around 100,000+ experiments, brute-force becomes slow enough that you'd want approximate nearest neighbor methods — something like FAISS or HNSW indexing. Those trade a small amount of accuracy for dramatically faster query times. The architecture supports that swap because the similarity engine is a clean abstraction — you'd just change the backend from scikit-learn's NearestNeighbors to a FAISS index.
>
> The precomputation pipeline — fetching, embedding, feature combination — would also need to scale. We already run it on SDSC Expanse with 8 CPUs and 40 GB RAM for the full dataset, and it's embarrassingly parallel, so it scales linearly with more cores."

### Q: "What are the limitations?"

> "Three main ones. First, similarity quality depends on what the researcher values — our default weights are a reasonable starting point but not personalized. Users who need precise control can use the faceted search directly.
>
> Second, ENCODE is one of several genomic data repositories. Many labs publish to GEO or host data in paper supplementals. We scoped to ENCODE as a proof of concept, but the architecture generalizes — you could plug in any metadata-described dataset catalog.
>
> Third, the embedding model is general-purpose. A biomedical-specific model would likely improve similarity quality for domain terminology. That's our top priority for next steps."

### Q: "What was the hardest engineering challenge?"

> "Parsing ENCODE's API responses. The JSON structures are deeply nested and inconsistent — organism name might be buried three levels deep under replicates, and the field structure varies across experiments. Our parsing method handles 3+ format variations per field with defensive fallbacks at every level. That was more work than the ML pipeline itself."

### Q: "Why Streamlit instead of a React frontend?"

> "The team is Python-focused, and this is fundamentally a data exploration tool — not a CRUD app. Streamlit lets us go from DataFrame to interactive web app in pure Python without writing any frontend code. The trade-off is less UI customization, but for a data science tool focused on search and visualization, the reactive model fits naturally."

### Q: "What would you do with more time?"

> [Show hidden Future Implementations slide]
> "Three things. Domain-specific embeddings with PubMedBERT to improve similarity quality. User-tunable weights so researchers can adjust the importance of each feature at runtime. And extending to GEO and other repositories to increase coverage — which would also push us toward approximate nearest neighbors for scalability."

### Q: "How is this different from just using ENCODE's search?"

> "ENCODE's search does filtering — show me all mouse ChIP-seq experiments in brain tissue. That returns a flat, unranked list. MetaENCODE does ranking — given this specific experiment, which of the 27,000 others is the closest match across *all* metadata dimensions simultaneously? Filtering answers 'what matches these criteria.' Ranking answers 'what is most similar to *this*.'"
