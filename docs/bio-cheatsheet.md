# MetaENCODE Biology Cheat Sheet

**For all team members — read this in 10 minutes and you can confidently field any biology question from the judges.**

The judges are DS/ML/engineering people, not biologists. They'll ask biology questions only to check that *you* understand the domain you're working in. You don't need deep knowledge — you need clear, confident framing.

---

## The Genome in 60 Seconds

Every cell in your body contains a copy of your genome — roughly 3 billion characters (A, T, C, G) of DNA. Think of it as an instruction manual.

Only about **1.5% of the genome contains genes** — the instructions for building proteins. Proteins are the workers that carry out cellular functions.

The other **98.5% was historically called "junk DNA"** but we now know it contains **regulatory elements** — molecular switches and dimmers that control *when*, *where*, and *how much* each gene is active.

**Key intuition:** A liver cell and a brain cell have *identical DNA*. The difference is which regulatory switches are turned on. Understanding those switches is the key to understanding how cells specialize, how diseases develop, and how organisms grow.

---

## What is ENCODE?

**ENCODE** stands for **Encyclopedia of DNA Elements**. It's a massive public research project funded by the NIH (US National Institutes of Health) that catalogs the functional elements of the genome — primarily those regulatory switches.

ENCODE contains **~27,000 experiments**, each measuring one type of molecular activity in one biological context. The data is **freely available** — no login or authentication required — through a public portal and REST API.

Think of ENCODE as a **parts list for the genome**: here are all the switches, here's where they are, and here's what they're doing in each cell type.

---

## What is an "Experiment" / "Dataset"?

Each ENCODE experiment measures **one molecular signal** in **one biological sample**. It's described by structured metadata:

| Field | What it means | Example |
|-------|--------------|---------|
| **Assay type** | The experimental technique used | ChIP-seq, RNA-seq, ATAC-seq |
| **Organism** | The species | Human (Homo sapiens), Mouse (Mus musculus) |
| **Biosample** | The cell or tissue being studied | K562 (a leukemia cell line), liver, brain |
| **Target** | The specific molecule being measured (if applicable) | H3K27ac, CTCF, p53 |
| **Lab** | The research lab that produced the data | Various ENCODE consortium labs |
| **Life stage** | The developmental stage of the organism | adult, embryonic, child |
| **Accession** | Unique identifier | ENCSR000AKS |

Each experiment also has a free-text **description** and **title** written by the producing lab.

---

## Assay Types (the 3 you need to know)

An **assay** is a laboratory technique for measuring something. Here are the three most common in ENCODE:

**ChIP-seq** (Chromatin Immunoprecipitation followed by sequencing): Identifies where specific proteins bind to DNA. Often used to find which regulatory switches are active. When someone mentions "H3K27ac ChIP-seq," they're mapping a specific chemical tag (H3K27ac) that marks active regulatory regions.

**RNA-seq** (RNA sequencing): Measures gene activity by reading the messenger RNA molecules a cell produces. Higher RNA levels = more actively used gene. This tells you which genes are turned on in a given cell type.

**ATAC-seq** (Assay for Transposase-Accessible Chromatin): Measures which parts of the genome are physically "open" and accessible — open chromatin is where regulatory activity happens. Think of it as finding which pages of the instruction manual are currently being read.

---

## Biosample, Organism, Developmental Stage

**Biosample:** The biological material being tested. Could be a specific cell line (K562, HeLa — these are standardized cells grown in labs worldwide), a tissue (liver, brain, heart), or a primary cell type (T cells, fibroblasts).

**Organism:** Almost all ENCODE experiments are in human or mouse. Mouse is used because it's a well-understood model organism — insights from mouse experiments often translate to humans.

**Developmental stage / Life stage:** When in the organism's life the sample was taken — adult, embryonic, newborn, etc. This matters because gene regulation changes dramatically during development.

---

## Biosample Hierarchy ("Slims")

ENCODE organizes biosamples into higher-level categories called "slims." MetaENCODE uses four classification systems so different researchers can browse from their preferred perspective:

| System | What it organizes by | Example: K562 maps to... |
|--------|---------------------|--------------------------|
| **Organ system** | Which organ | blood |
| **Cell type** | What kind of cell | leukemia cell line |
| **Germ layer** | Embryonic origin layer | mesoderm |
| **Body system** | Physiological system | immune system |

Don't worry about memorizing these. The key point: the same biosample can be viewed through multiple lenses, and MetaENCODE supports all four.

---

## The H3K27ac Shortcut

If you need to mention a specific molecular target, use **H3K27ac**. Explain it as:

> "H3K27ac is a chemical tag — specifically an acetyl group — found on a histone protein. Its presence marks active regulatory regions of the genome. If you see H3K27ac somewhere in the DNA, that region is probably acting as a switch that's turned on."

You do not need to go deeper than this. If a judge asks more, say: "The specifics of histone biochemistry are outside the scope of what MetaENCODE operates on — we treat it as a metadata label, just like any other assay target."

---

## The User Story (memorize this one)

> A researcher ran an experiment measuring which parts of the genome are active in developing mouse brain — specifically an H3K27ac ChIP-seq experiment. That experiment failed due to technical problems. They need a substitute from a public catalog.
>
> They go to ENCODE's portal and filter to "mouse brain ChIP-seq." They get back dozens of results — but those results are unranked. Which one is the *closest* match to their specific experimental context?
>
> They open MetaENCODE, find their experiment (or one like it), and click "Similar Datasets." MetaENCODE ranks the entire ENCODE catalog by multi-dimensional similarity — weighing text description, assay type, organism, cell type, lab, and replicate counts — and surfaces the 10 most similar experiments, scored and linked directly to ENCODE for download.

---

## "If a Judge Asks X, Say Y"

### "What is bioinformatics?"
> "Bioinformatics is the application of computational and data science methods to biological data. In our case, we're applying NLP, recommendation systems, and information retrieval techniques to a catalog of genomic experiments. The biology gives us the domain and the data; the data science gives us the methods."

### "Why can't researchers just use ENCODE's own search?"
> "ENCODE's search does filtering — it can show you all experiments matching certain criteria. What it can't do is *rank* which experiment is the closest match to a specific reference point across multiple metadata dimensions simultaneously. That ranking is what MetaENCODE adds."

### "What are these datasets, actually? What's in them?"
> "Each dataset is an experiment that measured one molecular signal — like gene activity or protein-DNA binding — in one biological sample. The raw data is sequencing reads — millions of short DNA fragments. But MetaENCODE doesn't touch the raw data. We operate entirely on the experiment metadata — the structured descriptions of what was measured, in what organism, in what tissue, by which lab."

### "Is this actually useful to real researchers?"
> "Yes — finding related datasets is a routine task in genomics research. Researchers do this to find controls, replicates, or substitutes for failed experiments. Right now they do it manually by browsing ENCODE and eyeballing metadata. MetaENCODE automates and quantifies that process. Our faculty advisor from UBIC confirmed this is a real pain point."

### "Why only ENCODE? What about other databases?"
> "ENCODE is one of several major genomic data repositories — GEO is another large one, and many labs host data in paper supplementals. We scoped to ENCODE as a proof of concept because it has a clean REST API and rich structured metadata. The architecture is repository-agnostic — you could plug in any metadata-described dataset catalog."

### "What's a cosine similarity score of 0.85 mean in practice?"
> "It means two experiments share about 85% of their combined metadata signal — similar descriptions, same or related assay types, same organism, related cell types. In practice, a score above 0.8 typically means the experiments are measuring related things in related biological contexts. Below 0.5, the experiments are fairly different."

---

## Terms You Might Hear (and What They Mean)

| Term | Plain English |
|------|--------------|
| **Chromatin** | The physical structure of DNA wrapped around proteins. "Open chromatin" = accessible for gene regulation. |
| **Histone** | Proteins that DNA wraps around. Chemical modifications to histones act as regulatory signals. |
| **Transcription factor** | A protein that binds to DNA to turn genes on or off. |
| **Ontology** | A structured vocabulary of biological terms with defined relationships — like a taxonomy. |
| **cCRE** | Candidate cis-Regulatory Element — a region of DNA that ENCODE has identified as a potential regulatory switch. |
| **Replicate** | A repeated measurement. Biological replicates use different samples; technical replicates repeat the same sample. More replicates = more confidence. |
| **Accession** | A unique ID for an ENCODE experiment (starts with ENCSR). |

---

*Read this once, then skim it again before the presentation. You don't need to memorize everything — you need to sound confident that you understand the domain your system operates in. If a question goes deeper than this document, it's perfectly fine to say: "The specific biology there is beyond what our system models — we operate on the metadata layer, not the raw experimental data."*
