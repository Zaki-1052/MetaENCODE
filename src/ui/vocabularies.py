# src/ui/vocabularies.py
"""Vocabulary definitions for ENCODE dataset search and filtering.

This module provides comprehensive dictionaries of biological terms used
for autocomplete, filtering, and NLP-based term matching in the MetaENCODE
search interface.
"""

from typing import Dict, List, Set

# =============================================================================
# ASSAY TYPES
# =============================================================================
# Official ENCODE assay_term_name values from schema
# Source: https://github.com/ENCODE-DCC/encoded/blob/dev/src/encoded/schemas/mixins.json
ASSAY_TYPES: Dict[str, str] = {
    # Chromatin accessibility
    "ATAC-seq": "ATAC-seq (Chromatin Accessibility)",
    "DNase-seq": "DNase-seq (Chromatin Accessibility)",
    "FAIRE-seq": "FAIRE-seq (Chromatin Accessibility)",
    "MNase-seq": "MNase-seq (Nucleosome Mapping)",
    # ChIP-seq and variants
    "ChIP-seq": "ChIP-seq (Chromatin Immunoprecipitation)",
    "CUT&RUN": "CUT&RUN (Chromatin Profiling)",
    "CUT&Tag": "CUT&Tag (Chromatin Profiling)",
    "Mint-ChIP-seq": "Mint-ChIP-seq (Low-input ChIP)",
    # Chromatin conformation (NOTE: ENCODE uses "HiC" not "Hi-C")
    "HiC": "HiC (Chromatin Conformation)",
    "capture Hi-C": "capture Hi-C (Promoter Contacts)",
    "ChIA-PET": "ChIA-PET (Chromatin Interaction)",
    "PLAC-seq": "PLAC-seq (Promoter-Enhancer Contacts)",
    "5C": "5C (Chromatin Conformation)",
    "4C": "4C (Chromatin Conformation)",
    "genotype phasing by HiC": "genotype phasing by HiC",
    "SPRITE": "SPRITE (3D Chromatin)",
    "SPRITE-IP": "SPRITE-IP (3D Chromatin)",
    # Transcription - RNA-seq variants
    "RNA-seq": "RNA-seq (Transcriptome)",
    "polyA plus RNA-seq": "polyA plus RNA-seq (mRNA)",
    "polyA minus RNA-seq": "polyA minus RNA-seq (Non-polyA)",
    "small RNA-seq": "small RNA-seq (Small RNAs)",
    "microRNA-seq": "microRNA-seq (miRNAs)",
    "long read RNA-seq": "long read RNA-seq (Isoform)",
    "long read single-cell RNA-seq": "long read single-cell RNA-seq",
    "direct RNA-seq": "direct RNA-seq",
    # TSS mapping
    "CAGE": "CAGE (TSS Mapping)",
    "RAMPAGE": "RAMPAGE (TSS Mapping)",
    "RNA-PET": "RNA-PET (Transcript Boundaries)",
    # Nascent transcription
    "PRO-seq": "PRO-seq (Nascent Transcription)",
    "PRO-cap": "PRO-cap (Nascent Transcription)",
    "GRO-seq": "GRO-seq (Nascent Transcription)",
    "GRO-cap": "GRO-cap (Nascent Transcription)",
    "Bru-seq": "Bru-seq (Nascent Transcription)",
    "BruChase-seq": "BruChase-seq (Nascent Transcription)",
    "BruUV-seq": "BruUV-seq (Nascent Transcription)",
    # Single-cell assays
    "single-cell RNA sequencing assay": "scRNA-seq (Single-Cell Transcriptome)",
    "single-nucleus ATAC-seq": "snATAC-seq (Single-Nucleus Accessibility)",
    # DNA methylation
    "whole-genome shotgun bisulfite sequencing": "WGBS (Whole-Genome Bisulfite)",
    "RRBS": "RRBS (Reduced Representation Bisulfite)",
    "MeDIP-seq": "MeDIP-seq (Methylation)",
    "TAB-seq": "TAB-seq (5hmC Mapping)",
    # RNA-protein interactions
    "eCLIP": "eCLIP (RNA-Protein Binding)",
    "iCLIP": "iCLIP (RNA-Protein Binding)",
    "RIP-seq": "RIP-seq (RNA-Protein Binding)",
    "RIP-chip": "RIP-chip (RNA-Protein Binding)",
    "RNA Bind-n-Seq": "RNA Bind-n-Seq",
    "icLASER": "icLASER (RNA Structure)",
    "icSHAPE": "icSHAPE (RNA Structure)",
    # Translation
    "Ribo-seq": "Ribo-seq (Translation)",
    # Replication
    "Repli-seq": "Repli-seq (Replication Timing)",
    "Repli-chip": "Repli-chip (Replication Timing)",
    # Perturbation assays
    "genetic modification followed by DNase-seq": "CRISPR-DNase",
    "CRISPR genome editing followed by RNA-seq": "CRISPR followed by RNA-seq",
    "CRISPRi followed by RNA-seq": "CRISPRi followed by RNA-seq",
    "shRNA knockdown followed by RNA-seq": "shRNA knockdown RNA-seq",
    "siRNA knockdown followed by RNA-seq": "siRNA knockdown RNA-seq",
    "genomic perturbation followed by RT-qPCR": "Perturbation RT-qPCR",
    # Other specialized assays
    "seqFISH": "seqFISH (Spatial Transcriptomics)",
    "PAS-seq": "PAS-seq (PolyA Site)",
    "microRNA counts": "microRNA counts",
    "Switchgear": "Switchgear",
    "Circulome-seq": "Circulome-seq",
    "Clone-seq": "Clone-seq",
    "DNA-PET": "DNA-PET",
    "MRE-seq": "MRE-seq (Methylation)",
    # RACE assays
    "3' RACE": "3' RACE",
    "5' RACE": "5' RACE",
    "5' RLM RACE": "5' RLM RACE",
    # Array-based assays
    "comparative genomic hybridization by array": "CGH Array",
    "DNA methylation profiling by array assay": "Methylation Array",
    "transcription profiling by array assay": "Expression Array",
    # Proteomics
    "protein sequencing by tandem mass spectrometry assay": "MS/MS Proteomics",
    "LC/MS label-free quantitative proteomics": "LC/MS Proteomics",
    "LC-MS/MS isobaric label quantitative proteomics": "LC-MS/MS Proteomics",
    # Other genomics
    "whole genome sequencing assay": "WGS (Whole Genome Sequencing)",
    "genotyping by high throughput sequencing assay": "Genotyping",
}

# Common assay type aliases for matching user input
# Keys must match ASSAY_TYPES keys exactly
ASSAY_ALIASES: Dict[str, List[str]] = {
    "ChIP-seq": ["chip", "chipseq", "chip-seq", "chromatin immunoprecipitation"],
    "RNA-seq": ["rna", "rnaseq", "rna-seq", "transcriptome", "expression"],
    "ATAC-seq": ["atac", "atacseq", "atac-seq", "chromatin accessibility"],
    "DNase-seq": ["dnase", "dnaseseq", "dnase-seq", "dhs"],
    "HiC": ["hic", "hi-c", "Hi-C", "chromatin conformation", "3d genome"],
    "whole-genome shotgun bisulfite sequencing": ["wgbs", "bisulfite", "methylation", "dna methylation"],
    "eCLIP": ["eclip", "clip", "rna binding"],
    "CUT&RUN": ["cut and run", "cutandrun", "cutnrun"],
    "CUT&Tag": ["cut and tag", "cutandtag", "cutntag"],
    "single-cell RNA sequencing assay": ["scrna", "scrnaseq", "single cell rna"],
    "single-nucleus ATAC-seq": ["snatac", "snatacsep", "single nucleus atac"],
}

# =============================================================================
# ORGANISMS AND GENOME ASSEMBLIES
# =============================================================================
ORGANISMS: Dict[str, Dict[str, str]] = {
    "human": {
        "display_name": "Human (Homo sapiens)",
        "scientific_name": "Homo sapiens",
        "assembly": "hg38",
        "alt_assembly": "GRCh38",
        "previous_assembly": "hg19",
    },
    "mouse": {
        "display_name": "Mouse (Mus musculus)",
        "scientific_name": "Mus musculus",
        "assembly": "mm10",
        "alt_assembly": "GRCm38",
        "newer_assembly": "mm39",
    },
    "worm": {
        "display_name": "C. elegans",
        "scientific_name": "Caenorhabditis elegans",
        "assembly": "ce11",
        "alt_assembly": "WBcel235",
    },
    "fly": {
        "display_name": "D. melanogaster",
        "scientific_name": "Drosophila melanogaster",
        "assembly": "dm6",
        "alt_assembly": "BDGP6",
    },
}

# =============================================================================
# HISTONE MODIFICATIONS AND CHROMATIN TARGETS
# =============================================================================
HISTONE_MODIFICATIONS: Dict[str, Dict[str, str]] = {
    # Active histone marks
    "H3K27ac": {
        "full_name": "H3K27ac",
        "description": "Active enhancers and promoters",
        "category": "active",
    },
    "H3K4me1": {
        "full_name": "H3K4me1",
        "description": "Enhancers (poised or active)",
        "category": "enhancer",
    },
    "H3K4me2": {
        "full_name": "H3K4me2",
        "description": "Active promoters and enhancers",
        "category": "active",
    },
    "H3K4me3": {
        "full_name": "H3K4me3",
        "description": "Active promoters",
        "category": "promoter",
    },
    "H3K9ac": {
        "full_name": "H3K9ac",
        "description": "Active chromatin",
        "category": "active",
    },
    "H3K36me3": {
        "full_name": "H3K36me3",
        "description": "Transcribed gene bodies",
        "category": "transcription",
    },
    "H3K79me2": {
        "full_name": "H3K79me2",
        "description": "Active transcription",
        "category": "transcription",
    },
    "H4K20me1": {
        "full_name": "H4K20me1",
        "description": "Active transcription",
        "category": "transcription",
    },
    # Repressive histone marks
    "H3K27me3": {
        "full_name": "H3K27me3",
        "description": "Polycomb repression",
        "category": "repressive",
    },
    "H3K9me3": {
        "full_name": "H3K9me3",
        "description": "Heterochromatin",
        "category": "repressive",
    },
    "H3K9me2": {
        "full_name": "H3K9me2",
        "description": "Facultative heterochromatin",
        "category": "repressive",
    },
    "H4K20me3": {
        "full_name": "H4K20me3",
        "description": "Constitutive heterochromatin",
        "category": "repressive",
    },
    # Transcription factors and other targets
    "CTCF": {
        "full_name": "CTCF",
        "description": "Insulator / chromatin organizer",
        "category": "tf",
    },
    "POLR2A": {
        "full_name": "RNA Polymerase II",
        "description": "Active transcription",
        "category": "tf",
    },
    "EP300": {
        "full_name": "p300",
        "description": "Active enhancers",
        "category": "tf",
    },
    "RAD21": {
        "full_name": "RAD21 (Cohesin)",
        "description": "Chromatin loops",
        "category": "tf",
    },
    "SMC3": {
        "full_name": "SMC3 (Cohesin)",
        "description": "Chromatin loops",
        "category": "tf",
    },
}

# Aliases for histone modification searches
HISTONE_ALIASES: Dict[str, List[str]] = {
    "H3K27ac": ["h3k27ac", "k27ac", "h3 k27ac", "acetylation k27"],
    "H3K4me1": ["h3k4me1", "k4me1", "h3 k4me1", "monomethyl k4"],
    "H3K4me3": ["h3k4me3", "k4me3", "h3 k4me3", "trimethyl k4"],
    "H3K27me3": ["h3k27me3", "k27me3", "h3 k27me3", "polycomb"],
    "H3K9me3": ["h3k9me3", "k9me3", "h3 k9me3", "heterochromatin"],
    "H3K36me3": ["h3k36me3", "k36me3", "h3 k36me3", "gene body"],
    "CTCF": ["ctcf", "insulator", "boundary"],
}

# =============================================================================
# BODY PARTS AND ORGAN SYSTEMS
# =============================================================================
BODY_PARTS: Dict[str, Dict[str, List[str]]] = {
    "brain": {
        "display_name": "Brain / Nervous System",
        "tissues": [
            "brain",
            "cerebral cortex",
            "prefrontal cortex",
            "frontal cortex",
            "temporal lobe",
            "parietal lobe",
            "occipital lobe",
            "hippocampus",
            "amygdala",
            "hypothalamus",
            "thalamus",
            "cerebellum",
            "hindbrain",
            "midbrain",
            "forebrain",
            "brainstem",
            "medulla oblongata",
            "pons",
            "striatum",
            "caudate nucleus",
            "putamen",
            "substantia nigra",
            "spinal cord",
            "neural tube",
            "neuron",
            "astrocyte",
            "microglia",
            "oligodendrocyte",
            "neural progenitor cell",
            "neural stem cell",
            "neuroblast",
        ],
        "aliases": ["nervous system", "cns", "central nervous system", "neural"],
    },
    "heart": {
        "display_name": "Heart / Cardiovascular",
        "tissues": [
            "heart",
            "cardiac muscle",
            "cardiomyocyte",
            "left ventricle",
            "right ventricle",
            "left atrium",
            "right atrium",
            "aorta",
            "coronary artery",
            "endocardium",
            "myocardium",
            "pericardium",
            "cardiac progenitor",
        ],
        "aliases": ["cardiovascular", "cardiac", "circulatory"],
    },
    "liver": {
        "display_name": "Liver",
        "tissues": [
            "liver",
            "hepatocyte",
            "hepatic stellate cell",
            "Kupffer cell",
            "bile duct",
            "hepatoblast",
            "fetal liver",
        ],
        "aliases": ["hepatic"],
    },
    "kidney": {
        "display_name": "Kidney",
        "tissues": [
            "kidney",
            "renal cortex",
            "renal medulla",
            "nephron",
            "glomerulus",
            "proximal tubule",
            "distal tubule",
            "collecting duct",
            "podocyte",
            "renal epithelial cell",
        ],
        "aliases": ["renal"],
    },
    "lung": {
        "display_name": "Lung / Respiratory",
        "tissues": [
            "lung",
            "bronchus",
            "bronchiole",
            "alveolus",
            "trachea",
            "airway epithelium",
            "type I pneumocyte",
            "type II pneumocyte",
            "lung fibroblast",
        ],
        "aliases": ["respiratory", "pulmonary"],
    },
    "blood": {
        "display_name": "Blood / Immune",
        "tissues": [
            "blood",
            "whole blood",
            "peripheral blood",
            "bone marrow",
            "spleen",
            "thymus",
            "lymph node",
            "T cell",
            "CD4-positive T cell",
            "CD8-positive T cell",
            "regulatory T cell",
            "naive T cell",
            "B cell",
            "naive B cell",
            "plasma cell",
            "NK cell",
            "natural killer cell",
            "monocyte",
            "macrophage",
            "dendritic cell",
            "neutrophil",
            "eosinophil",
            "basophil",
            "mast cell",
            "megakaryocyte",
            "platelet",
            "erythrocyte",
            "red blood cell",
            "hematopoietic stem cell",
            "common myeloid progenitor",
            "granulocyte",
            "PBMC",
        ],
        "aliases": ["hematopoietic", "immune", "lymphoid", "myeloid"],
    },
    "gut": {
        "display_name": "Gastrointestinal",
        "tissues": [
            "intestine",
            "small intestine",
            "large intestine",
            "colon",
            "duodenum",
            "jejunum",
            "ileum",
            "rectum",
            "stomach",
            "esophagus",
            "intestinal epithelium",
            "enterocyte",
            "goblet cell",
            "Paneth cell",
            "enteroendocrine cell",
            "intestinal stem cell",
        ],
        "aliases": ["digestive", "gastrointestinal", "gi tract", "intestinal"],
    },
    "skin": {
        "display_name": "Skin / Integumentary",
        "tissues": [
            "skin",
            "epidermis",
            "dermis",
            "keratinocyte",
            "melanocyte",
            "fibroblast",
            "hair follicle",
            "sebaceous gland",
            "sweat gland",
        ],
        "aliases": ["integumentary", "dermal", "epidermal"],
    },
    "muscle": {
        "display_name": "Muscle",
        "tissues": [
            "skeletal muscle",
            "smooth muscle",
            "myocyte",
            "myoblast",
            "myotube",
            "satellite cell",
            "muscle stem cell",
        ],
        "aliases": ["muscular", "myogenic"],
    },
    "bone": {
        "display_name": "Bone / Skeletal",
        "tissues": [
            "bone",
            "osteoblast",
            "osteocyte",
            "osteoclast",
            "cartilage",
            "chondrocyte",
            "bone marrow stromal cell",
            "mesenchymal stem cell",
        ],
        "aliases": ["skeletal", "osseous"],
    },
    "adipose": {
        "display_name": "Adipose Tissue",
        "tissues": [
            "adipose tissue",
            "white adipose",
            "brown adipose",
            "adipocyte",
            "preadipocyte",
            "subcutaneous fat",
            "visceral fat",
        ],
        "aliases": ["fat", "fatty tissue"],
    },
    "reproductive": {
        "display_name": "Reproductive",
        "tissues": [
            "testis",
            "ovary",
            "uterus",
            "endometrium",
            "placenta",
            "prostate",
            "breast",
            "mammary gland",
            "mammary epithelium",
            "spermatocyte",
            "spermatogonia",
            "oocyte",
            "granulosa cell",
            "Leydig cell",
            "Sertoli cell",
        ],
        "aliases": ["gonad", "germline"],
    },
    "embryonic": {
        "display_name": "Embryonic / Stem Cells",
        "tissues": [
            "embryo",
            "blastocyst",
            "inner cell mass",
            "trophoblast",
            "embryonic stem cell",
            "ES cell",
            "ESC",
            "induced pluripotent stem cell",
            "iPSC",
            "iPS cell",
            "embryoid body",
            "primitive streak",
            "epiblast",
            "mesoderm",
            "endoderm",
            "ectoderm",
        ],
        "aliases": ["stem cell", "pluripotent", "developmental"],
    },
    "cell_line": {
        "display_name": "Cell Lines",
        "tissues": [
            # ENCODE Tier 1 (highest priority)
            "K562",
            "GM12878",
            "H1-hESC",
            "H1",  # Alias for H1-hESC
            # ENCODE Tier 2
            "A549",
            "HeLa-S3",
            "HepG2",
            "HUVEC",
            "IMR-90",
            "MCF-7",
            "SK-N-SH",
            # Other common cell lines
            "H9",
            "HeLa",
            "HEK293",
            "293T",
            "HCT116",
            "NHEK",
            "Caco-2",
            "Jurkat",
            "U2OS",
            "PC-3",
            "LNCaP",
        ],
        "aliases": ["immortalized", "cancer cell line", "transformed"],
    },
}

# =============================================================================
# TISSUE SYNONYMS FOR NLP MATCHING
# =============================================================================
# Maps related terms that should match each other
TISSUE_SYNONYMS: Dict[str, Set[str]] = {
    "cerebellum": {"hindbrain", "metencephalon", "cerebellar"},
    "hindbrain": {"cerebellum", "rhombencephalon", "metencephalon"},
    "hippocampus": {"hippocampal formation", "dentate gyrus", "ca1", "ca3"},
    "cortex": {"cerebral cortex", "neocortex", "cortical"},
    "frontal cortex": {"prefrontal cortex", "frontal lobe", "pfc"},
    "liver": {"hepatic", "hepatocyte"},
    "kidney": {"renal", "nephric"},
    "heart": {"cardiac", "myocardial"},
    "lung": {"pulmonary", "respiratory"},
    "blood": {"hematopoietic", "peripheral blood", "whole blood"},
    "bone marrow": {"marrow", "hematopoietic stem cell niche"},
    "spleen": {"splenic", "lymphoid organ"},
    "thymus": {"thymic", "thymocyte"},
    "intestine": {"gut", "bowel", "enteric"},
    "colon": {"large intestine", "colonic"},
    "small intestine": {"duodenum", "jejunum", "ileum"},
    "stomach": {"gastric"},
    "skin": {"dermal", "epidermal", "cutaneous"},
    "muscle": {"muscular", "myogenic", "skeletal muscle"},
    "fat": {"adipose", "adipocyte"},
    "embryo": {"embryonic", "fetal", "developmental"},
    "stem cell": {"progenitor", "pluripotent", "multipotent"},
}

# =============================================================================
# DEVELOPMENTAL STAGES AND AGE
# =============================================================================
DEVELOPMENTAL_STAGES: Dict[str, Dict[str, any]] = {
    # Mouse developmental stages
    "E0": {"species": "mouse", "description": "Fertilization", "days": 0},
    "E0.5": {"species": "mouse", "description": "1-cell stage", "days": 0.5},
    "E1.5": {"species": "mouse", "description": "2-cell stage", "days": 1.5},
    "E2.5": {"species": "mouse", "description": "8-cell stage", "days": 2.5},
    "E3.5": {"species": "mouse", "description": "Blastocyst", "days": 3.5},
    "E4.5": {"species": "mouse", "description": "Implantation", "days": 4.5},
    "E6.5": {"species": "mouse", "description": "Gastrulation", "days": 6.5},
    "E7.5": {"species": "mouse", "description": "Early somite", "days": 7.5},
    "E8.5": {"species": "mouse", "description": "Neural tube closure", "days": 8.5},
    "E9.5": {"species": "mouse", "description": "Limb bud formation", "days": 9.5},
    "E10.5": {"species": "mouse", "description": "Organogenesis", "days": 10.5},
    "E11.5": {"species": "mouse", "description": "Organogenesis", "days": 11.5},
    "E12.5": {"species": "mouse", "description": "Mid-gestation", "days": 12.5},
    "E13.5": {"species": "mouse", "description": "Mid-gestation", "days": 13.5},
    "E14.5": {"species": "mouse", "description": "Late organogenesis", "days": 14.5},
    "E15.5": {"species": "mouse", "description": "Fetal", "days": 15.5},
    "E16.5": {"species": "mouse", "description": "Fetal", "days": 16.5},
    "E17.5": {"species": "mouse", "description": "Late fetal", "days": 17.5},
    "E18.5": {"species": "mouse", "description": "Perinatal", "days": 18.5},
    "P0": {"species": "mouse", "description": "Newborn", "days": 19},
    "P1": {"species": "mouse", "description": "Postnatal day 1", "days": 20},
    "P7": {"species": "mouse", "description": "1 week old", "days": 26},
    "P14": {"species": "mouse", "description": "2 weeks old", "days": 33},
    "P21": {"species": "mouse", "description": "3 weeks (weaning)", "days": 40},
    "P28": {"species": "mouse", "description": "4 weeks", "days": 47},
    "P56": {"species": "mouse", "description": "8 weeks (adult)", "days": 75},
    "P60": {"species": "mouse", "description": "~8 weeks (adult)", "days": 79},
    "P90": {"species": "mouse", "description": "~3 months (adult)", "days": 109},
    # Week-based stages (commonly used)
    "2 weeks": {"species": "mouse", "description": "Juvenile", "days": 33},
    "4 weeks": {"species": "mouse", "description": "Juvenile/adolescent", "days": 47},
    "6 weeks": {"species": "mouse", "description": "Young adult", "days": 61},
    "8 weeks": {"species": "mouse", "description": "Adult", "days": 75},
    "10 weeks": {"species": "mouse", "description": "Adult", "days": 89},
    "12 weeks": {"species": "mouse", "description": "Adult", "days": 103},
    # Month-based stages
    "2 months": {"species": "mouse", "description": "Adult", "days": 75},
    "3 months": {"species": "mouse", "description": "Adult", "days": 109},
    "6 months": {"species": "mouse", "description": "Middle-aged", "days": 200},
    "12 months": {"species": "mouse", "description": "Middle-aged", "days": 380},
    "18 months": {"species": "mouse", "description": "Aged", "days": 560},
    "24 months": {"species": "mouse", "description": "Aged", "days": 740},
    # Human developmental stages
    "embryonic": {"species": "human", "description": "0-8 weeks gestation", "days": 0},
    "fetal": {"species": "human", "description": "8-40 weeks gestation", "days": 56},
    "newborn": {"species": "human", "description": "0-28 days", "days": 280},
    "infant": {"species": "human", "description": "1-12 months", "days": 308},
    "child": {"species": "human", "description": "1-12 years", "days": 645},
    "adolescent": {"species": "human", "description": "12-18 years", "days": 4650},
    "adult": {"species": "human", "description": "18+ years", "days": 6840},
}

# Age pattern aliases for search matching
AGE_ALIASES: Dict[str, List[str]] = {
    "P0": ["p0", "newborn", "birth", "postnatal day 0"],
    "P7": ["p7", "1 week", "one week", "1w"],
    "P14": ["p14", "2 weeks", "two weeks", "2w"],
    "P21": ["p21", "3 weeks", "three weeks", "3w", "weaning"],
    "P56": ["p56", "8 weeks", "eight weeks", "8w", "adult"],
    "P60": ["p60", "~8 weeks", "adult"],
    "E10.5": ["e10.5", "e10", "embryonic day 10"],
    "E12.5": ["e12.5", "e12", "embryonic day 12"],
    "E14.5": ["e14.5", "e14", "embryonic day 14"],
    "E16.5": ["e16.5", "e16", "embryonic day 16"],
    "E18.5": ["e18.5", "e18", "embryonic day 18", "late fetal"],
}


# =============================================================================
# LABS (Common ENCODE labs)
# =============================================================================
COMMON_LABS: List[str] = [
    "Bradley Bernstein, Broad",
    "Bing Ren, UCSD",
    "John Stamatoyannopoulos, UW",
    "Michael Snyder, Stanford",
    "Barbara Wold, Caltech",
    "Brenton Graveley, UConn",
    "Richard Myers, HAIB",
    "Thomas Gingeras, CSHL",
    "Peggy Farnham, USC",
    "Ali Mortazavi, UCI",
    "Kevin White, UChicago",
    "Job Dekker, UMass",
    "Len Pennacchio, LBNL",
    "Ross Hardison, PennState",
    "Howard Chang, Stanford",
]


def get_all_assay_types() -> List[str]:
    """Return sorted list of all assay type keys."""
    return sorted(ASSAY_TYPES.keys())


def get_all_organisms() -> List[str]:
    """Return list of organism keys."""
    return list(ORGANISMS.keys())


def get_organism_display(organism: str) -> str:
    """Get display name with genome assembly for an organism."""
    if organism in ORGANISMS:
        info = ORGANISMS[organism]
        return f"{info['display_name']} [{info['assembly']}]"
    return organism


def get_all_histone_mods() -> List[str]:
    """Return sorted list of all histone modification keys."""
    return sorted(HISTONE_MODIFICATIONS.keys())


def get_all_body_parts() -> List[str]:
    """Return list of body part keys."""
    return list(BODY_PARTS.keys())


def get_tissues_for_body_part(body_part: str) -> List[str]:
    """Return list of tissues for a given body part."""
    if body_part in BODY_PARTS:
        return BODY_PARTS[body_part]["tissues"]
    return []


def get_all_developmental_stages() -> List[str]:
    """Return sorted list of all developmental stage keys."""
    return list(DEVELOPMENTAL_STAGES.keys())
