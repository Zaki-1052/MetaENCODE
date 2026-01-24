# src/ui/vocabularies.py
"""Vocabulary definitions for ENCODE dataset search and filtering.

This module provides dictionaries of biological terms used for autocomplete,
filtering, and NLP-based term matching in the MetaENCODE search interface.

DATA SOURCE: All vocabulary values are derived from ENCODE API faceted search,
ensuring they match actual ENCODE database values.

Last updated: 2026-01-23
Source: ENCODE API /search/?type=Experiment (27,398 experiments)
"""

from typing import Any, Dict, List, Set

# =============================================================================
# ASSAY TYPES (from ENCODE API, ordered by experiment count)
# Filter parameter: assay_term_name
# =============================================================================
# Mapping from API value -> display name
ASSAY_TYPES: Dict[str, str] = {
    # High-volume assays (1000+ experiments)
    "ChIP-seq": "ChIP-seq (12,569 experiments)",
    "DNase-seq": "DNase-seq (3,582 experiments)",
    "RNA-seq": "RNA-seq (1,270 experiments)",
    "Mint-ChIP-seq": "Mint-ChIP-seq (1,162 experiments)",
    # Medium-volume assays (100-999 experiments)
    "polyA plus RNA-seq": "polyA plus RNA-seq (943 experiments)",
    "shRNA knockdown followed by RNA-seq": "shRNA knockdown RNA-seq (668 experiments)",
    "eCLIP": "eCLIP (587 experiments)",
    "ATAC-seq": "ATAC-seq (560 experiments)",
    "single-cell RNA sequencing assay": "scRNA-seq (524 experiments)",
    "HiC": "Hi-C (501 experiments)",
    "CRISPR genome editing followed by RNA-seq": "CRISPR RNA-seq (470 experiments)",
    "microRNA-seq": "microRNA-seq (418 experiments)",
    "single-nucleus ATAC-seq": "snATAC-seq (370 experiments)",
    "transcription profiling by array assay": "Expression Array (291 experiments)",
    "whole-genome shotgun bisulfite sequencing": "WGBS (278 experiments)",
    "DNA methylation profiling by array assay": "Methylation Array (257 experiments)",
    "PRO-cap": "PRO-cap (224 experiments)",
    "RRBS": "RRBS (218 experiments)",
    "RNA Bind-n-Seq": "RNA Bind-n-Seq (216 experiments)",
    "small RNA-seq": "small RNA-seq (216 experiments)",
    "ChIA-PET": "ChIA-PET (213 experiments)",
    "long read RNA-seq": "long read RNA-seq (203 experiments)",
    "RAMPAGE": "RAMPAGE (168 experiments)",
    "comparative genomic hybridization by array": "CGH Array (142 experiments)",
    "CAGE": "CAGE (121 experiments)",
    "Repli-seq": "Repli-seq (119 experiments)",
    "microRNA counts": "microRNA counts (115 experiments)",
    "siRNA knockdown followed by RNA-seq": "siRNA knockdown RNA-seq (113 experiments)",
    # Lower-volume assays (10-99 experiments)
    "CRISPRi followed by RNA-seq": "CRISPRi RNA-seq (77 experiments)",
    "RIP-seq": "RIP-seq (73 experiments)",
    "MRE-seq": "MRE-seq (68 experiments)",
    "long read single-cell RNA-seq": "long read scRNA-seq (64 experiments)",
    "Repli-chip": "Repli-chip (63 experiments)",
    "MeDIP-seq": "MeDIP-seq (55 experiments)",
    "PAS-seq": "PAS-seq (54 experiments)",
    "whole genome sequencing assay": "WGS (52 experiments)",
    "Bru-seq": "Bru-seq (49 experiments)",
    "genetic modification followed by DNase-seq": "CRISPR-DNase (46 experiments)",
    "FAIRE-seq": "FAIRE-seq (37 experiments)",
    "BruChase-seq": "BruChase-seq (32 experiments)",
    "polyA minus RNA-seq": "polyA minus RNA-seq (32 experiments)",
    "RIP-chip": "RIP-chip (32 experiments)",
    "RNA-PET": "RNA-PET (31 experiments)",
    "PRO-seq": "PRO-seq (22 experiments)",
    "BruUV-seq": "BruUV-seq (16 experiments)",
    "protein sequencing by tandem mass spectrometry assay": "MS/MS Proteomics",
    "5C": "5C (13 experiments)",
    # Rare assays (<10 experiments)
    "TAB-seq": "TAB-seq (8 experiments)",
    "iCLIP": "iCLIP (7 experiments)",
    "DNA-PET": "DNA-PET (6 experiments)",
    "icSHAPE": "icSHAPE (6 experiments)",
    "seqFISH": "seqFISH (5 experiments)",
    "GRO-cap": "GRO-cap (4 experiments)",
    "SPRITE": "SPRITE (3 experiments)",
    "MNase-seq": "MNase-seq (2 experiments)",
    "Switchgear": "Switchgear (2 experiments)",
    "capture Hi-C": "capture Hi-C (2 experiments)",
    "5' RLM RACE": "5' RLM RACE (2 experiments)",
    "icLASER": "icLASER (2 experiments)",
    "Circulome-seq": "Circulome-seq (1 experiment)",
}

# Common assay type aliases for matching user input
ASSAY_ALIASES: Dict[str, List[str]] = {
    "ChIP-seq": ["chip", "chipseq", "chip-seq", "chromatin immunoprecipitation"],
    "RNA-seq": ["rna", "rnaseq", "rna-seq", "transcriptome", "expression"],
    "ATAC-seq": ["atac", "atacseq", "atac-seq", "chromatin accessibility"],
    "DNase-seq": ["dnase", "dnaseseq", "dnase-seq", "dhs"],
    "HiC": ["hic", "hi-c", "Hi-C", "chromatin conformation", "3d genome"],
    "whole-genome shotgun bisulfite sequencing": [
        "wgbs",
        "bisulfite",
        "methylation",
        "dna methylation",
    ],
    "eCLIP": ["eclip", "clip", "rna binding"],
    "single-cell RNA sequencing assay": ["scrna", "scrnaseq", "single cell rna"],
    "single-nucleus ATAC-seq": ["snatac", "snatacsep", "single nucleus atac"],
    "polyA plus RNA-seq": ["polya", "mrna-seq", "mrna"],
    "microRNA-seq": ["mirna", "microrna", "mir-seq"],
}

# =============================================================================
# ORGANISMS (from ENCODE API, ordered by experiment count)
# Filter parameter: replicates.library.biosample.donor.organism.scientific_name
# =============================================================================
ORGANISMS: Dict[str, Dict[str, str]] = {
    "human": {
        "display_name": "Human (Homo sapiens)",
        "scientific_name": "Homo sapiens",
        "assembly": "hg38",
        "alt_assembly": "GRCh38",
        "previous_assembly": "hg19",
        "experiment_count": "35994",  # from replicates
    },
    "mouse": {
        "display_name": "Mouse (Mus musculus)",
        "scientific_name": "Mus musculus",
        "assembly": "mm10",
        "alt_assembly": "GRCm38",
        "newer_assembly": "mm39",
        "experiment_count": "7690",
    },
    "fly": {
        "display_name": "D. melanogaster",
        "scientific_name": "Drosophila melanogaster",
        "assembly": "dm6",
        "alt_assembly": "BDGP6",
        "experiment_count": "3734",
    },
    "worm": {
        "display_name": "C. elegans",
        "scientific_name": "Caenorhabditis elegans",
        "assembly": "ce11",
        "alt_assembly": "WBcel235",
        "experiment_count": "2485",
    },
}

# Additional organisms with fewer experiments (for reference)
OTHER_ORGANISMS: List[str] = [
    "Drosophila pseudoobscura",
    "Drosophila simulans",
    "Trichechus manatus",  # Manatee
    "Drosophila mojavensis",
    "Drosophila virilis",
    "Drosophila ananassae",
    "Drosophila yakuba",
]

# =============================================================================
# BIOSAMPLES (from ENCODE API, top 100 by experiment count)
# Filter parameter: biosample_ontology.term_name
# =============================================================================
# Top biosamples ordered by experiment count
TOP_BIOSAMPLES: List[str] = [
    "whole organism",  # 3247
    "K562",  # 2571
    "HepG2",  # 1793
    "dorsolateral prefrontal cortex",  # 605
    "A549",  # 538
    "GM12878",  # 490
    "HCT116",  # 442
    "MCF-7",  # 356
    "heart",  # 328
    "adrenal gland",  # 320
    "heart left ventricle",  # 313
    "HEK293",  # 291
    "liver",  # 286
    "H1",  # 234
    "cell-free sample",  # 216
    "stomach",  # 208
    "CD4-positive, alpha-beta T cell",  # 208
    "SK-N-SH",  # 196
    "spleen",  # 196
    "layer of hippocampus",  # 193
    "heart right ventricle",  # 170
    "lung",  # 165
    "naive thymus-derived CD4-positive, alpha-beta T cell",  # 160
    "HeLa-S3",  # 159
    "kidney",  # 156
    "T-cell",  # 154
    "WTC11",  # 152
    "gastrocnemius",  # 151
    "BLaER1",  # 144
    "CD14-positive monocyte",  # 137
    "left cerebral cortex",  # 133
    "forebrain",  # 127
    "brain",  # 127
    "midbrain",  # 119
    "ovary",  # 117
    "hindbrain",  # 116
    "IMR-90",  # 114
    "transverse colon",  # 114
    "sigmoid colon",  # 112
    "cerebellum",  # 111
    "pancreas",  # 108
    "foreskin keratinocyte",  # 99
    "macrophage",  # 98
    "limb",  # 96
    "MEL",  # 94
    "B cell",  # 80
    "thyroid gland",  # 79
    "mesenchymal stem cell",  # 78
    "H9",  # 85
    "placenta",  # 85
]

# ENCODE Tier 1 cell lines (highest priority for standardization)
TIER1_CELL_LINES: List[str] = ["K562", "GM12878", "H1"]

# ENCODE Tier 2 cell lines
TIER2_CELL_LINES: List[str] = [
    "A549",
    "HeLa-S3",
    "HepG2",
    "IMR-90",
    "MCF-7",
    "SK-N-SH",
    "HCT116",
]

# =============================================================================
# TARGETS - ChIP-seq/CUT&RUN targets (from ENCODE API, top by experiment count)
# Filter parameter: target.label
# =============================================================================
# Histone modifications with detailed info (top targets from ENCODE)
HISTONE_MODIFICATIONS: Dict[str, Dict[str, str]] = {
    # Active histone marks
    "H3K4me3": {
        "full_name": "H3K4me3",
        "description": "Active promoters (854 experiments)",
        "category": "promoter",
    },
    "H3K27ac": {
        "full_name": "H3K27ac",
        "description": "Active enhancers and promoters (757 experiments)",
        "category": "active",
    },
    "H3K4me1": {
        "full_name": "H3K4me1",
        "description": "Enhancers (691 experiments)",
        "category": "enhancer",
    },
    "H3K36me3": {
        "full_name": "H3K36me3",
        "description": "Transcribed gene bodies (672 experiments)",
        "category": "transcription",
    },
    "H3K9ac": {
        "full_name": "H3K9ac",
        "description": "Active chromatin (203 experiments)",
        "category": "active",
    },
    "H3K4me2": {
        "full_name": "H3K4me2",
        "description": "Active promoters and enhancers (146 experiments)",
        "category": "active",
    },
    "H3K79me2": {
        "full_name": "H3K79me2",
        "description": "Active transcription (56 experiments)",
        "category": "transcription",
    },
    "H4K20me1": {
        "full_name": "H4K20me1",
        "description": "Active transcription (54 experiments)",
        "category": "transcription",
    },
    "H2AFZ": {
        "full_name": "H2A.Z",
        "description": "Promoters and enhancers (51 experiments)",
        "category": "active",
    },
    # Repressive histone marks
    "H3K27me3": {
        "full_name": "H3K27me3",
        "description": "Polycomb repression (718 experiments)",
        "category": "repressive",
    },
    "H3K9me3": {
        "full_name": "H3K9me3",
        "description": "Heterochromatin (643 experiments)",
        "category": "repressive",
    },
    # Transcription factors and other targets
    "CTCF": {
        "full_name": "CTCF",
        "description": "Insulator / chromatin organizer (608 experiments)",
        "category": "tf",
    },
    "POLR2A": {
        "full_name": "RNA Polymerase II",
        "description": "Active transcription (225 experiments)",
        "category": "tf",
    },
    "EP300": {
        "full_name": "p300",
        "description": "Active enhancers (97 experiments)",
        "category": "tf",
    },
    "RAD21": {
        "full_name": "RAD21 (Cohesin)",
        "description": "Chromatin loops (83 experiments)",
        "category": "tf",
    },
}

# Top 50 targets from ENCODE API (for autocomplete)
TOP_TARGETS: List[str] = [
    "H3K4me3",
    "H3K27ac",
    "H3K27me3",
    "H3K4me1",
    "H3K36me3",
    "H3K9me3",
    "CTCF",
    "POLR2A",
    "H3K9ac",
    "H3K4me2",
    "EP300",
    "RAD21",
    "H3K79me2",
    "H4K20me1",
    "H2AFZ",
    "POLR2AphosphoS5",
    "NR3C1",
    "JUN",
    "CEBPB",
    "REST",
    "MAX",
    "EZH2",
    "MYC",
    "JUND",
    "SMC3",
    "FOXA1",
    "TCF7L2",
    "SPI1",
    "GATA2",
    "FOS",
    "RUNX1",
    "FOXA2",
    "STAT3",
    "RXRA",
    "MAZ",
    "ESR1",
    "GABPA",
    "USF1",
    "YY1",
    "TAF1",
]

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
# LIFE STAGES (from ENCODE API, ordered by experiment count)
# Filter parameter: replicates.library.biosample.life_stage
# =============================================================================
LIFE_STAGES: List[str] = [
    "adult",  # 25196
    "embryonic",  # 9573
    "unknown",  # 4861
    "child",  # 4606
    "postnatal",  # 1992
    "newborn",  # 743
    "young adult",  # 466
    "L4 larva",  # 437 (C. elegans)
    "L1 larva",  # 393 (C. elegans)
    "prepupa",  # 282 (Drosophila)
    "L3 larva",  # 265 (C. elegans)
    "late embryonic",  # 236
    "wandering third instar larva",  # 218 (Drosophila)
    "L2 larva",  # 214 (C. elegans)
    "mixed stage (embryonic)",  # 143
    "third instar larva",  # 128 (Drosophila)
    "pupa",  # 94 (Drosophila)
    "early embryonic",  # 57
    "midembryonic",  # 27
    "L4/young adult",  # 10 (C. elegans)
    "larva",  # 9
    "first instar larva",  # 7 (Drosophila)
    "second instar larva",  # 6 (Drosophila)
    "fetal",  # 4
    "dauer",  # 4 (C. elegans)
]

# =============================================================================
# LABS (from ENCODE API, ordered by experiment count)
# Filter parameter: lab.title
# =============================================================================
COMMON_LABS: List[str] = [
    "John Stamatoyannopoulos, UW",  # 4855
    "Michael Snyder, Stanford",  # 3645
    "Bradley Bernstein, Broad",  # 2965
    "Bing Ren, UCSD",  # 2600
    "J. Michael Cherry, Stanford",  # 2163
    "Brenton Graveley, UConn",  # 1731
    "Barbara Wold, Caltech",  # 1418
    "Gene Yeo, UCSD",  # 1215
    "Richard Myers, HAIB",  # 1101
    "Thomas Gingeras, CSHL",  # 915
    "Ali Mortazavi, UCI",  # 878
    "Kevin White, UChicago",  # 655
    "Peggy Farnham, USC",  # 534
    "Job Dekker, UMass",  # 501
    "Tim Reddy, Duke",  # 413
    "Jesse Engreitz, Stanford",  # 372
    "Xiang-Dong Fu, UCSD",  # 301
    "Eric Mendenhall, UAB",  # 252
    "Mark Gerstein, Yale",  # 199
    "Len Pennacchio, LBNL",  # 182
]

# =============================================================================
# BODY PARTS AND ORGAN SYSTEMS (curated for UI organization)
# =============================================================================
BODY_PARTS: Dict[str, Dict[str, Any]] = {
    "brain": {
        "display_name": "Brain / Nervous System",
        "tissues": [
            "brain",
            "dorsolateral prefrontal cortex",
            "layer of hippocampus",
            "left cerebral cortex",
            "forebrain",
            "midbrain",
            "hindbrain",
            "cerebellum",
            "substantia nigra",
            "head of caudate nucleus",
            "caudate nucleus",
            "putamen",
            "angular gyrus",
            "anterior cingulate cortex",
            "middle frontal area",
            "superior temporal gyrus",
            "spinal cord",
            "tibial nerve",
        ],
        "aliases": [
            "nervous system",
            "cns",
            "central nervous system",
            "neural",
            "cortex",
        ],
    },
    "heart": {
        "display_name": "Heart / Cardiovascular",
        "tissues": [
            "heart",
            "heart left ventricle",
            "heart right ventricle",
            "left cardiac atrium",
            "right cardiac atrium",
            "aorta",
            "coronary artery",
            "thoracic aorta",
        ],
        "aliases": ["cardiovascular", "cardiac", "circulatory"],
    },
    "liver": {
        "display_name": "Liver",
        "tissues": [
            "liver",
            "right lobe of liver",
            "HepG2",
        ],
        "aliases": ["hepatic"],
    },
    "kidney": {
        "display_name": "Kidney",
        "tissues": [
            "kidney",
            "renal cortex",
            "renal medulla",
        ],
        "aliases": ["renal"],
    },
    "lung": {
        "display_name": "Lung / Respiratory",
        "tissues": [
            "lung",
            "upper lobe of left lung",
            "A549",
        ],
        "aliases": ["respiratory", "pulmonary"],
    },
    "blood": {
        "display_name": "Blood / Immune",
        "tissues": [
            "K562",
            "GM12878",
            "CD4-positive, alpha-beta T cell",
            "naive thymus-derived CD4-positive, alpha-beta T cell",
            "T-cell",
            "CD14-positive monocyte",
            "macrophage",
            "B cell",
            "spleen",
            "thymus",
            "bone marrow",
        ],
        "aliases": ["hematopoietic", "immune", "lymphoid", "myeloid"],
    },
    "gut": {
        "display_name": "Gastrointestinal",
        "tissues": [
            "stomach",
            "transverse colon",
            "sigmoid colon",
            "small intestine",
            "esophagus",
            "esophagus muscularis mucosa",
            "HCT116",
        ],
        "aliases": ["digestive", "gastrointestinal", "gi tract", "intestinal"],
    },
    "reproductive": {
        "display_name": "Reproductive",
        "tissues": [
            "ovary",
            "testis",
            "placenta",
            "uterus",
            "prostate",
            "MCF-7",
        ],
        "aliases": ["gonad", "germline"],
    },
    "muscle": {
        "display_name": "Muscle",
        "tissues": [
            "gastrocnemius",
            "gastrocnemius medialis",
            "skeletal muscle tissue",
            "psoas muscle",
        ],
        "aliases": ["muscular", "myogenic"],
    },
    "skin": {
        "display_name": "Skin / Integumentary",
        "tissues": [
            "foreskin keratinocyte",
            "skin of body",
            "keratinocyte",
        ],
        "aliases": ["integumentary", "dermal", "epidermal"],
    },
    "embryonic": {
        "display_name": "Embryonic / Stem Cells",
        "tissues": [
            "H1",
            "H9",
            "WTC11",
            "embryonic facial prominence",
            "limb",
            "whole organism",
        ],
        "aliases": ["stem cell", "pluripotent", "developmental", "esc", "ipsc"],
    },
    "cell_line": {
        "display_name": "Cell Lines",
        "tissues": [
            "K562",
            "HepG2",
            "A549",
            "GM12878",
            "HCT116",
            "MCF-7",
            "SK-N-SH",
            "HEK293",
            "H1",
            "HeLa-S3",
            "WTC11",
            "BLaER1",
            "IMR-90",
            "MEL",
            "H9",
        ],
        "aliases": ["immortalized", "cancer cell line", "transformed"],
    },
}

# =============================================================================
# TISSUE SYNONYMS FOR NLP MATCHING
# =============================================================================
TISSUE_SYNONYMS: Dict[str, Set[str]] = {
    "cerebellum": {"hindbrain", "metencephalon", "cerebellar"},
    "hindbrain": {"cerebellum", "rhombencephalon", "metencephalon"},
    "hippocampus": {"hippocampal formation", "layer of hippocampus"},
    "cortex": {
        "cerebral cortex",
        "neocortex",
        "cortical",
        "dorsolateral prefrontal cortex",
    },
    "liver": {"hepatic", "HepG2", "right lobe of liver"},
    "kidney": {"renal", "nephric"},
    "heart": {"cardiac", "myocardial", "heart left ventricle", "heart right ventricle"},
    "lung": {"pulmonary", "respiratory", "A549"},
    "blood": {"hematopoietic", "K562", "GM12878"},
    "colon": {
        "large intestine",
        "colonic",
        "transverse colon",
        "sigmoid colon",
        "HCT116",
    },
    "muscle": {"muscular", "myogenic", "gastrocnemius"},
}

# =============================================================================
# DEVELOPMENTAL STAGES AND AGE
# =============================================================================
DEVELOPMENTAL_STAGES: Dict[str, Dict[str, Any]] = {
    # Mouse developmental stages
    "E10.5": {"species": "mouse", "description": "Organogenesis", "days": 10.5},
    "E11.5": {"species": "mouse", "description": "Organogenesis", "days": 11.5},
    "E12.5": {"species": "mouse", "description": "Mid-gestation", "days": 12.5},
    "E13.5": {"species": "mouse", "description": "Mid-gestation", "days": 13.5},
    "E14.5": {"species": "mouse", "description": "Late organogenesis", "days": 14.5},
    "E15.5": {"species": "mouse", "description": "Fetal", "days": 15.5},
    "E16.5": {"species": "mouse", "description": "Fetal", "days": 16.5},
    "P0": {"species": "mouse", "description": "Newborn", "days": 19},
    "P7": {"species": "mouse", "description": "1 week old", "days": 26},
    "P14": {"species": "mouse", "description": "2 weeks old", "days": 33},
    "P21": {"species": "mouse", "description": "3 weeks (weaning)", "days": 40},
    "P56": {"species": "mouse", "description": "8 weeks (adult)", "days": 75},
    "P60": {"species": "mouse", "description": "~8 weeks (adult)", "days": 79},
    "8 weeks": {"species": "mouse", "description": "Adult", "days": 75},
    "adult": {"species": "human", "description": "18+ years", "days": 6840},
    "embryonic": {"species": "human", "description": "0-8 weeks gestation", "days": 0},
    "fetal": {"species": "human", "description": "8-40 weeks gestation", "days": 56},
    "newborn": {"species": "human", "description": "0-28 days", "days": 280},
    "child": {"species": "human", "description": "1-12 years", "days": 645},
}

# Age pattern aliases for search matching
AGE_ALIASES: Dict[str, List[str]] = {
    "P0": ["p0", "newborn", "birth", "postnatal day 0"],
    "P7": ["p7", "1 week", "one week", "1w"],
    "P14": ["p14", "2 weeks", "two weeks", "2w"],
    "P21": ["p21", "3 weeks", "three weeks", "3w", "weaning"],
    "P56": ["p56", "8 weeks", "eight weeks", "8w", "adult"],
    "P60": ["p60", "~8 weeks", "adult"],
    "E14.5": ["e14.5", "e14", "embryonic day 14"],
    "adult": ["adult", "grown", "mature"],
    "embryonic": ["embryonic", "embryo", "developmental"],
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_all_assay_types() -> List[str]:
    """Return list of all assay type keys, ordered by experiment count."""
    return list(ASSAY_TYPES.keys())


def get_all_organisms() -> List[str]:
    """Return list of organism keys."""
    return list(ORGANISMS.keys())


def get_organism_display(organism: str) -> str:
    """Get display name with genome assembly for an organism."""
    if organism in ORGANISMS:
        info = ORGANISMS[organism]
        return f"{info['display_name']} [{info['assembly']}]"
    return organism


def get_organism_scientific_name(organism: str) -> str:
    """Get scientific name for filtering ENCODE API."""
    if organism in ORGANISMS:
        return ORGANISMS[organism]["scientific_name"]
    return organism


def get_all_histone_mods() -> List[str]:
    """Return list of all histone modification/target keys."""
    return list(HISTONE_MODIFICATIONS.keys())


def get_all_body_parts() -> List[str]:
    """Return list of body part keys."""
    return list(BODY_PARTS.keys())


def get_tissues_for_body_part(body_part: str) -> List[str]:
    """Return list of tissues for a given body part."""
    if body_part in BODY_PARTS:
        return BODY_PARTS[body_part]["tissues"]
    return []


def get_all_developmental_stages() -> List[str]:
    """Return list of all developmental stage keys."""
    return list(DEVELOPMENTAL_STAGES.keys())


def get_top_biosamples(limit: int = 50) -> List[str]:
    """Return top biosamples by experiment count."""
    return TOP_BIOSAMPLES[:limit]


def get_top_targets(limit: int = 20) -> List[str]:
    """Return top targets by experiment count."""
    return TOP_TARGETS[:limit]
