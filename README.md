# EvoFATE

EvoFATE is a computational toolkit for analyzing high-throughput long-read single-cell RNA sequencing (LR-scRNAseq) data to jointly reconstruct Evolutionary (point mutation) and Fate (gene expression) trajectories in individual cells. EvoFATE includes three major steps:
1. *Genetic Graph Constructor* to build genotype graphs from single-cell mutation data using Node2Vec, capturing relationships among cells based on shared mutations;
2. *Evolutionary Lineage Mapper* to calculate genetic timing, infer clone lineage layouts, and project cells in an evolution-informed embedding space;
3. *EvoFATE Integrator* to integrate transcriptomic profiles with genetic structure using BGRL with GAT backbone, perform co-projection of modalities, and compute EvoFATE time for ordered trajectory reconstruction.
   
Leveraging graph representation learning framework, EvoFATE dynamically co-optimizes genotype and phenotype information to reveal joint evolutionary and fate dynamics at both clonal and single-cell levels.
