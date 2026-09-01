# EvoFATE

EvoFATE is a graph-learning framework for joint reconstruction of tumor-cell evolution and cell-fate organization from single-cell genotype–phenotype measurements. It is designed for datasets in which single-cell SNV and transcriptomic profiles are measured from the same cells, including LR-scRNA-seq-enabled co-profiling experiments.

## Scientific overview

EvoFATE first constructs a genetic cell–cell connectivity graph from single-cell SNV profiles. The graph supports genetic clone definition from shared mutation patterns and aggregation of mutation evidence across genetically related cells. Clone-level consensus mutation profiles are inferred with graph-informed Bayesian posterior probabilities, reducing sensitivity to sparse observations and isolated erroneous SNV calls.

Directional relationships between clones are inferred from consensus profiles through mutation gains and losses. Edmonds’ algorithm is then used to obtain a globally consistent directed clonal lineage, providing an ordered representation of tumor evolution.

Gene-expression profiles are subsequently used as node features on the genetic graph. A graph attention network (GAT) is trained in a bootstrapped graph representation learning (BGRL) framework to learn a unified genotype–phenotype representation. BGRL optimizes consistency between augmented views of node features and graph structure through an asymmetric online–target architecture without explicit negative sampling, supporting efficient analysis of large cell populations.

For visualization, EvoFATE combines evolutionary ordering with local cell-state organization in a lineage-guided two-dimensional projection. Canonical correlation analysis (CCA) provides guidance from inferred genetic progression, while UMAP preserves local structure in the unified representation. This enables genetic lineages and cell-fate transitions, including reversible and convergent trajectories, to be examined on a common map.

## Workflow

1. Construct mutation-derived cell connectivity and define genetic clones.
2. Infer graph-supported consensus mutation profiles and a directed clonal lineage.
3. Integrate transcriptomic node features with the genetic graph using BGRL and GAT.
4. Generate a CCA- and UMAP-based lineage-guided projection.
5. Quantify state remodeling at clonal and single-cell resolution and identify progression-associated genes.

## Public API

Calculation functions are available through `evofate.tl`; plotting functions are available through `evofate.pl`.

### Genetic reconstruction

- `tl.cal_genetic_connectivities(adata)` constructs the mutation-derived cell graph.
- `tl.define_clones(adata)` assigns cells to genetic clones.
- `tl.cal_consensus_profile(adata)` estimates graph-supported Bayesian clone consensus profiles.
- `tl.cal_clone_connectivity(adata)` calculates clone-level connectivity.
- `tl.cal_tree_layout(adata)` infers and lays out the directed clonal lineage.

### Integration and projection

- `tl.cal_evofate_embedding(adata)` learns the unified representation with BGRL and a GAT backbone.
- `tl.cal_linear_projection(adata)` constructs the CCA lineage scaffold.
- `tl.cal_lineage_guided_projection(adata)` generates the lineage-guided two-dimensional projection.

### Downstream analysis

- `tl.cal_clonal_evofate(adata)` quantifies within-clone state dispersion and state changes across clonal edges.
- `tl.cal_single_cell_evofate(adata)` delineates lineage progression and state variation at single-cell resolution.
- `tl.cal_progression_features(adata, expression)` tests expression features for structured variation along inferred progression paths.
- `tl.select_progression_features(adata)` selects and stores important progression-associated genes.

Available plotting functions include `plot_consensus_profile`, `plot_filtered_mutations`, `plot_lineage_tree`, `plot_lineage_tree_w_piechart`, `plot_embedding`, `plot_clonal_evofate`, `plot_single_cell_evofate`, and `plot_progression_features`.

## Installation

From the repository root:

```bash
python -m pip install -e .
```

Dependencies include Scanpy, PyTorch, PyTorch Geometric, igraph, UMAP, and Seaborn. See [`pyproject.toml`](pyproject.toml) for the package specification.

## Example data and tutorial

The [`example/`](example/) directory contains the mLung17 example and [`mLung17.ipynb`](example/mLung17.ipynb). The tutorial performs expression quality control, mitochondrial filtering, library-size normalization, log transformation, highly variable gene selection, PCA, neighborhood construction, Leiden clustering, graph integration, lineage-guided projection, and downstream evolutionary-fate analysis.

Example inputs are:

- `mLung17_mutation.csv`: cell-by-mutation calls encoded as `1` (mutant), `-1` (wild type), and `0` (missing).
- `mLung17_mutation_meta.csv`: mutation annotations and genomic metadata.
- `mLung17_cell_meta.csv`: barcode, sample, cell-type, and cell-state annotations.
- `mLung17_gene_counts_input.csv`: raw gene-count table with gene metadata.
- `mLung17_expression.csv`: prepared expression values retained for reference.

Generated figures are saved under `example/figures/`, and the completed AnnData object is written to `example/mLung17_snv_example.h5ad`.

## Data requirements

New datasets should provide cell-aligned SNV and expression measurements together with mutation and cell metadata. Cell identifiers must be unique and consistently ordered across modalities. Raw counts should be retained for quality control and normalization; `tl.cal_progression_features()` should receive normalized, log1p-transformed expression values.

## License

See [`Copyright.txt`](Copyright.txt).
