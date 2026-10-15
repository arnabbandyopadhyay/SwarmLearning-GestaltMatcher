# Post-Analysis

Post-hoc evaluation notebooks comparing the **Swarm-trained** model against the
**centralized baseline** (and the GMDB / AIDY reference models) for the
GestaltMatcher / GMDB face-embedding work.

The notebooks cover five themes:

- **Accuracy** — Top-1/5/10/30 performance on the test sets, split by rare vs
  frequent syndromes and by gallery composition.
- **Statistical robustness** — paired central-vs-decentral comparisons and
  bootstrap confidence intervals.
- **Calibration** — how well predicted confidences match accuracy.
- **Uncertainty & fairness** — per-class uncertainty and
  accuracy across demographic groups.
- **Embedding structure** — clustering and manifold structure of the learned
  representations.


## Prerequisites

**Python packages**

- `numpy`, `pandas`
- `torch` / `torchvision`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `plotly`
- `umap-learn`, `colorcet`
- `tqdm`
- `openpyxl`

**Inputs**

- The GMDB metadata splits, gallery/test files, and syndrome lookup table.
- Saved model checkpoints (central, swarm, and reference models).
- Precomputed face-embedding / representation files for the ensemble and paired
  comparison notebooks.
- An aggregated results workbook for the overall accuracy summary.


## Notebook reference

### `plot_all_accuracy.ipynb`
Overall accuracy comparison across the GMDB, Central, Swarm, and AIDY models (Main text Figure 2B and 2C).
Parses an aggregated results workbook and extracts Top-N accuracy for frequent
and rare syndromes across the different test/gallery sections. 

- **Consumes:** aggregated results workbook.
- **Produces:** grouped bar plots (`model_comparison_accuracy.png` / `.pdf`),
  a per-model/dataset heatmap, and box/strip plots of accuracy.

### `evaluate_ensemble_bootstrap.ipynb`
Evaluates an ensemble (averaging cosine distances across models) with
syndrome-level bootstrap resampling and 95% confidence intervals, across the
Frequent/Rare test and gallery combinations.

- **Consumes:** precomputed representation files, gallery/test splits, and the
  syndrome lookup table.
- **Produces:** bootstrap distribution and summary CSVs (mean accuracy, SE, CI
  per Top-N and experiment).

### `plot_accuracy_with_error.ipynb`
Paired Central-vs-decentralized comparison built from precomputed embedding
files. Computes Top-k metrics including accuracy difference, agreement/
disagreement, error overlap and complementarity, and directional advantage (Supplementary figure S2, S3 and S5).

- **Consumes:** central and decentralized embedding/representation files
- **Produces:** a summary CSV in a comparison output folder and a multi-panel
  figure (accuracy, error overlap, marginal gain per Top-k step).


### `ECE_frequent_vs_rare.ipynb`
Model calibration analysis. Computes Expected Calibration Error (ECE) and Brier
score and plots reliability diagrams with Wilson confidence intervals,
comparing central vs swarm models overall and for frequent (>50 images) vs rare
(<50 images) diseases (Main text Figure 3).

- **Consumes:** saved central/swarm checkpoints and GMDB metadata.
- **Produces:** ECE/Brier values and reliability-diagram figures.

### `bald_plot.ipynb`
Epistemic uncertainty via Monte-Carlo Dropout (BALD). Computes mutual
information per class for swarm vs central models
(ResNet-50 and ResNet-100), visualized against class sample counts (Supplementary Figure S6).

- **Consumes:** saved swarm and central checkpoints 
- **Produces:** per-class uncertainty plots.

### `accuracy_by_age_sex_top_30.ipynb`
Demographic robustness and fairness. Evaluates Top-N accuracy across age
(baby/kid/adult), sex (male/female), and ethnicity groups, plus per-syndrome
accuracy, for central vs swarm (Main text Figure 4)

- **Consumes:** saved central/swarm checkpoints, GMDB metadata (with age,
  gender, ethnicity), and a label mapping.
- **Produces:** combined demographic bar plot (`combined_plot.png` / `.pdf`)
  and per-syndrome / per-ethnicity accuracy bars.

### `clustering_top_30.ipynb`
Embedding-space analysis for a set of well-known syndromes. Computes clustering
quality metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz), distance-to-
centroid distributions, UMAP projections, and a dendrogram
colored by disease family — central vs swarm (Main text Figure 5).

- **Consumes:** saved central/swarm checkpoints, aligned images, and syndrome
  metadata.
- **Produces:** UMAP scatter plots, box/violin plots, and a
  disease-family-colored dendrogram




