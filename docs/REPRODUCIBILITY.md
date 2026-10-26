# From scripts to the article

Run the steps in the README in order. The table maps each item of the article to the script that produces it.

| Article item | Produced by | File |
|---|---|---|
| Figure 1, study design | `tools/figures/figure1.py` | `results/manuscript/Figure1_final.png` |
| Figure 2, cohort selection and profiles | `tools/figures/figure2.py` | `results/manuscript/Figure2_final.png` |
| Figure 3, inputs and model structure | `tools/figures/figure3.py` | `results/manuscript/Figure3_final.png` |
| Figure 4, development discrimination | `scripts/12_figures_tables.py` | `results/manuscript/fig2_development_heatmap.png` |
| Figure 5, external validation | `scripts/12_figures_tables.py` | `results/manuscript/fig3_external_forest.png` |
| Figure 6, learning curve | `scripts/12_figures_tables.py` | `results/manuscript/fig4_learning_curve.png` |
| Figure 7, calibration and decision curves | `scripts/12_figures_tables.py` | `results/manuscript/fig5_calibration_dca.png` |
| Figure 8, TreeSHAP | `scripts/12_figures_tables.py` | `results/manuscript/fig6_shap.png` |
| Supplementary Figure 1, recalibration | `scripts/13_recalibration.py` | `results/manuscript/figS_recalibration.png` |
| Supplementary Figure 2, neural comparators | `scripts/17_additional_dl.py` | `results/manuscript/figS_deep_learning.png` |
| Table 1, cohort characteristics | `scripts/12_figures_tables.py` | `results/manuscript/table1_cohorts.csv` |
| Table 2, data sources and definitions | `config/sources.json`, `src/biobridge/prepare.py` | descriptive |
| Table 3, module features and coverage | `scripts/03_check_genesets.py` | console report |
| Tables 4-5, settings and model grids | `config/protocol.yaml` | descriptive |
| Supplementary Tables S1-S13 | `scripts/14_supplementary_tables.py` | `results/manuscript/supplementary/` |
| Supplementary Tables S14-S16 | `scripts/17_additional_dl.py` | `results/additional_dl/` |
| Submission package | `tools/build_publication_package.py` | `publication/` |

Figures 4-8 are renumbered in the article; the file names keep the numbering of the script. Figures 1-3 are schematic; Figure 2 uses the cohort counts of Table 1. `scripts/15_figure1_design.py` produces an earlier draft of Figure 1 and is kept because it is part of the locked code.

## Determinism

All random steps use fixed seeds (folds 20261010; models 42, 43, 44; bootstrap seeds in the scripts). Gradient boosting and neural networks run on CPU with a fixed number of threads. Small numerical differences can occur with other library versions, operating systems or thread counts; install `requirements-lock.txt` to reproduce the published numbers.

## Run time

On a laptop with four CPU threads: development about 7.7 hours, external validation 2-3 hours, additional neural comparators about 30 minutes, all other steps minutes.
