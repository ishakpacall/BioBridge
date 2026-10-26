# Implementation notes

These are implementation choices of the released code that affect how results should be read. They are reported in the Limitations of the article. None of them uses test-set outcomes for fitting or selection. The code was not changed after the results were produced, so that the repository reproduces the article exactly.

| Topic | What the code does | Where |
|---|---|---|
| Preprocessing and inner holdout | Gene selection (top-500 rank variance), reference statistics for standardisation and median imputation are computed on the whole outer training set before the 25% inner holdout is drawn. These steps use no outcomes, but the inner holdout is not fully separated from preprocessing. | `src/biobridge/features.py`, `src/biobridge/models.py` |
| Transductive standardisation | In transfer and external settings the unlabelled target cohort is the reference for standardisation and PAM50 centring. Predictions therefore assume access to a batch of target expression profiles. | `src/biobridge/features.py` |
| XGBoost early stopping | Candidate configurations are ranked with the booster returned after early stopping (including the patience rounds); the final refit uses `best_iteration + 1` rounds. | `src/biobridge/models.py` |
| XGBoost-AFT initialisation | The initial prediction is the maximum-likelihood intercept of a censored intercept-only model, passed as `base_score = exp(mu0)`. The library default (0.5) starts every patient near log(0.5) log-months. | `src/biobridge/models.py` |
| Harrell's C | Pairs with equal observed times are not counted as comparable; tied risk scores count one half. | `src/biobridge/metrics.py` |
| DeepSurv loss | Partial likelihood over risk sets ordered by observed time, without a grouped Breslow or Efron correction for tied times. | `src/biobridge/deepsurv_worker.py` |
| PAM50 correlation | Spearman correlation computed from ordinal ranks obtained by sorting; ties are ordered by the sort rather than given average ranks. | `src/biobridge/features.py` |
| Survival forests | scikit-survival defaults: bootstrap sampling for both random survival forests and extra survival trees; survival curves average terminal-node Kaplan-Meier estimates. | `src/biobridge/models.py` |
| Discrete-time networks | 20 intervals at quantiles of training event times; duplicate cut points removed; horizons beyond the grid are clipped to its support. Output layers start at the empirical baseline hazard. | `src/biobridge/dl_worker.py` |
| Bootstrap intervals | Paired patient bootstrap (B = 1,000) conditional on the fitted models; training variability is not included. | `src/biobridge/metrics.py` |
| Learning-curve intervals | The 15 replicates per training size share patients, so their t-intervals are approximate. | `scripts/09_learning_curve.py` |
| Recalibration | The intercept update targets the Kaplan-Meier 5-year risk of the same external cohort in which it is then assessed (apparent performance). | `scripts/13_recalibration.py` |
| Messages | Some console and error messages are in Turkish. They do not affect results. | various |

Sensitivity analyses for these choices are planned for a later release; they will be added as separate scripts so that the published results remain reproducible from tag `v1.0.0`.
