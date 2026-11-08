# Learn: regularised regression, from basics to advanced

Eight notebooks that build the ideas used in this project. Each mixes theory, from-scratch code and
the project's real UK data.

| # | Notebook | What you learn |
|---|---|---|
| 1 | [Why regularise](01_why_regularise.ipynb) | OLS overfitting when p ≈ n, bias-variance, the nowcasting problem |
| 2 | [Ridge](02_ridge.ipynb) | Closed form, SVD shrinkage, effective df, standardisation, correlated predictors |
| 3 | [Lasso](03_lasso.ipynb) | Soft-thresholding, coordinate descent from scratch, paths, weaknesses |
| 4 | [Elastic net](04_elastic_net.ipynb) | The grouping effect, tuning two parameters, selection stability on the real data |
| 5 | [Choosing lambda](05_choosing_lambda.ipynb) | Time-series CV vs shuffled CV, the 1-SE rule, information criteria |
| 6 | [Adaptive and group lasso](06_adaptive_and_group_lasso.ipynb) | Oracle property, group selection, FISTA |
| 7 | [Regularisation vs factor models](07_regularisation_vs_factors.ipynb) | PCR vs ridge, dense vs sparse signals |
| 8 | [Pitfalls](08_pitfalls.ipynb) | Selection instability, post-selection inference, leakage, revisions |

Run them from this folder after `python scripts/run_pipeline.py --data` has built `data/processed/`.

**Further reading**
* Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*, ch. 3 (free online).
* Hastie, Tibshirani and Wainwright, *Statistical Learning with Sparsity* (free online).
* Zou (2006) adaptive lasso; Yuan and Lin (2006) group lasso; Zou and Hastie (2005) elastic net.
* Bańbura, Giannone, Modugno and Reichlin (2013), "Now-casting and the real-time data flow".
