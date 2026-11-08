# Predictions written BEFORE the evaluation was run (training of the predictors was still in progress)
Timestamp: $(date) -- see file mtime. Nothing below was revised after seeing evaluation output; deviations are reported in the paper.

Setting: hybrid nominal+neural one-step residual predictors; 25-step position error e_j; 90% target; calibration on ID episodes only;
shifts = mu-drop, ice, cornering-stiffness drop (cs), mass jump; "POST" windows = origins at/after the shift.

H1  Native intervals of the epistemic estimators (Ens, MCD, LL-Laplace, epistemic only) under-cover at 90% even in distribution (per-step coverage < 0.85 and joint coverage < 0.5).
H2  The heteroscedastic NLL ensemble (Het) is the best calibrated native estimator in distribution (|per-step coverage - 0.9| smallest among native estimators) but under-covers after shifts.
H3  The spread of Ens/MCD/LL-Laplace does not react to dynamics shifts: median spread at j=25 after the shift is < 1.5x its ID value, whereas the median error grows by > 1.5x for ice (and for mu-drop).
H4  The shift-detection AUROC of the spread (POST vs ID windows) is < 0.7 for Ens/MCD/LL-Laplace and > 0.9 for the delay-one residual monitor (Mon) for ice.
H5  Normalising conformal scores by the learned spread does not raise post-shift joint coverage above that of unnormalised split-CP by more than 0.05 for any shift.
H6  Every statically calibrated layer has joint coverage < 0.9 after the mu-drop, ice and stiffness shifts.
H7  Across the 80 test episodes of the mu-drop set, the Spearman correlation between shift severity and the post-shift median spread at j=25 is < 0.3 for Ens/MCD/LL-Laplace and > 0.5 for Mon.
H8  In distribution, the learned models are more accurate than the nominal model (median e_25 at least 2x smaller) but after the ice shift they are not (ratio of median e_25 learned/nominal > 0.8).
Thu Oct 22 07:02:49 EEST 2026
