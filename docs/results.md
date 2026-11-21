# Results

All numbers are on the held-out **test** set, produced by `lead-scoring train` with the
model selected on validation PR-AUC and calibrated (isotonic) on the validation set.
UCI Bank Marketing, `bank-additional-full`, 41,188 contacts, call duration excluded.

| Split | Features | Selected model | ROC-AUC | PR-AUC | Test base rate | Lift@10% | Capture@10% | Brier |
|---|---|---|---|---|---|---|---|---|
| random | all | hist_gradient_boosting | 0.812 | 0.471 | 11.3% | 4.81x | 48% | 0.074 |
| random | client | hist_gradient_boosting | 0.734 | 0.355 | 11.3% | 3.39x | 34% | 0.085 |
| time | all | logistic_regression | 0.561 | 0.436 | 38.5% | 1.23x | 12% | 0.295 |
| time | client | hist_gradient_boosting | 0.589 | 0.443 | 38.5% | 1.21x | 12% | 0.289 |

## Drift between the training period and the test period (`lead-scoring drift`)

| Feature | PSI | Status |
|---|---|---|
| cons_price_idx | 14.082 | alert |
| cons_conf_idx | 14.082 | alert |
| nr_employed | 14.060 | alert |
| emp_var_rate | 13.610 | alert |
| euribor3m | 8.486 | alert |
| poutcome | 1.506 | alert |
| previous | 1.452 | alert |
| month | 1.385 | alert |
| pdays | 1.039 | alert |
| contact | 0.654 | alert |
| default | 0.471 | alert |
| job | 0.469 | alert |
| age | 0.276 | alert |
| education | 0.121 | watch |
| campaign | 0.085 | stable |
| marital | 0.085 | stable |
| day_of_week | 0.045 | stable |
| housing | 0.009 | stable |
| loan | 0.000 | stable |

alerts: job, default, contact, month, poutcome, age, previous, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed, pdays
