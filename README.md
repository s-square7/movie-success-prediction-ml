# Movie Success Prediction with Decision Trees and XGBoost

Regression and classification on a movie-performance dataset, comparing an unpruned
Decision Tree against pre-pruned, post-pruned (cost-complexity), cross-validated and
XGBoost models.

**Author:** Shuvam Saren  
**Programme:** Summer Training and Internship Programme on ML & Agentic AI  
**Organised by:** Electronics & ICT Academy, Indian Institute of Technology Roorkee

---

## Problem Statement

Two supervised learning tasks are built on the same movie dataset:

| Task | Target | Type |
|---|---|---|
| Regression | `Collection` (box-office collection) | Continuous |
| Classification | `Start_Tech_Oscar` (award nomination) | Binary |

The goal is not only to fit a model but to study **overfitting in Decision Trees** and
show how pruning, hyperparameter search and boosting each address it.

## Dataset

`data/Movie_regression.csv` (18 features) and `data/Movie_classification.csv`
(19 features, including the binary target). Features cover marketing and production
expense, budget, multiplex coverage, movie length, actor/director/producer/critic
ratings, trailer views, 3D availability, genre, and number of multiplexes.

Preprocessing: rows with missing values dropped, categorical columns (`3D_available`,
`Genre`) one-hot encoded with `drop_first=True`, 80/20 train-test split
(`random_state=42`, stratified for classification).

> **Note:** the classification file retains `Collection` as a predictor. Since box-office
> collection is only known after release, this is post-outcome information relative to a
> pre-release award prediction — worth keeping in mind when reading the classification
> feature importances.

## Method

1. **Baseline** — fully grown `DecisionTreeRegressor` / `DecisionTreeClassifier`
2. **Level 1 — Pre-pruning** — `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_leaf_nodes`
3. **Level 2 — Post-pruning** — cost-complexity pruning, sweeping `ccp_alpha` over the full pruning path
4. **Level 3 — Cross-validation** — 5-fold CV plus `GridSearchCV` over the pruning hyperparameters
5. **Level 4 — XGBoost** — `XGBRegressor` and `XGBClassifier` with feature-importance analysis

## Results

### Regression — target `Collection`

| Model | Test RMSE | Test MAE | Test R² | Depth | Leaves |
|---|---|---|---|---|---|
| Unpruned Decision Tree | 7618.69 | 5789.90 | 0.7611 | 20 | 366 |
| Pre-pruned Decision Tree | 6410.80 | 5199.82 | 0.8308 | 5 | 20 |
| Post-pruned Decision Tree | 6790.54 | 5402.59 | 0.8102 | 5 | 14 |
| Cross-validated Decision Tree | 6463.39 | 5251.92 | 0.8281 | 5 | 22 |
| **XGBoost Regressor** | **5469.19** | **4095.64** | **0.8769** | NA | NA |

Best post-pruning alpha: `ccp_alpha = 1,868,492.89`.
Best grid parameters: `max_depth=5, min_samples_leaf=5, min_samples_split=2, max_leaf_nodes=None` (CV R² = 0.7628).

### Classification — target `Start_Tech_Oscar`

| Model | Accuracy | Precision | Recall | F1 | Depth | Leaves |
|---|---|---|---|---|---|---|
| Unpruned Decision Tree | 0.6162 | 0.6481 | 0.6481 | 0.6481 | 17 | 79 |
| Pre-pruned Decision Tree | 0.5960 | 0.6750 | 0.5000 | 0.5745 | 5 | 14 |
| Post-pruned Decision Tree | 0.6768 | 0.6833 | 0.7593 | 0.7193 | 14 | 33 |
| Cross-validated Decision Tree | 0.6465 | 0.6418 | 0.7963 | 0.7107 | 6 | 10 |
| **XGBoost Classifier** | **0.6768** | **0.6833** | **0.7593** | **0.7193** | NA | NA |

Best post-pruning alpha: `ccp_alpha = 0.005891`.
Best grid parameters: `max_depth=7, max_leaf_nodes=10, min_samples_leaf=5, min_samples_split=2` (CV accuracy = 0.5975).

### Takeaways

- The unpruned tree reached a **perfect training score (R² = 1.00, accuracy = 1.00)** in both
  tasks while testing far lower — textbook overfitting.
- Pruning cut the regression tree from 366 leaves to 14–22 while *raising* test R² from
  0.76 to ~0.83. Complexity fell and accuracy rose at the same time.
- `max_depth` had the largest single effect; `max_leaf_nodes` mattered most for
  interpretability.
- XGBoost gave the best predictive numbers on both tasks. A pruned single tree remains
  the better choice when the model has to be *explained* to a business audience.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   ├── Movie_regression.csv
│   └── Movie_classification.csv
├── notebooks/
│   └── decision_tree_pruning_cv_xgboost.ipynb
└── src/
    └── run_pipeline.py
```

## Getting Started

```bash
git clone https://github.com/<your-username>/movie-success-prediction-ml.git
cd movie-success-prediction-ml

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
jupyter notebook notebooks/decision_tree_pruning_cv_xgboost.ipynb
```

To run everything headless instead:

```bash
python src/run_pipeline.py
```

## Tech Stack

Python 3.10+ · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Seaborn · Jupyter

## Acknowledgement

Completed as part of the Summer Training and Internship Programme on Machine Learning &
Agentic AI conducted by the Electronics & ICT Academy, IIT Roorkee.

## License

MIT — see [LICENSE](LICENSE).
