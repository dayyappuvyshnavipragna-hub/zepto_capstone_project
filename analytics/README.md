# Module 2 — Analytics Pipeline

## Run order

```bash
python 01_eda.py
python 02_modeling.py
```

`01_eda.py` is the only script that calls `sns.load_dataset("titanic")`. It immediately saves `titanic.csv`. `02_modeling.py` reads that CSV and does not reload the dataset from Seaborn.

## Missing-value rule

The measured missing percentages are printed by `01_eda.py` and saved in `eda_results.txt`.

The rule implemented is:

- under 5%: drop affected rows;
- 5%–30%: impute;
- above 30%: numeric columns are dropped; categorical columns receive an explicit `"Missing"` category.

The exact percentages produced by the execution are part of the generated `eda_results.txt`.

## Fare interpretation

The script calculates mean, median and mode. The skew conclusion is explicitly based on their ordering.

## Multivariate chart interpretations

1. **Survival by sex and class:** this compares survival rates across both sex and passenger class. The chart is intended to show that survival was not distributed uniformly across demographic/class groups.
2. **Fare by survival:** the box plot compares fare distributions for survivors and non-survivors and helps inspect the association between fare level and survival.
3. **Age vs fare:** the scatter plot shows how age and fare vary together while survival is encoded as the hue.
4. **Class and sex survival:** this gives another grouped view of the joint relationship between passenger class, sex and survival.

The generated plots are supporting artifacts; the numerical outputs and these written interpretations are the primary evidence.

## Modeling design

The train/test split is stratified because the target contains two classes and stratification preserves approximately the same class proportions in both splits.

Preprocessing is inside scikit-learn pipelines:

- numeric columns: median imputation + `StandardScaler`;
- categorical columns: most-frequent imputation + one-hot encoding.

The preprocessor is fitted only on `X_train` and then transforms `X_test`.

Three classifiers are trained on the same split: Logistic Regression, Decision Tree and Random Forest. All are evaluated with accuracy, precision, recall, F1, confusion matrix and ROC/AUC.

For imbalance handling, Logistic Regression is compared as baseline, `class_weight="balanced"`, and SMOTE applied only inside the training pipeline.

Random Forest tuning uses `GridSearchCV` over `n_estimators`, `max_depth`, and `max_features`, with `oob_score=True`.

The regression side-task predicts `fare` from the remaining available features and reports MAE, RMSE, R² and Adjusted R². The residual plot is used to assess whether residual spread appears non-random or heteroscedastic.

The final deployable artifact is `best_classifier_pipeline.joblib`, which contains both preprocessing and the estimator and can predict directly from raw feature columns.
