from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, mean_absolute_error,
    mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


ROOT = Path(__file__).resolve().parent
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(ROOT / "titanic.csv")

target = "survived"

# ============================================================
# 2. CLASSIFICATION DATA
# ============================================================

model_df = df.copy()

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = model_df[features]
y = model_df[target]

print("Class balance:")
print(y.value_counts())
print(y.value_counts(normalize=True))

# ============================================================
# 3. STRATIFIED TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

# ============================================================
# 4. TRAIN-ONLY PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numeric_features
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_features
    )
])

# ============================================================
# 5. THREE CLASSIFIERS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

results = {}
fitted_pipelines = {}

for name, estimator in models.items():

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    pipe.fit(X_train, y_train)
    fitted_pipelines[name] = pipe

    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, pred)

    auc = roc_auc_score(
        y_test,
        prob
    )

    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(
            y_test,
            pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            pred,
            zero_division=0
        ),
        "auc": auc,
        "confusion_matrix": cm.tolist()
    }

    # ROC curve
    fpr, tpr, _ = roc_curve(
        y_test,
        prob
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} AUC={auc:.3f}"
    )


# ============================================================
# 6. ROC CURVES
# ============================================================

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "roc_curves.png"
)

plt.close()


comparison = pd.DataFrame(results).T

print("\nClassifier comparison:")
print(
    comparison[
        [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc"
        ]
    ]
)

# ============================================================
# 7. DECISION TREE VISUALIZATION
# ============================================================

tree_pipe = fitted_pipelines["Decision Tree"]

feature_names = (
    tree_pipe
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

plt.figure(
    figsize=(22, 12)
)

plot_tree(
    tree_pipe.named_steps["model"],
    feature_names=feature_names,
    class_names=[
        "Not Survived",
        "Survived"
    ],
    filled=True,
    max_depth=3,
    fontsize=7
)

plt.title(
    "Decision Tree"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "decision_tree.png"
)

plt.close()

# ============================================================
# 8. CONFUSION MATRICES
# ============================================================

for name, pipe in fitted_pipelines.items():

    pred = pipe.predict(X_test)

    cm = confusion_matrix(
        y_test,
        pred
    )

    plt.figure(
        figsize=(5, 4)
    )

    plt.imshow(cm)

    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    safe = (
        name
        .lower()
        .replace(" ", "_")
    )

    plt.savefig(
        PLOT_DIR / f"cm_{safe}.png"
    )

    plt.close()

# ============================================================
# 9. CLASS IMBALANCE
# ============================================================

baseline = ImbPipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

baseline.fit(
    X_train,
    y_train
)

balanced_pre = ColumnTransformer([
    (
        "num",
        Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]),
        numeric_features
    ),
    (
        "cat",
        Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]),
        categorical_features
    )
])

balanced = Pipeline([
    (
        "preprocessor",
        balanced_pre
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

balanced.fit(
    X_train,
    y_train
)

# SMOTE is applied only to the training pipeline.
smote_pipe = ImbPipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "smote",
        SMOTE(random_state=42)
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

smote_pipe.fit(
    X_train,
    y_train
)

imbalance_rows = []

for name, pipe in [
    ("baseline", baseline),
    ("class_weight_balanced", balanced),
    ("SMOTE_train_only", smote_pipe)
]:

    pred = pipe.predict(X_test)

    imbalance_rows.append({
        "strategy": name,
        "precision": precision_score(
            y_test,
            pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            pred,
            zero_division=0
        )
    })

imbalance_df = pd.DataFrame(
    imbalance_rows
)

print("\nImbalance comparison:")
print(imbalance_df)

# ============================================================
# 10. RANDOM FOREST GRID SEARCH
# ============================================================

rf_pipe = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestClassifier(
            random_state=42,
            oob_score=True
        )
    )
])

param_grid = {
    "model__n_estimators": [
        100,
        200
    ],

    "model__max_depth": [
        None,
        5,
        10
    ],

    "model__max_features": [
        "sqrt",
        "log2"
    ]
}

grid = GridSearchCV(
    rf_pipe,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid.fit(
    X_train,
    y_train
)

best_rf = grid.best_estimator_

oob_score = (
    best_rf
    .named_steps["model"]
    .oob_score_
)

print(
    "\nBest RF parameters:",
    grid.best_params_
)

print(
    "Best RF OOB score:",
    oob_score
)

# ============================================================
# 11. FARE REGRESSION
# ============================================================

reg_df = df.drop(
    columns=["fare"]
).copy()

reg_features = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked",
    "alone",
    "adult_male"
]

reg_features = [
    c for c in reg_features
    if c in reg_df.columns
]

Xr = reg_df[reg_features]
yr = df["fare"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr,
    yr,
    test_size=0.20,
    random_state=42
)

reg_num = [
    c for c in reg_features
    if pd.api.types.is_numeric_dtype(
        Xr[c]
    )
]

reg_cat = [
    c for c in reg_features
    if c not in reg_num
]

reg_pre = ColumnTransformer([
    (
        "num",
        Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]),
        reg_num
    ),
    (
        "cat",
        Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]),
        reg_cat
    )
])

reg_pipe = Pipeline([
    (
        "preprocessor",
        reg_pre
    ),
    (
        "model",
        LinearRegression()
    )
])

reg_pipe.fit(
    Xr_train,
    yr_train
)

reg_pred = reg_pipe.predict(
    Xr_test
)

mae = mean_absolute_error(
    yr_test,
    reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        yr_test,
        reg_pred
    )
)

r2 = r2_score(
    yr_test,
    reg_pred
)

n = len(yr_test)

p = len(
    reg_pipe
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

adjusted_r2 = (
    1
    - (1 - r2)
    * (n - 1)
    / (n - p - 1)
)

print("\nRegression:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("Adjusted R2:", adjusted_r2)

# ============================================================
# 12. RESIDUAL PLOT
# ============================================================

residuals = yr_test - reg_pred

plt.figure(
    figsize=(7, 5)
)

plt.scatter(
    reg_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Fare"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Fare Regression Residual Plot"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "regression_residuals.png"
)

plt.close()

# ============================================================
# 13. HETEROSCEDASTICITY CONCLUSION
# ============================================================

# Simple visual diagnostic:
# compare residual spread in lower and higher predicted fare ranges.

residual_check = pd.DataFrame({
    "predicted": reg_pred,
    "residual": residuals
}).sort_values("predicted")

half = len(residual_check) // 2

low_group = residual_check.iloc[:half]["residual"]
high_group = residual_check.iloc[half:]["residual"]

low_std = low_group.std()
high_std = high_group.std()

if high_std > low_std * 1.25:
    hetero_conclusion = (
        "The residual spread is larger at higher predicted fare values, "
        "suggesting possible heteroscedasticity."
    )
elif low_std > high_std * 1.25:
    hetero_conclusion = (
        "The residual spread is larger at lower predicted fare values, "
        "suggesting possible heteroscedasticity."
    )
else:
    hetero_conclusion = (
        "The residual spread is relatively similar across the predicted "
        "fare range, so strong heteroscedasticity is not evident from "
        "this residual-spread check."
    )

print("\nHeteroscedasticity diagnostic:")
print(
    f"Lower predicted-fare residual SD: {low_std:.4f}"
)
print(
    f"Higher predicted-fare residual SD: {high_std:.4f}"
)
print(
    "Conclusion:",
    hetero_conclusion
)

# ============================================================
# 14. SELECT DEPLOYABLE CLASSIFIER
# ============================================================

# F1 is used as the selection criterion because the dataset
# has an imbalanced target and F1 balances precision and recall.

best_name = comparison["f1"].idxmax()

best_pipeline = fitted_pipelines[
    best_name
]

dump(
    best_pipeline,
    ROOT / "best_classifier_pipeline.joblib"
)

# ============================================================
# 15. RELOAD AND TEST DEPLOYED PIPELINE
# ============================================================

loaded = load(
    ROOT / "best_classifier_pipeline.joblib"
)

sample = X_test.iloc[[0]]

print(
    "\nReloaded pipeline prediction on raw input:"
)

print(
    loaded.predict(sample)
)

# ============================================================
# 16. FINAL CLASSIFICATION TABLE
# ============================================================

final_classification = comparison[
    [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc"
    ]
].copy()

final_classification[
    "model_type"
] = "classification"

print(
    "\nFinal classification table:"
)

print(
    final_classification
)

# ============================================================
# 17. FINAL REGRESSION METRICS
# ============================================================

regression_metrics = pd.DataFrame(
    [{
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adjusted_r2
    }],
    index=["Linear Regression"]
)

print(
    "\nFinal regression metrics:"
)

print(
    regression_metrics
)

# ============================================================
# 18. FINAL CLASSIFIER INTERPRETATION
# ============================================================

best_row = comparison.loc[best_name]

recommendation = (
    f"Based on the test-set F1 criterion, the deployable classifier "
    f"selected for this project is {best_name}. "
    f"It achieved accuracy={best_row['accuracy']:.4f}, "
    f"precision={best_row['precision']:.4f}, "
    f"recall={best_row['recall']:.4f}, "
    f"F1={best_row['f1']:.4f}, and AUC={best_row['auc']:.4f}. "
    f"The Logistic Regression model achieved F1={comparison.loc['Logistic Regression', 'f1']:.4f} "
    f"and AUC={comparison.loc['Logistic Regression', 'auc']:.4f}, "
    f"while the Decision Tree achieved F1={comparison.loc['Decision Tree', 'f1']:.4f} "
    f"and AUC={comparison.loc['Decision Tree', 'auc']:.4f}. "
    f"The F1-based selection criterion was used because the target classes "
    f"are imbalanced, with approximately 61.6% non-survivors and 38.4% survivors."
)

# ============================================================
# 19. SAVE COMPLETE MODEL RESULTS
# ============================================================

with open(
    ROOT / "model_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MODULE 2 - MODELING RESULTS\n"
    )

    f.write(
        "============================\n\n"
    )

    f.write(
        "CLASS BALANCE\n"
    )

    f.write(
        y.value_counts().to_string()
    )

    f.write(
        "\n\nClass proportions:\n"
    )

    f.write(
        y.value_counts(
            normalize=True
        ).to_string()
    )

    f.write(
        "\n\nCLASSIFICATION\n"
    )

    f.write(
        final_classification.to_string()
    )

    f.write(
        "\n\nCONFUSION MATRICES\n"
    )

    for name in results:
        f.write(
            f"\n{name}:\n"
        )
        f.write(
            str(
                results[name][
                    "confusion_matrix"
                ]
            )
        )

    f.write(
        "\n\nIMBALANCE COMPARISON\n"
    )

    f.write(
        imbalance_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nRANDOM FOREST GRID SEARCH\n"
    )

    f.write(
        f"Best parameters: {grid.best_params_}\n"
    )

    f.write(
        f"OOB score: {oob_score:.6f}\n"
    )

    f.write(
        "\n\nREGRESSION\n"
    )

    f.write(
        regression_metrics.to_string()
    )

    f.write(
        "\n\nHETEROSCEDASTICITY DIAGNOSTIC\n"
    )

    f.write(
        f"Lower predicted-fare residual SD: "
        f"{low_std:.6f}\n"
    )

    f.write(
        f"Higher predicted-fare residual SD: "
        f"{high_std:.6f}\n"
    )

    f.write(
        f"Conclusion: {hetero_conclusion}\n"
    )

    f.write(
        "\n\nFINAL CLASSIFIER INTERPRETATION\n"
    )

    f.write(
        recommendation
    )

    f.write(
        "\n\nDEPLOYED CLASSIFIER\n"
    )

    f.write(
        best_name
    )

print(
    "\nMODEL RESULTS SAVED TO:"
)

print(
    ROOT / "model_results.txt"
)

print(
    "\nDEPLOYED CLASSIFIER:"
)

print(
    best_name
)