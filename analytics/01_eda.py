from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. LOAD TITANIC DATA ONCE
# ============================================================

df = sns.load_dataset("titanic")

# Save the raw dataset immediately after loading.
df.to_csv(
    ROOT / "titanic.csv",
    index=False
)

print("INFO")
df.info()

print("\nDESCRIBE")
print(df.describe(include="all"))

print("\nSHAPE")
print(df.shape)

# ============================================================
# 2. MISSING VALUE ANALYSIS
# ============================================================

missing = df.isna().mean().mul(100)
missing = missing[missing > 0]

print("\nMissing percentages:")
print(missing.round(2))

# ============================================================
# 3. MISSING VALUE HANDLING
# ============================================================

clean = df.copy()

cleaning_decisions = []

for col in missing.index:

    pct = missing[col]

    if pct < 5:

        # Less than 5% missing:
        # drop affected rows.
        clean = clean.dropna(
            subset=[col]
        )

        cleaning_decisions.append(
            f"{col}: {pct:.2f}% missing -> "
            f"rows with missing values dropped because "
            f"missingness is below 5%."
        )

    elif pct <= 30:

        # 5% to 30% missing:
        # numeric -> median
        # categorical -> mode
        if pd.api.types.is_numeric_dtype(
            clean[col]
        ):

            median_value = clean[col].median()

            clean[col] = clean[col].fillna(
                median_value
            )

            cleaning_decisions.append(
                f"{col}: {pct:.2f}% missing -> "
                f"numeric values imputed using median "
                f"({median_value:.4f})."
            )

        else:

            mode_value = clean[col].mode()[0]

            clean[col] = clean[col].fillna(
                mode_value
            )

            cleaning_decisions.append(
                f"{col}: {pct:.2f}% missing -> "
                f"categorical values imputed using mode "
                f"({mode_value})."
            )

    else:

        # More than 30% missing:
        # For categorical variables, explicitly represent
        # missingness as a category.
        if pd.api.types.is_categorical_dtype(
            clean[col]
        ):

            clean[col] = (
                clean[col]
                .cat
                .add_categories(["Missing"])
            )

            clean[col] = clean[col].fillna(
                "Missing"
            )

            cleaning_decisions.append(
                f"{col}: {pct:.2f}% missing -> "
                f"retained as categorical 'Missing' "
                f"because missingness exceeds 30%."
            )

        elif pd.api.types.is_numeric_dtype(
            clean[col]
        ):

            clean = clean.drop(
                columns=[col]
            )

            cleaning_decisions.append(
                f"{col}: {pct:.2f}% missing -> "
                f"numeric column dropped because "
                f"missingness exceeds 30%."
            )

        else:

            clean[col] = clean[col].fillna(
                "Missing"
            )

            cleaning_decisions.append(
                f"{col}: {pct:.2f}% missing -> "
                f"missing values explicitly labelled "
                f"'Missing'."
            )

print("\nMissing-value cleaning decisions:")

for decision in cleaning_decisions:
    print("-", decision)

print(
    "\nClean shape:",
    clean.shape
)

# ============================================================
# 4. IQR OUTLIERS
# ============================================================

def iqr_outliers(series):

    q1, q3 = series.quantile(
        [0.25, 0.75]
    )

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (
        (series < lower)
        |
        (series > upper)
    )

    return (
        int(mask.sum()),
        lower,
        upper
    )


print("\nIQR OUTLIER ANALYSIS")

iqr_results = {}

for col in ["age", "fare"]:

    count, lower, upper = (
        iqr_outliers(
            clean[col]
        )
    )

    iqr_results[col] = (
        count,
        lower,
        upper
    )

    print(
        f"{col} IQR outliers: "
        f"{count}; "
        f"bounds=({lower:.3f}, {upper:.3f})"
    )

# ============================================================
# 5. FARE STATISTICS AND SKEWNESS
# ============================================================

fare_mean = clean["fare"].mean()
fare_median = clean["fare"].median()
fare_mode = clean["fare"].mode().iloc[0]

print("\nFare mean/median/mode:")

print(
    "mean =",
    fare_mean
)

print(
    "median =",
    fare_median
)

print(
    "mode =",
    fare_mode
)

if (
    fare_mean
    >
    fare_median
    >
    fare_mode
):

    skew_text = (
        "right-skewed because "
        "mean > median > mode."
    )

elif (
    fare_mean
    <
    fare_median
    <
    fare_mode
):

    skew_text = (
        "left-skewed because "
        "mean < median < mode."
    )

else:

    skew_text = (
        "not strictly classified "
        "using mean/median/mode ordering."
    )

print(
    "Fare distribution:",
    skew_text
)

# ============================================================
# 6. SURVIVAL ANALYSIS USING BOOLEAN MASKS
# ============================================================

# ----- Sex -----

female_mask = (
    clean["sex"] == "female"
)

male_mask = (
    clean["sex"] == "male"
)

female_survival = (
    clean.loc[
        female_mask,
        "survived"
    ].mean()
)

male_survival = (
    clean.loc[
        male_mask,
        "survived"
    ].mean()
)

sex_survival = pd.Series({
    "female": female_survival,
    "male": male_survival
})

print("\nSurvival by sex using boolean masks:")
print(sex_survival)

# ----- Passenger class -----

pclass_survival = {}

for pclass in sorted(
    clean["pclass"].dropna().unique()
):

    mask = (
        clean["pclass"] == pclass
    )

    pclass_survival[pclass] = (
        clean.loc[
            mask,
            "survived"
        ].mean()
    )

pclass_survival = pd.Series(
    pclass_survival
)

print("\nSurvival by pclass using boolean masks:")
print(pclass_survival)

# ----- Sex + Passenger class -----

sex_pclass_survival = {}

for sex in ["female", "male"]:

    for pclass in sorted(
        clean["pclass"].dropna().unique()
    ):

        mask = (
            (clean["sex"] == sex)
            &
            (clean["pclass"] == pclass)
        )

        sex_pclass_survival[
            (sex, pclass)
        ] = clean.loc[
            mask,
            "survived"
        ].mean()

sex_pclass_survival = pd.Series(
    sex_pclass_survival
)

print(
    "\nSurvival by sex and pclass "
    "using boolean masks:"
)

print(
    sex_pclass_survival
)

# ============================================================
# 7. CORRELATION ANALYSIS
# ============================================================

corr_cols = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr = clean[
    corr_cols
].corr()

print("\nCorrelation matrix:")
print(corr)

pairs = []

for i in range(
    len(corr_cols)
):

    for j in range(
        i + 1,
        len(corr_cols)
    ):

        pairs.append(
            (
                corr_cols[i],
                corr_cols[j],
                corr.iloc[i, j],
                abs(corr.iloc[i, j])
            )
        )

pairs = sorted(
    pairs,
    key=lambda x: x[3],
    reverse=True
)

print(
    "\nTwo strongest absolute "
    "off-diagonal correlations:"
)

for pair in pairs[:2]:
    print(pair)

# ============================================================
# 8. AGE AND FARE HISTOGRAMS + BOXPLOTS
# ============================================================

for col in ["age", "fare"]:

    plt.figure(
        figsize=(7, 5)
    )

    sns.histplot(
        clean[col],
        kde=True
    )

    plt.title(
        f"{col.title()} Histogram"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"{col}_hist.png"
    )

    plt.close()

    plt.figure(
        figsize=(7, 5)
    )

    sns.boxplot(
        x=clean[col]
    )

    plt.title(
        f"{col.title()} Box Plot"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"{col}_box.png"
    )

    plt.close()

# ============================================================
# 9. CORRELATION HEATMAP
# ============================================================

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Titanic Numeric Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "correlation_heatmap.png"
)

plt.close()

# ============================================================
# 10. MULTIVARIATE CHART 1
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    data=clean,
    x="sex",
    y="survived",
    hue="pclass"
)

plt.title(
    "Survival Rate by Sex and Passenger Class"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "survival_sex_pclass.png"
)

plt.close()

interpretation_1 = (
    "Survival rates differ by both sex and passenger class. "
    "Within the displayed groups, female passengers have higher "
    "survival rates than male passengers, while survival also "
    "varies across passenger classes. The chart shows that sex "
    "and pclass jointly provide useful descriptive information "
    "about survival outcomes."
)

# ============================================================
# 11. MULTIVARIATE CHART 2
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    data=clean,
    x="survived",
    y="fare"
)

plt.title(
    "Fare Distribution by Survival"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "fare_survival.png"
)

plt.close()

interpretation_2 = (
    "The fare distributions differ between passengers who "
    "survived and those who did not. The survival groups contain "
    "substantial overlap, indicating that fare alone does not "
    "separate the two outcomes completely. The boxplot also shows "
    "high-fare observations that contribute to the right-skewed "
    "fare distribution."
)

# ============================================================
# 12. MULTIVARIATE CHART 3
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.scatterplot(
    data=clean,
    x="age",
    y="fare",
    hue="survived",
    alpha=0.7
)

plt.title(
    "Age vs Fare by Survival"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "age_fare_survival.png"
)

plt.close()

interpretation_3 = (
    "The scatter plot shows the relationship between passenger "
    "age and fare while distinguishing survival status. Fare values "
    "vary widely across ages, and the survival groups overlap in "
    "much of the age-fare space. This indicates that age and fare "
    "together provide descriptive patterns but do not form a simple "
    "linear separation of survival outcomes."
)

# ============================================================
# 13. MULTIVARIATE CHART 4
# ============================================================

plt.figure(
    figsize=(8, 5)
)

sns.barplot(
    data=clean,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title(
    "Class and Sex Survival Pattern"
)

plt.tight_layout()

plt.savefig(
    PLOT_DIR / "class_sex_survival.png"
)

plt.close()

interpretation_4 = (
    "Survival varies across passenger class and sex simultaneously. "
    "The displayed class groups show different survival rates, and "
    "the difference between male and female survival is visible "
    "within the passenger classes. This supports examining sex and "
    "pclass together rather than interpreting either variable alone."
)

# ============================================================
# 14. Z-SCORE STANDARDIZATION
# ============================================================

standardized = clean.copy()

for col in [
    "age",
    "fare"
]:

    mean = standardized[col].mean()

    std = standardized[col].std()

    standardized[
        col + "_z"
    ] = (
        standardized[col] - mean
    ) / std

print("\nStandardization check:")

print(
    standardized[
        [
            "age_z",
            "fare_z"
        ]
    ].agg(
        ["mean", "std"]
    )
)

# ============================================================
# 15. SAVE COMPLETE EDA RESULTS
# ============================================================

with open(
    ROOT / "eda_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MODULE 2 - EDA RESULTS\n"
    )

    f.write(
        "======================\n\n"
    )

    f.write(
        "DATASET SHAPE\n"
    )

    f.write(
        str(df.shape)
    )

    f.write(
        "\n\nMISSING PERCENTAGES\n"
    )

    f.write(
        missing.round(2).to_string()
    )

    f.write(
        "\n\nMISSING-VALUE CLEANING DECISIONS\n"
    )

    for decision in cleaning_decisions:

        f.write(
            "- " + decision + "\n"
        )

    f.write(
        f"\nClean shape: {clean.shape}\n"
    )

    f.write(
        "\nIQR OUTLIERS\n"
    )

    for col in [
        "age",
        "fare"
    ]:

        count, lower, upper = (
            iqr_results[col]
        )

        f.write(
            f"{col}: "
            f"{count} outliers, "
            f"bounds=({lower:.4f}, "
            f"{upper:.4f})\n"
        )

    f.write(
        "\nFARE STATISTICS\n"
    )

    f.write(
        f"mean={fare_mean}\n"
    )

    f.write(
        f"median={fare_median}\n"
    )

    f.write(
        f"mode={fare_mode}\n"
    )

    f.write(
        f"Skew conclusion: {skew_text}\n"
    )

    f.write(
        "\nSURVIVAL BY SEX - BOOLEAN MASKS\n"
    )

    f.write(
        sex_survival.to_string()
    )

    f.write(
        "\n\nSURVIVAL BY PCLASS - BOOLEAN MASKS\n"
    )

    f.write(
        pclass_survival.to_string()
    )

    f.write(
        "\n\nSURVIVAL BY SEX AND PCLASS - BOOLEAN MASKS\n"
    )

    f.write(
        sex_pclass_survival.to_string()
    )

    f.write(
        "\n\nCORRELATION MATRIX\n"
    )

    f.write(
        corr.to_string()
    )

    f.write(
        "\n\nTWO STRONGEST CORRELATIONS\n"
    )

    for pair in pairs[:2]:

        f.write(
            str(pair) + "\n"
        )

    f.write(
        "\nMULTIVARIATE INTERPRETATIONS\n\n"
    )

    f.write(
        "Chart 1 - Survival by Sex and Passenger Class:\n"
    )

    f.write(
        interpretation_1 + "\n\n"
    )

    f.write(
        "Chart 2 - Fare Distribution by Survival:\n"
    )

    f.write(
        interpretation_2 + "\n\n"
    )

    f.write(
        "Chart 3 - Age vs Fare by Survival:\n"
    )

    f.write(
        interpretation_3 + "\n\n"
    )

    f.write(
        "Chart 4 - Class and Sex Survival Pattern:\n"
    )

    f.write(
        interpretation_4 + "\n\n"
    )

    f.write(
        "Z-SCORE STANDARDIZATION\n"
    )

    f.write(
        standardized[
            [
                "age_z",
                "fare_z"
            ]
        ].agg(
            ["mean", "std"]
        ).to_string()
    )

print(
    "\nEDA RESULTS SAVED TO:"
)

print(
    ROOT / "eda_results.txt"
)