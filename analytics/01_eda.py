from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# The only network/cache dataset load in this module.
df = sns.load_dataset("titanic")
df.to_csv(ROOT / "titanic.csv", index=False)

print("INFO")
df.info()
print("\nDESCRIBE")
print(df.describe(include="all"))
print("\nSHAPE")
print(df.shape)

missing = df.isna().mean().mul(100)
missing = missing[missing > 0]
print("\nMissing percentages:")
print(missing.round(2))

# EDA cleaning according to the assignment threshold.
clean = df.copy()

for col in missing.index:

    pct = missing[col]

    if pct < 5:
        clean = clean.dropna(subset=[col])

    elif pct <= 30:
        if pd.api.types.is_numeric_dtype(clean[col]):
            clean[col] = clean[col].fillna(clean[col].median())
        else:
            clean[col] = clean[col].fillna(clean[col].mode()[0])

    else:
        # Keep high-missing categorical data as an explicit category.
        if pd.api.types.is_numeric_dtype(clean[col]):
            clean = clean.drop(columns=[col])
        else:
            if str(clean[col].dtype) == "category":
                clean[col] = clean[col].cat.add_categories(["Missing"])
            clean[col] = clean[col].fillna("Missing")

print("\nClean shape:", clean.shape)

def iqr_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    mask = (series < lo) | (series > hi)
    return int(mask.sum()), lo, hi

for col in ["age", "fare"]:
    count, lo, hi = iqr_outliers(clean[col])
    print(f"{col} IQR outliers: {count}; bounds=({lo:.3f}, {hi:.3f})")

print("\nFare mean/median/mode:")
fare_mean = clean["fare"].mean()
fare_median = clean["fare"].median()
fare_mode = clean["fare"].mode().iloc[0]
print("mean =", fare_mean)
print("median =", fare_median)
print("mode =", fare_mode)

if fare_mean > fare_median > fare_mode:
    skew_text = "right-skewed"
elif fare_mean < fare_median < fare_mode:
    skew_text = "left-skewed"
else:
    skew_text = "not strictly classified by the mean/median/mode ordering"
print("Fare distribution:", skew_text)

print("\nSurvival by sex:")
print(clean.groupby("sex")["survived"].mean())

print("\nSurvival by pclass:")
print(clean.groupby("pclass")["survived"].mean())

print("\nSurvival by sex and pclass:")
print(clean.groupby(["sex", "pclass"])["survived"].mean())

corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = clean[corr_cols].corr()
print("\nCorrelation matrix:")
print(corr)

pairs = []
for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        pairs.append((corr_cols[i], corr_cols[j], corr.iloc[i, j], abs(corr.iloc[i, j])))
pairs = sorted(pairs, key=lambda x: x[3], reverse=True)
print("\nTwo strongest absolute off-diagonal correlations:")
for p in pairs[:2]:
    print(p)

# Histograms and box plots for age and fare.
for col in ["age", "fare"]:
    plt.figure(figsize=(7, 5))
    sns.histplot(clean[col], kde=True)
    plt.title(f"{col.title()} Histogram")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{col}_hist.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.boxplot(x=clean[col])
    plt.title(f"{col.title()} Box Plot")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{col}_box.png")
    plt.close()

# Correlation heatmap.
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Titanic Numeric Correlation Heatmap")
plt.tight_layout()
plt.savefig(PLOT_DIR / "correlation_heatmap.png")
plt.close()

# Four multivariate charts.
plt.figure(figsize=(8, 5))
sns.barplot(data=clean, x="sex", y="survived", hue="pclass")
plt.title("Survival Rate by Sex and Passenger Class")
plt.tight_layout()
plt.savefig(PLOT_DIR / "survival_sex_pclass.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=clean, x="survived", y="fare")
plt.title("Fare Distribution by Survival")
plt.tight_layout()
plt.savefig(PLOT_DIR / "fare_survival.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=clean, x="age", y="fare", hue="survived", alpha=0.7)
plt.title("Age vs Fare by Survival")
plt.tight_layout()
plt.savefig(PLOT_DIR / "age_fare_survival.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=clean, x="pclass", y="survived", hue="sex")
plt.title("Class and Sex Survival Pattern")
plt.tight_layout()
plt.savefig(PLOT_DIR / "class_sex_survival.png")
plt.close()

# Exploratory z-score standardization on the full cleaned DataFrame.
standardized = clean.copy()
for col in ["age", "fare"]:
    mean = standardized[col].mean()
    std = standardized[col].std()
    standardized[col + "_z"] = (standardized[col] - mean) / std

print("\nStandardization check:")
print(standardized[["age_z", "fare_z"]].agg(["mean", "std"]))

# Save a text summary for the README/reporting.
with open(ROOT / "eda_results.txt", "w", encoding="utf-8") as f:
    f.write("Missing percentages:\n")
    f.write(missing.round(2).to_string())
    f.write("\n\nIQR outliers:\n")
    for col in ["age", "fare"]:
        count, lo, hi = iqr_outliers(clean[col])
        f.write(f"{col}: {count}, bounds=({lo}, {hi})\n")
    f.write("\nFare statistics:\n")
    f.write(f"mean={fare_mean}\nmedian={fare_median}\nmode={fare_mode}\n")
    f.write(f"skew conclusion={skew_text}\n\n")
    f.write("Survival by sex:\n")
    f.write(clean.groupby("sex")["survived"].mean().to_string())
    f.write("\n\nSurvival by pclass:\n")
    f.write(clean.groupby("pclass")["survived"].mean().to_string())
    f.write("\n\nSurvival by sex and pclass:\n")
    f.write(clean.groupby(["sex", "pclass"])["survived"].mean().to_string())
    f.write("\n\nCorrelation matrix:\n")
    f.write(corr.to_string())
    f.write("\n\nTwo strongest correlations:\n")
    for p in pairs[:2]:
        f.write(str(p) + "\n")