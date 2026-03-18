# Data Science Bundle
> Skills: EDA · Polars · SHAP · Statistical Analysis
> Usage: Paste this URL at session start for all data science tasks.

---

## 1. Exploratory Data Analysis (EDA)
**Trigger:** "explore/analyze/summarize a data file"
**Supports:** 200+ formats — CSV, FASTQ, PDB, TIFF, HDF5, NPY, DICOM, mzML, etc.

**Workflow:**
1. Detect file type by extension
2. Load format-specific analysis approach
3. Perform structure + quality + stats analysis
4. Generate markdown report → `{filename}_eda_report.md`

**Report sections:** Basic Info · File Type Details · Data Analysis · Key Findings · Recommendations

**By data type:**
```python
# Tabular (CSV/Excel)
import pandas as pd
df = pd.read_csv('data.csv')
print(df.shape, df.dtypes, df.isnull().sum(), df.describe())

# Sequence (FASTA/FASTQ)
from Bio import SeqIO
seqs = list(SeqIO.parse('reads.fastq', 'fastq'))
# → count, length dist, GC content, quality scores

# Arrays (NPY/HDF5)
import numpy as np, h5py
arr = np.load('data.npy')
print(arr.shape, arr.dtype, arr.min(), arr.max(), np.isnan(arr).sum())
```

**Common libs by category:**
- Bioinformatics: `biopython`, `pysam`
- Chemistry: `rdkit`, `mdanalysis`
- Microscopy: `tifffile`, `nd2reader`, `pydicom`
- General: `pandas`, `numpy`, `h5py`

---

## 2. Polars
**Trigger:** "fast dataframe", "pandas too slow", "1-100GB dataset", "ETL pipeline"

**Core pattern:**
```python
import polars as pl

# Lazy (preferred for large data)
result = (
    pl.scan_csv("large.csv")
    .filter(pl.col("age") > 25)
    .with_columns(pl.col("value") * 10)
    .group_by("city").agg(pl.col("age").mean())
    .collect()
)

# Window functions
df.with_columns(
    avg_by_city=pl.col("age").mean().over("city"),
    rank=pl.col("salary").rank().over("city")
)

# Joins / concat
df1.join(df2, on="id", how="left")
pl.concat([df1, df2], how="vertical")
```

**Pandas → Polars quick map:**
| Pandas | Polars |
|--------|--------|
| `df[df["x"] > 10]` | `df.filter(pl.col("x") > 10)` |
| `df.assign(x=...)` | `df.with_columns(x=...)` |
| `df.groupby().transform()` | `df.with_columns(...).over()` |

**Performance rules:** use lazy, select early, avoid `map_elements`, use `streaming=True` for huge data.

---

## 3. SHAP (Model Explainability)
**Trigger:** "explain model", "feature importance", "why did model predict", "SHAP plots"

**Explainer selection:**
```python
import shap

# Tree models (XGBoost, LightGBM, RF) — fastest
explainer = shap.TreeExplainer(model)

# Neural nets (TF/PyTorch)
explainer = shap.DeepExplainer(model, background_data)

# Linear models
explainer = shap.LinearExplainer(model, X_train)

# Any black-box (slow)
explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))

# Auto-select
explainer = shap.Explainer(model)
```

**Core workflow:**
```python
shap_values = explainer(X_test)

# Global importance
shap.plots.beeswarm(shap_values)     # distribution + magnitude
shap.plots.bar(shap_values)          # mean absolute SHAP

# Single prediction
shap.plots.waterfall(shap_values[0]) # feature contributions
shap.plots.force(shap_values[0])     # additive force view

# Feature relationships
shap.plots.scatter(shap_values[:, "feature_name"])
```

**Interpretation:** positive SHAP = pushes prediction up · magnitude = strength · values sum to prediction − baseline

---

## 4. Statistical Analysis
**Trigger:** "hypothesis test", "t-test", "ANOVA", "regression", "p-value", "APA report"

**Test selection quick ref:**
| Scenario | Test |
|----------|------|
| 2 groups, continuous, normal | Independent t-test |
| 2 groups, non-normal | Mann-Whitney U |
| 2 groups, paired | Paired t-test / Wilcoxon |
| 3+ groups, normal | One-way ANOVA → Tukey HSD |
| 3+ groups, non-normal | Kruskal-Wallis |
| 2 continuous vars | Pearson / Spearman correlation |
| Binary outcome | Logistic regression |

**Core pattern (pingouin):**
```python
import pingouin as pg

# T-test with effect size
result = pg.ttest(group_a, group_b, correction='auto')
# → T, dof, p-val, cohen-d, CI95%

# ANOVA + post-hoc
aov = pg.anova(dv='score', between='group', data=df, detailed=True)
if aov['p-unc'].values[0] < 0.05:
    posthoc = pg.pairwise_tukey(dv='score', between='group', data=df)
```

**APA report format:**
```
t(98) = 3.82, p < .001, d = 0.77, 95% CI [0.36, 1.18]
F(2, 147) = 8.45, p < .001, η²_p = .10
```

**Effect size benchmarks:**
| Test | Small | Medium | Large |
|------|-------|--------|-------|
| Cohen's d | 0.20 | 0.50 | 0.80 |
| η²_p | 0.01 | 0.06 | 0.14 |
| r | 0.10 | 0.30 | 0.50 |

**Always:** check assumptions first (`pg.normality`, `pg.homoscedasticity`) · report effect sizes + CIs · distinguish statistical from practical significance.
