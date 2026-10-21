"""
=============================================================================
CROSS-BORDER CREDIT PASSPORT SYSTEM
STEP 2(A): DATA VISUALIZATION (02_data_visualization.py)
=============================================================================
What this script generates:
1. Target Distribution Comparison (Lending Club vs. FICO HELOC)
2. FICO Score vs. Default Rate Curve (The credit risk cliff)
3. Income & Debt-to-Income (DTI) Distribution with Outlier detection
4. FICO HELOC Special Negative Codes (-7, -8, -9) breakdown
5. Correlation Heatmap across key financial variables
All charts are saved into: reports/visualizations/
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for clean, professional charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FICO_PATH = os.path.join(DATA_DIR, "fico", "C:\\Users\\Bhargavi Pandey\\Downloads\\Cross-Border Credit Passport Project\\heloc_dataset_v1.csv")
LC_PATH = os.path.join(DATA_DIR, "lending_club", "C:\\Users\\Bhargavi Pandey\\Downloads\\Cross-Border Credit Passport Project\\LC_loans_granting_model_dataset.csv")


def plot_target_distributions():
    """
    Graph 1: Compares class imbalance across datasets.
    Helps explain why predicting credit default is an imbalanced problem.
    """
    print(" -> Generating Graph 1: Target Distributions (Class Balance)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. FICO HELOC Target
    if os.path.exists(FICO_PATH):
        df_fico = pd.read_csv(FICO_PATH, usecols=["RiskPerformance"])
        fico_counts = df_fico["RiskPerformance"].value_counts(normalize=True) * 100
        bars1 = axes[0].bar(fico_counts.index, fico_counts.values, color=["#e74c3c", "#2ecc71"], width=0.5)
        axes[0].set_title("FICO HELOC: Target ('RiskPerformance')", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("Percentage (%)")
        axes[0].set_ylim(0, 100)
        for bar in bars1:
            yval = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', fontweight='bold')

    # 2. Lending Club Target
    if os.path.exists(LC_PATH):
        df_lc = pd.read_csv(LC_PATH, usecols=["Default"], nrows=500000)
        lc_counts = df_lc["Default"].value_counts(normalize=True) * 100
        labels = ["Fully Paid (0)", "Defaulted (1)"]
        bars2 = axes[1].bar(labels, [lc_counts.get(0, 0), lc_counts.get(1, 0)], color=["#2ecc71", "#e74c3c"], width=0.5)
        axes[1].set_title("Lending Club: Target ('Default' Outcome)", fontsize=13, fontweight='bold')
        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_ylim(0, 100)
        for bar in bars2:
            yval = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "01_target_distributions.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_fico_vs_risk():
    """
    Graph 2: FICO Score vs. Default Rate Curve.
    Visually proves how credit scores correlate with default risk.
    """
    if not os.path.exists(LC_PATH):
        return
    print(" -> Generating Graph 2: FICO Score vs Default Rate...")
    
    df = pd.read_csv(LC_PATH, usecols=["fico_n", "Default"], nrows=500000)
    # Group FICO scores into 10-point buckets
    df["fico_bins"] = pd.cut(df["fico_n"], bins=np.arange(640, 850, 15))
    grouped = df.groupby("fico_bins", observed=False)["Default"].agg(["mean", "count"]).reset_index()
    grouped["default_rate_pct"] = grouped["mean"] * 100
    grouped["fico_label"] = grouped["fico_bins"].astype(str)

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(grouped)), grouped["default_rate_pct"], marker='o', linewidth=2.5, color="#c0392b")
    plt.xticks(range(len(grouped)), grouped["fico_label"], rotation=40, ha="right")
    plt.title("Credit Risk Curve: Higher FICO Score Lowers Default Rate", fontsize=14, fontweight='bold')
    plt.xlabel("FICO Score Range (US Standard)")
    plt.ylabel("Default Rate (%)")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Highlight the drop
    for i, txt in enumerate(grouped["default_rate_pct"]):
        plt.annotate(f"{txt:.1f}%", (i, txt + 0.8), ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "02_fico_vs_default_risk.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_financial_distributions_and_outliers():
    """
    Graph 3: Distribution and Outlier analysis for Income and DTI.
    """
    if not os.path.exists(LC_PATH):
        return
    print(" -> Generating Graph 3: Income & Debt-to-Income (DTI) Outliers...")

    df = pd.read_csv(LC_PATH, usecols=["revenue", "dti_n", "loan_amnt", "Default"], nrows=300000)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # A. Annual Income Distribution (Capped at 98th percentile for visualization)
    income_cap = df["revenue"].quantile(0.98)
    sns.histplot(df[df["revenue"] <= income_cap]["revenue"], bins=40, kde=True, ax=axes[0, 0], color="#2980b9")
    axes[0, 0].set_title("Annual Income Distribution (Filtered to 98th percentile)", fontweight='bold')
    axes[0, 0].set_xlabel("Annual Income ($)")

    # B. Income Boxplot by Default Status (shows if defaulters earn less)
    sns.boxplot(data=df[df["revenue"] <= income_cap], x="Default", y="revenue", ax=axes[0, 1], palette=["#2ecc71", "#e74c3c"])
    axes[0, 1].set_title("Annual Income: Fully Paid (0) vs Defaulted (1)", fontweight='bold')
    axes[0, 1].set_xticklabels(["Fully Paid (0)", "Defaulted (1)"])
    axes[0, 1].set_ylabel("Annual Income ($)")

    # C. Debt-to-Income (DTI) Distribution
    dti_cap = df["dti_n"].quantile(0.99)
    sns.histplot(df[df["dti_n"] <= dti_cap]["dti_n"], bins=40, kde=True, ax=axes[1, 0], color="#8e44ad")
    axes[1, 0].set_title("Debt-to-Income Ratio (DTI %) Distribution", fontweight='bold')
    axes[1, 0].set_xlabel("DTI (%) - (Debt Payments / Income)")

    # D. DTI Boxplot by Default Status
    sns.boxplot(data=df[df["dti_n"] <= dti_cap], x="Default", y="dti_n", ax=axes[1, 1], palette=["#2ecc71", "#e74c3c"])
    axes[1, 1].set_title("DTI: Defaulters Carry Higher Debt-to-Income", fontweight='bold')
    axes[1, 1].set_xticklabels(["Fully Paid (0)", "Defaulted (1)"])
    axes[1, 1].set_ylabel("DTI (%)")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "03_income_and_dti_analysis.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_fico_special_codes():
    """
    Graph 4: Visualizing FICO HELOC Special Negative Values (-7, -8, -9).
    Exposes the hidden codes representing 'No Inquiries' or 'Never Delinquent'.
    """
    if not os.path.exists(FICO_PATH):
        return
    print(" -> Generating Graph 4: FICO HELOC Special Codes Breakdown...")
    
    df = pd.read_csv(FICO_PATH)
    sample_cols = ["MSinceMostRecentDelq", "MSinceMostRecentInqexcl7days", "NetFractionInstallBurden"]
    
    code_summary = []
    for col in sample_cols:
        total = len(df)
        c_reg = (df[col] >= 0).sum() / total * 100
        c_neg8 = (df[col] == -8).sum() / total * 100
        c_neg7 = (df[col] == -7).sum() / total * 100
        code_summary.append({
            "Feature": col,
            "Valid Positive Number": c_reg,
            "Code -8 (Never Occurred / N/A)": c_neg8,
            "Code -7 (Condition Not Met)": c_neg7
        })
    
    df_codes = pd.DataFrame(code_summary).set_index("Feature")
    
    ax = df_codes.plot(kind="barh", stacked=True, figsize=(13, 5), color=["#27ae60", "#e67e22", "#95a5a6"])
    plt.title("Understanding FICO's Negative Numbers: Missing vs. 'Never Delinquent'", fontsize=13, fontweight='bold')
    plt.xlabel("Percentage of Applicants (%)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "04_fico_special_codes_breakdown.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_correlation_heatmap():
    """
    Graph 5: Correlation Matrix.
    Shows which variables are strongly linked to each other.
    """
    if not os.path.exists(LC_PATH):
        return
    print(" -> Generating Graph 5: Correlation Heatmap...")
    
    cols = ["revenue", "loan_amnt", "dti_n", "fico_n", "Default"]
    df = pd.read_csv(LC_PATH, usecols=cols, nrows=300000).dropna()
    
    # Rename columns to human-friendly labels for the chart
    df = df.rename(columns={
        "revenue": "Annual Income",
        "loan_amnt": "Loan Amount",
        "dti_n": "Debt-to-Income",
        "fico_n": "FICO Score",
        "Default": "Default Risk"
    })
    
    corr = df.corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-0.3, vmax=0.3, fmt=".2f", linewidths=1)
    plt.title("Correlation Heatmap: How Key Financial Features Relate", fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, "05_correlation_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    Saved: {save_path}")


def main():
    print("\n" + "#" * 70)
    print(" STEP 2 (PART B): GENERATING KEY DATASET VISUALIZATIONS")
    print("#" * 70)
    
    plot_target_distributions()
    plot_fico_vs_risk()
    plot_financial_distributions_and_outliers()
    plot_fico_special_codes()
    plot_correlation_heatmap()
    
    print("\n" + "=" * 70)
    print(f" ALL 5 GRAPHS SAVED SUCCESSFULLY IN:\n {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()