"""
eda.py — Exploratory Data Analysis via parameter sweeps on the SIR simulation.
Can be run standalone (uses a built-in mock SIR) or with the real environment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ── lightweight SIR simulation (no gym dependency) ───────────────────────────
# Replace this with the real env once Person 1 is done.

def run_sir_no_intervention(population, beta, gamma, initial_infected, max_steps=200):
    """Pure SIR simulation with no intervention, returns trajectory dict."""
    S = population - initial_infected
    I = initial_infected
    R = 0
    S_hist, I_hist, R_hist = [S], [I], [R]

    for _ in range(max_steps):
        if I == 0:
            break
        new_infected  = min(S, int(beta * S * I / population))
        new_recovered = min(I, int(gamma * I))
        S -= new_infected
        I += new_infected - new_recovered
        R += new_recovered
        S = max(S, 0)
        I = max(I, 0)
        S_hist.append(S)
        I_hist.append(I)
        R_hist.append(R)

    return {
        "S": S_hist, "I": I_hist, "R": R_hist,
        "total_infections": R_hist[-1],
        "peak_infections":  max(I_hist),
        "duration":         len(I_hist),
    }


# ── EDA plots ────────────────────────────────────────────────────────────────

def eda_beta_sweep(save=True):
    """How does infection rate affect outcomes?"""
    beta_values = np.linspace(0.05, 0.9, 30)
    total_inf, peak_inf, durations = [], [], []

    for beta in beta_values:
        result = run_sir_no_intervention(
            population=1000, beta=beta, gamma=0.05,
            initial_infected=10, max_steps=300
        )
        total_inf.append(result["total_infections"])
        peak_inf.append(result["peak_infections"])
        durations.append(result["duration"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Effect of Infection Rate (β) on Epidemic Outcomes", fontsize=13, fontweight="bold")

    axes[0].plot(beta_values, total_inf, color="#F44336", marker="o", markersize=3)
    axes[0].set_title("Total Infections")
    axes[0].set_xlabel("β"); axes[0].set_ylabel("Count"); axes[0].grid(alpha=0.3)

    axes[1].plot(beta_values, peak_inf, color="#FF9800", marker="o", markersize=3)
    axes[1].set_title("Peak Infections")
    axes[1].set_xlabel("β"); axes[1].set_ylabel("Count"); axes[1].grid(alpha=0.3)

    axes[2].plot(beta_values, durations, color="#2196F3", marker="o", markersize=3)
    axes[2].set_title("Epidemic Duration")
    axes[2].set_xlabel("β"); axes[2].set_ylabel("Steps"); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig("eda_beta_sweep.png", dpi=150, bbox_inches="tight")
        print("Saved: eda_beta_sweep.png")
    plt.show()


def eda_population_sweep(save=True):
    """How does population size affect outcomes?"""
    pop_values = range(100, 5001, 200)
    total_inf, peak_inf = [], []

    for pop in pop_values:
        result = run_sir_no_intervention(
            population=pop, beta=0.3, gamma=0.05,
            initial_infected=max(1, pop // 100), max_steps=300
        )
        total_inf.append(result["total_infections"])
        peak_inf.append(result["peak_infections"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Effect of Population Size on Epidemic Outcomes", fontsize=13, fontweight="bold")

    axes[0].plot(list(pop_values), total_inf, color="#F44336", marker="o", markersize=3)
    axes[0].set_title("Total Infections vs Population")
    axes[0].set_xlabel("Population"); axes[0].set_ylabel("Total Infections"); axes[0].grid(alpha=0.3)

    axes[1].plot(list(pop_values), peak_inf, color="#2196F3", marker="o", markersize=3)
    axes[1].set_title("Peak Infections vs Population")
    axes[1].set_xlabel("Population"); axes[1].set_ylabel("Peak Infections"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig("eda_population_sweep.png", dpi=150, bbox_inches="tight")
        print("Saved: eda_population_sweep.png")
    plt.show()


def eda_beta_gamma_heatmap(save=True):
    """Heatmap of total infections across beta x gamma space."""
    beta_values  = np.linspace(0.1, 0.9, 20)
    gamma_values = np.linspace(0.01, 0.2, 20)
    matrix = np.zeros((len(beta_values), len(gamma_values)))

    for i, beta in enumerate(beta_values):
        for j, gamma in enumerate(gamma_values):
            result = run_sir_no_intervention(
                population=1000, beta=beta, gamma=gamma,
                initial_infected=10, max_steps=300
            )
            matrix[i, j] = result["total_infections"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="YlOrRd",
                   extent=[gamma_values.min(), gamma_values.max(),
                            beta_values.min(),  beta_values.max()])
    plt.colorbar(im, ax=ax, label="Total Infections")
    ax.set_xlabel("Recovery Rate (γ)")
    ax.set_ylabel("Infection Rate (β)")
    ax.set_title("Total Infections Across β–γ Parameter Space")

    plt.tight_layout()
    if save:
        plt.savefig("eda_heatmap.png", dpi=150, bbox_inches="tight")
        print("Saved: eda_heatmap.png")
    plt.show()


def eda_initial_infected_sweep(save=True):
    """How does the number of initial infected affect the outbreak?"""
    init_values = range(1, 101, 5)
    total_inf, peak_inf = [], []

    for init in init_values:
        result = run_sir_no_intervention(
            population=1000, beta=0.3, gamma=0.05,
            initial_infected=init, max_steps=300
        )
        total_inf.append(result["total_infections"])
        peak_inf.append(result["peak_infections"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Effect of Initial Infected Count on Epidemic", fontsize=13, fontweight="bold")

    axes[0].plot(list(init_values), total_inf, color="#9C27B0", marker="o", markersize=4)
    axes[0].set_title("Total Infections")
    axes[0].set_xlabel("Initial Infected"); axes[0].set_ylabel("Count"); axes[0].grid(alpha=0.3)

    axes[1].plot(list(init_values), peak_inf, color="#00BCD4", marker="o", markersize=4)
    axes[1].set_title("Peak Infections")
    axes[1].set_xlabel("Initial Infected"); axes[1].set_ylabel("Count"); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig("eda_initial_infected.png", dpi=150, bbox_inches="tight")
        print("Saved: eda_initial_infected.png")
    plt.show()


def build_sweep_dataframe():
    """
    Runs a grid of simulations across beta, gamma, population, and initial_infected.
    Returns a pandas DataFrame — one row per simulation run.
    This is the 'dataset' used for five-number summary and correlation analysis.
    """
    rows = []
    beta_values    = np.linspace(0.05, 0.9, 15)
    gamma_values   = np.linspace(0.01, 0.2, 10)
    pop_values     = [200, 500, 1000, 2000, 5000]
    init_values    = [5, 10, 20, 50]

    for beta in beta_values:
        for gamma in gamma_values:
            for pop in pop_values:
                for init in init_values:
                    if init >= pop:
                        continue
                    result = run_sir_no_intervention(
                        population=pop, beta=beta, gamma=gamma,
                        initial_infected=init, max_steps=300
                    )
                    rows.append({
                        "beta":             beta,
                        "gamma":            gamma,
                        "population":       pop,
                        "initial_infected": init,
                        "total_infections": result["total_infections"],
                        "peak_infections":  result["peak_infections"],
                        "duration":         result["duration"],
                    })

    return pd.DataFrame(rows)


def eda_five_number_summary(df=None, save=True):
    """Prints and saves a five-number summary of all simulation outcomes."""
    if df is None:
        print("Building dataset for five-number summary...")
        df = build_sweep_dataframe()

    outcomes = ["total_infections", "peak_infections", "duration"]
    summary  = df[outcomes].describe(percentiles=[0.25, 0.5, 0.75])
    # keep only the five-number stats
    summary  = summary.loc[["min", "25%", "50%", "75%", "max"]]
    summary.index = ["Min", "Q1", "Median", "Q3", "Max"]

    print("\n=== Five-Number Summary of Simulation Outcomes ===")
    print(summary.to_string())

    # plot as a table figure
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText=summary.round(1).values,
        rowLabels=summary.index,
        colLabels=["Total Infections", "Peak Infections", "Duration (steps)"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)
    ax.set_title("Five-Number Summary of Simulation Outcomes",
                 fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    if save:
        plt.savefig("eda_five_number_summary.png", dpi=150, bbox_inches="tight")
        print("Saved: eda_five_number_summary.png")
    plt.show()
    return df


def eda_correlation_analysis(df=None, save=True):
    """Correlation heatmap between parameters and outcomes."""
    if df is None:
        print("Building dataset for correlation analysis...")
        df = build_sweep_dataframe()

    corr = df.corr(numeric_only=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Correlation Analysis", fontsize=13, fontweight="bold")

    # full correlation heatmap
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=axes[0]
    )
    axes[0].set_title("Full Correlation Matrix")

    # bar chart: correlation of each parameter with total_infections
    params = ["beta", "gamma", "population", "initial_infected"]
    corr_with_target = corr["total_infections"][params]
    colors = ["#F44336" if v > 0 else "#2196F3" for v in corr_with_target]
    axes[1].barh(params, corr_with_target, color=colors, edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Correlation with Total Infections")
    axes[1].set_title("Parameter Correlation with Total Infections")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig("eda_correlation.png", dpi=150, bbox_inches="tight")
        print("Saved: eda_correlation.png")
    plt.show()
    return df


def eda_boxplots(df=None, save=True):
    """Box plots showing distribution of outcomes — covers outlier detection."""
    if df is None:
        print("Building dataset for box plots...")
        df = build_sweep_dataframe()

    outcomes = ["total_infections", "peak_infections", "duration"]
    labels   = ["Total Infections", "Peak Infections", "Duration (steps)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Distribution of Simulation Outcomes (Box Plots)", fontsize=13, fontweight="bold")

    for ax, col, label in zip(axes, outcomes, labels):
        ax.boxplot(df[col], patch_artist=True,
                   boxprops=dict(facecolor="#90CAF9", color="#1565C0"),
                   medianprops=dict(color="#F44336", linewidth=2),
                   whiskerprops=dict(color="#1565C0"),
                   capprops=dict(color="#1565C0"),
                   flierprops=dict(marker="o", color="#FF9800", alpha=0.4))
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig("eda_boxplots.png", dpi=150, bbox_inches="tight")
        print("Saved: eda_boxplots.png")
    plt.show()
    return df


def run_all_eda():
    print("Running EDA — this may take 1-2 minutes...")
    eda_beta_sweep()
    eda_population_sweep()
    eda_beta_gamma_heatmap()
    eda_initial_infected_sweep()

    # build dataset once and reuse for all three
    print("Building full sweep dataset (this is the slow part)...")
    df = build_sweep_dataframe()
    print(f"Dataset built: {len(df)} simulation runs.")

    eda_five_number_summary(df)
    eda_correlation_analysis(df)
    eda_boxplots(df)

    print("\nAll EDA plots saved.")


if __name__ == "__main__":
    run_all_eda()
