"""
eda.py — Exploratory Data Analysis via parameter sweeps on the SIR simulation.
Can be run standalone (uses a built-in mock SIR) or with the real environment.
"""

import numpy as np
import matplotlib.pyplot as plt


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


def run_all_eda():
    print("Running EDA — this may take ~30 seconds...")
    eda_beta_sweep()
    eda_population_sweep()
    eda_beta_gamma_heatmap()
    eda_initial_infected_sweep()
    print("All EDA plots saved.")


if __name__ == "__main__":
    run_all_eda()
