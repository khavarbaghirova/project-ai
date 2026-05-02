import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _save_or_show(path):
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    else:
        plt.show()
    plt.close()


# ── single episode ────────────────────────────────────────────────────────────

def plot_episode(episode: dict, title="SIR Episode", save_path=None):
    """Plots S, I, R curves for a single episode."""
    fig, ax = plt.subplots(figsize=(9, 5))
    steps = range(len(episode["S"]))

    ax.plot(steps, episode["S"], label="Susceptible", color="#2196F3", linewidth=2)
    ax.plot(steps, episode["I"], label="Infected",    color="#F44336", linewidth=2)
    ax.plot(steps, episode["R"], label="Recovered",   color="#4CAF50", linewidth=2)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Population")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    _save_or_show(save_path)


# ── policy comparison ─────────────────────────────────────────────────────────

def plot_policy_comparison(comparison: dict, save_path=None):
    """
    Bar chart comparing policies across 3 metrics:
    total infections, peak infections, duration.
    """
    names   = list(comparison.keys())
    metrics = ["mean_total_infections", "mean_peak_infections", "mean_duration"]
    labels  = ["Total Infections", "Peak Infections", "Duration (steps)"]
    colors  = ["#F44336", "#FF9800", "#2196F3"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Policy Comparison", fontsize=14, fontweight="bold")

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        values = [comparison[n][metric] for n in names]
        stds   = [comparison[n][metric.replace("mean", "std")] for n in names]
        bars = ax.bar(names, values, yerr=stds, color=color, alpha=0.8,
                      capsize=5, edgecolor="white")
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save_or_show(save_path)


def plot_infection_curves_comparison(comparison: dict, save_path=None):
    """
    Plots mean infection curve (I over time) for each policy on the same axes.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = cm.tab10(np.linspace(0, 1, len(comparison)))

    for (name, metrics), color in zip(comparison.items(), colors):
        episodes = metrics["episodes"]
        max_len  = max(len(ep["I"]) for ep in episodes)
        # pad shorter episodes with their last value
        padded = [ep["I"] + [ep["I"][-1]] * (max_len - len(ep["I"])) for ep in episodes]
        mean_I = np.mean(padded, axis=0)
        std_I  = np.std(padded, axis=0)

        ax.plot(mean_I, label=name, color=color, linewidth=2)
        ax.fill_between(range(max_len),
                        mean_I - std_I, mean_I + std_I,
                        color=color, alpha=0.15)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Infected")
    ax.set_title("Mean Infection Curve by Policy")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_or_show(save_path)


# ── EDA / parameter sweeps ────────────────────────────────────────────────────

def plot_beta_sweep(beta_values, total_infections, save_path=None):
    """Line plot: infection rate (beta) vs total infections."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(beta_values, total_infections, marker="o", color="#F44336", linewidth=2)
    ax.set_xlabel("Infection Rate (β)")
    ax.set_ylabel("Total Infections")
    ax.set_title("Effect of Infection Rate on Total Infections")
    ax.grid(alpha=0.3)
    _save_or_show(save_path)


def plot_population_sweep(pop_values, total_infections, save_path=None):
    """Line plot: population size vs total infections."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pop_values, total_infections, marker="s", color="#2196F3", linewidth=2)
    ax.set_xlabel("Population Size")
    ax.set_ylabel("Total Infections")
    ax.set_title("Effect of Population Size on Total Infections")
    ax.grid(alpha=0.3)
    _save_or_show(save_path)


def plot_parameter_heatmap(beta_values, gamma_values, infection_matrix, save_path=None):
    """
    Heatmap of total infections across beta x gamma grid.
    infection_matrix shape: (len(beta_values), len(gamma_values))
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(infection_matrix, aspect="auto", origin="lower",
                   cmap="YlOrRd",
                   extent=[min(gamma_values), max(gamma_values),
                            min(beta_values),  max(beta_values)])
    plt.colorbar(im, ax=ax, label="Total Infections")
    ax.set_xlabel("Recovery Rate (γ)")
    ax.set_ylabel("Infection Rate (β)")
    ax.set_title("Total Infections Across β–γ Parameter Space")
    _save_or_show(save_path)


def plot_reward_curve(rewards_per_episode: list, save_path=None):
    """Plots training reward curve for the RL agent (Person 2 hands you this list)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rewards_per_episode, color="#9C27B0", alpha=0.4, linewidth=1, label="Episode reward")

    # smoothed moving average
    window = max(1, len(rewards_per_episode) // 20)
    smoothed = np.convolve(rewards_per_episode,
                           np.ones(window) / window, mode="valid")
    ax.plot(range(window - 1, len(rewards_per_episode)),
            smoothed, color="#9C27B0", linewidth=2, label=f"Smoothed (w={window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("RL Agent Training Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_or_show(save_path)
