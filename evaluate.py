import numpy as np


def run_episode(env, policy, max_steps=100):
    """
    Runs a single episode using the given policy.
    
    Args:
        env: SIR environment with reset() and step() methods
        policy: callable(state) -> action
        max_steps: maximum number of steps
    
    Returns:
        dict with trajectory and summary metrics
    """
    state = env.reset()
    S_hist, I_hist, R_hist = [], [], []
    total_reward = 0
    done = False
    step = 0

    while not done and step < max_steps:
        S, I, R = state
        S_hist.append(S)
        I_hist.append(I)
        R_hist.append(R)

        action = policy(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        step += 1

    # Append final state
    S_hist.append(state[0])
    I_hist.append(state[1])
    R_hist.append(state[2])

    total_infections = R_hist[-1]  # everyone who recovered was infected
    peak_infections = max(I_hist)
    duration = len(I_hist)

    return {
        "S": S_hist,
        "I": I_hist,
        "R": R_hist,
        "total_infections": total_infections,
        "peak_infections": peak_infections,
        "duration": duration,
        "total_reward": total_reward,
    }


def evaluate_policy(env, policy, n_episodes=20, max_steps=100):
    """
    Runs multiple episodes and returns aggregated metrics.
    
    Returns:
        dict with mean and std of each metric
    """
    results = [run_episode(env, policy, max_steps) for _ in range(n_episodes)]

    total_infections = [r["total_infections"] for r in results]
    peak_infections  = [r["peak_infections"]  for r in results]
    durations        = [r["duration"]         for r in results]
    rewards          = [r["total_reward"]     for r in results]

    return {
        "mean_total_infections": np.mean(total_infections),
        "std_total_infections":  np.std(total_infections),
        "mean_peak_infections":  np.mean(peak_infections),
        "std_peak_infections":   np.std(peak_infections),
        "mean_duration":         np.mean(durations),
        "std_duration":          np.std(durations),
        "mean_reward":           np.mean(rewards),
        "std_reward":            np.std(rewards),
        "episodes":              results,
    }


def compare_policies(env, policies: dict, n_episodes=20, max_steps=100):
    """
    Evaluates multiple policies and returns a comparison dict.
    
    Args:
        policies: dict of {name: policy_fn}
    
    Returns:
        dict of {name: metrics}
    """
    comparison = {}
    for name, policy in policies.items():
        print(f"Evaluating: {name}...")
        comparison[name] = evaluate_policy(env, policy, n_episodes, max_steps)
    return comparison


def print_comparison_table(comparison: dict):
    """Prints a simple comparison table to console."""
    header = f"{'Policy':<25} {'Total Inf':>12} {'Peak Inf':>12} {'Duration':>12} {'Reward':>12}"
    print(header)
    print("-" * len(header))
    for name, metrics in comparison.items():
        print(
            f"{name:<25} "
            f"{metrics['mean_total_infections']:>10.1f}  "
            f"{metrics['mean_peak_infections']:>10.1f}  "
            f"{metrics['mean_duration']:>10.1f}  "
            f"{metrics['mean_reward']:>10.2f}"
        )
