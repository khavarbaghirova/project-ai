from sir_env import SIREnv
from agent import QLearningAgent
from baselines import random_policy, no_intervention_policy, threshold_policy
from evaluate import compare_policies, print_comparison_table
from plot import plot_policy_comparison, plot_infection_curves_comparison, plot_reward_curve
import json

# load trained agent
agent = QLearningAgent()
agent.load("q_table.pkl")
agent.epsilon = 0.0  # no exploration, pure greedy

env = SIREnv()

policies = {
    "No Intervention": no_intervention_policy,
    "Random":          random_policy,
    "Threshold":       threshold_policy,
    "RL Agent":        agent.select_action,
}

print("\nEvaluating policies...")
results = compare_policies(env, policies, n_episodes=100)
print_comparison_table(results)

# save plots
plot_policy_comparison(results, save_path="comparison_bars.png")
plot_infection_curves_comparison(results, save_path="infection_curves.png")

with open("training_log.json") as f:
    log = json.load(f)
plot_reward_curve(log["episode_rewards"], save_path="training_curve.png")

print("\nDone. Plots saved.")
