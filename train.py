import numpy as np
import json
from collections import defaultdict

# Environment import
from sir_env import SIREnv

from agent import QLearningAgent
from config import N_EPISODES, RANDOM_SEED, MAX_STEPS, DISCOUNT


def train(n_episodes: int = N_EPISODES, seed: int = RANDOM_SEED, verbose: bool = True):
    """
    Run the full training loop.
    Returns the trained agent and a log dict of episode metrics.
    """
    env   = SIREnv(seed=seed)
    agent = QLearningAgent()

    log = defaultdict(list)          # {episode_rewards, episode_lengths, …}

    for ep in range(1, n_episodes + 1):

        state      = env.reset()
        total_reward = 0.0
        ep_length    = 0
        td_errors    = []

        for _ in range(MAX_STEPS):
            action                     = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            td_err = agent.update(state, action, reward, next_state, done)
            td_errors.append(td_err)

            state         = next_state
            total_reward += reward
            ep_length    += 1

            if done:
                break

        agent.decay_epsilon()

        # logging 
        log["episode_rewards"].append(total_reward)
        log["episode_lengths"].append(ep_length)
        log["mean_td_error"].append(float(np.mean(np.abs(td_errors))))
        log["epsilon"].append(agent.epsilon)

        if verbose and (ep % 100 == 0 or ep == 1):
            avg_r  = np.mean(log["episode_rewards"][-100:])
            stats  = agent.stats()
            print(
                f"Episode {ep:5d}/{n_episodes} | "
                f"avg_reward(100)={avg_r:7.3f} | "
                f"ε={stats['epsilon']:.4f} | "
                f"states={stats['states_visited']:5d} | "
                f"Q_max={stats['q_max']:.3f}"
            )

    #  save artefacts 
    agent.save("q_table.pkl")
    with open("training_log.json", "w") as f:
        json.dump(dict(log), f, indent=2)
    print("[Train] training_log.json saved.")

    return agent, dict(log)


if __name__ == "__main__":
    agent, log = train()
