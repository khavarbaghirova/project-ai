import numpy as np
import pickle
from config import (
    ACTIONS, N_ACTIONS,
    S_BINS, I_BINS, R_BINS,
    POPULATION,
    ALPHA, DISCOUNT, EPSILON, EPSILON_MIN, EPSILON_DECAY,
)


def discretize_state(state: tuple, s_bins=S_BINS, i_bins=I_BINS, r_bins=R_BINS) -> tuple:
    """
    Convert a raw-integer (S, I, R) tuple into discrete bin indices.
    State values are integers in [0, POPULATION] (agreed Day 1 spec).
    Returns a tuple of ints usable as a Q-table key.
    """
    S, I, R = state
    s_idx = min(int(S / POPULATION * s_bins), s_bins - 1)
    i_idx = min(int(I / POPULATION * i_bins), i_bins - 1)
    r_idx = min(int(R / POPULATION * r_bins), r_bins - 1)
    return (s_idx, i_idx, r_idx)


class QLearningAgent:
    """
    Tabular Q-Learning agent for epidemic intervention.

    State  : (S, I, R) — raw integers (discretized internally)
    Actions: 0 = do nothing, 1 = vaccinate, 2 = quarantine
    """

    def __init__(
        self,
        alpha: float = ALPHA,
        discount: float = DISCOUNT,
        epsilon: float = EPSILON,
        epsilon_min: float = EPSILON_MIN,
        epsilon_decay: float = EPSILON_DECAY,
    ):
        self.alpha         = alpha        # learning rate
        self.discount      = discount     # discount factor (γ)
        self.epsilon       = epsilon      # exploration rate
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: keys are discrete state tuples → numpy arrays of shape (N_ACTIONS,)
        self.q_table: dict[tuple, np.ndarray] = {}

    # Q-table helpers
    def _get_q(self, disc_state: tuple) -> np.ndarray:
        """Return Q-values for a state, initialising to zeros if unseen."""
        if disc_state not in self.q_table:
            self.q_table[disc_state] = np.zeros(N_ACTIONS)
        return self.q_table[disc_state]

    # Policy
    def select_action(self, state: tuple) -> int:
        """
        Epsilon-greedy action selection.
        state: raw integer (S, I, R) tuple
        """
        disc = discretize_state(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)       # explore
        return int(np.argmax(self._get_q(disc)))      # exploit


    # Learning
    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> float:
        """
        Q-Learning (off-policy TD) update.

        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]

        Returns the TD error (useful for diagnostics).
        """
        disc       = discretize_state(state)
        disc_next  = discretize_state(next_state)

        q_current  = self._get_q(disc)[action]
        q_next_max = 0.0 if done else np.max(self._get_q(disc_next))

        td_target  = reward + self.discount * q_next_max
        td_error   = td_target - q_current

        self._get_q(disc)[action] += self.alpha * td_error
        return td_error


    # Epsilon decay
    def decay_epsilon(self):
        """Call once per episode to reduce exploration over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Persistence
    def save(self, path: str = "q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)
        print(f"[Agent] Saved Q-table ({len(self.q_table)} states) → {path}")

    def load(self, path: str = "q_table.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]
        print(f"[Agent] Loaded Q-table ({len(self.q_table)} states) ← {path}")

    # Diagnostics
    def stats(self) -> dict:
        """Return a summary dict for logging/debugging."""
        all_q = np.concatenate(list(self.q_table.values())) if self.q_table else np.array([0])
        return {
            "states_visited": len(self.q_table),
            "epsilon":        round(self.epsilon, 4),
            "q_mean":         round(float(np.mean(all_q)), 4),
            "q_max":          round(float(np.max(all_q)), 4),
        }
