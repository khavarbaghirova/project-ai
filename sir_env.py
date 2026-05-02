"""
sir_env.py — SIR Epidemic Environment (Gym-style)
Usage:
    from sir_env import SIREnv
    env = SIREnv()
    state = env.reset()
    state, reward, done, info = env.step(action)
"""

import random
from config import (
    POPULATION, INITIAL_INFECTED,
    BETA, GAMMA,
    VACCINE_EFFECT, QUARANTINE_EFFECT,
    MAX_STEPS,
    REWARD_INFECTION_WEIGHT, REWARD_ACTION_WEIGHT,
    ACTION_COSTS, N_ACTIONS,
    RANDOM_SEED,
)


class SIREnv:
    """
    Gym-style SIR epidemic environment.

    State : tuple (S, I, R) — raw integer counts
    Actions:
        0 = do nothing
        1 = vaccinate   (moves VACCINE_EFFECT * S people directly to R)
        2 = quarantine  (multiplies beta by QUARANTINE_EFFECT this step)

    Reward:
        reward = -REWARD_INFECTION_WEIGHT * new_infections
                 - REWARD_ACTION_WEIGHT   * action_cost
    """

    #  Construction                                                        

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        random.seed(seed)

        # will be populated by reset()
        self.S: int = 0
        self.I: int = 0
        self.R: int = 0
        self.step_count: int = 0
        self.done: bool = False

        # for render / history
        self._history: list[dict] = []

    #  Public API                                                          

    def reset(self) -> tuple[int, int, int]:
        """Reset to initial conditions. Returns starting state."""
        self.I = INITIAL_INFECTED
        self.S = POPULATION - self.I
        self.R = 0
        self.step_count = 0
        self.done = False
        self._history = [{"S": self.S, "I": self.I, "R": self.R,
                           "action": None, "reward": 0.0}]
        return self._state()

    def step(self, action: int) -> tuple[tuple[int, int, int], float, bool, dict]:
        """
        Apply action, advance SIR by one day.

        Parameters
        ----------
        action : int — 0, 1, or 2

        Returns
        -------
        state  : (S, I, R)
        reward : float
        done   : bool
        info   : dict with extra diagnostics
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if action not in range(N_ACTIONS):
            raise ValueError(f"Invalid action {action}. Must be in {list(range(N_ACTIONS))}.")

        # --- 1. Apply action pre-effects ---
        beta_effective = BETA

        if action == 1:          # vaccinate
            vaccinated = int(VACCINE_EFFECT * self.S)
            self.S -= vaccinated
            self.R += vaccinated

        elif action == 2:        # quarantine
            beta_effective = BETA * QUARANTINE_EFFECT

        # --- 2. SIR transition (discrete, integer) ---
        S_prev = self.S
        new_infections = self._compute_new_infections(beta_effective)
        new_recoveries = self._compute_new_recoveries()

        # clamp to valid ranges
        new_infections = min(new_infections, self.S)
        new_recoveries = min(new_recoveries, self.I)

        self.S -= new_infections
        self.I += new_infections - new_recoveries
        self.R += new_recoveries

        # safety: keep everything non-negative
        self.S = max(self.S, 0)
        self.I = max(self.I, 0)
        self.R = max(self.R, 0)

        # --- 3. Reward ---
        action_cost = ACTION_COSTS[action]
        reward = (- REWARD_INFECTION_WEIGHT * new_infections
                  - REWARD_ACTION_WEIGHT    * action_cost)

        # --- 4. Termination ---
        self.step_count += 1
        self.done = (self.I == 0) or (self.step_count >= MAX_STEPS)

        # --- 5. Bookkeeping ---
        state = self._state()
        info = {
            "new_infections": new_infections,
            "new_recoveries": new_recoveries,
            "action_cost": action_cost,
            "beta_effective": beta_effective,
            "step": self.step_count,
        }
        self._history.append({"S": self.S, "I": self.I, "R": self.R,
                               "action": action, "reward": reward})

        return state, reward, self.done, info

    def render(self, mode: str = "text") -> None:
        """Print a simple text summary of the current state."""
        bar_width = 40
        total = POPULATION

        s_bar = int(bar_width * self.S / total)
        i_bar = int(bar_width * self.I / total)
        r_bar = int(bar_width * self.R / total)

        print(f"Day {self.step_count:>3} | "
              f"S={self.S:>5} {'█' * s_bar:<{bar_width}} | "
              f"I={self.I:>5} {'█' * i_bar:<{bar_width}} | "
              f"R={self.R:>5} {'█' * r_bar:<{bar_width}}")

    @property
    def history(self) -> list[dict]:
        """Full episode history (list of dicts with S, I, R, action, reward)."""
        return self._history

    @property
    def action_space_n(self) -> int:
        return N_ACTIONS

    @property
    def observation(self) -> tuple[int, int, int]:
        return self._state()

    #  Internal helpers                                                    

    def _state(self) -> tuple[int, int, int]:
        return (self.S, self.I, self.R)

    def _compute_new_infections(self, beta: float) -> int:
        """
        Expected new infections = beta * S * I / N.
        We draw from a Binomial to keep the model stochastic.
        """
        if self.S == 0 or self.I == 0:
            return 0
        prob = 1.0 - (1.0 - beta * self.I / POPULATION) ** self.S
        prob = max(0.0, min(1.0, prob))   # numerical safety
        return self._binomial(self.S, prob)

    def _compute_new_recoveries(self) -> int:
        """Each infected individual recovers with probability GAMMA."""
        if self.I == 0:
            return 0
        return self._binomial(self.I, GAMMA)

    @staticmethod
    def _binomial(n: int, p: float) -> int:
        """Simple binomial draw using Python's random module."""
        if n <= 0 or p <= 0:
            return 0
        if p >= 1:
            return n
        return sum(1 for _ in range(n) if random.random() < p)


#  Quick standalone test — run: python sir_env.py                     

if __name__ == "__main__":
    print("=== SIR Environment standalone test ===\n")
    env = SIREnv(seed=42)
    state = env.reset()
    print(f"Initial state: S={state[0]}, I={state[1]}, R={state[2]}\n")

    total_reward = 0.0
    total_infections = 0

    for day in range(MAX_STEPS):
        action = random.choice([0, 1, 2])   # random policy for testing
        state, reward, done, info = env.step(action)
        total_reward += reward
        total_infections += info["new_infections"]
        env.render()

        if done:
            print(f"\nEpisode ended on day {day + 1}.")
            break

    print(f"\nTotal reward     : {total_reward:.2f}")
    print(f"Total infections : {total_infections}")
    print(f"Final state      : S={state[0]}, I={state[1]}, R={state[2]}")
