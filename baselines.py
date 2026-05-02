import random


def random_policy(state):
    """Randomly chooses an action: 0=nothing, 1=vaccinate, 2=quarantine"""
    return random.randint(0, 2)


def no_intervention_policy(state):
    """Always does nothing"""
    return 0


def always_vaccinate_policy(state):
    """Always vaccinates — useful as an upper-bound baseline"""
    return 1


def always_quarantine_policy(state):
    """Always quarantines — useful as an upper-bound baseline"""
    return 2


def threshold_policy(state, threshold=0.1):
    """
    Simple rule-based policy:
    - If infected fraction > threshold, quarantine
    - Otherwise do nothing
    """
    S, I, R = state
    total = S + I + R
    if total == 0:
        return 0
    if I / total > threshold:
        return 2  # quarantine
    return 0  # nothing
