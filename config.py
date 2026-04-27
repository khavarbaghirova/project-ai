"""
config.py — Shared configuration 
All teammates import from here. Do NOT hardcode these values elsewhere
"""

# Population & SIR parameters 
POPULATION        = 1000
INITIAL_INFECTED  = 10
BETA              = 0.3          # infection rate
GAMMA             = 0.05         # recovery rate
VACCINE_EFFECT    = 0.1          # fraction of S vaccinated per step  
QUARANTINE_EFFECT = 0.5          # beta multiplier during quarantine   
MAX_STEPS         = 100          # episode length (days)  

# Actions 
# 0 = do nothing, 1 = vaccinate, 2 = quarantine
ACTIONS   = [0, 1, 2]
N_ACTIONS = len(ACTIONS)

# Action costs — 0 for nothing, 1 for vaccinate or quarantine
ACTION_COSTS = {
    0: 0,
    1: 1,
    2: 1,
}

# Reward function 
# reward = -new_infections - 0.5 * action_cost
REWARD_INFECTION_WEIGHT = 1.0
REWARD_ACTION_WEIGHT    = 0.5

# State discretisation 
# State is raw integers (S, I, R) each in [0, POPULATION]
S_BINS = 20
I_BINS = 20
R_BINS = 10

# RL hyper-parameters 
ALPHA         = 0.1
DISCOUNT      = 0.95    
EPSILON       = 1.0
EPSILON_MIN   = 0.05
EPSILON_DECAY = 0.995
N_EPISODES    = 2000

# Evaluation 
EVAL_EPISODES = 100
RANDOM_SEED   = 42
