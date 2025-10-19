# =======================
# Simulation Configuration
# =======================

# Environment size and population
N_AGENTS = 1000
GRID_WIDTH = 50
GRID_HEIGHT = 50
STEPS = 1000

# Multipliers (core dynamic factors)
FAMILY_MULTIPLIER = 1.0
HEALTHCARE_MULTIPLIER = 1.0
WEALTH_INFLUENCE_FACTOR = 1.0
INTERACTION_MULTIPLIER = 1.0

# Seeding setup
UND_RATIO = 0.1                    # fraction of undecided agents
MAJORITY_PARTY = "Conservatism"

# =======================
# Party Distributions (initial population %)
# Must sum to ~1.0 (will normalize automatically)
# =======================
CONSERVATISM_PERC = 0.6
SOCIALISM_PERC = 0.15
LIBERALISM_PERC = 0.15

# =======================
# Belief Spread (stddev for each party)
# =======================
CONSERVATISM_STD = 5
SOCIALISM_STD = 5
LIBERALISM_STD = 5

# =======================
# Family Structure
# =======================
MIN_FAMILY_SIZE = 3
MAX_FAMILY_SIZE = 10

# =======================
# Health and Wealth Factors
# =======================
SICKNESS_CHANCE = 0.05        # 5% chance of getting sick per step
WEALTH_PARTY_FOCUS = "Conservatism"
WEALTH_FOCUS_VALUE = "normal"  # “low” | “high” | “normal”
WEALTH_OTHER_VALUE = "normal"

# =======================
# Susceptibility Setup
# =======================
SUSC_PARTY_FOCUS = "Conservatism"
SUSC_FOCUS_VALUE = "low"    # “low” | “high” | “normal”
SUSC_OTHER_VALUE = "normal"

# =======================
# Random Seed (for reproducibility)
# =======================
RANDOM_SEED = 80
