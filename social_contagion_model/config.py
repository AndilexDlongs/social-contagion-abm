# =======================
# Simulation Configuration
# =======================

# Environment size and population
N_AGENTS = 1000
GRID_WIDTH = 50
GRID_HEIGHT = 50
STEPS = 1000

# Multipliers (core dynamic factors)
FAMILY_MULTIPLIER = 0.8
HEALTHCARE_MULTIPLIER = 0.2
WEALTH_INFLUENCE_FACTOR = 0.6
INTERACTION_MULTIPLIER = 0.4

# Seeding setup
UND_RATIO = 0.1                    # fraction of undecided agents
MAJORITY_PARTY = "Socialism"

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
SOCIALISM_STD = 10
LIBERALISM_STD = 10

# =======================
# Family Structure
# =======================
MIN_FAMILY_SIZE = 2
MAX_FAMILY_SIZE = 3

# =======================
# Health and Wealth Factors
# =======================
SICKNESS_CHANCE = 0.01        # 5% chance of getting sick per step

# percentage of stubborn agents (low susceptibility)
CONSERVATISM_SUSC = 0.9
SOCIALISM_SUSC = 0.3
LIBERALISM_SUSC = 0.3

CONSERVATISM_WEALTH = 1
SOCIALISM_WEALTH = 0.3
LIBERALISM_WEALTH = 0.3

# =======================
# Random Seed (for reproducibility)
# =======================
RANDOM_SEED = 80
