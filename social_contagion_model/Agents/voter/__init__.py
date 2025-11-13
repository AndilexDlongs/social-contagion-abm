import numpy as np
from mesa.discrete_space import CellAgent

from .party_logic import PartyLogic
from .interaction import InteractionLogic
from .environment import EnvironmentLogic
from .economics import EconomicsLogic
from .health import HealthLogic
from .voting import VotingLogic
from .step_control import StepControl


class VoterAgent(
    CellAgent,
    PartyLogic,
    InteractionLogic,
    EnvironmentLogic,
    EconomicsLogic,
    HealthLogic,
    VotingLogic,
    StepControl,
):
    """Voter agent with attributes and a party preference."""

    def __init__(self, model, cell):
        super().__init__(model)

        # Attributes (named for clarity)
        self.LawAndOrder = np.random.uniform(0, 100)
        self.EconomicEquality = np.random.uniform(0, 100)
        self.SocialWelfare = np.random.uniform(0, 100)
        self.party_affiliation = "Undecided"
        self.original_party_affiliation = "Undecided"
        self.distance = self.party_distance()
        self.low = np.random.uniform(0, 0.1)
        self.mid = np.random.uniform(0.3, 0.8)
        self.high = np.random.uniform(1.6, 2)
        self.susceptibility = np.random.choice([self.low, self.high])
        self.switched_this_step = False
        self.switched_in_support = False
        self.switched_in_rebellion = False
        self.movement_tracker = {
            "death reaction": 0,
            "policy influence": 0,
            "family ripple": 0,
            "wealth comparison": 0,
        }
        # This will be used to track why an agent switched parties
        self.switch_cause = []
        self.has_interacted = False
        self.interacted_with = None
        self.interacted_within_party = False
        self.interacted_cross_party = False
        self.in_support = False
        self.wealth = 0
        # track the economic dissatisfaction of the agent
        self.wealth_dissatisfaction = 0
        self.dissatisfaction_threshold = 4  # threshold for economic dissatisfaction
        self.satisfaction_threshold = -8  # threshold for economic satisfaction
        self.dissatisfaction_multiplier = 5  # multiplier for dissatisfaction increase
        self.significant_difference = 50  # threshold for significant distance
        self.education = "Primary"  # Example additional attribute
        self.health_care = "Public"  # Example additional attribute
        self.healthy = True
        self.alive = True
        self.wealth_influence_factor = model.wealth_influence_factor
        self.interaction_multiplier = model.interaction_multiplier
        self.family_id = None
        self.cell = cell
        self.sickness_chance = 0.05
        self.family_members = None
        self.family_size = None
        self.interacted_within_family = False

        if self.model.majority_party == self.party_affiliation:
            self.in_support = True

    # ---------------------------
    # Helper methods
    # ---------------------------
    def get_members(self):
        """Return the IDs of this agent's family members."""
        if hasattr(self, "family"):
            return [m.unique_id for m in self.family.members]
        return []

    def evaluate_susceptibility(self):
        if self.susceptibility > 0.3:
            if np.random.random() < 0.7:
                self.susceptibility = self.mid
            else:
                self.susceptibility = self.high

    def belief_vector(self):
        return np.array(
            [self.LawAndOrder, self.EconomicEquality, self.SocialWelfare]
        )

    def update_from_vector(self, vec):
        self.LawAndOrder, self.EconomicEquality, self.SocialWelfare = vec

    @staticmethod
    def reflect(vec, lower=0, upper=100):
        """
        Reflects values in `vec` back into range [lower, upper].
        Works for scalars or NumPy arrays.
        """
        vec = np.asarray(vec, dtype=float)
        reflected = np.empty_like(vec)

        for i, val in np.ndenumerate(vec):
            while val < lower or val > upper:
                if val > upper:
                    val = upper - (val - upper)
                elif val < lower:
                    val = lower + (lower - val)
            reflected[i] = val
        return reflected

    def __repr__(self):
        return (
            f"Law&Order: {self.LawAndOrder:.1f}, "
            f"EconEquality: {self.EconomicEquality:.1f}, "
            f"SocWelfare: {self.SocialWelfare:.1f}, "
            f"Party: {self.party_affiliation}"
        )
