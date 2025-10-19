import numpy as np
from mesa.discrete_space import CellAgent


class Party:
    def __init__(self, name, LawAndOrder, EconomicEquality, SocialWelfare, radius):
        self.name = name
        self.LawAndOrder = LawAndOrder
        self.EconomicEquality = EconomicEquality
        self.SocialWelfare = SocialWelfare
        self.current_party_in_power = "Undecided" # Example initial state this needs to be a global variable
        self.radius = radius  # how tolerant/inclusive the party is

    def center_vector(self):
        """Return the party's attribute profile as a vector."""
        return np.array([self.LawAndOrder, self.EconomicEquality, self.SocialWelfare])       

class FamilyAgent:
    """Group-level agent representing a family unit."""
    family_counter = 0  # static counter shared across all FamilyAgents

    def __init__(self, model, members):
        self.model = model
        self.members = members  # list of VoterAgents
        self.id = FamilyAgent.family_counter  # unique family number
        FamilyAgent.family_counter += 1

        self.family_multiplier = model.family_multiplier
        self.healthcare_multiplier = model.healthcare_multiplier

        # assign this family to its members
        for member in members:
            member.family = self
            member.family_id = self.id   


    def ripple_influence(self, initiator, old_vec, new_vec):
        """Triggered when one family member changes belief significantly."""
        for member in self.members:
            if member == initiator:
                continue
            # ideological distance
            dist = np.linalg.norm(member.belief_vector() - initiator.belief_vector())
            # standard distance decay-based probability
            prob = np.exp(-0.07 * dist) 
            if np.random.random() < prob:
                # small movement toward the initiator's shift
                adj_vec = member.belief_vector() + self.family_multiplier * member.susceptibility * (new_vec - old_vec)
                member.update_from_vector(self.reflect(adj_vec))
                member.update_affiliation_and_support(old_party=member.party_affiliation)

    def react_to_death(self, deceased_member):
        """
        When a family member dies, surviving members react emotionally
        by moving away from the ruling party's ideology (blaming the government).
        """
        for member in self.members:
            if member.alive:

                # If the ruling party is in power
                ruling_party = self.model.majority_party

                # If the member supports the ruling party → move away from it
                if member.party_affiliation == ruling_party:
                    new_vec = member.move_closer_to_other_party_vector()
                    # Amplify the shift using the multiplier
                    amplified_vec = member.belief_vector() + self.family_multiplier * (new_vec - member.belief_vector())

                # If they already oppose the ruling party → reinforce their own stance
                else:
                    new_vec = member.move_closer_to_own_party_vector()
                    amplified_vec = member.belief_vector() + self.healthcare_multiplier * (new_vec - member.belief_vector())
                # Apply reflection and update
                member.update_from_vector(member.reflect(amplified_vec))
                member.update_affiliation_and_support(old_party=member.party_affiliation)

    @staticmethod
    def reflect(vec, lower=0, upper=100):
        """
        Reflects values in `vec` back into range [lower, upper].
        Works for scalars or NumPy arrays.
        """
        vec = np.asarray(vec, dtype=float)
        range_size = upper - lower
        reflected = np.empty_like(vec)

        for i, val in np.ndenumerate(vec):
            while val < lower or val > upper:
                if val > upper:
                    val = upper - (val - upper)
                elif val < lower:
                    val = lower + (lower - val)
            reflected[i] = val
        return reflected

class VoterAgent(CellAgent): 
    """ Voter agent with attributes and a party preference. """

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
        self.susceptibility = np.random.choice([self.low, self.high]) # np.random.uniform(0, 1) # low # 
        self.switched_this_step = False
        self.switched_in_support = False
        self.switched_in_rebellion = False
        self.switch_cause = None # This will be used to track why an agent switched parties
        self.has_interacted = False
        self.interacted_with = None
        self.interacted_within_party = False
        self.in_support = False
        self.wealth = 0
        self.wealth_dissatisfaction = 0 # track the economic dissatisfaction of the agent and can be totalled for the model
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
        self.susc_party_focus = "Undecided"
        self.susc_focus_value = None
        self.susc_other_value = None
        self.wealth_party_focus = "Undecided"
        self.wealth_focus_value = None
        self.wealth_other_value = None
        self.sickness_chance = 0.05
        self.family_members = None
        self.family_size = None
        self.interacted_within_family = False

        if(self.model.majority_party == self.party_affiliation):
            self.in_support = True

        # Set susceptibility based on model configuration
        if self.susc_focus_value == "low":
            if self.susc_other_value == "normal":
                if self.party_affiliation == self.susc_party_focus:
                    if np.random.random < 0.9:
                        self.susceptibility = self.low # this makes majority low
                    else:
                        self.susceptibility = self.high
                else:
                    self.susceptibility = np.random.choice([self.low, self.high])
            elif self.susc_other_value == "high":
                if self.party_affiliation == self.susc_party_focus:
                    if np.random.random < 0.9:
                        self.susceptibility = self.low  # this makes majority low
                    else:
                        self.susceptibility = self.high
                else:
                    if np.random.random < 0.9:          
                        self.susceptibility = self.high  # this makes majority of other parties high
                    else:
                        self.susceptibility = self.low
        elif self.susc_focus_value == "high":
            if self.susc_other_value == "normal":
                if self.party_affiliation == self.susc_party_focus:
                    if np.random.random < 0.9:
                        self.susceptibility = self.high # this makes majority high
                    else:
                        self.susceptibility = self.low
                else:
                    self.susceptibility = np.random.choice([self.low, self.high])
            elif self.susc_other_value == "low":
                if self.party_affiliation == self.susc_party_focus:
                    if np.random.random < 0.9:
                        self.susceptibility = self.high  # this makes majority high
                    else:
                        self.susceptibility = self.low
                else:
                    if np.random.random < 0.9:          
                        self.susceptibility = self.low  # this makes majority of other parties high
                    else:
                        self.susceptibility = self.high
        else :
            self.susceptibility = np.random.choice([self.low, self.high])

    # ---------------------------
    # Wealth Initialization Method
    # ---------------------------
    def initialize_wealth(self):
        """Assign initial wealth based on model configuration and party affiliation."""
        raw_wealth = np.random.beta(2, 5) * 100
        beta_cutoff = 26

        focus = self.wealth_focus_value
        other = self.wealth_other_value
        party = self.party_affiliation
        focus_party = self.wealth_party_focus

        if focus == "low" and other == "normal":
            if party == focus_party:
                self.wealth = raw_wealth if raw_wealth < beta_cutoff else np.random.uniform(0, beta_cutoff)
            else:
                self.wealth = raw_wealth
        elif focus == "high" and other == "normal":
            if party == focus_party:
                self.wealth = raw_wealth if raw_wealth > beta_cutoff else np.random.uniform(beta_cutoff, 100)
            else:
                self.wealth = raw_wealth
        elif focus == "low" and other == "high":
            if party == focus_party:
                self.wealth = raw_wealth if raw_wealth < beta_cutoff else np.random.uniform(0, beta_cutoff)
            else:
                self.wealth = raw_wealth if raw_wealth > beta_cutoff else np.random.uniform(beta_cutoff, 100)
        elif focus == "high" and other == "low":
            if party == focus_party:
                self.wealth = raw_wealth if raw_wealth > beta_cutoff else np.random.uniform(beta_cutoff, 100)
            else:
                self.wealth = raw_wealth if raw_wealth < beta_cutoff else np.random.uniform(0, beta_cutoff)
        else:
            self.wealth = raw_wealth


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
        return np.array([self.LawAndOrder, self.EconomicEquality, self.SocialWelfare])

    def update_from_vector(self, vec):
        self.LawAndOrder, self.EconomicEquality, self.SocialWelfare = vec

    def assign_party(self):
        """Check which party (if any) the agent belongs to."""
        for p in self.model.parties:
            if np.linalg.norm(self.belief_vector() - p.center_vector()) <= p.radius:
                return p.name, 0.0
        undecided_center = np.array([50, 50, 50])
        return "Undecided", np.linalg.norm(self.belief_vector() - undecided_center)
    
    def update_affiliation_and_support(self, old_party=None): # might have to add a "cause" variable
        """
        Assign party based on current beliefs and check if agent switched.
        Also updates whether the agent is in support or in rebellion
        relative to the current party in power.
        """
        # Assign party
        self.party_affiliation, undecided_distance = self.assign_party()

        # Update ideological distance
        if self.party_affiliation == "Undecided":
            self.distance = undecided_distance
        else:
            self.distance = self.party_distance()

        # Compare to old party (if provided)
        if old_party is not None:
            self.switched_this_step = (self.party_affiliation != old_party)

        # Check support vs rebellion against the government
        current_gov = self.model.majority_party 
        if self.party_affiliation == current_gov:
            self.in_support = True
            if self.switched_this_step:
                self.switched_in_support = True
        else:
            self.in_support = False
            if self.switched_this_step and old_party == current_gov:
                self.switched_in_rebellion = True

    
    def party_center(self):
        if self.party_affiliation == "Undecided":
            return self.belief_vector()
        for p in self.model.parties:
            if p.name == self.party_affiliation:
                return p.center_vector()

    def party_distance(self):
        return np.linalg.norm(self.party_center() - self.belief_vector())
    
    def distance_from_party(self, party):
        for p in self.model.parties:
            if p.name == party:
                return p.center_vector()
            
    def distance_from_nearest_party(self):
        # Find nearest party
        if self.party_affiliation != "Undecided":
            return  # Only applies to undecided agents

        nearest_party = None
        nearest_distance = float("inf")

        for p in self.model.parties:
            d = np.linalg.norm(self.belief_vector() - p.center_vector())
            if d < nearest_distance:
                nearest_distance = d
                nearest_party = p

        return nearest_distance, nearest_party

    def original_party_distance(self):
        original_party = self.original_party_affiliation
        party_center = self.distance_from_party(original_party)
        return np.linalg.norm(party_center - self.belief_vector())

    def move(self):
        """ Move to a random neighboring cell. """    
        self.cell = self.cell.neighborhood.select_random_cell()

    @staticmethod
    def reflect(vec, lower=0, upper=100):
        """
        Reflects values in `vec` back into range [lower, upper].
        Works for scalars or NumPy arrays.
        """
        vec = np.asarray(vec, dtype=float)
        range_size = upper - lower
        reflected = np.empty_like(vec)

        for i, val in np.ndenumerate(vec):
            while val < lower or val > upper:
                if val > upper:
                    val = upper - (val - upper)
                elif val < lower:
                    val = lower + (lower - val)
            reflected[i] = val
        return reflected
        

    # ---------------------------
    # Belief Interaction Rules
    # ---------------------------

    def mutual_persuasion(self, other):
        if self.interacted_within_family or other.interacted_within_family:
            new_self = self.belief_vector() + self.family.family_multiplier * self.susceptibility * (other.belief_vector() - self.belief_vector())
            new_other = other.belief_vector() + other.family.family_multiplier * other.susceptibility * (self.belief_vector() - other.belief_vector())
        else:
            new_self = self.belief_vector() + self.interaction_multiplier * self.susceptibility * (other.belief_vector() - self.belief_vector())
            new_other = other.belief_vector() + other.interaction_multiplier * other.susceptibility * (self.belief_vector() - other.belief_vector())
        return new_self, new_other

    def other_convinces_self(self, other):
        reinforce = (other.party_center() - other.belief_vector())

        if self.interacted_within_family or other.interacted_within_family:
            new_self = self.belief_vector() + self.family.family_multiplier * self.susceptibility * (other.belief_vector() - self.belief_vector())
            new_other = other.belief_vector() + other.family.family_multiplier * other.susceptibility * reinforce
        else:
            new_self = self.belief_vector() + self.interaction_multiplier * self.susceptibility * (other.belief_vector() - self.belief_vector())
            new_other = other.belief_vector() + other.interaction_multiplier * other.susceptibility * reinforce

        return new_self, new_other

    def self_convinces_other(self, other):
        reinforce = (self.party_center() - self.belief_vector())
        
        if self.interacted_within_family or other.interacted_within_family:
            new_other = other.belief_vector() + other.family.family_multiplier * other.susceptibility * (self.belief_vector() - other.belief_vector())
            new_self = self.belief_vector() + self.family.family_multiplier * self.susceptibility * reinforce
        else:
            new_other = other.belief_vector() + other.interaction_multiplier * other.susceptibility * (self.belief_vector() - other.belief_vector())
            new_self = self.belief_vector() + self.interaction_multiplier * self.susceptibility * reinforce

        return new_self, new_other

    def disagreement(self, other):  # go closer to own party center
        if self.interacted_within_family or other.interacted_within_family:
            new_self = self.belief_vector() + self.family.family_multiplier * self.susceptibility * (self.party_center() - self.belief_vector())
            new_other = other.belief_vector() + other.family.family_multiplier * other.susceptibility * (other.party_center() - other.belief_vector())
        else:
            new_self = self.belief_vector() + self.interaction_multiplier * self.susceptibility * (self.party_center() - self.belief_vector())
            new_other = other.belief_vector() + other.interaction_multiplier * other.susceptibility * (other.party_center() - other.belief_vector())
        
        return new_self, new_other

    # overall susceptibilities might have to go lower for stubborn and higher for naive
    #def choose_rule(self, other):
    #    """Decide which interaction rule to apply based on susceptibility."""
    #    if self.susceptibility > 0.3 and other.susceptibility > 0.3:
    #        return "mutual"
    #    elif self.susceptibility > 0.7 and other.susceptibility < 0.3:
    #        return "otherconvince"
    #    elif self.susceptibility < 0.3 and other.susceptibility > 0.7:
    #        return "selfconvince"
    #    else:
    #        return "disagree"
    
    def choose_rule(self, other):
        # """Decide which interaction rule to apply based on susceptibility."""
        # if self.susceptibility > 0.55 and other.susceptibility < 0.45:
        #    return "otherconvince"
        # elif self.susceptibility < 0.45 and other.susceptibility > 0.55:
        #    return "selfconvince"
        # elif self.susceptibility > 0.05 and other.susceptibility > 0.05: # might have to be last
        #    return "mutual"
        # else:
        #    return "disagree"

        self_susc = self.susceptibility 
        other_susc = other.susceptibility       
       # --- Both Low ---
        if self_susc <= 0.1 and other_susc <= 0.1:
            if self_susc <= 0.05 and other_susc <= 0.05:
                return "disagree"
            elif self_susc > other_susc and (self_susc <= 0.1 or other_susc <= 0.1):
                return "otherconvince"
            elif self_susc < other_susc and (self_susc <= 0.1 or other_susc <= 0.1):
                return "selfconvince"
            else:
                return "mutual"

        # --- One low, one higher ---
        elif self_susc <= 0.1 and other_susc >= 0.3:
            return "selfconvince"
        elif self_susc >= 0.3 and other_susc <= 0.1:
            return "otherconvince"

        # --- Both Midrange (0.3–0.8) ---
        elif (0.3 <= self_susc <= 0.8) and (0.3 <= other_susc <= 0.8):
            if self_susc <= 0.55 and other_susc <= 0.55:
                return "disagree"
            elif self_susc > other_susc and (self_susc <= 0.8 or other_susc <= 0.8):
                return "otherconvince"
            elif self_susc < other_susc and (self_susc <= 0.8 or other_susc <= 0.8):
                return "selfconvince"
            else:
                return "mutual"

        # --- Mid vs High ---
        elif (0.3 <= self_susc <= 0.8) and other_susc >= 1.6:
            return "selfconvince"
        elif self_susc >= 1.6 and (0.3 <= other_susc <= 0.8):
            return "otherconvince"

        # --- Both High (>1.6) ---
        elif self_susc >= 1.6 and other_susc >= 1.6:
            if self_susc <= 1.8 and other_susc <= 1.8:
                return "disagree"
            elif self_susc > other_susc and (self_susc <= 2.0 or other_susc <= 2.0):
                return "otherconvince"
            elif self_susc < other_susc and (self_susc <= 2.0 or other_susc <= 2.0):
                return "selfconvince"
            else:
                return "mutual"

        # --- Fallback ---
        else:
            return "disagree"

    # dictionary to define interaction rules
    interaction_rules = {
    "mutual": mutual_persuasion,
    "otherconvince": other_convinces_self,
    "selfconvince": self_convinces_other,
    "disagree": disagreement,
    }

    # def media_influence(self, media_bias):
    #    """ Influence from media. """
    #    self.political_bias = self.political_bias + self.susceptibility * media_bias

    def policy_influence(self, other):
        
        ## making sure that when families interact, a ripple ##
        if hasattr(self, "family") and hasattr(other, "family"):
            if self.family == other.family:
                # Skip ripple triggers if family interaction
                self.interacted_within_family = True
            else:
                self.interacted_within_family = False
        else:
            self.interacted_within_family = False


        rule = self.choose_rule(other)
        new_self, new_other = self.interaction_rules[rule](self, other) # what's happening here?
        old_self = self.belief_vector().copy()
        old_other = other.belief_vector().copy()

        # Update beliefs (clamp between 0–100)
        #self.update_from_vector(np.clip(new_self, 0, 100))
        #other.update_from_vector(np.clip(new_other, 0, 100))
        self.update_from_vector(self.reflect(new_self))
        other.update_from_vector(self.reflect(new_other))


        # Update party affiliation
        self.update_affiliation_and_support(old_party=self.party_affiliation)
        other.update_affiliation_and_support(old_party=other.party_affiliation)

        # --- Family ripple trigger ---
        if not self.interacted_within_family:
            if hasattr(self, "family") and self.family:
                self.family.ripple_influence(self, old_self, new_self)
            if hasattr(other, "family") and other.family:
                other.family.ripple_influence(other, old_other, new_other)

    
    # ---------------------------
    # Environment Interaction Rules
    # ---------------------------

    def other_party_distance(self, vec):
        return np.linalg.norm(vec - self.belief_vector())
    
    def move_closer_to_other_party_vector(self):
        shortest_distance = 200
        for p in self.model.parties:
            party_distance = self.other_party_distance(p.center_vector())
            if p.name != self.party_affiliation and party_distance < shortest_distance:
                shortest_distance = np.linalg.norm(p.center_vector() - self.belief_vector())
                center_vector = p.center_vector()

        new_self = self.belief_vector() + self.susceptibility * (center_vector - self.belief_vector())
        return new_self
    
    def move_closer_to_own_party_vector(self):
        new_self = self.belief_vector() + self.susceptibility * (self.party_center() - self.belief_vector())
        return new_self
    
    def move_closer_to_majority_party_vector(self):
        for p in self.model.parties:
            if p.name == self.model.majority_party:
                center_vector = p.center_vector()
                new_self = self.belief_vector() + self.susceptibility * (center_vector - self.belief_vector())
                return new_self
        return self.belief_vector()  # if no majority party, return current beliefs


    # ---------------------------
    # Perception control
    # ---------------------------

    def adjust_economic_view(self):
        """Adjust ideological position when dissatisfaction thresholds are reached."""
        if self.wealth_dissatisfaction > self.dissatisfaction_threshold:
            # Too dissatisfied → shift away from own or majority party
            if self.model.majority_party == self.party_affiliation:
                new_self = self.move_closer_to_other_party_vector()
                wealth_effect_new = self.belief_vector() + self.wealth_influence_factor * (new_self - self.belief_vector())
            else:
                new_self = self.move_closer_to_own_party_vector()
                wealth_effect_new = self.belief_vector() + self.wealth_influence_factor * (new_self - self.belief_vector())
            self.update_from_vector(self.reflect(wealth_effect_new))
            self.update_affiliation_and_support(old_party=self.party_affiliation)
            self.wealth_dissatisfaction = 0

        elif self.wealth_dissatisfaction < self.satisfaction_threshold:
            # Very satisfied → go toward majority party since they're controlling the economy
            if self.model.majority_party == self.party_affiliation:
                new_self = self.move_closer_to_own_party_vector()
                wealth_effect_new = self.belief_vector() + self.wealth_influence_factor * (new_self - self.belief_vector())
            else:
                new_self = self.move_closer_to_majority_party_vector()
                wealth_effect_new = self.belief_vector() + self.wealth_influence_factor * (new_self - self.belief_vector())
            self.update_from_vector(self.reflect(wealth_effect_new))
            self.update_affiliation_and_support(old_party=self.party_affiliation)
            self.wealth_dissatisfaction = 0

    def compare_wealth(self, other):
        wealth_gap = other.wealth - self.wealth

        # Emotional response (accumulate dissatisfaction)
        if wealth_gap > self.significant_difference:
            self.wealth_dissatisfaction += 1
        elif wealth_gap < -self.significant_difference:
            self.wealth_dissatisfaction -= 1

        # Trigger ideological adjustment if thresholds reached
        self.adjust_economic_view()


    def perceive_health_status(self):
        """
        Models random sickness events during simulation.
        Agents can fall sick based on a small probability each step.
        """
        if self.alive and self.healthy:
            # % chance of getting sick per step (tune this)
            if self.random.random() < self.sickness_chance:
                self.healthy = False

    def perceive_healthcare(self): 
        if not self.healthy and self.alive:  # Only act if sick and alive
            if self.wealth_dissatisfaction <= 0:
                self.health_care = "Private"
                self.healthy = True  # More likely to be healthy
            else:
                self.health_care = "Public"
                if self.random.random() < 0.6:
                    self.healthy = False  # More likely to be sick
                    self.alive = False  # Agent dies if sick and using public healthcare
                    if hasattr(self, "family") and self.family:
                        self.family.react_to_death(self)
                else:
                    self.healthy = True


    def give_wealth(self, other):
        # compare wealth levels
        self.compare_wealth(other)
        other.compare_wealth(self)
                    
        # give some wealth to other agent 
        if self.wealth > 1:
            other.wealth += 1 # might increase others satisfaction because they are getting wealth
            # other.wealth_dissatisfaction -= 1
            self.wealth -= 1        

    # ---------------------------
    # Step control
    # ---------------------------
    
    def interact(self):
        if self.has_interacted:
            return

        others = [a for a in self.cell.agents if a != self and not a.has_interacted]
        if others:
            other = self.random.choice(others)

            self.evaluate_susceptibility
            other.evaluate_susceptibility

            self.policy_influence(other)
            self.give_wealth(other)

            # mark both as having interacted
            self.has_interacted = True
            self.interacted_with = int(other.unique_id)
            if self.party_affiliation == other.party_affiliation:
                self.interacted_within_party = True
                other.interacted_within_party = True
            other.has_interacted = True
            other.interacted_with = int(self.unique_id)

    def force_vote(self):
        """Final decision for undecided agents based on susceptibility and history."""

        # Only undecided agents should act
        if self.party_affiliation != "Undecided":
            return

        original = self.original_party_affiliation

        # --------------------------
        # Case 1: Always been undecided
        # --------------------------
        if original == "Undecided":
            # 60% chance that this logic happens at all
            if np.random.random() < 0.6:
                if self.susceptibility == self.high:
                    # Naive agents: 80% chance to pick nearest party
                    if np.random.random() < 0.8:
                        nearest_distance, nearest_party = self.distance_from_nearest_party()
                        if nearest_party is not None:
                            self.party_affiliation = nearest_party.name
                        else:
                            self.party_affiliation = "Undecided"
                    else:
                        self.party_affiliation = "Undecided"
                elif self.susceptibility == self.low:
                    # Stubborn agents: 80% chance to remain undecided
                    if np.random.random() < 0.8:
                        self.party_affiliation = "Undecided"
                    else:
                        nearest_distance, nearest_party = self.distance_from_nearest_party()
                        if nearest_party is not None:
                            self.party_affiliation = nearest_party.name
            else:
                # 40% of the time do nothing (remain undecided)
                self.party_affiliation = "Undecided"

            return

        # --------------------------
        # Case 2: Had an original party
        # --------------------------
        if np.random.random() < 0.8:
            if self.susceptibility == self.high:
                # Naive agents: 80% chance to pick nearest party
                if np.random.random() < 0.8:
                    nearest_distance, nearest_party = self.distance_from_nearest_party()
                    if nearest_party is not None:
                        self.party_affiliation = nearest_party.name
                    else:
                        self.party_affiliation = "Undecided"
                else:
                    self.party_affiliation = "Undecided"
            elif self.susceptibility == self.low:
                # Stubborn agents: 80% chance to go back to their original party
                if np.random.random() < 0.8:
                    self.party_affiliation = original
                else:
                    self.party_affiliation = "Undecided"
        else:
            # 20% of the time do nothing
            self.party_affiliation = "Undecided"



    def reset(self):
        """Reset interaction flag for this agent."""
        self.has_interacted = False
        self.switched_this_step = False
        self.switched_in_support = False
        self.switched_in_rebellion = False
        self.interacted_with = None
        

    def __repr__(self):
        return (f"Law&Order: {self.LawAndOrder:.1f}, "
                f"EconEquality: {self.EconomicEquality:.1f}, "
                f"SocWelfare: {self.SocialWelfare:.1f}, "
                f"Party: {self.party_affiliation}")
    
