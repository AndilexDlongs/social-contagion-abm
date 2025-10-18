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
    def __init__(self, model, members):
        self.model = model
        self.members = members  # list of VoterAgents
        for member in members:
            member.family = self

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
                adj_vec = member.belief_vector() + member.susceptibility * (new_vec - old_vec)
                member.update_from_vector(np.clip(adj_vec, 0, 100))
                member.update_affiliation_and_support(old_party=member.party_affiliation)

    def react_to_death(self, deceased_member):
        deceased_party = deceased_member.party_affiliation

        for member in self.members:
            if member.alive:
                # Emotional/economic reaction
                member.wealth_dissatisfaction += 2
                member.in_support = False
                member.switched_in_rebellion = True

                # Find the party center of the deceased
                deceased_party_center = None
                for p in self.model.parties:
                    if p.name == deceased_party:
                        deceased_party_center = p.center_vector()
                        break

                # If found, move away from that party center
                if deceased_party_center is not None:
                    # Vector pointing *away* from the deceased's party ideology
                    direction_away = member.belief_vector() - deceased_party_center

                    # Normalize direction (unit vector)
                    if np.linalg.norm(direction_away) > 0:
                        direction_away = direction_away / np.linalg.norm(direction_away)

                    # Move slightly in that direction (e.g., 5 units away)
                    new_vec = member.belief_vector() + direction_away * 5 * member.susceptibility
                    member.update_from_vector(np.clip(new_vec, 0, 100))
                    member.update_affiliation_and_support(old_party=member.party_affiliation)


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
        self.wealth = None
        self.wealth_dissatisfaction = 0 # track the economic dissatisfaction of the agent and can be totalled for the model
        self.dissatisfaction_threshold = 4  # threshold for economic dissatisfaction
        self.satisfaction_threshold = -8  # threshold for economic satisfaction
        self.dissatisfaction_multiplier = 5  # multiplier for dissatisfaction increase
        self.significant_difference = 50  # threshold for significant distance
        self.education = "Primary"  # Example additional attribute
        self.health_care = "Public"  # Example additional attribute
        self.healthy = True
        self.alive = True
        self.cell = cell

        if(self.model.majority_party == self.party_affiliation):
            self.in_support = True

    # ---------------------------
    # Helper methods
    # ---------------------------
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
        

    # ---------------------------
    # Belief Interaction Rules
    # ---------------------------

    def mutual_persuasion(self, other):
        new_self = self.belief_vector() + self.susceptibility * (other.belief_vector() - self.belief_vector())
        new_other = other.belief_vector() + other.susceptibility * (self.belief_vector() - other.belief_vector())
        return new_self, new_other

    def other_convinces_self(self, other):
        #old_susc_other = other.susceptibility
        #if old_susc_other > 1.5:
        #    other.susceptibility = other.mid

        new_self = self.belief_vector() + self.susceptibility * (other.belief_vector() - self.belief_vector())
        reinforce = (other.party_center() - other.belief_vector())
        new_other = other.belief_vector() + other.susceptibility * reinforce

        #other.susceptibility = old_susc_other
        return new_self, new_other

    def self_convinces_other(self, other):
        #old_susc_self = self.susceptibility
        #if old_susc_self > 1.5:
        #    self.susceptibility = self.mid

        new_other = other.belief_vector() + other.susceptibility * (self.belief_vector() - other.belief_vector())
        reinforce = (self.party_center() - self.belief_vector())
        new_self = self.belief_vector() + self.susceptibility * reinforce

        #self.susceptibility = old_susc_self
        return new_self, new_other

    def disagreement(self, other):  # go closer to own party center
        new_self = self.belief_vector() + self.susceptibility * (self.party_center() - self.belief_vector())
        new_other = other.belief_vector() + other.susceptibility * (other.party_center() - other.belief_vector())
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
        def reflect(vec, lower=0, upper=100):
            """Reflects out-of-bound values back into range [lower, upper]."""
            range_size = upper - lower
            reflected = np.copy(vec)
            for i, val in enumerate(reflected):
                while val < lower or val > upper:
                    if val > upper:
                        val = upper - (val - upper)  # reflect downward
                    elif val < lower:
                        val = lower + (lower - val)  # reflect upward
                reflected[i] = val
            return reflected
        
        ## making sure that when families interact, a ripple ##
        if hasattr(self, "family") and hasattr(other, "family"):
            if self.family == other.family:
                # Skip ripple triggers if family interaction
                family_interaction = True
            else:
                family_interaction = False
        else:
            family_interaction = False


        rule = self.choose_rule(other)
        new_self, new_other = self.interaction_rules[rule](self, other) # what's happening here?
        old_self = self.belief_vector().copy()
        old_other = other.belief_vector().copy()

        # Update beliefs (clamp between 0–100)
        #self.update_from_vector(np.clip(new_self, 0, 100))
        #other.update_from_vector(np.clip(new_other, 0, 100))
        self.update_from_vector(reflect(new_self))
        other.update_from_vector(reflect(new_other))


        # Update party affiliation
        self.update_affiliation_and_support(old_party=self.party_affiliation)
        other.update_affiliation_and_support(old_party=other.party_affiliation)

        # --- Family ripple trigger ---
        if not family_interaction:
            if hasattr(self, "family") and self.family:
                self.family.ripple_influence(self, old_self, new_self)
            if hasattr(other, "family") and other.family:
                other.family.ripple_influence(other, old_other, new_other)

    
    # ---------------------------
    # Environment Interaction Rules
    # ---------------------------
    # def maybe_join_nearest_party(self, distance_threshold=20, join_prob=0.8):
    #     """Undecided agent may join a nearby party if close enough."""
    #     if self.party_affiliation != "Undecided":
    #         return  # Only applies to undecided agents

    #     nearest_party = None
    #     nearest_distance = float("inf")

    #     # Find nearest party
    #     for p in self.model.parties:
    #         d = np.linalg.norm(self.belief_vector() - p.center_vector())
    #         if d < nearest_distance:
    #             nearest_distance = d
    #             nearest_party = p

    #     # Only consider joining if within threshold
    #     if nearest_distance <= distance_threshold:
    #         if np.random.random() < join_prob:
    #             # Move slightly toward that party center
    #             new_self = self.belief_vector()  + self.susceptibility * (
    #                nearest_party.center_vector() - self.belief_vector()
    #             )
    #             self.update_from_vector(self.reflect(new_self))  # use reflection
    #             self.party_affiliation = nearest_party.name
    #             self.distance = self.party_distance()


    def other_party_distance(self, vec):
        return np.linalg.norm(vec - self.belief_vector())
    
    def move_closer_to_other_party_vector(self):
        shortest_distance = 500
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
            else:
                new_self = self.move_closer_to_own_party_vector()
            self.update_from_vector(np.clip(new_self, 0, 100))
            self.update_affiliation_and_support(old_party=self.party_affiliation)
            self.wealth_dissatisfaction = 0

        elif self.wealth_dissatisfaction < self.satisfaction_threshold:
            # Very satisfied → reinforce beliefs
            new_self = self.move_closer_to_own_party_vector()
            self.update_from_vector(np.clip(new_self, 0, 100))
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
            if self.random.random() < 0.05:
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

        if self.party_affiliation == "Undecided":
            self.force_vote()

    def force_vote(self, turnout_prob=0.6, loyalty_radius=35):
        """Encourage undecided agents to make a final choice based on history and proximity."""
        
        if self.party_affiliation != "Undecided":
            return  # Only applies to undecided agents

        # --------------------------
        # Case 1: No prior affiliation
        # --------------------------
        original = self.original_party_affiliation
        if original == "Undecided":
            nearest_distance, nearest_party = self.distance_from_nearest_party()

            # Only act if they decide to vote
            if np.random.random() < turnout_prob and nearest_party and nearest_distance < loyalty_radius:
                self.party_affiliation = nearest_party.name

            # Move slightly toward nearest party (if exists)
            if nearest_party is not None:
                new_vec = self.belief_vector() + self.susceptibility * (
                    nearest_party.center_vector() - self.belief_vector()
                )
                self.update_from_vector(np.clip(new_vec, 0, 100))
                self.update_affiliation_and_support(old_party=self.party_affiliation)
            else:
                # fallback: just reinforce current beliefs
                self.update_from_vector(np.clip(self.belief_vector(), 0, 100))
                self.update_affiliation_and_support(old_party=self.party_affiliation)

            return

        # --------------------------
        # Case 2: Had an original party
        # --------------------------
        if np.random.random() < turnout_prob:
            dist_from_orig = self.original_party_distance()
            nearest_distance, nearest_party = self.distance_from_nearest_party()

            # More loyal voters rejoin their old party
            if dist_from_orig < loyalty_radius or np.random.random() < 0.7:
                self.party_affiliation = original
            elif nearest_party is not None:
                # Otherwise join the nearest viable party
                self.party_affiliation = nearest_party.name

        # Update internal distance
        self.distance = self.party_distance()



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
