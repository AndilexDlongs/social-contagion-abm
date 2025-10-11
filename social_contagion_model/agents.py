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

class VoterAgent(CellAgent): 
    """ Voter agent with attributes and a party preference. """

    def __init__(self, model, cell):
        super().__init__(model)

        # Attributes (named for clarity)
        self.LawAndOrder = np.random.uniform(0, 100)
        self.EconomicEquality = np.random.uniform(0, 100)
        self.SocialWelfare = np.random.uniform(0, 100)
        self.party_affiliation = "Undecided"
        self.distance = self.party_distance()
        self.susceptibility = np.random.uniform(0, 1)
        self.switched_this_step = False
        self.switched_in_support = False
        self.switched_in_rebellion = False
        self.switch_cause = None # This will be used to track why an agent switched parties
        self.has_interacted = False
        self.interacted_with = None
        self.in_support = False
        self.wealth = 1
        self.wealth_dissatisfaction = 0 # track the economic dissatisfaction of the agent and can be totalled for the model
        self.dissatisfation_threshold = 4  # threshold for economic dissatisfaction
        self.satisfaction_threshold = -8  # threshold for economic satisfaction
        self.dissatisfaction_multiplier = 5  # multiplier for dissatisfaction increase
        self.significant_difference = 50  # threshold for significant distance
        self.education = "Primary"  # Example additional attribute
        self.health_care = "Public"  # Example additional attribute
        self.cell = cell

        if(self.model.majority_party == self.party_affiliation):
            self.in_support = True

    # ---------------------------
    # Helper methods
    # ---------------------------

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
        new_self = self.belief_vector() + self.susceptibility * (other.belief_vector() - self.belief_vector())
        reinforce = (other.party_center() - other.belief_vector())
        new_other = other.belief_vector() + other.susceptibility * reinforce
        return new_self, new_other

    def self_convinces_other(self, other):
        new_other = other.belief_vector() + other.susceptibility * (self.belief_vector() - other.belief_vector())
        reinforce = (self.party_center() - self.belief_vector())
        new_self = self.belief_vector() + self.susceptibility * reinforce
        return new_self, new_other

    def disagreement(self, other):  # go closer to own party center
        new_self = self.belief_vector() + self.susceptibility * (self.party_center() - self.belief_vector())
        new_other = other.belief_vector() + other.susceptibility * (other.party_center() - other.belief_vector())
        return new_self, new_other

    def choose_rule(self, other):
        """Decide which interaction rule to apply based on susceptibility."""
        if self.susceptibility > 0.3 and other.susceptibility > 0.3:
            return "mutual"
        elif self.susceptibility > 0.7 and other.susceptibility < 0.3:
            return "otherconvince"
        elif self.susceptibility < 0.3 and other.susceptibility > 0.7:
            return "selfconvince"
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
        rule = self.choose_rule(other)
        new_self, new_other = self.interaction_rules[rule](self, other) # what's happening here?

        # Update beliefs (clamp between 0–100)
        self.update_from_vector(np.clip(new_self, 0, 100))
        other.update_from_vector(np.clip(new_other, 0, 100))

        # Update party affiliation
        self.update_affiliation_and_support(old_party=self.party_affiliation)
        other.update_affiliation_and_support(old_party=other.party_affiliation)

    
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
        if self.wealth_dissatisfaction > self.dissatisfation_threshold:
            if self.model.majority_party == self.party_affiliation:
                new_self = self.move_closer_to_other_party_vector()
                self.update_from_vector(np.clip(new_self, 0, 100))
                self.update_affiliation_and_support(old_party=self.party_affiliation)
            else: 
                new_self = self.move_closer_to_own_party_vector()
                self.update_from_vector(np.clip(new_self, 0, 100))
                self.update_affiliation_and_support(old_party=self.party_affiliation)

            self.wealth_dissatisfaction = 0  # reset dissatisfaction after adjustment
        elif self.wealth_dissatisfaction < self.satisfaction_threshold:
            if self.model.majority_party == self.party_affiliation:
                new_self = self.move_closer_to_own_party_vector()
                self.update_from_vector(np.clip(new_self, 0, 100))
                self.update_affiliation_and_support(old_party=self.party_affiliation)
            else: 
                new_self = self.move_closer_to_majority_party_vector()
                self.update_from_vector(np.clip(new_self, 0, 100))
                self.update_affiliation_and_support(old_party=self.party_affiliation)
                
            self.wealth_dissatisfaction = 0  # reset dissatisfaction after adjustment

    def compare_wealth(self,other): # later gonna have to make changes based off of disatisfaction level
        if (self.wealth + self.significant_difference < other.wealth) and (self.model.majority_party == self.party_affiliation): # move closer to other party
            new_self = self.move_closer_to_other_party_vector()
            self.update_from_vector(np.clip(new_self, 0, 100))
            self.update_affiliation_and_support(old_party=self.party_affiliation)
        elif(self.wealth + self.significant_difference > other.wealth) and (self.model.majority_party == self.party_affiliation): # move closer to own party
            new_self = self.move_closer_to_own_party_vector()
            self.update_from_vector(np.clip(new_self, 0, 100))
            self.update_affiliation_and_support(old_party=self.party_affiliation)
        elif(self.wealth + self.significant_difference > other.wealth) and (self.model.majority_party != self.party_affiliation): # move closer to majority party
            new_self = self.move_closer_to_majority_party_vector()
            self.update_from_vector(np.clip(new_self, 0, 100))
            self.update_affiliation_and_support(old_party=self.party_affiliation)
        
    def perceive_economy(self):
        average_wealth = self.model.state["average_wealth"]

        if average_wealth > self.wealth:
            self.wealth_dissatisfaction += 1
            self.adjust_economic_view()
        else:
            self.wealth_dissatisfaction -= 1
            self.adjust_economic_view()

    def perceive_healthcare(self):
        pass  # Placeholder for future implementation


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
            self.policy_influence(other)
            self.give_wealth(other)

            # mark both as having interacted
            self.has_interacted = True
            self.interacted_with = int(other.unique_id)
            other.has_interacted = True
            other.interacted_with = int(self.unique_id)

    def perceive_environment(self):
        self.perceive_economy()

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
