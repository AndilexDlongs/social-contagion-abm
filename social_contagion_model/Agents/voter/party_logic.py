import numpy as np


class PartyLogic:
    def assign_party(self):
        """Check which party (if any) the agent belongs to."""
        for p in self.model.parties:
            if np.linalg.norm(self.belief_vector() - p.center_vector()) <= p.radius:
                return p.name, 0.0
        undecided_center = np.array([50, 50, 50])
        return "Undecided", np.linalg.norm(
            self.belief_vector() - undecided_center
        )

    def update_affiliation_and_support(self, old_party=None):
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
            self.switched_this_step = self.party_affiliation != old_party

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

    @staticmethod
    def distance_from_opinions(vec_2, vec_1):
        return np.linalg.norm(vec_2 - vec_1)

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
