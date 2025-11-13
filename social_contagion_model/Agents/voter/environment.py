import numpy as np


class EnvironmentLogic:
    def move(self):
        """Move to a random neighboring cell."""
        self.cell = self.cell.neighborhood.select_random_cell()

    def other_party_distance(self, vec):
        return np.linalg.norm(vec - self.belief_vector())

    def move_closer_to_other_party_vector(self):
        shortest_distance = 200
        center_vector = self.belief_vector()

        for p in self.model.parties:
            party_distance = self.other_party_distance(p.center_vector())
            if p.name != self.party_affiliation and party_distance < shortest_distance:
                shortest_distance = np.linalg.norm(
                    p.center_vector() - self.belief_vector()
                )
                center_vector = p.center_vector()

        new_self = self.belief_vector() + self.susceptibility * (
            center_vector - self.belief_vector()
        )
        return new_self

    def move_closer_to_own_party_vector(self):
        new_self = self.belief_vector() + self.susceptibility * (
            self.party_center() - self.belief_vector()
        )
        return new_self

    def move_closer_to_majority_party_vector(self):
        for p in self.model.parties:
            if p.name == self.model.majority_party:
                center_vector = p.center_vector()
                new_self = self.belief_vector() + self.susceptibility * (
                    center_vector - self.belief_vector()
                )
                return new_self
        # if no majority party, return current beliefs
        return self.belief_vector()
