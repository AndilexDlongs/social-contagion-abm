import numpy as np


class VotingLogic:
    def force_vote(self):
        """Final decision for undecided agents based on susceptibility and history."""

        # Only undecided agents should act
        if self.party_affiliation != "Undecided":
            return

        original = self.original_party_affiliation

        # Case 1: Always been undecided
        if original == "Undecided":
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

        # Case 2: Had an original party
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
