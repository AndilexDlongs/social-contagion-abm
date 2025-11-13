import numpy as np

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
            old_party = member.party_affiliation
            old_vec = member.belief_vector()
            # ideological distance
            dist = np.linalg.norm(member.belief_vector() - initiator.belief_vector())
            # standard distance decay-based probability
            prob = np.exp(-0.07 * dist) 
            if np.random.random() < prob:
                # small movement toward the initiator's shift
                adj_vec = member.belief_vector() + self.family_multiplier * member.susceptibility * (new_vec - old_vec)
                member.update_from_vector(self.reflect(adj_vec))
                member.update_affiliation_and_support(old_party=member.party_affiliation)
            
            new_party = member.party_affiliation
            new_vec = member.belief_vector()

            member.movement_tracker["family_ripple"] = int(np.linalg.norm(new_vec - old_vec))

            if old_party != new_party:
                member.switched_this_step = True
                member.switch_cause.append("family_ripple")

    def react_to_death(self, deceased_member):
        """
        When a family member dies, surviving members react emotionally
        by moving away from the ruling party's ideology (blaming the government).
        """
        for member in self.members:
            if member.alive:
                old_party = member.party_affiliation
                old_vec = member.belief_vector()
                # If the ruling party is in power
                ruling_party = self.model.majority_party

                # If the member supports the ruling party → move away from it
                if member.party_affiliation == ruling_party:
                    new_vec = member.move_closer_to_other_party_vector()
                    # Amplify the shift using the multiplier
                    amplified_vec = member.belief_vector() + self.healthcare_multiplier * (new_vec - member.belief_vector())

                # If they already oppose the ruling party → reinforce their own stance
                else:
                    new_vec = member.move_closer_to_own_party_vector()
                    amplified_vec = member.belief_vector() + self.healthcare_multiplier * (new_vec - member.belief_vector())
                # Apply reflection and update
                member.update_from_vector(member.reflect(amplified_vec))
                member.update_affiliation_and_support(old_party=member.party_affiliation)

                new_party = member.party_affiliation
                new_vec = member.belief_vector()

                member.movement_tracker["death reaction"] = int(np.linalg.norm(new_vec - old_vec))

                if old_party != new_party:
                    member.switched_this_step = True
                    member.switch_cause.append("death_shock")

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