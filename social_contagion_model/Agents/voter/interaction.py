import numpy as np


class InteractionLogic:
    # ---------------------------
    # Belief Interaction Rules
    # ---------------------------

    def mutual_persuasion(self, other):
        if self.interacted_within_family or other.interacted_within_family:
            self_factor = (
                self.family.family_multiplier
                if hasattr(self, "family") and self.family
                else self.model.interaction_multiplier
            )
            other_factor = (
                other.family.family_multiplier
                if hasattr(other, "family") and other.family
                else other.model.interaction_multiplier
            )
            new_self = self.belief_vector() + self_factor * self.susceptibility * (
                other.belief_vector() - self.belief_vector()
            )
            new_other = other.belief_vector() + other_factor * other.susceptibility * (
                self.belief_vector() - other.belief_vector()
            )
        else:
            new_self = self.belief_vector() + self.interaction_multiplier * self.susceptibility * (
                other.belief_vector() - self.belief_vector()
            )
            new_other = other.belief_vector() + other.interaction_multiplier * other.susceptibility * (
                self.belief_vector() - other.belief_vector()
            )
        return new_self, new_other

    def other_convinces_self(self, other):
        reinforce = other.party_center() - other.belief_vector()

        if self.interacted_within_family or other.interacted_within_family:
            self_factor = (
                self.family.family_multiplier
                if hasattr(self, "family") and self.family
                else self.model.interaction_multiplier
            )
            other_factor = (
                other.family.family_multiplier
                if hasattr(other, "family") and other.family
                else other.model.interaction_multiplier
            )
            new_self = self.belief_vector() + self_factor * self.susceptibility * (
                other.belief_vector() - self.belief_vector()
            )
            new_other = other.belief_vector() + other_factor * other.susceptibility * reinforce
        else:
            new_self = self.belief_vector() + self.interaction_multiplier * self.susceptibility * (
                other.belief_vector() - self.belief_vector()
            )
            new_other = other.belief_vector() + other.interaction_multiplier * other.susceptibility * reinforce

        return new_self, new_other

    def self_convinces_other(self, other):
        reinforce = self.party_center() - self.belief_vector()

        if self.interacted_within_family or other.interacted_within_family:
            self_factor = (
                self.family.family_multiplier
                if hasattr(self, "family") and self.family
                else self.model.interaction_multiplier
            )
            other_factor = (
                other.family.family_multiplier
                if hasattr(other, "family") and other.family
                else other.model.interaction_multiplier
            )
            new_other = other.belief_vector() + other_factor * other.susceptibility * (
                self.belief_vector() - other.belief_vector()
            )
            new_self = self.belief_vector() + self_factor * self.susceptibility * reinforce
        else:
            new_other = other.belief_vector() + other.interaction_multiplier * other.susceptibility * (
                self.belief_vector() - other.belief_vector()
            )
            new_self = self.belief_vector() + self.interaction_multiplier * self.susceptibility * reinforce

        return new_self, new_other

    def disagreement(self, other):
        """Both move closer to own party center."""
        if self.interacted_within_family or other.interacted_within_family:
            self_factor = (
                self.family.family_multiplier
                if hasattr(self, "family") and self.family
                else self.model.interaction_multiplier
            )
            other_factor = (
                other.family.family_multiplier
                if hasattr(other, "family") and other.family
                else other.model.interaction_multiplier
            )
            new_self = self.belief_vector() + self_factor * self.susceptibility * (
                self.party_center() - self.belief_vector()
            )
            new_other = other.belief_vector() + other_factor * other.susceptibility * (
                other.party_center() - other.belief_vector()
            )
        else:
            new_self = self.belief_vector() + self.interaction_multiplier * self.susceptibility * (
                self.party_center() - self.belief_vector()
            )
            new_other = other.belief_vector() + other.interaction_multiplier * other.susceptibility * (
                other.party_center() - other.belief_vector()
            )

        return new_self, new_other

    def choose_rule(self, other):
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
        elif 0.3 <= self_susc <= 0.8 and 0.3 <= other_susc <= 0.8:
            if self_susc <= 0.55 and other_susc <= 0.55:
                return "disagree"
            elif self_susc > other_susc and (self_susc <= 0.8 or other_susc <= 0.8):
                return "otherconvince"
            elif self_susc < other_susc and (self_susc <= 0.8 or other_susc <= 0.8):
                return "selfconvince"
            else:
                return "mutual"

        # --- Mid vs High ---
        elif 0.3 <= self_susc <= 0.8 and other_susc >= 1.6:
            return "selfconvince"
        elif self_susc >= 1.6 and 0.3 <= other_susc <= 0.8:
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

    def policy_influence(self, other):
        # making sure that when families interact, a ripple
        if hasattr(self, "family") and hasattr(other, "family"):
            if self.family == other.family:
                # Skip ripple triggers if family interaction
                self.interacted_within_family = True
                other.interacted_within_family = True
            else:
                self.interacted_within_family = False
                other.interacted_within_family = False
        else:
            self.interacted_within_family = False
            other.interacted_within_family = False

        old_self = self.belief_vector().copy()
        old_other = other.belief_vector().copy()

        rule = self.choose_rule(other)
        new_self, new_other = self.interaction_rules[rule](self, other)

        other_old_party = other.party_affiliation
        self_old_party = self.party_affiliation

        self.update_from_vector(self.reflect(new_self))
        other.update_from_vector(self.reflect(new_other))

        new_self_vec = self.belief_vector()
        new_other_vec = other.belief_vector()

        self.movement_tracker["policy influence"] = int(
            np.linalg.norm(new_self_vec - old_self)
        )
        other.movement_tracker["policy influence"] = int(
            np.linalg.norm(new_other_vec - old_other)
        )

        # Update party affiliation
        self.update_affiliation_and_support(old_party=self.party_affiliation)
        other.update_affiliation_and_support(old_party=other.party_affiliation)

        other_new_party = other.party_affiliation
        self_new_party = self.party_affiliation

        # Track switches and causes (same logic as your original)
        if other_old_party != other_new_party:
            if other.interacted_within_family:
                if other.party_affiliation == self.party_affiliation:
                    other.switched_this_step = True
                    other.interacted_within_family = True
                    other.interacted_within_party = True
                    other.switch_cause.append("family_interaction_within_party")
                else:
                    other.switched_this_step = True
                    other.interacted_cross_party = True
                    other.switch_cause.append("family_interaction_cross_party")
            else:
                if other.party_affiliation == self.party_affiliation:
                    other.switched_this_step = True
                    other.interacted_within_party = True
                    other.switch_cause.append("interaction_within_party")
                else:
                    other.switched_this_step = True
                    other.interacted_cross_party = True
                    other.switch_cause.append("interaction_cross_party")

        if self_old_party != self_new_party:
            if self.interacted_within_family:
                if self.party_affiliation == other.party_affiliation:
                    self.switched_this_step = True
                    self.interacted_within_family = True
                    self.interacted_within_party = True
                    self.switch_cause.append("family_interaction_within_party")
                else:
                    self.switched_this_step = True
                    self.interacted_within_family = False
                    self.interacted_cross_party = True
                    self.switch_cause.append("family_interaction_cross_party")
            else:
                if self.party_affiliation == other.party_affiliation:
                    self.switched_this_step = True
                    self.interacted_within_party = True
                    self.switch_cause.append("interaction_within_party")
                else:
                    self.switched_this_step = True
                    self.interacted_cross_party = True
                    self.switch_cause.append("interaction_cross_party")

        # --- Family ripple trigger ---
        if not self.interacted_within_family:
            if hasattr(self, "family") and self.family:
                self.family.ripple_influence(self, old_self, new_self)
            if hasattr(other, "family") and other.family:
                other.family.ripple_influence(other, old_other, new_other)
