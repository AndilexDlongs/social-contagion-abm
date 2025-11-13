import numpy as np


class EconomicsLogic:
    # ---------------------------
    # Wealth Initialization Method
    # ---------------------------
    def initialize_wealth(self, conservatism_wealth, socialism_wealth, liberalism_wealth):
        """Initializes wealth based on party affiliation and wealth distribution percentages."""
        raw_wealth = np.random.beta(2, 5) * 100  # Random wealth generation
        beta_cutoff = 26  # Threshold for wealth categorization (can be adjusted)

        # Wealth focus per party
        if self.party_affiliation == "Conservatism":
            if np.random.random() < conservatism_wealth:
                self.wealth = (
                    raw_wealth
                    if raw_wealth > beta_cutoff
                    else np.random.uniform(0, beta_cutoff)
                )
            else:
                self.wealth = raw_wealth

        elif self.party_affiliation == "Socialism":
            if np.random.random() < socialism_wealth:
                self.wealth = (
                    raw_wealth
                    if raw_wealth > beta_cutoff
                    else np.random.uniform(0, beta_cutoff)
                )
            else:
                self.wealth = raw_wealth

        elif self.party_affiliation == "Liberalism":
            if np.random.random() < liberalism_wealth:
                self.wealth = (
                    raw_wealth
                    if raw_wealth > beta_cutoff
                    else np.random.uniform(0, beta_cutoff)
                )
            else:
                self.wealth = raw_wealth

    def initialize_susceptibility(self, conservatism_susc, socialism_susc, liberalism_susc):
        if self.party_affiliation == "Conservatism":
            if np.random.random() < conservatism_susc:
                self.susceptibility = self.low
            else:
                self.susceptibility = self.high
        elif self.party_affiliation == "Socialism":
            if np.random.random() < socialism_susc:
                self.susceptibility = self.low
            else:
                self.susceptibility = self.high
        elif self.party_affiliation == "Liberalism":
            if np.random.random() < liberalism_susc:
                self.susceptibility = self.low
            else:
                self.susceptibility = self.high

    def adjust_economic_view(self):
        """Adjust ideological position when dissatisfaction thresholds are reached."""
        old_vec = self.belief_vector()

        if self.wealth_dissatisfaction > self.dissatisfaction_threshold:
            old_party = self.party_affiliation
            # Too dissatisfied → shift away from own or majority party
            if self.model.majority_party == self.party_affiliation:
                new_self = self.move_closer_to_other_party_vector()
                wealth_effect_new = self.belief_vector() + self.wealth_influence_factor * (
                    new_self - self.belief_vector()
                )
            else:
                new_self = self.move_closer_to_own_party_vector()
                wealth_effect_new = self.belief_vector() + self.wealth_influence_factor * (
                    new_self - self.belief_vector()
                )

            self.update_from_vector(self.reflect(wealth_effect_new))
            self.update_affiliation_and_support(old_party=self.party_affiliation)
            self.wealth_dissatisfaction = 0

            new_party = self.party_affiliation
            if old_party != new_party:
                self.switch_cause.append("dissatisfied_wealth")

        elif self.wealth_dissatisfaction < self.satisfaction_threshold:
            old_party = self.party_affiliation
            # Very satisfied → go toward majority party since they're controlling the economy
            if self.model.majority_party == self.party_affiliation:
                new_self = self.move_closer_to_own_party_vector()
                wealth_effect_new = self.belief_vector() + self.wealth_influence_factor * (
                    new_self - self.belief_vector()
                )
            else:
                new_self = self.move_closer_to_majority_party_vector()
                wealth_effect_new = self.belief_vector() + self.wealth_influence_factor * (
                    new_self - self.belief_vector()
                )

            self.update_from_vector(self.reflect(wealth_effect_new))
            self.update_affiliation_and_support(old_party=self.party_affiliation)
            self.wealth_dissatisfaction = 0

            new_party = self.party_affiliation
            if old_party != new_party:
                self.switch_cause.append("satisfied_wealth")

        new_vec = self.belief_vector()
        self.movement_tracker["wealth comparison"] = int(
            np.linalg.norm(new_vec - old_vec)
        )

    def compare_wealth(self, other):
        wealth_gap = other.wealth - self.wealth

        # Emotional response (accumulate dissatisfaction)
        if wealth_gap > self.significant_difference:
            self.wealth_dissatisfaction += 1
        elif wealth_gap < -self.significant_difference:
            self.wealth_dissatisfaction -= 1

        # Trigger ideological adjustment if thresholds reached
        self.adjust_economic_view()

    def give_wealth(self, other):
        # compare wealth levels
        self.compare_wealth(other)
        other.compare_wealth(self)

        # give some wealth to other agent
        if self.wealth > 1:
            other.wealth += 1
            self.wealth -= 1
