class StepControl:
    def interact(self):
        if self.has_interacted:
            return

        others = [a for a in self.cell.agents if a != self and not a.has_interacted]
        if others:
            other = self.random.choice(others)

            # FIX: call methods, not references
            self.evaluate_susceptibility()
            other.evaluate_susceptibility()

            self.policy_influence(other)
            self.give_wealth(other)

            # mark both as having interacted
            self.has_interacted = True
            self.interacted_with = int(other.unique_id)
            other.has_interacted = True
            other.interacted_with = int(self.unique_id)

    def reset(self):
        """Reset interaction and tracking flags for this agent."""
        self.has_interacted = False
        self.switched_this_step = False
        self.switched_in_support = False
        self.switched_in_rebellion = False
        self.interacted_with = None
        self.interacted_within_party = False
        self.interacted_cross_party = False
        self.switch_cause = []
        self.movement_tracker = {
            "death reaction": 0,
            "policy influence": 0,
            "family ripple": 0,
            "wealth comparison": 0,
        }
        if self.alive:
            self.healthy = True
