import numpy as np


class HealthLogic:
    def perceive_health_status(self):
        """
        Models random sickness events during simulation.
        Agents can fall sick based on a small probability each step.
        """
        if self.alive and self.healthy:
            if np.random.random() < self.sickness_chance:
                self.healthy = False

    def perceive_healthcare(self):
        if self.healthy is False and self.alive:
            if self.wealth >= 10:  # assuming average wealth is around 27
                self.health_care = "Private"
                self.healthy = True
            else:
                self.health_care = "Public"
                if np.random.random() < 0.4:
                    self.alive = False
                    if hasattr(self, "family") and self.family:
                        self.family.react_to_death(self)
