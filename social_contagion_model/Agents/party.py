import numpy as np

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
