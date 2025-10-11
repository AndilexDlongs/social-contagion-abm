import numpy as np


class Seeder:
    def __init__(self, parties, undecided_ratio=0.1, strategy="proximity", 
                 majority_party=None):
        """
        parties: list of party objects (with .center or .belief_vector attribute)
        undecided_ratio: float, fraction of agents to be undecided
        strategy: str, determines seeding strategy ("proximity", "fixed_split", etc.)
        """
        self.parties = parties
        self.undecided_ratio = undecided_ratio
        self.strategy = strategy
        self.majority_party = majority_party

    def assign_agents(self, num_agents):
        """
        Returns a list of dicts per agent:
        {
          "beliefs": np.array([..]),
          "affiliation": "<party|Undecided>",
          "wealth": float
          # (later: education, healthcare, etc.)
        }
        """
        base = self._assign_beliefs(num_agents)  # <- keep your existing logic here
        # attach attributes
        out = []

        for beliefs, affiliation in base:
            attrs = {
                "beliefs": beliefs,
                "affiliation": affiliation,
                "wealth": int(np.random.beta(2, 5) * 100), 
            }
            out.append(attrs)
        return out
    
    def _assign_beliefs(self, num_agents):
        if self.strategy == "proximity":
            return self._proximity_based(num_agents)
        elif self.strategy == "fixed_split":
            return self._fixed_split(num_agents)
        elif self.strategy == "equal_distribution":
            return self._equal_distribution(num_agents)
        else:
            raise ValueError(f"Unknown seeding strategy: {self.strategy}")

    def _proximity_based(self, num_agents):
        """
        Each agent starts near some party's center (Gaussian noise).
        A few go to undecided.
        """
        agents = []
        num_undecided = int(num_agents * self.undecided_ratio)
        num_decided = num_agents - num_undecided

        for _ in range(num_decided):
            party = np.random.choice(self.parties)
            vec = party.center_vector() + np.random.normal(0, 10, size=len(party.center_vector()))  # adjust spread
            agents.append((vec, party.name))

        # Undecided (sampled until they are NOT in any party radius)
        for _ in range(num_undecided):
            while True:
                vec = np.random.uniform(0, 100, size=len(self.parties[0].center_vector()))
                # Check distance from all party centers
                if all(np.linalg.norm(vec - p.center_vector()) > p.radius for p in self.parties):
                    agents.append((vec, "Undecided"))
                    break

        while len(agents) < num_agents:
            party = np.random.choice(self.parties)
            vec = party.center_vector() + np.random.normal(0, 10, size=len(party.center_vector()))
            agents.append((vec, party.name))
                
        return agents

    def _fixed_split(self, num_agents):
        """
        Example: half to one party, rest equally split, some undecided.
        """
        agents = []
        num_undecided = int(num_agents * self.undecided_ratio)
        remaining = num_agents - num_undecided

        # Half go to first party
        first_party_count = remaining // 2

        majority = self.majority_party
        majority_party = next(p for p in self.parties if p.name == majority)

        for _ in range(first_party_count):
            vec = majority_party.center_vector() + np.random.normal(0, 10, size=len(self.parties[0].center_vector()))
            agents.append((vec, majority_party.name))

        # Rest split equally
        others = [p for p in self.parties if p.name != majority_party.name]
        rest_count = remaining - first_party_count
        per_party = max(1, rest_count // max(1, len(others)))
                        
        for party in others:
            for _ in range(per_party):
                vec = party.center_vector() + np.random.normal(0, 10, size=len(party.center_vector()))
                agents.append((vec, party.name))

        # Undecided (sampled until they are NOT in any party radius)
        for _ in range(num_undecided):
            while True:
                vec = np.random.uniform(0, 100, size=len(self.parties[0].center_vector()))
                # Check distance from all party centers
                if all(np.linalg.norm(vec - p.center_vector()) > p.radius for p in self.parties):
                    agents.append((vec, "Undecided"))
                    break

        while len(agents) < num_agents:
            party = np.random.choice(self.parties)
            vec = party.center_vector() + np.random.normal(0, 10, size=len(party.center_vector()))
            agents.append((vec, party.name))

        return agents[:num_agents]

    def _equal_distribution(self, num_agents):
        """
        Every party gets an equal share, no undecided (unless ratio > 0).
        """
        agents = []
        num_undecided = int(num_agents * self.undecided_ratio)
        remaining = num_agents - num_undecided
        per_party = remaining // len(self.parties)

        for party in self.parties:
            for _ in range(per_party):
                vec = party.center_vector() + np.random.normal(0, 10, size=len(party.center_vector()))
                agents.append((vec, party.name))

        # Undecided (sampled until they are NOT in any party radius)
        for _ in range(num_undecided):
            while True:
                vec = np.random.uniform(0, 100, size=len(self.parties[0].center_vector()))
                # Check distance from all party centers
                if all(np.linalg.norm(vec - p.center_vector()) > p.radius for p in self.parties):
                    agents.append((vec, "Undecided"))
                    break

        while len(agents) < num_agents:
            party = np.random.choice(self.parties)
            vec = party.center_vector() + np.random.normal(0, 10, size=len(party.center_vector()))
            agents.append((vec, party.name))

        return agents[:num_agents]
