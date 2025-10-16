import numpy as np

class Seeder:
    """
    Seeder assigns initial beliefs, party affiliations, and attributes (like wealth)
    for agents based on different strategies:
      - proximity: Gaussian spread around party centers
      - fixed_split: majority party bias, others split equally
      - equal_distribution: uniform across parties
    If `fixed_seed` is provided, results are reproducible across runs.
    """

    def __init__(self, parties, undecided_ratio=0.1, strategy="proximity", 
                 majority_party=None, fixed_seed=None):
        self.parties = parties
        self.undecided_ratio = undecided_ratio
        self.strategy = strategy
        self.majority_party = majority_party
        self.fixed_seed = fixed_seed  # reproducibility control

    # --------------------------
    # Main Entry
    # --------------------------
    def assign_agents(self, num_agents):
        # Make seeding reproducible when fixed_seed is set
        if self.fixed_seed is not None:
            np.random.seed(self.fixed_seed)

        base = self._assign_beliefs(num_agents)
        out = []

        for beliefs, affiliation in base:
            attrs = {
                "beliefs": beliefs,
                "affiliation": affiliation,
                "wealth": int(np.random.beta(2, 5) * 100),  # right-skewed wealth
            }
            out.append(attrs)
        return out

    # --------------------------
    # Strategy Routing
    # --------------------------
    def _assign_beliefs(self, num_agents):
        if self.strategy == "proximity":
            return self._proximity_based(num_agents)
        elif self.strategy == "fixed_split":
            return self._fixed_split(num_agents)
        elif self.strategy == "equal_distribution":
            return self._equal_distribution(num_agents)
        else:
            raise ValueError(f"Unknown seeding strategy: {self.strategy}")

    # --------------------------
    # Shared helper: nearest-party rule
    # --------------------------
    def _assign_affiliation_by_distance(self, vec):
        """Assign nearest party if within its radius, else Undecided (includes tie-breaker)."""
        distances = {p.name: np.linalg.norm(vec - p.center_vector()) for p in self.parties}
        sorted_parties = sorted(distances.items(), key=lambda x: x[1])
        closest_party, closest_distance = sorted_parties[0]

        # Random tie-breaker when two parties are almost equidistant
        if len(sorted_parties) > 1 and abs(sorted_parties[0][1] - sorted_parties[1][1]) < 3:
            closest_party = np.random.choice([sorted_parties[0][0], sorted_parties[1][0]])

        # Assign only if within that party’s radius
        radius = next(p.radius for p in self.parties if p.name == closest_party)
        return closest_party if closest_distance <= radius else "Undecided"

    # --------------------------
    # Strategy 1: Proximity-based
    # --------------------------
    def _proximity_based(self, num_agents):
        agents = []
        num_undecided = int(num_agents * self.undecided_ratio)
        num_decided = num_agents - num_undecided

        for _ in range(num_decided):
            party = np.random.choice(self.parties)
            vec = party.center_vector() + np.random.normal(0, 10, size=len(party.center_vector()))
            agents.append((vec, party.name))

        for _ in range(num_undecided):
            vec = np.random.uniform(0, 100, size=len(self.parties[0].center_vector()))
            affiliation = self._assign_affiliation_by_distance(vec)
            agents.append((vec, affiliation))

        while len(agents) < num_agents:
            party = np.random.choice(self.parties)
            vec = party.center_vector() + np.random.normal(0, 10, size=len(party.center_vector()))
            agents.append((vec, party.name))
        return agents[:num_agents]

    # --------------------------
    # Strategy 2: Fixed Split
    # --------------------------
    def _fixed_split(self, num_agents):
        agents = []
        num_undecided = int(num_agents * self.undecided_ratio)
        remaining = num_agents - num_undecided

        # Majority party gets half
        majority_party = next(p for p in self.parties if p.name == self.majority_party)
        first_party_count = remaining // 2

        for _ in range(first_party_count):
            vec = majority_party.center_vector() + np.random.normal(0, 10, size=len(majority_party.center_vector()))
            agents.append((vec, majority_party.name))

        # Other parties share remaining
        others = [p for p in self.parties if p.name != majority_party.name]
        rest_count = remaining - first_party_count
        per_party = max(1, rest_count // max(1, len(others)))

        for party in others:
            for _ in range(per_party):
                vec = party.center_vector() + np.random.normal(0, 10, size=len(party.center_vector()))
                agents.append((vec, party.name))

        # Undecided with nearest-party fallback
        for _ in range(num_undecided):
            vec = np.random.uniform(0, 100, size=len(self.parties[0].center_vector()))
            affiliation = self._assign_affiliation_by_distance(vec)
            agents.append((vec, affiliation))

        while len(agents) < num_agents:
            party = np.random.choice(self.parties)
            vec = party.center_vector() + np.random.normal(0, 10, size=len(party.center_vector()))
            agents.append((vec, party.name))
        return agents[:num_agents]

    # --------------------------
    # Strategy 3: Equal Distribution
    # --------------------------
    def _equal_distribution(self, num_agents):
        agents = []
        num_undecided = int(num_agents * self.undecided_ratio)
        remaining = num_agents - num_undecided
        per_party = remaining // len(self.parties)

        for party in self.parties:
            for _ in range(per_party):
                vec = party.center_vector() + np.random.normal(0, 10, size=len(party.center_vector()))
                agents.append((vec, party.name))

        for _ in range(num_undecided):
            vec = np.random.uniform(0, 100, size=len(self.parties[0].center_vector()))
            affiliation = self._assign_affiliation_by_distance(vec)
            agents.append((vec, affiliation))

        while len(agents) < num_agents:
            party = np.random.choice(self.parties)
            vec = party.center_vector() + np.random.normal(0, 10, size=len(party.center_vector()))
            agents.append((vec, party.name))
        return agents[:num_agents]
