import numpy as np

class Seeder:
    """
    Seeder assigns initial beliefs and affiliations using
    explicit party percentages and per-party standard deviations.

    Example:
    Seeder(
        parties,
        party_distribution={"Conservatism": 0.4, "Socialism": 0.35, "Liberalism": 0.25},
        party_stddev={"Conservatism": 10, "Socialism": 8, "Liberalism": 12},
        undecided_ratio=0.1
    )
    """

    def __init__(self, parties,
                 party_distribution=None,
                 party_stddev=None,
                 undecided_ratio=0.1,
                 majority_party=None,
                 fixed_seed=None):
        self.parties = parties
        self.party_distribution = party_distribution or {
            p.name: 1 / len(parties) for p in parties
        }
        self.party_stddev = party_stddev or {
            p.name: 10 for p in parties
        }
        self.undecided_ratio = undecided_ratio
        self.majority_party = majority_party
        self.fixed_seed = fixed_seed

        # Normalize party percentages if they don’t sum to 1
        total = sum(self.party_distribution.values())
        if total != 1.0:
            self.party_distribution = {k: v / total for k, v in self.party_distribution.items()}

    def assign_agents(self, num_agents):
        """Return list of dicts: [{beliefs, affiliation, wealth}, ...]"""
        if self.fixed_seed is not None:
            np.random.seed(self.fixed_seed)

        out = []
        num_undecided = int(num_agents * self.undecided_ratio)
        remaining = num_agents - num_undecided

        # Assign decided agents by party proportions
        for party in self.parties:
            party_name = party.name
            count = int(remaining * self.party_distribution.get(party_name, 0))
            stddev = self.party_stddev.get(party_name, 10)

            for _ in range(count):
                vec = party.center_vector() + np.random.normal(0, stddev, size=len(party.center_vector()))
                out.append({
                    "beliefs": vec,
                    "affiliation": party_name,
                    "wealth": None
                })

        # Assign undecideds
        for _ in range(num_undecided):
            vec = np.random.uniform(0, 100, size=len(self.parties[0].center_vector()))
            distances = {p.name: np.linalg.norm(vec - p.center_vector()) for p in self.parties}
            closest_party = min(distances, key=distances.get)
            affiliation = closest_party if np.random.random() < 0.4 else "Undecided"
            out.append({"beliefs": vec, "affiliation": affiliation, "wealth": None})

        # Fill remainder if rounding errors left out some
        while len(out) < num_agents:
            p = np.random.choice(self.parties)
            vec = p.center_vector() + np.random.normal(0, 10, size=len(p.center_vector()))
            out.append({"beliefs": vec, "affiliation": p.name, "wealth": None})

        return out[:num_agents]
