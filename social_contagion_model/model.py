import mesa 
import numpy as np
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid


from agents import VoterAgent, Party, FamilyAgent
from seeder import Seeder
from utils import (count_Conservatism, count_Liberalism, count_Socialism,
                   count_Undecided, count_in_support, num_interactions,
                   num_switches, num_switches_in_rebellion,
                   num_switches_in_support, vote_counts)

class Environment(mesa.Model):
    """Environment with agents, parties, and interactions."""

    def __init__(self, n=5, width=3, height=3, seed=None,
                seeding_strategy="fixed_split", 
                undecided_ratio=0.1, 
                majority_party="Conservatism"):
    
        super().__init__()
        self.seed = seed
        self.num_agents = n
        self.grid = OrthogonalMooreGrid(
            (width, height), 
            torus=True, 
            capacity=100, 
            random=self.random
        )

        # visualization helper
        self.steps = 0

        # Attributes
        self.attribute_names = ["LawAndOrder", "EconomicEquality", "SocialWelfare"]
        self.belief_dim = len(self.attribute_names)
        self.neutral_center = np.array([50] * self.belief_dim)
        
        # Experiment state
        self.majority_party = majority_party # majority part must be in that state variable
        self.state = { 
            "average_wealth": 0
        }

        # Parties
        self.parties = [
            Party("Conservatism", LawAndOrder=80, EconomicEquality=20, SocialWelfare=40, radius=30),
            Party("Socialism",    LawAndOrder=30, EconomicEquality=80, SocialWelfare=70, radius=30),
            Party("Liberalism",   LawAndOrder=50, EconomicEquality=50, SocialWelfare=80, radius=30),
        ]

        # --------------------------
        # NEW: Seeder Integration
        # --------------------------
        seeder = Seeder(
            self.parties, 
            undecided_ratio=undecided_ratio, 
            strategy=seeding_strategy,
            majority_party=majority_party,
            fixed_seed=80
        )

        init_agents = seeder.assign_agents(self.num_agents)

        for data in init_agents:
            agent = VoterAgent(self, cell=None)
            # beliefs + affiliation
            agent.update_from_vector(data["beliefs"])

            affiliation = data["affiliation"]
            
            # Recompute party affiliation and distance using agent’s logic
            if affiliation == "Undecided":
                agent.party_affiliation, undecided_distance = agent.assign_party()
                agent.original_party_affiliation = agent.party_affiliation
                agent.distance = undecided_distance
            else:
                agent.party_affiliation = affiliation
                agent.original_party_affiliation = agent.party_affiliation
                agent.distance = agent.party_distance()

            # attributes
            agent.wealth = data["wealth"]
            # (later) agent.education = data["education"]; agent.health_care = data["healthcare"]
            # grid placement
            cell = self.random.choice(self.grid.all_cells.cells)
            cell.add_agent(agent)
            agent.cell = cell

        # ----------------------------
        # FAMILY INITIALIZATION
        # ----------------------------

        # self.family_agents = []
        # agents_list = list(self.agents)
        # self.random.shuffle(agents_list)

        # for i, agent in enumerate(agents_list):
        #     # Skip if already assigned to a family
        #     if hasattr(agent, "family") and agent.family:
        #         continue

        #     family_members = [agent]

        #     # Max family size 
        #     max_size = self.random.randint(3, 10)

        #     for other in agents_list:
        #         if other == agent or (hasattr(other, "family") and other.family):
        #             continue

        #         # Find if this other agent is closer to *another* party center than their own
        #         own_party_center = agent.party_center()
        #         distances = {p.name: np.linalg.norm(other.belief_vector() - p.center_vector()) for p in self.parties}

        #         # Check if this agent is near the ideological border
        #         closest_party = min(distances, key=distances.get)
        #         if closest_party == agent.party_affiliation and distances[closest_party] < 20 and len(family_members) < 3:
        #             family_members.append(other)
        #             other.family = True
        #         elif len(family_members) < max_size and self.random.random() < 0.2:
        #             # Add random members to fill out family diversity
        #             family_members.append(other)
        #             other.family = True

        #         if len(family_members) >= max_size:
        #             break

        #     # Assign the family group
        #     family = FamilyAgent(self, family_members)
        #     for m in family_members:
        #         m.family = family
        #     self.family_agents.append(family)


        self.datacollector = DataCollector(
            model_reporters={
                "num_switches": num_switches,
                "num_interactions": num_interactions,
                "vote_Conservatism": count_Conservatism,
                "vote_Socialism": count_Socialism,
                "vote_Liberalism": count_Liberalism,
                "vote_Undecided": count_Undecided,
                "current_party_in_power": lambda m: m.majority_party,
                "count_in_support": count_in_support,
                "num_switches_in_support": num_switches_in_support,
                "num_switches_in_rebellion": num_switches_in_rebellion,
                "avg_wealth": lambda m: m.state["average_wealth"],
            },
            agent_reporters={
                "belief_vector": lambda a: a.belief_vector().tolist(),
                "distance_from_party": lambda a: a.distance,
                "party": lambda a: a.party_affiliation,
                "susceptibility": lambda a: a.susceptibility,
                "switched": lambda a: a.switched_this_step,
                "has_interacted": lambda a: a.has_interacted,
                "interacted_with": lambda a: a.interacted_with,
                "wealth": lambda a: a.wealth,
            },
        )

        
        self.datacollector.collect(self)
        self.evaluate_majority_party()
        
        for a in self.agents:
            a.update_affiliation_and_support()

        self.aggregate_environment_state()
        self.agents.shuffle_do("reset")

    def aggregate_environment_state(self):
        """Compute and store global environmental metrics."""
        self.state["average_wealth"] = np.mean([a.wealth for a in self.agents])
        

    def evaluate_majority_party(self):
        df = self.datacollector.get_model_vars_dataframe()
        conservatism = df["vote_Conservatism"].iloc[-1] 
        socialism = df["vote_Socialism"].iloc[-1] 
        liberalism = df["vote_Liberalism"].iloc[-1]

        if conservatism > socialism and conservatism > liberalism:
            self.majority_party = "Conservatism" 
        elif socialism > conservatism and socialism > liberalism:   
            self.majority_party = "Socialism"
        elif liberalism > conservatism and liberalism > socialism:
            self.majority_party = "Liberalism"

        for party in self.parties:
            party.current_party_in_power = self.majority_party

    def step(self):
        """ Advance the model by one step. """
        self.agents.shuffle_do("move")
        self.agents.shuffle_do("interact")
        self.aggregate_environment_state()
        self.datacollector.collect(self)
        self.agents.shuffle_do("reset")
        self.steps += 1

    # def media_campaign(self, bias):
    #    """ Conduct a media campaign with a specific bias. """
    #   self.agents.do("media_influence", bias)
        