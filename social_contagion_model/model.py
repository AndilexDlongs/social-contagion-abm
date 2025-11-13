import mesa
import numpy as np
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid

from .Agents.voter import VoterAgent
from .Agents.family import FamilyAgent
from .Agents.party import Party
from .seeder import Seeder
from .utils import (
    count_Conservatism, count_Liberalism, count_Socialism, 
    count_Undecided, count_in_support, num_interactions,
    num_switches, num_switches_in_rebellion, num_switches_in_support,
    vote_counts, num_interactions_in_party, num_interactions_cross_party,
    count_sick_agents
)

class Environment(mesa.Model):
    """Environment with agents, parties, and interactions."""

    def __init__(self, n=5, width=3, height=3, seed=None, 
                undecided_ratio=0.1, 
                majority_party="Conservatism", 
                family_multiplier=1.2,
                healthcare_multiplier=1.6, 
                wealth_influence_factor=1, 
                interaction_multiplier=0.8,
                conservatism_perc=0.4,
                socialism_perc=0.35,
                liberalism_perc=0.25,
                conservatism_std=10,
                socialism_std=10,
                liberalism_std=10, 
                conservatism_susc=0.5, 
                socialism_susc=0.5,
                liberalism_susc=0.5,
                conservatism_wealth=0.5, 
                socialism_wealth=0.5, 
                liberalism_wealth=0.5,
                sickness_chance=0.05,
                min_family_size=3,
                max_family_size=10):
        
        super().__init__()
        self.seed = seed
        self.num_agents = n
        self.family_multiplier = family_multiplier
        self.healthcare_multiplier = healthcare_multiplier
        self.wealth_influence_factor = wealth_influence_factor
        self.interaction_multiplier = interaction_multiplier
        self.grid = OrthogonalMooreGrid(
            (width, height), 
            torus=True, 
            capacity=100, 
            random=self.random
        )

        # visualization helper
        self.step_count = 0

        # Attributes
        self.attribute_names = ["LawAndOrder", "EconomicEquality", "SocialWelfare"]
        self.belief_dim = len(self.attribute_names)
        self.neutral_center = np.array([50] * self.belief_dim)
        
        # Experiment state
        self.majority_party = majority_party # majority part must be in that state variable
        self.state = { 
            "average_wealth": 0
        }
        self.death_count = 0
        self.sickness_from_deaths = 0

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
        majority_party=majority_party,
        fixed_seed=80,
        party_distribution={
            "Conservatism": conservatism_perc,
            "Socialism": socialism_perc,
            "Liberalism": liberalism_perc
        },
        party_stddev={
            "Conservatism": conservatism_std,
            "Socialism": socialism_std,
            "Liberalism": liberalism_std
        }
)


        init_agents = seeder.assign_agents(self.num_agents)

        for data in init_agents:
            agent = VoterAgent(self, cell=None)
            # Pass global susceptibility config so agent logic triggers correctly

            agent.sickness_chance = sickness_chance
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
            agent.initialize_wealth(
                conservatism_wealth=conservatism_wealth, 
                socialism_wealth=socialism_wealth, 
                liberalism_wealth=liberalism_wealth
            )
            agent.initialize_susceptibility(
                conservatism_susc=conservatism_susc, 
                socialism_susc=socialism_susc,
                liberalism_susc=liberalism_susc,
            )
            # (later) agent.education = data["education"]; agent.health_care = data["healthcare"]
            # grid placement
            cell = self.random.choice(self.grid.all_cells.cells)
            cell.add_agent(agent)
            agent.cell = cell

        # ----------------------------
        # FAMILY INITIALIZATION
        # ----------------------------

        self.family_agents = []
        agents_list = list(self.agents)
        self.random.shuffle(agents_list)

        for i, agent in enumerate(agents_list):
            # Skip if already assigned to a family
            if hasattr(agent, "family") and agent.family:
                continue

            family_members = [agent]

            # Max family size 
            family_size = self.random.randint(min_family_size, max_family_size)

            for other in agents_list:
                if other == agent or (hasattr(other, "family") and other.family):
                    continue

                # Find if this other agent is closer to *another* party center than their own
                own_party_center = agent.party_center()
                distances = {p.name: np.linalg.norm(other.belief_vector() - p.center_vector()) for p in self.parties}

                # Check if this agent is near the ideological border
                closest_party = min(distances, key=distances.get)
                if closest_party == agent.party_affiliation and distances[closest_party] < 20 and len(family_members) < family_size:
                    family_members.append(other)
                    # other.family = True
                elif len(family_members) < family_size and self.random.random() < 0.2:
                    # Add random members to fill out family diversity
                    family_members.append(other)
                   # other.family = True

                if len(family_members) >= family_size:
                    break

            # Assign the family group
            if len(family_members) >= min_family_size:
                family = FamilyAgent(self, family_members)
                for m in family_members:
                    m.family = family # Assigns the family object reference to each member.
                self.family_agents.append(family)


        for i, agent in enumerate(agents_list):
            agent.family_members = agent.get_members()
            agent.family_size = len(agent.family_members) 

         


        self.datacollector = DataCollector(
            model_reporters={
                "num_switches": num_switches,
                "num_interactions": num_interactions,
                "num_interactions_in_party": num_interactions_in_party,
                "num_interactions_cross_party": num_interactions_cross_party,
                "vote_Conservatism": count_Conservatism,
                "vote_Socialism": count_Socialism,
                "vote_Liberalism": count_Liberalism,
                "vote_Undecided": count_Undecided,
                "current_party_in_power": lambda m: m.majority_party,
                "count_in_support": count_in_support,
                "num_switches_in_support": num_switches_in_support,
                "num_switches_in_rebellion": num_switches_in_rebellion,
                "avg_wealth": lambda m: m.state["average_wealth"],
                "death_count": lambda m: m.death_count,
                "sickness_count": lambda m: m.count_sick_agents() + m.sickness_from_deaths
            },
            agent_reporters={
                "belief_vector": lambda a: a.belief_vector().tolist(),
                "distance_from_party": lambda a: a.distance,
                "party": lambda a: a.party_affiliation,
                "original_party": lambda a: a.original_party_affiliation,
                "susceptibility": lambda a: a.susceptibility,
                "switched": lambda a: a.switched_this_step,
                "has_interacted": lambda a: a.has_interacted,
                "interacted_with": lambda a: a.interacted_with,
                "interacted_within_party": lambda a: a.interacted_within_party,
                "wealth": lambda a: a.wealth,
                "in_support": lambda a: a.in_support,
                "family_id": lambda a: a.family_id,
                "family_members": lambda a: a.family_members,
                "family_size": lambda a: a.family_size,
                "switched_this_step": lambda a: a.switched_this_step,
                "switch_cause": lambda a: a.switch_cause if a.switch_cause else "none",
                "alive": lambda a: a.alive,
                "healthcare_access": lambda a: a.health_care,
                "healthy": lambda a: a.healthy,
                "movement_tracker": lambda a: a.movement_tracker
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
        
    def count_sick_agents(self):
        """Count how many agents are currently sick."""
        return sum(1 for a in self.agents if getattr(a, "healthy", True) is False)

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
        self.agents.shuffle_do("perceive_health_status")
        self.agents.shuffle_do("perceive_healthcare")
        self.agents.shuffle_do("interact")
        self.aggregate_environment_state()

        # Remove dead agents
        dead = [a for a in self.agents if not a.alive]
        for d in dead:
            if d.cell and d in d.cell.agents:
                d.cell.agents.remove(d)
            self.agents.remove(d)
            self.death_count += 1
            self.sickness_from_deaths += 1

        self.datacollector.collect(self)
        self.agents.shuffle_do("reset")
        self.death_count = 0  # reset death count after collection
        self.sickness_from_deaths = 0  # reset sickness from deaths after collection
        self.step_count += 1
        