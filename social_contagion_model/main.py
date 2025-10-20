
import numpy as np
from model import Environment
import json
from config import (
    N_AGENTS, GRID_WIDTH, GRID_HEIGHT, STEPS,
    MAJORITY_PARTY, UND_RATIO,
    FAMILY_MULTIPLIER, HEALTHCARE_MULTIPLIER,
    WEALTH_INFLUENCE_FACTOR, INTERACTION_MULTIPLIER,
    CONSERVATISM_PERC, SOCIALISM_PERC, LIBERALISM_PERC,
    CONSERVATISM_STD, SOCIALISM_STD, LIBERALISM_STD,
    SICKNESS_CHANCE, MIN_FAMILY_SIZE, MAX_FAMILY_SIZE,
    CONSERVATISM_SUSC, SOCIALISM_SUSC, LIBERALISM_SUSC, 
    CONSERVATISM_WEALTH, SOCIALISM_WEALTH, LIBERALISM_WEALTH 
)
import os
from plots import (
    plot_initial_party_affiliations,
    plot_belief_scatter_2d,
    plot_belief_scatter_3d, 
    plot_party_composition,
    plot_avg_distance_per_party,
    plot_distance_std,
    plot_support_vs_rebellion,
    plot_avg_susceptibility,
    plot_fraction_original_party
)

# Ensure the "results" directory exists
os.makedirs("results", exist_ok=True)

def main():
    # Initialize the environment
    env = Environment(
        n=N_AGENTS,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        undecided_ratio=UND_RATIO,
        majority_party=MAJORITY_PARTY,

        # Core multipliers
        family_multiplier=FAMILY_MULTIPLIER,
        healthcare_multiplier=HEALTHCARE_MULTIPLIER,
        wealth_influence_factor=WEALTH_INFLUENCE_FACTOR,
        interaction_multiplier=INTERACTION_MULTIPLIER,

        # Susceptibility setup
        # susc_party_focus=SUSC_PARTY_FOCUS,
        # susc_focus_value=SUSC_FOCUS_VALUE,
        # susc_other_value=SUSC_OTHER_VALUE,

        # Wealth setup
        # wealth_party_focus=WEALTH_PARTY_FOCUS,
        # wealth_focus_value=WEALTH_FOCUS_VALUE,
        # wealth_other_value=WEALTH_OTHER_VALUE,

        # Susceptibility setup
        conservatism_susc=CONSERVATISM_SUSC, 
        socialism_susc=SOCIALISM_SUSC,
        liberalism_susc=LIBERALISM_SUSC,

        # Wealth setup
        conservatism_wealth=CONSERVATISM_WEALTH, 
        socialism_wealth=SOCIALISM_WEALTH, 
        liberalism_wealth=LIBERALISM_WEALTH,

        # Party distribution
        conservatism_perc=CONSERVATISM_PERC,
        socialism_perc=SOCIALISM_PERC,
        liberalism_perc=LIBERALISM_PERC,

        # Std devs
        conservatism_std=CONSERVATISM_STD,
        socialism_std=SOCIALISM_STD,
        liberalism_std=LIBERALISM_STD,

        # Health & family
        sickness_chance=SICKNESS_CHANCE,
        min_family_size=MIN_FAMILY_SIZE,
        max_family_size=MAX_FAMILY_SIZE
    )

    # Create unique folder for this run (based on random seed or ID)
    run_id = env.seed or np.random.randint(1000)
    run_folder = f"results/run_{run_id}"
    
    # Create subfolders for plots and csv
    os.makedirs(f"{run_folder}/plots", exist_ok=True)
    os.makedirs(f"{run_folder}/csv", exist_ok=True)

    # --- Before simulation ---
    plot_initial_party_affiliations(env, folder=f"{run_folder}/plots", save=True, stage="initial")
    plot_belief_scatter_2d(env, folder=f"{run_folder}/plots", save=True, stage="initial")
    plot_belief_scatter_3d(env, folder=f"{run_folder}/plots", save=True, stage="initial")

    # display_df = env.datacollector.get_agent_vars_dataframe()
    # avg_susc = display_df.groupby("party")["susceptibility"].mean()
    # print("Average susceptibility:", avg_susc)

    # --- Simulation loop ---
    for step in range(STEPS):
        env.step()
        if step in (98, 198, 298, 398, 498, 598, 698, 798, 898, 998):  # Evaluate majority party at specific steps # how many elections
            for agent in env.agents:
                agent.force_vote()
            env.evaluate_majority_party()

    # --- Save results ---
    model_df = env.datacollector.get_model_vars_dataframe()
    agent_df = env.datacollector.get_agent_vars_dataframe()

    model_views = model_df[[
        "current_party_in_power", "count_in_support",
        "num_switches_in_support", "num_switches_in_rebellion",
        "avg_wealth"
    ]]
    model_views.to_csv(f"{run_folder}/csv/model_views.csv")

    dist_party_df = agent_df[["distance_from_party", "party", "has_interacted", "wealth"]]
    dist_party_df.unstack(level="AgentID").to_csv(f"{run_folder}/csv/dist_party_unstacked.csv")

    agent_attributes = agent_df[["distance_from_party", "susceptibility", "party", "belief_vector", "wealth"]]
    agent_attributes.to_csv(f"{run_folder}/csv/agent_attributes.csv")

    family_details = agent_df[["family_id", "family_members"]]
    family_details.to_csv(f"{run_folder}/csv/family_details.csv")
    
    # display_df2 = env.datacollector.get_agent_vars_dataframe()
    # avg_susc_after = display_df2.groupby("party")["susceptibility"].mean()
    # print("Average susceptibility after:", avg_susc_after)

    # Calculate the average of 'num_switches_in_rebellion'
    avg_rebellion_switches = model_views["num_switches_in_rebellion"].mean()

    print("Average number of switches in rebellion:", avg_rebellion_switches)

    model_df.to_csv(f"{run_folder}/csv/full_model_data.csv")
    agent_df.to_csv(f"{run_folder}/csv/full_agent_data.csv")

    # --- After simulation ---
    plot_initial_party_affiliations(env, folder=f"{run_folder}/plots", save=True, stage="final")
    plot_belief_scatter_2d(env, folder=f"{run_folder}/plots", save=True, stage="final")
    plot_belief_scatter_3d(env, folder=f"{run_folder}/plots", save=True, stage="final")

    plot_party_composition(env, folder=f"{run_folder}/plots")
    plot_avg_distance_per_party(env, folder=f"{run_folder}/plots")
    plot_distance_std(env, folder=f"{run_folder}/plots")
    plot_support_vs_rebellion(env, folder=f"{run_folder}/plots")
    plot_avg_susceptibility(env, folder=f"{run_folder}/plots")
    plot_fraction_original_party(env, folder=f"{run_folder}/plots")

    config_dict = {
        "N_AGENTS": N_AGENTS,
        "GRID_WIDTH": GRID_WIDTH,
        "GRID_HEIGHT": GRID_HEIGHT,
        "STEPS": STEPS,
        "MAJORITY_PARTY": MAJORITY_PARTY,
        "UND_RATIO": UND_RATIO,
        "FAMILY_MULTIPLIER": FAMILY_MULTIPLIER,
        "HEALTHCARE_MULTIPLIER": HEALTHCARE_MULTIPLIER,
        "WEALTH_INFLUENCE_FACTOR": WEALTH_INFLUENCE_FACTOR,
        "INTERACTION_MULTIPLIER": INTERACTION_MULTIPLIER,
        "CONSERVATISM_SUSC" : CONSERVATISM_SUSC,
        "SOCIALISM_SUSC" : SOCIALISM_SUSC,
        "LIBERALISM_SUSC" : LIBERALISM_SUSC,
        "CONSERVATISM_WEALTH" : CONSERVATISM_WEALTH,
        "SOCIALISM_WEALTH" : SOCIALISM_WEALTH,
        "LIBERALISM_WEALTH" : LIBERALISM_WEALTH,
        "CONSERVATISM_PERC": CONSERVATISM_PERC,
        "SOCIALISM_PERC": SOCIALISM_PERC,
        "LIBERALISM_PERC": LIBERALISM_PERC,
        "CONSERVATISM_STD": CONSERVATISM_STD,
        "SOCIALISM_STD": SOCIALISM_STD,
        "LIBERALISM_STD": LIBERALISM_STD,
        "SICKNESS_CHANCE": SICKNESS_CHANCE,
        "MIN_FAMILY_SIZE": MIN_FAMILY_SIZE,
        "MAX_FAMILY_SIZE": MAX_FAMILY_SIZE
    }

    # Save the config dictionary as a JSON file
    def save_config_as_json(filename=f"{run_folder}/config.json"):
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"Config saved to {filename}")

# Call the function to save the config as a JSON file
    save_config_as_json()

if __name__ == "__main__":
    main()
