import numpy as np
from model import Environment
from config import (
    N_AGENTS, GRID_WIDTH, GRID_HEIGHT, STEPS,
    MAJORITY_PARTY, UND_RATIO,
    FAMILY_MULTIPLIER, HEALTHCARE_MULTIPLIER,
    WEALTH_INFLUENCE_FACTOR, INTERACTION_MULTIPLIER,
    SUSC_PARTY_FOCUS, SUSC_FOCUS_VALUE, SUSC_OTHER_VALUE,
    WEALTH_PARTY_FOCUS, WEALTH_FOCUS_VALUE, WEALTH_OTHER_VALUE,
    CONSERVATISM_PERC, SOCIALISM_PERC, LIBERALISM_PERC,
    CONSERVATISM_STD, SOCIALISM_STD, LIBERALISM_STD,
    SICKNESS_CHANCE, MIN_FAMILY_SIZE, MAX_FAMILY_SIZE
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

os.makedirs("results/csv", exist_ok=True)

def main():
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
        susc_party_focus=SUSC_PARTY_FOCUS,
        susc_focus_value=SUSC_FOCUS_VALUE,
        susc_other_value=SUSC_OTHER_VALUE,

        # Wealth setup
        wealth_party_focus=WEALTH_PARTY_FOCUS,
        wealth_focus_value=WEALTH_FOCUS_VALUE,
        wealth_other_value=WEALTH_OTHER_VALUE,

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

    # this is where you change the nanm of the folder
    run_folder = f"results/plots/run_{env.seed or np.random.randint(1000)}"
    os.makedirs(run_folder, exist_ok=True)

    # --- Before simulation ---
    plot_initial_party_affiliations(env, save=True, stage="initial")
    plot_belief_scatter_2d(env, save=True, stage="initial")
    plot_belief_scatter_3d(env, save=True, stage="initial")

    display_df = env.datacollector.get_agent_vars_dataframe()
    avg_susc = display_df.groupby("party")["susceptibility"].mean()
    print("Average susceptibility:", avg_susc)

    # --- Simulation loop ---
    for step in range(STEPS):
        env.step()
        if step in (48, 98):  # Evaluate majority party at specific steps
            for agent in env.agents:
                agent.force_vote()
            env.evaluate_majority_party()

    # --- Save results ---
    model_df = env.datacollector.get_model_vars_dataframe()
    agent_df = env.datacollector.get_agent_vars_dataframe()

    os.makedirs("results/csv", exist_ok=True)
    model_views = model_df[[
        "current_party_in_power", "count_in_support",
        "num_switches_in_support", "num_switches_in_rebellion",
        "avg_wealth"
    ]]
    model_views.to_csv("results/csv/model_views.csv")

    dist_party_df = agent_df[["distance_from_party", "party", "has_interacted", "wealth"]]
    dist_party_df.unstack(level="AgentID").to_csv("results/csv/dist_party_unstacked.csv")

    agent_attributes = agent_df[["distance_from_party", "susceptibility", "party", "belief_vector", "wealth"]]
    agent_attributes.to_csv("results/csv/agent_attributes.csv")

    display_df2 = env.datacollector.get_agent_vars_dataframe()
    avg_susc_after = display_df2.groupby("party")["susceptibility"].mean()
    print("Average susceptibility after:", avg_susc_after)

    model_df.to_csv("results/csv/full_model_data.csv")
    agent_df.to_csv("results/csv/full_agent_data.csv")

    # --- After simulation ---
    plot_initial_party_affiliations(env, save=True, stage="final")
    plot_belief_scatter_2d(env, save=True, stage="final")
    plot_belief_scatter_3d(env, save=True, stage="final")

    plot_party_composition(env, folder=run_folder)
    plot_avg_distance_per_party(env, folder=run_folder)
    plot_distance_std(env, folder=run_folder)
    plot_support_vs_rebellion(env, folder=run_folder)
    plot_avg_susceptibility(env, folder=run_folder)
    plot_fraction_original_party(env, folder=run_folder)

if __name__ == "__main__":
    main()
