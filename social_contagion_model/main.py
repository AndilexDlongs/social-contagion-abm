from model import Environment
from config import N_AGENTS, GRID_WIDTH, GRID_HEIGHT, STEPS, MAJORITY_PARTY, UND_RATIO, SEED_STRATEGY
import os
from plots import (
    plot_initial_party_affiliations,
    plot_belief_scatter_2d,
    plot_belief_scatter_3d
)

os.makedirs("results/csv", exist_ok=True)

def main():
    env = Environment(
        n=N_AGENTS, width=GRID_WIDTH, height=GRID_HEIGHT,
        seeding_strategy=SEED_STRATEGY,
        undecided_ratio=UND_RATIO,
        majority_party=MAJORITY_PARTY
    )
     # --- Before simulation ---
    plot_initial_party_affiliations(env, save=True, stage="initial")
    plot_belief_scatter_2d(env, save=True, stage="initial")
    plot_belief_scatter_3d(env, save=True, stage="initial")

    display_df = env.datacollector.get_agent_vars_dataframe()
    avg_susc = display_df.groupby("party")["susceptibility"].mean()  # this is a float
    print("Average susceptibility:", avg_susc)


    # Run simulation steps
    for _ in range(STEPS):
        env.step()

    # Trigger late-stage force voting (e.g., last 10% of steps)
    if _ > int(0.9 * STEPS):
        for agent in env.agents:
            agent.force_vote(turnout_prob=0.6, loyalty_radius=35)


        if _ == 48 or _ == 98:  # Evaluate majority party at specific steps
            env.evaluate_majority_party()

    # Collect and save data
    model_df = env.datacollector.get_model_vars_dataframe()
    agent_df = env.datacollector.get_agent_vars_dataframe()

    model_views = model_df[["current_party_in_power", "count_in_support", "num_switches_in_support", "num_switches_in_rebellion", "avg_wealth"]]
    model_views.to_csv("results/csv/model_views.csv")

    dist_party_df = agent_df[["distance_from_party", "party", "has_interacted", "wealth"]]
    dist_party_unstacked = dist_party_df.unstack(level="AgentID")
    dist_party_unstacked.to_csv("results/csv/dist_party_unstacked.csv")

    
    agent_attributes = agent_df[["distance_from_party", "susceptibility", "party", "belief_vector", "wealth"]]
    agent_attributes.to_csv("results/csv/agent_attributes.csv")
    
    display_df2 = env.datacollector.get_agent_vars_dataframe()
    avg_susc_after = display_df2.groupby("party")["susceptibility"].mean()
    print("Average susceptibility after:", avg_susc_after)

    # agent_df.to_csv("results/csv/agent_full_data.csv")
    agent_df.to_csv("results/csv/agent_full_data.csv")


    #print("\nFinal Belief Vectors by Agent:")
    #for agent_id, row in agent_df.iterrows():
    #    print(f"Agent {agent_id}: {row['belief_vector']}")


    # --- After simulation ---
    plot_initial_party_affiliations(env, save=True, stage="final")
    plot_belief_scatter_2d(env, save=True, stage="final")
    plot_belief_scatter_3d(env, save=True, stage="final")



if __name__ == "__main__":
    main()
