import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from mesa.visualization import SolaraViz, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle
from social_contagion_model.model import Environment
import matplotlib.pyplot as plt
import solara
from social_contagion_model.config import (
    N_AGENTS, GRID_WIDTH, GRID_HEIGHT,
    MAJORITY_PARTY, UND_RATIO,
    FAMILY_MULTIPLIER, HEALTHCARE_MULTIPLIER,
    WEALTH_INFLUENCE_FACTOR, INTERACTION_MULTIPLIER,
    CONSERVATISM_PERC, SOCIALISM_PERC, LIBERALISM_PERC,
    CONSERVATISM_STD, SOCIALISM_STD, LIBERALISM_STD,
    SICKNESS_CHANCE, MIN_FAMILY_SIZE, MAX_FAMILY_SIZE,
    CONSERVATISM_SUSC, SOCIALISM_SUSC, LIBERALISM_SUSC, 
    CONSERVATISM_WEALTH, SOCIALISM_WEALTH, LIBERALISM_WEALTH 
)

# -------------------------------------------------
# Colour maps
# -------------------------------------------------
PARTY_COLOURS = {
    "Conservatism": "red",
    "Socialism": "blue",
    "Liberalism": "green",
    "Undecided": "gray"
}

# -------------------------------------------------
# Display name map (for legend/labels only)
# -------------------------------------------------
DISPLAY_NAMES = {
    "Conservatism": "Red",
    "Socialism": "Blue",
    "Liberalism": "Green",
    "Undecided": "Undecided"
}

def dynamic_agent_portrayal(agent):
    """Always get the live colour from the agent's current party."""
    return AgentPortrayalStyle(
        color=PARTY_COLOURS.get(agent.party_affiliation, "black"),
        marker="o",
        size=50
    )

# -------------------------------------------------
# Model setup
# -------------------------------------------------
social_contagion = Environment(
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

renderer = SpaceRenderer(model=social_contagion, backend="matplotlib")
renderer.draw_structure(lw=2, ls="solid", color="black", alpha=0.1)
# renderer.draw_agents(agent_portrayal)

renderer.draw_agents(dynamic_agent_portrayal)
renderer.dynamic = True

def post_process(ax):
    """Customize the matplotlib axes after rendering."""
    ax.set_title("Social Contagion Model")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")

    # # 🔒 Keep the grid constant (fix axis limits)
    # ax.set_xlim(0, GRID_WIDTH)
    # ax.set_ylim(0, GRID_HEIGHT)


renderer.post_process = post_process

# -------------------------------------------------
# Reactive binding for model.step
# -------------------------------------------------
def bind_solara_reactive(model):
    """Attach or reset reactive step counter on the model."""
    # If it already exists, reset it to zero
    if getattr(model, "_solara_reactive_step", None) is not None:
        model._solara_reactive_step.value = 0
    else:
        model._solara_reactive_step = solara.reactive(0)

        old_step = model.step

        def wrapped_step():
            old_step()
            # Force redraw of live agents only
            renderer.clear()
            renderer.draw_agents(dynamic_agent_portrayal)
            
            model._solara_reactive_step.value += 1


        model.step = wrapped_step

# Initially bind for the original model
bind_solara_reactive(social_contagion)

@solara.component
def Live3DBeliefScatter(model):
    """Interactive 3D scatter of agents in ideological space (updates live)."""
    if not hasattr(model, "_solara_reactive_step"):
        return solara.FigurePlotly(go.Figure())

    step = model._solara_reactive_step

    X = [a.LawAndOrder for a in model.agents]
    Y = [a.EconomicEquality for a in model.agents]
    Z = [a.SocialWelfare for a in model.agents]
    colours = [PARTY_COLOURS[a.party_affiliation] for a in model.agents]

    scatter = go.Scatter3d(
        x=X, y=Y, z=Z,
        mode="markers",
        marker=dict(size=2, color=colours, opacity=0.8),
        name="Agents"
    )

    sphere_traces = []
    for p in model.parties:
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        x = p.LawAndOrder + p.radius * np.cos(u) * np.sin(v)
        y = p.EconomicEquality + p.radius * np.sin(u) * np.sin(v)
        z = p.SocialWelfare + p.radius * np.cos(v)
        sphere_traces.append(
            go.Surface(
                x=x, y=y, z=z,
                showscale=False,
                opacity=0.1,
                surfacecolor=np.ones_like(x),
                colorscale=[[0, PARTY_COLOURS[p.name]], [1, PARTY_COLOURS[p.name]]],
                name=p.name
            )
        )

    centers = go.Scatter3d(
        x=[p.LawAndOrder for p in model.parties],
        y=[p.EconomicEquality for p in model.parties],
        z=[p.SocialWelfare for p in model.parties],
        mode="markers",
        marker=dict(symbol="diamond", size=8,
                    color=[PARTY_COLOURS[p.name] for p in model.parties],
                    line=dict(width=1, color="black")),
        name="Party Centers"
    )

    fig = go.Figure(data=[scatter, centers] + sphere_traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="Law & Order",
            yaxis_title="Economic Equality",
            zaxis_title="Social Welfare"
        ),
        title=f"3D Belief Space (Step {step.value})",
        margin=dict(l=0, r=0, b=0, t=40),
        height=500,
        showlegend=True
    )

    return solara.FigurePlotly(fig)

@solara.component
def PartyVotesLinePlot(model):
    """Live line plot of vote counts over time, with same reactive logic as PartyCompositionPlot."""
    # reactive trigger for re-render
    step = solara.use_reactive(0)

    def tick_updater():
        step.value += 1

    # Attach hook to model.step once
    if not getattr(model, "_solara_bound_votes", False):
        model._solara_bound_votes = True
        old_step = model.step
        def wrapped_step():
            old_step()
            tick_updater()
        model.step = wrapped_step

    df = model.datacollector.get_model_vars_dataframe()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Party Vote Counts Over Time")
        plt.close(fig)
        return solara.FigureMatplotlib(fig)

    # Now plot vote counts
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.plot(df.index, df["vote_Conservatism"], label="Red", color="red")
        ax.plot(df.index, df["vote_Socialism"], label="Blue", color="blue")
        ax.plot(df.index, df["vote_Liberalism"], label="Green", color="green")
        ax.plot(df.index, df["vote_Undecided"], label="Undecided", color="gray")
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Number of Agents {step.value}")
        ax.set_title(f"Party Vote Counts — Step {model.step_count}")
        ax.legend()
    finally:
        pass

    return solara.FigureMatplotlib(fig)

@solara.component
def PartyCompositionPlot(model):
    """Live filler plot showing % of each party at every step."""
    # reactive trigger for re-render
    step = solara.use_reactive(0)

    def tick_updater():
        step.value += 1

    # Attach hook to model.step once
    if not getattr(model, "_solara_bound", False):
        model._solara_bound = True
        old_step = model.step
        def wrapped_step():
            old_step()
            tick_updater()
        model.step = wrapped_step

    df = model.datacollector.get_model_vars_dataframe()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Party Growth Over Time")
        plt.close(fig)
        return solara.FigureMatplotlib(fig)

    # compute percentages
    df = df.copy()
    df["total"] = (
        df["vote_Conservatism"]
        + df["vote_Socialism"]
        + df["vote_Liberalism"]
        + df["vote_Undecided"]
    ).replace(0, 1)
    df["Conservatism%"] = df["vote_Conservatism"] / df["total"] * 100
    df["Socialism%"] = df["vote_Socialism"] / df["total"] * 100
    df["Liberalism%"] = df["vote_Liberalism"] / df["total"] * 100
    df["Undecided%"] = df["vote_Undecided"] / df["total"] * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.stackplot(
            df.index,
            df["Conservatism%"],
            df["Socialism%"],
            df["Liberalism%"],
            df["Undecided%"],
            labels=["Red", "Blue", "Green", "Undecided"],
            colors=["red", "blue", "green", "gray"],
            alpha=0.6,
        )
        ax.set_ylim(0, 100)
        ax.set_title(f"Party Growth After {model.step_count} Steps")
        ax.set_xlabel("Step")
        ax.set_ylabel(f"% of Population {step.value}")
        ax.legend(loc="upper right")
    finally:
        pass

    return solara.FigureMatplotlib(fig)

@solara.component
def AvgDistancePerPartyPlot(model):
    """Plot the average distance from party center per party over time."""
    # reactive trigger for re-render
    step = solara.use_reactive(0)

    def tick_updater():
        step.value += 1

    # Attach hook to model.step once
    if not getattr(model, "_solara_bound_avgdist", False):
        model._solara_bound_avgdist = True
        old_step = model.step
        def wrapped_step():
            old_step()
            tick_updater()
        model.step = wrapped_step

    # Agent-level data
    # Use the agent reporter dataframe
    df_agents = model.datacollector.get_agent_vars_dataframe()
    # That returns a MultiIndex (step, agent id) → with columns including “distance_from_party” and “party”
    if df_agents.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Avg Distance per Party (no data yet)")
        plt.close(fig)
        return solara.FigureMatplotlib(fig)

    # Compute average distance per party, by time step
    # Reset index so we can operate
    df2 = df_agents.reset_index()  # columns: “Step”, “AgentID”, “distance_from_party”, “party”, etc.

    # We group by Step and party affiliation
    # We’ll build a wide table: index = Step, columns = each party name, values = average distance
    grouped = df2.groupby(["Step", "party"])["distance_from_party"].mean().unstack(fill_value=0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        for party_name, colour in PARTY_COLOURS.items():
            if party_name == "Undecided":
                continue  # skip Undecided
            if party_name in grouped.columns:
                ax.plot(grouped.index, grouped[party_name],
                label=DISPLAY_NAMES.get(party_name, party_name),
                color=colour)
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Average Distance {step.value}")
        ax.set_title(f"Supporters Drift After {model.step_count} Steps")
        ax.legend(loc="upper right")
    finally:
        pass

    return solara.FigureMatplotlib(fig)

@solara.component
def AvgSusceptibilityPerPartyPlot(model):
    """Plot the average susceptibility per party (including Undecided)."""
    # reactive trigger for re-render
    step = solara.use_reactive(0)

    def tick_updater():
        step.value += 1

    # Attach hook to model.step once
    if not getattr(model, "_solara_bound_avgsusc", False):
        model._solara_bound_avgsusc = True
        old_step = model.step
        def wrapped_step():
            old_step()
            tick_updater()
        model.step = wrapped_step

    # Agent-level data
    df_agents = model.datacollector.get_agent_vars_dataframe()
    if df_agents.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Avg Susceptibility per Party (no data yet)")
        plt.close(fig)
        return solara.FigureMatplotlib(fig)

    # Reset index so we have Step, AgentID, etc.
    df2 = df_agents.reset_index()

    # Compute mean susceptibility by (Step, party)
    grouped = df2.groupby(["Step", "party"])["susceptibility"].mean().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        for party_name, colour in PARTY_COLOURS.items():
            if party_name in grouped.columns:
                ax.plot(grouped.index, grouped[party_name],
                label=DISPLAY_NAMES.get(party_name, party_name),
                color=colour)
        ax.set_ylabel(f"Average Susceptibility {step.value}")
        ax.set_title(f"Agents' Desire to Change Party After {model.step_count} Steps")
        ax.legend(loc="upper right")
    finally:
        pass

    return solara.FigureMatplotlib(fig)

@solara.component
def SupportVsRebellionStackPlot(model):
    """Stack plot: number of agents in support vs in rebellion over time."""
    # reactive trigger for re-render
    step = solara.use_reactive(0)

    def tick_updater():
        step.value += 1

    # Attach hook to model.step once
    if not getattr(model, "_solara_bound_support_stack", False):
        model._solara_bound_support_stack = True
        old_step = model.step
        def wrapped_step():
            old_step()
            tick_updater()
        model.step = wrapped_step

    # Use the model-level reporter “count_in_support” and (implicitly) number not in support
    # Or derive rebellion = total agents − in_support
    df = model.datacollector.get_model_vars_dataframe()
    if df.empty or "count_in_support" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Support vs Rebellion Over Time")
        plt.close(fig)
        return solara.FigureMatplotlib(fig)

    # Copy dataframe
    df2 = df.copy()
    # Total number of agents at each step = sum of all party vote counts (or add a reporter)
    df2["total_agents"] = (
        df2["vote_Conservatism"]
        + df2["vote_Socialism"]
        + df2["vote_Liberalism"]
        + df2["vote_Undecided"]
    )
    # in_support is given
    df2["in_support"] = df2["count_in_support"]
    # rebellion is the rest
    df2["in_rebellion"] = df2["total_agents"] - df2["in_support"]

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.stackplot(
            df2.index,
            df2["in_support"],
            df2["in_rebellion"],
            labels=["In Support", "In Rebellion"],
            colors=["blue", "red"],
            alpha=0.6,
        )
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Number of Agents {step.value}")
        ax.set_title(f"Support vs Rebellion — Step {model.step_count}")
        ax.legend(loc="upper right")
    finally:
        pass

    return solara.FigureMatplotlib(fig)

@solara.component
def DistanceStdOnlyPlot(model):
    """Plot only the standard deviation of distance
      per party over time."""
    # reactive trigger for re-render
    step = solara.use_reactive(0)

    def tick_updater():
        step.value += 1

    # Attach hook to model.step once
    if not getattr(model, "_solara_bound_stdonly", False):
        model._solara_bound_stdonly = True
        old_step = model.step
        def wrapped_step():
            old_step()
            tick_updater()
        model.step = wrapped_step

    df_agents = model.datacollector.get_agent_vars_dataframe()
    if df_agents.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Distance STD per Party (no data yet)")
        plt.close(fig)
        return solara.FigureMatplotlib(fig)

    df2 = df_agents.reset_index()
    # Group by step and party, compute std deviation
    std_tbl = df2.groupby(["Step", "party"])["distance_from_party"].std().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        for party_name, colour in PARTY_COLOURS.items():
            if party_name == "Undecided":
                continue  # skip Undecided
            if party_name in std_tbl.columns:
                ax.plot(std_tbl.index, std_tbl[party_name],
                        label=DISPLAY_NAMES.get(party_name, party_name),
                        color=colour)
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Standard Deviation of Distance {step.value}")
        ax.set_title(f"Party Spread After {model.step_count} Steps")
        ax.legend(loc="upper right")
    finally:
        pass

    return solara.FigureMatplotlib(fig)

@solara.component
def FractionOriginalPartyPlot(model):
    """Plot the fraction (%) of original‐party members still in their original party, per party, over time."""
    # reactive trigger for re-render
    step = solara.use_reactive(0)

    def tick_updater():
        step.value += 1

    # Attach hook to model.step once
    if not getattr(model, "_solara_bound_fraction_original", False):
        model._solara_bound_fraction_original = True
        old_step = model.step
        def wrapped_step():
            old_step()
            tick_updater()
        model.step = wrapped_step

    # Agent‐level data
    df_agents = model.datacollector.get_agent_vars_dataframe()
    if df_agents.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Fraction of Original Party Members (no data yet)")
        plt.close(fig)
        return solara.FigureMatplotlib(fig)

    df2 = df_agents.reset_index()  # columns: Step, AgentID, original_party, party, etc.

    # For each step and each party, compute:
    #   count of agents whose current party == original_party, divided by total agents currently in that party
    # We group by Step and party, but we need both counts and matches.

    grouped = df2.groupby(["Step", "party"])
    # total per (step,party)
    count_in_party = grouped.size().unstack(fill_value=0)
    # count of original‐still ones
    # mask where party == original_party, then group similarly
    df2["is_original"] = df2["party"] == df2["original_party"]
    grouped_orig = df2.groupby(["Step", "party"])["is_original"].sum().unstack(fill_value=0)

    # Now fraction = grouped_orig / count_in_party
    frac = grouped_orig.divide(count_in_party.where(count_in_party != 0), fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        for party_name, colour in PARTY_COLOURS.items():
            if party_name in frac.columns:
                ax.plot(frac.index, frac[party_name] * 100,
                        label=DISPLAY_NAMES.get(party_name, party_name),
                        color=colour)

        ax.set_xlabel("Step")
        ax.set_ylabel(f"% still in original party {step.value}")
        ax.set_title(f"Fraction of Original Members — Step {model.step_count}")
        ax.legend(loc="upper right")
    finally:
        pass

    return solara.FigureMatplotlib(fig)

@solara.component
def InteractionsWithinVsCrossPlot(model):
    """Plot number (or proportion) of interactions that are within party vs cross-party over time."""
    # reactive trigger for re-render
    step = solara.use_reactive(0)

    def tick_updater():
        step.value += 1

    if not getattr(model, "_solara_bound_interactions", False):
        model._solara_bound_interactions = True
        old = model.step
        def wrapped():
            old()
            tick_updater()
        model.step = wrapped

    # agent‐level data
    df_agents = model.datacollector.get_agent_vars_dataframe()
    if df_agents.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Interactions (within vs cross) — no data yet")
        plt.close(fig)
        return solara.FigureMatplotlib(fig)

    df2 = df_agents.reset_index()

    # For each step, sum how many agents had interacted and how many of those interactions were within party.
    grouped = df2.groupby("Step")
    # sum of “interacted_within_party” (assume it’s 0/1 or count) gives how many within-party interactions (or “within” events)
    within_sum = grouped["interacted_within_party"].sum()
    # total interactions (or interacting agents) could be those with has_interacted == True
    total_interactions = grouped["has_interacted"].sum()

    # cross = total_interactions − within_sum
    cross = total_interactions - within_sum

    # Option: plot raw counts or proportions
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.plot(within_sum.index, within_sum, label="Within-Party Interactions", color="blue")
        ax.plot(cross.index, cross, label="Cross-Party Interactions", color="red")
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Number of Interactions {step.value}")
        ax.set_title(f"Within vs Cross-Party Interactions — Step {model.step_count}")
        ax.legend(loc="upper right")
    finally:
        pass

    return solara.FigureMatplotlib(fig)

@solara.component
def InteractionTypePlot(model):
    """Plot number of within-party vs cross-party interactions over time."""
    # reactive trigger
    step = solara.use_reactive(0)

    def tick_updater():
        step.value += 1

    if not getattr(model, "_solara_bound_interaction_plot", False):
        model._solara_bound_interaction_plot = True
        old = model.step
        def wrapped():
            old()
            tick_updater()
        model.step = wrapped

    df = model.datacollector.get_model_vars_dataframe()
    if df.empty or "num_interactions_in_party" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Interaction types (no data yet)")
        plt.close(fig)
        return solara.FigureMatplotlib(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.plot(df.index, df["num_interactions_in_party"], label="Within Party", color="blue")
        ax.plot(df.index, df["num_interactions_cross_party"], label="Cross Party", color="red")
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Number of Interactions {step.value}")
        ax.set_title(f"Interaction Types — Step {model.step_count}")
        ax.legend(loc="upper right")
    finally:
        pass

    return solara.FigureMatplotlib(fig)


@solara.component
def PartyStatsTable(model):
    """Live table showing vote counts, average wealth, and deaths per party."""
    step = solara.use_reactive(0)

    def tick_updater():
        step.value += 1

    # Bind reactivity to model.step
    if not getattr(model, "_solara_bound_stats_table", False):
        model._solara_bound_stats_table = True
        old_step = model.step

        def wrapped_step():
            old_step()
            tick_updater()
        model.step = wrapped_step

    # Access agent-level data
    df_agents = model.datacollector.get_agent_vars_dataframe()
    if df_agents.empty:
        return solara.Text("No data yet — run the model for a few steps.")

    df2 = df_agents.reset_index()
    latest_step = df2["Step"].max()
    current = df2[df2["Step"] == latest_step]

    # --- Compute statistics ---
    parties = ["Conservatism", "Socialism", "Liberalism"]
    rows = []
    for party in parties:
        party_agents = current[current["party"] == party]
        if len(party_agents) == 0:
            votes = 0
            avg_wealth = 0
            deaths = 0
        else:
            votes = len(party_agents)
            avg_wealth = party_agents["wealth"].mean()
            deaths = len(party_agents[party_agents["alive"] == False]) if "alive" in party_agents else 0

        rows.append({
            "Party": party,
            "Votes": int(votes),
            "Avg Wealth": round(avg_wealth, 2) if not np.isnan(avg_wealth) else 0,
            "Deaths": int(deaths),
            "Step": step.value
        })

    df_summary = pd.DataFrame(rows)

    return solara.DataFrame(df_summary)

model_params = {
    "n": { 
        "type": "SliderInt", 
        "value": N_AGENTS, 
        "label": "Number of agents", 
        "min": 50, 
        "max": 1000,
        "step": 1,
    },
    "width": {
        "type": "SliderInt",
        "value": GRID_WIDTH,
        "label": "Grid Width",
        "min": 30,
        "max": 100,
        "step": 1,
    },
    "height": {
        "type": "SliderInt",
        "value": GRID_HEIGHT,
        "label": "Grid Height",
        "min": 30,
        "max": 100,
        "step": 1,
    },
    "conservatism_perc": {
        "type": "SliderFloat",
        "value": CONSERVATISM_PERC,
        "label": "Proportion of Red Agents",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "socialism_perc": {
        "type": "SliderFloat",
        "value": SOCIALISM_PERC,
        "label": "Proportion of Blue Agents",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "liberalism_perc": {
        "type": "SliderFloat",
        "value": LIBERALISM_PERC,
        "label": "Proportion of Green Agents",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "conservatism_wealth": {
        "type": "SliderFloat",
        "value": CONSERVATISM_WEALTH,
        "label": "Wealth class of Red Agents",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "socialism_wealth": {
        "type": "SliderFloat",
        "value": SOCIALISM_WEALTH,
        "label": "Wealth class of Blue Agents",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "liberalism_wealth": {
        "type": "SliderFloat",
        "value": LIBERALISM_WEALTH,
        "label": "Wealth class of Green Agents",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "conservatism_susc": {
        "type": "SliderFloat",
        "value": CONSERVATISM_SUSC,
        "label": "Proportion of stubborn Red Agents",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "socialism_susc": {
        "type": "SliderFloat",
        "value": SOCIALISM_SUSC,
        "label": "Proportion of stubborn Blue Agents",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "liberalism_susc": {
        "type": "SliderFloat",
        "value": LIBERALISM_SUSC,
        "label": "Proportion of stubborn Green Agents",
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
    },
    "max_family_size": {
        "type": "SliderInt",
        "value": MAX_FAMILY_SIZE,
        "label": "Largest family size",
        "min": 3,
        "max": 10,
        "step": 1,
    },
    "sickness_chance": {
        "type": "SliderFloat",
        "value": SICKNESS_CHANCE,
        "label": "Chances of falling sick",
        "min": 0.01,
        "max": 0.30,
        "step": 0.01,
    },
    "undecided_ratio": UND_RATIO,
    "majority_party": MAJORITY_PARTY,
}

# Keep your page definition exactly as it is
page = SolaraViz(
    social_contagion,
    renderer,
    components=[
        (PartyCompositionPlot, 0),
        # (InteractionTypePlot, 3),
        (AvgDistancePerPartyPlot, 1),
        (DistanceStdOnlyPlot, 1),
        # (SupportVsRebellionStackPlot, 3),
        (AvgSusceptibilityPerPartyPlot, 2),
        (FractionOriginalPartyPlot, 2),
    ],
    model_params=model_params,
    name="Social Contagion ABM",
)

if __name__ == "__main__":
    page