import plotly.graph_objects as go
import numpy as np
from mesa.visualization import SolaraViz, SpaceRenderer
from model import Environment
import matplotlib.pyplot as plt
import solara
from config import N_AGENTS, GRID_WIDTH, GRID_HEIGHT, MAJORITY_PARTY, UND_RATIO, SEED_STRATEGY
from mesa.visualization.components import AgentPortrayalStyle


# -------------------------------------------------
# Colour maps
# -------------------------------------------------
PARTY_COLOURS = {
    "Conservatism": "red",
    "Socialism": "blue",
    "Liberalism": "green",
    "Undecided": "gray"
}


def agent_portrayal(agent):
    """Visualize each agent in the 2D grid."""
    color = PARTY_COLOURS.get(agent.party_affiliation, "black")
    portrayal = AgentPortrayalStyle(color=color, marker="o", size=50)
    return portrayal


# -------------------------------------------------
# Model setup
# -------------------------------------------------
social_contagion = Environment(
    n=N_AGENTS,
    width=GRID_WIDTH,
    height=GRID_HEIGHT,
    seeding_strategy=SEED_STRATEGY,
    undecided_ratio=UND_RATIO,
    majority_party=MAJORITY_PARTY
)

model_params = {
    "n": {
        "type": "SliderInt",
        "value": N_AGENTS,
        "label": "Number of agents:",
        "min": 3,
        "max": 50,
        "step": 1,
    },
    "width": GRID_WIDTH,
    "height": GRID_HEIGHT,
    "seeding_strategy": SEED_STRATEGY,
    "undecided_ratio": UND_RATIO,
    "majority_party": MAJORITY_PARTY,
}


renderer = SpaceRenderer(model=social_contagion, backend="matplotlib")
renderer.draw_structure(lw=2, ls="solid", color="black", alpha=0.1)
renderer.draw_agents(agent_portrayal)


def post_process(ax):
    """Customize the matplotlib axes after rendering."""
    ax.set_title("Social Contagion Model")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")


renderer.post_process = post_process


# -------------------------------------------------
# Shared Solara reactive binding
# -------------------------------------------------
# def bind_solara_reactive(model):
#     """Ensure the model has a reactive step counter for live updates."""
#     if not hasattr(model, "_solara_reactive_step"):
#         model._solara_reactive_step = solara.reactive(0)

#         old_step = model.step

#         def wrapped_step():
#             old_step()
#             model._solara_reactive_step.value += 1

#         model.step = wrapped_step


# # Bind before visualization setup
# bind_solara_reactive(social_contagion)

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
            model._solara_reactive_step.value += 1

        model.step = wrapped_step

# Initially bind for the original model
bind_solara_reactive(social_contagion)

# -------------------------------------------------
# Custom patch for reset in SolaraViz's controller
# -------------------------------------------------
# def patch_viz_reset(viz):
#     """
#     Monkey-patch the ModelController.do_reset inside SolaraViz so that
#     after resetting the model it also resets our reactive counter.
#     """
#     # The SolaraViz component uses an internal ModelController
#     # We can try to locate it via viz (the SolaraViz instance)
#     # and wrap its do_reset.

#     # Try to access the controller object ( Mesa’s code names it `controller` or similar)
#     mc = None
#     if hasattr(viz, "controller"):
#         mc = getattr(viz, "controller")
#     elif hasattr(viz, "model_controller"):
#         mc = getattr(viz, "model_controller")
#     else:
#         # fallback: viz itself might implement do_reset
#         mc = viz

#     original_do_reset = getattr(mc, "do_reset", None)

#     def new_do_reset():
#         # First run the original logic to reinstantiate the model
#         if original_do_reset is not None:
#             original_do_reset()

#         # After reset, the viz’s model is replaced. We need to re-bind reactive.
#         # Access new model:
#         m = None
#         # The controller may store model in an attribute `model`
#         if hasattr(mc, "model"):
#             m = mc.model
#         # Or viz might store model
#         if hasattr(viz, "model"):
#             m = viz.model

#         # If it's a reactive wrapper, get .value
#         if hasattr(m, "value"):
#             m = m.value

#         if m is not None:
#             bind_solara_reactive(m)

#     # Replace do_reset
#     setattr(mc, "do_reset", new_do_reset)


# -------------------------------------------------
# Components
# -------------------------------------------------
# @solara.component
# def PartyVotesLinePlot(model):
#     """Live line plot of vote counts over time."""
#     step = model._solara_reactive_step
#     df = model.datacollector.get_model_vars_dataframe()

#     if df.empty:
#         fig, ax = plt.subplots(figsize=(6, 5))
#         ax.set_title("Votes over Time")
#         return solara.FigureMatplotlib(fig)

#     fig, ax = plt.subplots(figsize=(6, 5))
#     ax.plot(df.index, df["vote_Conservatism"], label="Conservatism", color="red")
#     ax.plot(df.index, df["vote_Socialism"], label="Socialism", color="blue")
#     ax.plot(df.index, df["vote_Liberalism"], label="Liberalism", color="green")
#     ax.plot(df.index, df["vote_Undecided"], label="Undecided", color="gray")
#     ax.set_xlabel("Step")
#     ax.set_ylabel("Number of Agents")
#     ax.set_title(f"Party Vote Counts Over Time (Step {step.value})")
#     ax.legend()

#     return solara.FigureMatplotlib(fig)

# @solara.component
# def PartyVotesLinePlot(model):

#     if not hasattr(model, "_solara_reactive_step"):
#         fig, ax = plt.subplots(figsize=(8, 6))
#         ax.set_title(f"Initializing … at Step {model.step_count}")
#         plt.close(fig)
#         return solara.FigureMatplotlib(fig)

    
#     step = model._solara_reactive_step
#     df = model.datacollector.get_model_vars_dataframe()

#     fig, ax = plt.subplots(figsize=(8, 6))
#     try:
#         if df.empty:
#             ax.set_title("Votes over Time")
#         else:
#             ax.plot(df.index, df["vote_Conservatism"], label="Conservatism", color="red")
#             ax.plot(df.index, df["vote_Socialism"], label="Socialism", color="blue")
#             ax.plot(df.index, df["vote_Liberalism"], label="Liberalism", color="green")
#             ax.plot(df.index, df["vote_Undecided"], label="Undecided", color="gray")
#             ax.set_xlabel("Step")
#             ax.set_ylabel(f"Number of Agents {step.value}")
#             ax.set_title(f"Party Vote Counts — Step {model.step_count}")
#             ax.legend()
#     finally:
#         plt.close(fig)

#     return solara.FigureMatplotlib(fig)

# @solara.component
# def PartyVotesLinePlot(model):
#     # reactive trigger
#     _ = model._solara_reactive_step  # ensure this component re-renders when step changes

#     df = model.datacollector.get_model_vars_dataframe()
#     if df.empty:
#         fig, ax = plt.subplots(figsize=(6, 5))
#         ax.set_title("Votes over Time (no data)")
#         plt.close(fig)
#         return solara.FigureMatplotlib(fig)

#     # At this point, df has data and columns
#     fig, ax = plt.subplots(figsize=(6, 5))
#     try:
#         ax.plot(df.index, df["vote_Conservatism"], label="Conservatism", color="red")
#         ax.plot(df.index, df["vote_Socialism"], label="Socialism", color="blue")
#         ax.plot(df.index, df["vote_Liberalism"], label="Liberalism", color="green")
#         ax.plot(df.index, df["vote_Undecided"], label="Undecided", color="gray")
#         ax.set_xlabel("Step")
#         ax.set_ylabel("Number of Agents")
#         ax.set_title(f"Party Vote Counts — Step {model.step_count}")
#         ax.legend()
#     finally:
#         plt.close(fig)

#     return solara.FigureMatplotlib(fig)


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
        ax.plot(df.index, df["vote_Conservatism"], label="Conservatism", color="red")
        ax.plot(df.index, df["vote_Socialism"], label="Socialism", color="blue")
        ax.plot(df.index, df["vote_Liberalism"], label="Liberalism", color="green")
        ax.plot(df.index, df["vote_Undecided"], label="Undecided", color="gray")
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Number of Agents {step.value}")
        ax.set_title(f"Party Vote Counts — Step {model.step_count}")
        ax.legend()
    finally:
        plt.close(fig)

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
        ax.set_title("Party Composition Over Time")
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
            labels=["Conservatism", "Socialism", "Liberalism", "Undecided"],
            colors=["red", "blue", "green", "gray"],
            alpha=0.6,
        )
        ax.set_ylim(0, 100)
        ax.set_title(f"Party Composition – Step {model.step_count}")
        ax.set_xlabel("Step")
        ax.set_ylabel(f"% of Population {step.value}")
        ax.legend(loc="upper right")
    finally:
        plt.close(fig)

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
                ax.plot(grouped.index, grouped[party_name], label=party_name, color=colour)
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Average Distance {step.value}")
        ax.set_title(f"Average Distance per Party — Step {model.step_count}")
        ax.legend(loc="upper right")
    finally:
        plt.close(fig)

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
                ax.plot(grouped.index, grouped[party_name], label=party_name, color=colour)
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Average Susceptibility {step.value}")
        ax.set_title(f"Average Susceptibility per Party — Step {model.step_count}")
        ax.legend(loc="upper right")
    finally:
        plt.close(fig)

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
        plt.close(fig)

    return solara.FigureMatplotlib(fig)

@solara.component
def DistanceStdOnlyPlot(model):
    """Plot only the standard deviation of distance per party over time."""
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
                ax.plot(std_tbl.index, std_tbl[party_name], label=party_name, color=colour)
        ax.set_xlabel("Step")
        ax.set_ylabel(f"Standard Deviation of Distance {step.value}")
        ax.set_title(f"Distance Std Dev per Party — Step {model.step_count}")
        ax.legend(loc="upper right")
    finally:
        plt.close(fig)

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
        ax.set_title("Fraction original party (no data yet)")
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
                ax.plot(frac.index, frac[party_name] * 100, label=party_name, color=colour)
        ax.set_xlabel("Step")
        ax.set_ylabel(f"% still in original party {step.value}")
        ax.set_title(f"Fraction of Original Members — Step {model.step_count}")
        ax.legend(loc="upper right")
    finally:
        plt.close(fig)

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
        plt.close(fig)

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
        plt.close(fig)

    return solara.FigureMatplotlib(fig)


# -------------------------------------------------
# SolaraViz page
# -------------------------------------------------
page = SolaraViz(
    social_contagion,
    renderer,
    components=[
        (PartyCompositionPlot,0),
        (InteractionTypePlot,0),
        (AvgDistancePerPartyPlot,1),
        (DistanceStdOnlyPlot,1),
        (SupportVsRebellionStackPlot,3),
        (AvgSusceptibilityPerPartyPlot,2),
        (FractionOriginalPartyPlot,2)
    ],
    model_params=model_params,
    name="Social Contagion ABM",
)


if __name__ == "__main__":
    page
