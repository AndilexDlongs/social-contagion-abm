from mesa.visualization import SolaraViz, SpaceRenderer
from model import Environment
import matplotlib.pyplot as plt
import solara
from config import N_AGENTS, GRID_WIDTH, GRID_HEIGHT, MAJORITY_PARTY, UND_RATIO, SEED_STRATEGY
from mesa.visualization.components import AgentPortrayalStyle


color_map = {
    0: "red",
    1: "blue",
    2: "green",
    3: "orange",
    4: "purple",
    5: "brown",
}

# Shared colour map
PARTY_COLOURS = {
    "Conservatism": "red",
    "Socialism": "blue",
    "Liberalism": "green",
    "Undecided": "gray"
}

def agent_portrayal(agent):
    color = PARTY_COLOURS.get(agent.party_affiliation, "black")
    portrayal = AgentPortrayalStyle(color=color, marker="o", size=50)

    # --- Draw lines to recently interacted agents ---
    if hasattr(agent, "interacted_with") and agent.interacted_with:
        for other in agent.interacted_with:
            if other.cell and agent.cell:
                x1, y1 = agent.cell.pos
                x2, y2 = other.cell.pos
                portrayal.lines.append(((x1, y1), (x2, y2), "red", 1.0))  # from->to line

    return portrayal


social_contagion = Environment(
        n=N_AGENTS, width=GRID_WIDTH, height=GRID_HEIGHT,
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

@solara.component
def PartyCompositionPlot(model):
    """Live filler plot showing % of each party over time."""
    # current_step = model.steps
    current_step = len(model.datacollector.get_model_vars_dataframe())

    # use Solara reactive trigger to re-render automatically
    step = solara.use_reactive(0)

    # --- Internal helper to trigger update on model.step() ---
    def tick_updater():
        step.value += 1

    # Attach our hook to Mesa model (executed once)
    if not hasattr(model, "_solara_bound"):
        model._solara_bound = True
        old_step = model.step
        def wrapped_step():
            old_step()
            tick_updater()
        model.step = wrapped_step


    df = model.datacollector.get_model_vars_dataframe()

    if len(df) == 0:
        return solara.FigureMatplotlib(plt.figure(figsize=(6, 5)))

    # Compute shares
    df["total"] = (
        df["vote_Conservatism"]
        + df["vote_Socialism"]
        + df["vote_Liberalism"]
        + df["vote_Undecided"]
    )
    df["Conservatism%"] = df["vote_Conservatism"] / df["total"] * 100
    df["Socialism%"] = df["vote_Socialism"] / df["total"] * 100
    df["Liberalism%"] = df["vote_Liberalism"] / df["total"] * 100
    df["Undecided%"] = df["vote_Undecided"] / df["total"] * 100

    fig, ax = plt.subplots(figsize=(6, 5))

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
    ax.set_title("Party Composition Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("% of Population")
    ax.legend(loc="upper right")

    return solara.FigureMatplotlib(fig)

import matplotlib.pyplot as plt
import solara

@solara.component
def PartyVotesLinePlot(model):
    """Line plot of vote counts for each party over time."""
    df = model.datacollector.get_model_vars_dataframe()
    if df.empty:
        fig, ax = plt.subplots(figsize=(6,5))
        ax.set_title("Votes over Time")
        return solara.FigureMatplotlib(fig)

    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(df.index, df["vote_Conservatism"], label="Conservatism", color="red")
    ax.plot(df.index, df["vote_Socialism"], label="Socialism", color="blue")
    ax.plot(df.index, df["vote_Liberalism"], label="Liberalism", color="green")
    ax.plot(df.index, df["vote_Undecided"], label="Undecided", color="gray")
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of Agents")
    ax.set_title("Party Vote Counts Over Time")
    ax.legend()

    return solara.FigureMatplotlib(fig)



page = SolaraViz(
    social_contagion,
    renderer,
    components=[PartyVotesLinePlot],  
    model_params=model_params,
    name="Social Contagion ABM",
)

# Expose Solara entry point
if __name__ == "__main__":
    page  # not executed here; Solara uses this object when you run "solara run app.py"
