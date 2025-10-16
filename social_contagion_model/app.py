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
if not hasattr(social_contagion, "_solara_reactive_step"):
    social_contagion._solara_reactive_step = solara.reactive(0)

    old_step = social_contagion.step
    def wrapped_step():
        old_step()
        social_contagion._solara_reactive_step.value += 1

    social_contagion.step = wrapped_step


# -------------------------------------------------
# Components
# -------------------------------------------------
@solara.component
def PartyVotesLinePlot(model):
    """Live line plot of vote counts over time."""
    step = model._solara_reactive_step
    df = model.datacollector.get_model_vars_dataframe()

    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title("Votes over Time")
        return solara.FigureMatplotlib(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(df.index, df["vote_Conservatism"], label="Conservatism", color="red")
    ax.plot(df.index, df["vote_Socialism"], label="Socialism", color="blue")
    ax.plot(df.index, df["vote_Liberalism"], label="Liberalism", color="green")
    ax.plot(df.index, df["vote_Undecided"], label="Undecided", color="gray")
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of Agents")
    ax.set_title(f"Party Vote Counts Over Time (Step {step.value})")
    ax.legend()

    return solara.FigureMatplotlib(fig)


@solara.component
def Live3DBeliefScatter(model):
    """Interactive 3D scatter of agents in ideological space (updates live)."""
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


# -------------------------------------------------
# SolaraViz page
# -------------------------------------------------
page = SolaraViz(
    social_contagion,
    renderer,
    components=[PartyVotesLinePlot, Live3DBeliefScatter],
    model_params=model_params,
    name="Social Contagion ABM",
)


if __name__ == "__main__":
    page
