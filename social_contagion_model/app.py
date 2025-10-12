from mesa.visualization import SolaraViz, SpaceRenderer
from model import Environment
from config import N_AGENTS, GRID_WIDTH, GRID_HEIGHT, STEPS, MAJORITY_PARTY, UND_RATIO, SEED_STRATEGY
from mesa.visualization.components import AgentPortrayalStyle


color_map = {
    0: "red",
    1: "blue",
    2: "green",
    3: "orange",
    4: "purple",
    5: "brown",
}

def agent_portrayal(agent):
    c = color_map.get(agent.unique_id, "gray")
    return AgentPortrayalStyle(color=c, marker="o", size=50)

social_contagion = Environment(
        n=N_AGENTS, width=GRID_WIDTH, height=GRID_HEIGHT,
        seeding_strategy=SEED_STRATEGY,
        undecided_ratio=UND_RATIO,
        majority_party=MAJORITY_PARTY
    )

model_params = {
    "n": {
        "type": "SliderInt",
        "value": 5,
        "label": "Number of agents:",
        "min": 3,
        "max": 6,
        "step": 1,
    },
    "width": 3,
    "height": 3
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

page = SolaraViz(
    social_contagion,
    renderer,
    #components=[BiasPlot],
    model_params=model_params,
    name="Social Contagion ABM",
)

# Expose Solara entry point
if __name__ == "__main__":
    page  # not executed here; Solara uses this object when you run "solara run app.py"
