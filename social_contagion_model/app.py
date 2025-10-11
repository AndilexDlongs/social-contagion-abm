from mesa.visualization import SolaraViz, SpaceRenderer
from model import Environment
from plots import agent_portrayal

social_contagion = Environment(n=5, width=3, height=3)
renderer = SpaceRenderer(model=social_contagion, backend="matplotlib")
renderer.draw_structure(lw=2, ls="solid", color="black", alpha=0.1)
renderer.draw_agents(agent_portrayal)

page = SolaraViz(
    social_contagion,
    renderer,
    name="Social Contagion ABM",
)
