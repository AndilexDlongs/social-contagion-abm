# plots.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
#from mpl_toolkits.mplot3d import Axes3
import os

# ensure folder structure exists
os.makedirs("results/plots", exist_ok=True)


# Shared colour map
PARTY_COLOURS = {
    "Conservatism": "red",
    "Socialism": "blue",
    "Liberalism": "green",
    "Undecided": "gray"
}

def plot_initial_party_affiliations(env, save=False, stage="final"):
    """
    Bar chart of how many agents are affiliated with each party.
    stage: 'initial', 'middle', 'final', etc. used to label saved plots.
    """
    votes = [a.party_affiliation for a in env.agents]
    counts = pd.Series(votes).value_counts().sort_index()

    plt.figure(figsize=(6, 5))
    ax = sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,       # assign hue to match palette use
        palette=[PARTY_COLOURS.get(p, "gray") for p in counts.index],
        legend=False
    )
    ax.set(
        title=f"Party Affiliations ({stage.title()})",
        xlabel="Party",
        ylabel="Number of Agents"
    )
    plt.tight_layout()

    if save:
        plt.savefig(f"results/plots/party_affiliations_{stage.lower()}.png", dpi=300)
    plt.show()


def plot_belief_scatter_2d(env, save=False, stage="final"):
    """
    Three side-by-side 2D projections of the ideological space.
    stage: 'initial', 'middle', 'final', etc.
    """
    X = [a.LawAndOrder for a in env.agents]
    Y = [a.EconomicEquality for a in env.agents]
    Z = [a.SocialWelfare for a in env.agents]
    colours = [PARTY_COLOURS[a.party_affiliation] for a in env.agents]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].scatter(X, Y, c=colours, s=50, alpha=0.7) # what does s and alpha mean
    axes[0].set(xlabel="Law & Order", ylabel="Economic Equality",
                title=f"L&O vs EconEq ({stage.title()})")

    axes[1].scatter(Y, Z, c=colours, s=50, alpha=0.7)
    axes[1].set(xlabel="Economic Equality", ylabel="Social Welfare",
                title=f"EconEq vs SocW ({stage.title()})")

    axes[2].scatter(X, Z, c=colours, s=50, alpha=0.7)
    axes[2].set(xlabel="Law & Order", ylabel="Social Welfare",
                title=f"L&O vs SocW ({stage.title()})")

    for ax in axes:
        ax.grid(True)

    plt.tight_layout()
    if save:
        plt.savefig(f"results/plots/belief_scatter_2d_{stage.lower()}.png", dpi=300) # whats dpi
    plt.show()


def plot_belief_scatter_3d(env, save=False, stage="final"):
    """
    3D scatter of agents and party spheres in ideological space.
    stage: 'initial', 'middle', 'final', etc.
    """
    X = [a.LawAndOrder for a in env.agents]
    Y = [a.EconomicEquality for a in env.agents]
    Z = [a.SocialWelfare for a in env.agents]
    colours = [PARTY_COLOURS[a.party_affiliation] for a in env.agents]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=colours, s=40, alpha=0.7)

    def draw_sphere(center, radius, colour):
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        ax.plot_surface(x, y, z, color=colour, alpha=0.1, linewidth=0)

    for p in env.parties:
        draw_sphere(p.center_vector(), p.radius, PARTY_COLOURS[p.name])
        ax.scatter(*p.center_vector(), c=PARTY_COLOURS[p.name],
                   marker="^", s=200, edgecolors="k")

    ax.set(
        xlabel="Law & Order",
        ylabel="Economic Equality",
        zlabel="Social Welfare",
        title=f"Agents in 3D Belief Space ({stage.title()})"
    )

    if save:
        plt.savefig(f"results/plots/belief_scatter_3d_{stage.lower()}.png", dpi=300)
    plt.show()
