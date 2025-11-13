import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Shared colour map
PARTY_COLOURS = {
    "Conservatism": "red",
    "Socialism": "blue",
    "Liberalism": "green",
    "Undecided": "gray"
}

# --------------------------------------------------------------------
# EXISTING FUNCTIONS BELOW
# --------------------------------------------------------------------

def plot_initial_party_affiliations(env, save=False, stage="final", folder="results/plots"):
    votes = [a.party_affiliation for a in env.agents]
    counts = pd.Series(votes).value_counts().sort_index()

    plt.figure(figsize=(6, 5))
    ax = sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=[PARTY_COLOURS.get(p, "gray") for p in counts.index],
        legend=False
    )
    ax.set(
        title=f"Party Affiliations ({stage.title()})",
        xlabel="Party",
        ylabel="Number of Agents"
    )
    plt.tight_layout()

    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    if save:
        plt.savefig(f"{folder}/party_affiliations_{stage.lower()}.png", dpi=300)
    plt.show()


def plot_belief_scatter_2d(env, save=False, stage="final", folder="results/plots"):
    X = [a.LawAndOrder for a in env.agents]
    Y = [a.EconomicEquality for a in env.agents]
    Z = [a.SocialWelfare for a in env.agents]
    colours = [PARTY_COLOURS[a.party_affiliation] for a in env.agents]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].scatter(X, Y, c=colours, s=50, alpha=0.7)
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

    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    if save:
        plt.savefig(f"{folder}/belief_scatter_2d_{stage.lower()}.png", dpi=300)
    plt.show()


def plot_belief_scatter_3d(env, save=False, stage="final", folder="results/plots"):
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
        xlabel="Law & Order", ylabel="Economic Equality", zlabel="Social Welfare", 
        title=f"Agents in 3D Belief Space ({stage.title()})"
    )

    # Increase the font size of the axis labels and title
    ax.xaxis.label.set_size(16)
    ax.yaxis.label.set_size(16)
    ax.zaxis.label.set_size(16)
    ax.title.set_size(16)

    # Ensure the folder exists
    os.makedirs(folder, exist_ok=True)

    if save:
        plt.savefig(f"{folder}/belief_scatter_3d_{stage.lower()}.png", dpi=300, bbox_inches="tight")
    plt.show()



# --------------------------------------------------------------------
# PURE MATPLOTLIB EQUIVALENTS OF SOLARA PLOTS
# --------------------------------------------------------------------

def plot_party_composition(env, folder="results/plots", filename="party_composition.png"):
    """Stack plot showing % of each party at every step."""
    df = env.datacollector.get_model_vars_dataframe()
    if df.empty:
        print("[WARN] No data for party composition plot.")
        return

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

    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {filename}")


def plot_avg_distance_per_party(env, folder="results/plots", filename="avg_distance.png"):
    """Average distance from party center per party over time."""
    df_agents = env.datacollector.get_agent_vars_dataframe()
    if df_agents.empty:
        print("[WARN] No data for avg distance plot.")
        return

    df = df_agents.reset_index()
    grouped = df.groupby(["Step", "party"])["distance_from_party"].mean().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    for party_name, colour in PARTY_COLOURS.items():
        if party_name == "Undecided":
                continue  # skip Undecided
        if party_name in grouped.columns:
            ax.plot(grouped.index, grouped[party_name], label=party_name, color=colour)
    ax.set_xlabel("Step")
    ax.set_ylabel("Average Distance")
    ax.set_title("Average Distance per Party")
    ax.legend(loc="upper right")

    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {filename}")


def plot_distance_std(env, folder="results/plots", filename="distance_std.png"):
    """Standard deviation of distance from party center per party."""
    df_agents = env.datacollector.get_agent_vars_dataframe()
    if df_agents.empty:
        print("[WARN] No data for distance std plot.")
        return

    df = df_agents.reset_index()
    grouped = df.groupby(["Step", "party"])["distance_from_party"].std().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    for party_name, colour in PARTY_COLOURS.items():
        if party_name in grouped.columns:
            ax.plot(grouped.index, grouped[party_name], label=party_name, color=colour)
    ax.set_xlabel("Step")
    ax.set_ylabel("Std. Deviation of Distance")
    ax.set_title("Distance Standard Deviation per Party")
    ax.legend(loc="upper right")

    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {filename}")


def plot_avg_susceptibility(env, folder="results/plots", filename="avg_susceptibility.png"):
    """Average susceptibility per party."""
    df_agents = env.datacollector.get_agent_vars_dataframe()
    if df_agents.empty:
        print("[WARN] No data for avg susceptibility plot.")
        return

    df = df_agents.reset_index()
    grouped = df.groupby(["Step", "party"])["susceptibility"].mean().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    for party_name, colour in PARTY_COLOURS.items():
        if party_name in grouped.columns:
            ax.plot(grouped.index, grouped[party_name], label=party_name, color=colour)
    ax.set_xlabel("Step")
    ax.set_ylabel("Average Susceptibility")
    ax.set_title("Average Susceptibility per Party")
    ax.legend(loc="upper right")

    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {filename}")


def plot_support_vs_rebellion(env, folder="results/plots", filename="support_vs_rebellion.png"):
    """Stack plot of number of agents in support vs rebellion."""
    df = env.datacollector.get_model_vars_dataframe()
    if df.empty or "count_in_support" not in df.columns:
        print("[WARN] No data for support vs rebellion plot.")
        return

    df2 = df.copy()
    df2["total_agents"] = (
        df2["vote_Conservatism"]
        + df2["vote_Socialism"]
        + df2["vote_Liberalism"]
        + df2["vote_Undecided"]
    )
    df2["in_support"] = df2["count_in_support"]
    df2["in_rebellion"] = df2["total_agents"] - df2["in_support"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.stackplot(
        df2.index,
        df2["in_support"],
        df2["in_rebellion"],
        labels=["In Support", "In Rebellion"],
        colors=["blue", "red"],
        alpha=0.6,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of Agents")
    ax.set_title("Support vs Rebellion Over Time")
    ax.legend(loc="upper right")

    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {filename}")


def plot_fraction_original_party(env, folder="results/plots", filename="fraction_original_party.png"):
    """Fraction of agents still in their original party."""
    df_agents = env.datacollector.get_agent_vars_dataframe()
    if df_agents.empty:
        print("[WARN] No data for fraction original party plot.")
        return

    df = df_agents.reset_index()
    df["is_original"] = df["party"] == df["original_party"]
    count_in_party = df.groupby(["Step", "party"]).size().unstack(fill_value=0)
    count_original = df.groupby(["Step", "party"])["is_original"].sum().unstack(fill_value=0)
    frac = count_original.divide(count_in_party.where(count_in_party != 0), fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    for party_name, colour in PARTY_COLOURS.items():
        if party_name in frac.columns:
            ax.plot(frac.index, frac[party_name] * 100, label=party_name, color=colour)
    ax.set_xlabel("Step")
    ax.set_ylabel("% Still in Original Party")
    ax.set_title("Fraction of Members Remaining in Original Party")
    ax.legend(loc="upper right")

    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {filename}")
