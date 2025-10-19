def num_switches(model):
    return sum(1 for a in model.agents if getattr(a, "switched_this_step", False))

def num_interactions(model):
    # count how many agents have interacted this step
    interacted = sum(1 for a in model.agents if getattr(a, "has_interacted", False))
    # each interaction sets two agents → divide by 2
    return interacted // 2

def count_in_support(model):
    return sum(1 for a in model.agents if a.in_support)

def num_switches_in_support(model):
    return sum(1 for a in model.agents if getattr(a, "switched_in_support", False))

def num_switches_in_rebellion(model):
    return sum(1 for a in model.agents if getattr(a, "switched_in_rebellion", False)) 

def vote_count_for(model, party_name):
    """Count how many agents are affiliated with `party_name`."""
    return sum(1 for a in model.agents if a.party_affiliation == party_name)

def vote_counts(model):
    """Return dict of counts per party (useful but not directly plotable)."""
    counts = {}
    for p in model.parties:
        counts[p.name] = vote_count_for(model, p.name)
    # You might also count “Undecided”
    counts["Undecided"] = sum(1 for a in model.agents if a.party_affiliation == "Undecided")
    return counts

def count_Conservatism(model):
    return vote_count_for(model, "Conservatism")

def count_Socialism(model):
    return vote_count_for(model, "Socialism")

def count_Liberalism(model):
    return vote_count_for(model, "Liberalism")

def count_Undecided(model):
    return vote_count_for(model, "Undecided")

def num_interactions_in_party(model):
    """
    Count how many *interactions* occurred between agents of the same party in this step.
    We assume each interaction sets has_interacted = True on both agents and
    sets interacted_within_party = True on both if same party.
    So sum of interacted_within_party across agents, divided by 2 (to avoid double count).
    """
    total_within_flags = sum(1 for a in model.agents if getattr(a, "interacted_within_party", False))
    return total_within_flags // 2  # integer division, each pair counted twice


def num_interactions_cross_party(model):
    """
    Count how many interactions in this step were cross-party (i.e. between agents of different parties).
    Cross = total interactions − within.
    We reuse your num_interactions and subtract within.
    """
    total = num_interactions(model)
    within = num_interactions_in_party(model)
    cross = total - within
    if cross < 0:
        cross = 0
    return cross

