def rank(network, damping, tolerance):

    def add_node(key):
        if key not in node_map:
            node_map[key] = set()
        if key not in node_pointers_count:
            node_pointers_count[key] = 0

    node_map = {}
    node_pointers_count = {}

    for origin, target in network:
        add_node(origin)
        add_node(target)
        if origin == target: continue

        if origin not in node_map[target]:
            node_map[target].add(origin)
            node_pointers_count[origin] += 1

    node_set = set(node_map.keys())

    for node, count in  node_pointers_count.items():
        if count == 0:  # No outflowing pointers
            node_pointers_count[node] = len(node_set)
            for n in  node_set:
                node_map[node].add(n)
            # Connect node with no outflow to each other node

    set_size = len(node_set)
    initial_val = 1/set_size

    ranking = {}
    for node in node_map.keys():
        ranking[node] = initial_val
        # initialise default rankings.

    new_ranking = {}
    difference = 1
    iterations = 0

    while difference > tolerance:
        new_ranking = {}
        for node, links in node_map.items():
            sum_term = sum(ranking[link] / node_pointers_count[link] for link in links)
            new_ranking[node] = (1-damping)/set_size + damping * sum_term

        difference = sum(abs(new_ranking[node]-ranking[node]) for node in new_ranking.keys())

        ranking = new_ranking
        iterations += 1

    return ranking
