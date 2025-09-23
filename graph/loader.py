import networkx as nx

def load_demo_graph(seed: int = 0):
    # small random graph for quick tests
    return nx.erdos_renyi_graph(n=100, p=0.08, seed=seed)
