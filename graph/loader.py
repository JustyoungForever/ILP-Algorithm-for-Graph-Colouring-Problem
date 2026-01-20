# graph/loader.py
import networkx as nx

def load_demo_graph(seed: int = 0):
    # small random graph for quick tests
    return nx.erdos_renyi_graph(n=100, p=0.5, seed=seed)


def load_rr100_d10(seed: int = 0) -> nx.Graph:
    """
    Hard instance used in experiments:
    Random regular graph with n=100, degree=10, edges=500.
    Equivalent to: nx.random_regular_graph(10, 100, seed=seed)
    """
    return nx.random_regular_graph(d=10, n=100, seed=seed)


def load_graph(graph_name: str, seed: int = 0) -> nx.Graph:
    """
    Dispatcher for main.py.
    """
    key = graph_name.strip().lower()
    if key in ("demo", "er", "erdos", "erdos_renyi"):
        return load_demo_graph(seed)
    if key in ("rr100_d10", "rr-100-10", "random_regular_100_10"):
        return load_rr100_d10(seed)
    raise ValueError(f"Unknown graph_name: {graph_name}")