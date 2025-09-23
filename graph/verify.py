from typing import Dict, Any, List, Tuple, Optional, Iterable

def verify_coloring(
    G,
    coloring: Dict[int, int],
    allowed_colors: Optional[Iterable[int]] = None,
    sample_conflicts: int = 10,
) -> Dict[str, Any]:

    report: Dict[str, Any] = {}

    V = set(G.nodes())
    nodes_colored = set(coloring.keys())

    # completeness check
    missing_nodes = sorted(V - nodes_colored)
    report["missing_nodes"] = missing_nodes

    bad_nodes = [v for v, c in coloring.items() if (c is None) or (not isinstance(c, int))]
    report["bad_nodes"] = bad_nodes

    used_colors = sorted(set(coloring.values()))
    report["used_colors"] = used_colors
    report["num_used_colors"] = len(used_colors)

    # color bound check
    out_of_range_nodes: List[int] = []
    allowed_set = set(allowed_colors) if allowed_colors is not None else None
    if allowed_set is not None:
        for v, c in coloring.items():
            if c not in allowed_set:
                out_of_range_nodes.append(v)
    report["out_of_range_nodes"] = out_of_range_nodes

    # conflicts check
    conflicts: List[Tuple[int, int, int, int]] = []
    for u, v in G.edges():
        cu = coloring.get(u, None)
        cv = coloring.get(v, None)
        if cu is None or cv is None or cu == cv:
            conflicts.append((u, v, cu, cv))
    report["num_conflicts"] = len(conflicts)
    report["conflicts_sample"] = conflicts[:sample_conflicts]
    # feasible check
    feasible = (
        len(missing_nodes) == 0 and
        len(bad_nodes) == 0 and
        len(out_of_range_nodes) == 0 and
        len(conflicts) == 0
    )
    report["feasible"] = feasible
    return report


def print_check_summary(report: Dict[str, Any], prefix: str = "[Check] ") -> None:

    feasible = report.get("feasible", False)
    num_conflicts = report.get("num_conflicts", -1)
    num_used = report.get("num_used_colors", -1)
    print(f"{prefix}feasible={feasible}|used_colors={num_used}|conflicts={num_conflicts}")
    if not feasible:
        miss = report.get("missing_nodes", [])
        oor  = report.get("out_of_range_nodes", [])
        bad  = report.get("bad_nodes", [])
        sample = report.get("conflicts_sample", [])
        if miss:
            print(f"{prefix}missing_nodes(sample) ={miss[:10]}")
        if oor:
            print(f"{prefix}out_of_range_nodes(sample) ={oor[:10]}")
        if bad:
            print(f"{prefix}bad_nodes(sample) ={bad[:10]}")
        if num_conflicts > 0:
            print(f"{prefix}conflicts_sample ={sample}")
