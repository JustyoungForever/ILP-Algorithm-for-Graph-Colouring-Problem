# visualisierung/draw.py
from __future__ import annotations
import os, re
from typing import Dict, Tuple, Set, List, Optional, Iterable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numbers
# Zähler für Schnappschüsse je (Schritt, Runde) → int
_SHOT_COUNTER = {}  # Schlüssel=(step_clean, round_id) -> int

PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#F4A261", "#8E44AD", "#F1C40F",
    "#7F8C8D", "#1ABC9C", "#D35400", "#27AE60", "#C2185B", "#5D6D7E",
]

# Layout-Cache pro Graph-Signatur
_POS_CACHE: Dict[int, Dict] = {}

def _graph_signature(G: nx.Graph) -> int:
    """Erzeuge eine stabile Signatur aus Knoten- und Kantenmengen, um Layouts zu cachen."""
    nodes_sig = tuple(sorted(G.nodes()))
    edges_sig = tuple(sorted(tuple(sorted(e)) for e in G.edges()))
    return hash((nodes_sig, edges_sig))

def _sanitize_step(step: str) -> str:
    """Schrittname bereinigen: Kleinbuchstaben, [a-z0-9-_], Mehrfach-Bindestriche zusammenfassen."""
    s = step.strip().lower()
    s = re.sub(r"[^a-z0-9\-_]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "step"

def ensure_outdir(out_dir: str) -> None:
    """Ausgabeverzeichnis sicherstellen (rekursiv anlegen)."""
    os.makedirs(out_dir, exist_ok=True)

def color_for_index(idx: int) -> str:
    """Farbwert aus Palette anhand des Farbindizes zurückgeben (zyklisch)."""
    return PALETTE[idx % len(PALETTE)]

def _get_layout(G: nx.Graph, seed: int = 42) -> Dict:
    """Spring-Layout je Graph-Signatur cachen, damit aufeinanderfolgende Bilder konsistent sind."""
    sig = _graph_signature(G)
    if sig in _POS_CACHE:
        return _POS_CACHE[sig]
    pos = nx.spring_layout(G, seed=seed)
    _POS_CACHE[sig] = pos
    return pos

def visualize_coloring(
    G: nx.Graph,
    coloring: Dict[int, int],
    step: str,
    round_id: int,
    out_dir: str = "visualisierung/picture",
    layout_seed: int = 42,
    pos: Optional[Dict] = None,
    show_labels: bool = True,
    figure_size: Tuple[float, float] = (8.0, 6.0),
    dpi: int = 220,
    *,
    only_colored: bool = True,
    allowed_colors: Optional[Iterable[int]] = None,
    conflict_nodes_fill_black: bool = True,
) -> str:
    """
    Visualisiert (standardmäßig) nur den bereits gefärbten Teilgraphen und hebt Unzulässigkeiten hervor:
      - Kanten zwischen gleichfarbigen Nachbarn → schwarz; deren Endknoten → schwarze Füllung
      - Knoten mit Farben außerhalb des erlaubten Spektrums → schwarze Füllung
      - Alle anderen gefärbten Knoten → Palettenfarben
      - Ungfärbte Knoten sowie Kanten mit mindestens einem ungefärbten Endpunkt → werden nicht gezeichnet
    """
    ensure_outdir(out_dir)
    step_clean = _sanitize_step(step)

    # Auswahl der zu zeichnenden Knoten: nur Knoten mit ganzzahliger Farbe (>= 0)
    # colored_nodes = {v for v, c in coloring.items() if isinstance(c, int) and c is not None and c >= 0}
    colored_nodes = {
        v for v, c in coloring.items()
        if c is not None and isinstance(c, numbers.Integral) and int(c) >= 0
    }


    if only_colored:
        nodes_to_draw = colored_nodes
    else:
        nodes_to_draw = set(G.nodes())

    # Kanten: nur zeichnen, wenn beide Endpunkte in nodes_to_draw liegen
    edges_to_draw = [(u, v) for (u, v) in G.edges() if u in nodes_to_draw and v in nodes_to_draw]

    allowed_set = set(allowed_colors) if allowed_colors is not None else None

    # Konfliktkanten: benachbarte Knoten mit identischer Farbe (Prüfung nur im gefärbten Teilgraphen)
    conflict_edges: List[Tuple[int, int]] = []
    for (u, v) in edges_to_draw:
        cu = coloring.get(u, None)
        cv = coloring.get(v, None)
        if cu is not None and cv is not None and cu == cv:
            conflict_edges.append((u, v))

    # Konfliktknoten: Endpunkte von Konfliktkanten
    conflict_nodes: Set[int] = set()
    for (u, v) in conflict_edges:
        conflict_nodes.add(u); conflict_nodes.add(v)

    # Knoten mit Farben außerhalb des erlaubten Spektrums (falls allowed_colors gesetzt ist)
    oor_nodes: Set[int] = set()
    if allowed_set is not None:
        for v in nodes_to_draw:
            c = coloring.get(v, None)
            if c not in allowed_set:
                oor_nodes.add(v)

    # Koordinaten vorbereiten (Layout ggf. aus dem Cache)
    if pos is None:
        pos = _get_layout(G, seed=layout_seed)

    # Zeichnen

    plt.figure(figsize=figure_size, dpi=dpi)

    # Nicht-Konfliktkanten (hellgrau)
    non_conflict_edges = [e for e in edges_to_draw if e not in conflict_edges and (e[1], e[0]) not in conflict_edges]
    if non_conflict_edges:
        nx.draw_networkx_edges(G, pos, edgelist=non_conflict_edges, width=0.6, alpha=0.25, edge_color="#CCCCCC")

    # Konfliktkanten (schwarz)
    if conflict_edges:
        nx.draw_networkx_edges(G, pos, edgelist=conflict_edges, width=1.6, alpha=0.95, edge_color="black")

    # Knotenkolorierung:
    #   - Konflikt-/OOR-Knoten → schwarz gefüllt
    #   - sonst Palettenfarbe (bzw. hellgrau für „nicht gesetzt“)
    node_colors = []
    node_edge_colors = []
    nodes_sorted = sorted(list(nodes_to_draw))
    for v in nodes_sorted:
        c = coloring.get(v, None)
        if (v in conflict_nodes) or (v in oor_nodes):
            fill = "black" if conflict_nodes_fill_black else "#000000"
            edgec = "black"
        else:
            fill = (
                color_for_index(int(c))
                if (c is not None and isinstance(c, numbers.Integral) and int(c) >= 0)
                else "#DDDDDD"
            )

            edgec = "#555555"
        node_colors.append(fill)
        node_edge_colors.append(edgec)
    if nodes_sorted:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes_sorted,
            node_color=node_colors,      
            edgecolors=node_edge_colors,
            linewidths=0.8,
            node_size=260
        )
    if show_labels:
        # Knotentexte: zeige den (ganzzahligen) Farbindex
        labels = {v: str(int(coloring[v])) for v in nodes_sorted if isinstance(coloring[v], numbers.Integral)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    # Titel-/Dateinamen-Metadaten: Konfliktanzahl und gefärbte Knotenanzahl
    conflicts_cnt = len(conflict_edges)
    key = (step_clean, int(round_id))
    cnt = _SHOT_COUNTER.get(key, 0) + 1
    _SHOT_COUNTER[key] = cnt
    title = (f"{step} — Round {round_id} — try={cnt:03d} — "
             f"conflicts={conflicts_cnt} (colored {len(nodes_to_draw)}/{G.number_of_nodes()})")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    # Dateiname & Speichern
    fname = f"step-{step_clean}_round-{round_id:03d}_try-{cnt:03d}_conflicts-{conflicts_cnt:03d}.png"
    fpath = os.path.join(out_dir, fname)

    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    return fpath
