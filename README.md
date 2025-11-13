
# 2-version
# ILP-Algorithm-for-Graph-Colouring-Problem

Inkrementelle LP-Heuristik für Vertex Coloring mit **zweistufigem Fixing** und drei modularen **Beschleunigern** (Side-Rounding, LP-guided Pack, UB-1 Greedy).  

---

## Verzeichnisstruktur

```bash
ILP-Algorithm-for-Graph-Colouring-Problem/
├── main.py
├── driver/
│   ├── iterate_lp.py            # Iterative Heuristik (v2, inkrementell)
│   └── accelerators.py          # drei Beschleunigern：side, lp_pack, greedy, compact_colors
├── graph/
│   ├── clique.py                # Greedy-Max-Clique (LB & Symmetriebrechung)
│   ├── cliques_enum.py          # Begrenzte Aufzählung größerer maximaler Cliquen (Cuts)
│   ├── dsatur.py                # DSATUR (Baseline/Init)
│   ├── slo.py                   # Smallest-Last (Init)
│   ├── kempe.py                 # Kempe-Ketten (Zwei-Farben-Tausch)
│   ├── local_search.py          # consolidate_colors + lp_guided_pack_highest_color
│   ├── ub1_greedy.py            # Rein graphische UB-1 Verdichtung
│   └── verify.py                # Prüfen: Vollständigkeit/Range/Konflikte
├── heuristics/
│   └── round_and_repair.py      # LP-geführt: Multi-Start Rounding + Repair (+Kempe)
└── ilp/
    ├── model.py                 # Assignment-LP (x,y) + Precedence + Clique-Fixing + Clique-Cuts
    ├── lp_solve.py              # Lösen & Extraktion (zLP, x*, y*, rc)
    ├── incremental.py           # IncrementalLP: Bounds ändern + Solve + Rollback
    └── fixing_policies.py       # Schritt-2 Fixierungs-Policy (prefix_shrink, strong_assign, rounded_support)
```

**Abhängigkeiten:** `python>=3.9`, `networkx`, `ortools` (GLOP).



---

## how to run


```bash
python3 main.py --algo iterlp2 --time 300 --seed 0 \
  --init-heuristic smallest_last \
  --fix-policy prefix_shrink+rounded_support \
  --max-fix-per-round 20 \
  --restarts 48 --perturb-y 1e-6 \
  2>&1 | tee -a logs/run_$(date +%F_%H%M%S).log
```
```bash
mmdc -i .\main_flow.mmd -o main_flow.svg -b transparent
mmdc -i .\explanation.mmd    -o .explanation.svg    -b transparent
```
### Wichtige Argumente

* `--algo`: `dsatur` | `iterlp` | `iterlp2` 
* `--init-heuristic`: `dsatur` | `smallest_last` (Initiale zulässige Färbung)
* `--time`: Zeitlimit (Sekunden)
* `--fix-policy`: Schritt-2 Fixierungsstrategie, mit `+` kombinierbar

  * `prefix_shrink` - setze **K = ceil(zLP)** (Y-Vektoren hinter dem Präfix auf 0)
  * `strong_assign` - setze **x[v,c*]=1**, wenn **x[v,c*] − 2nd >= strong-margin** (mit Rollback)
  * `rounded_support` - stützt Fixierungen durch die aktuelle Rundung (konservativ)
* `--strong-margin`: Schwelle für `strong_assign` (z. B. 0.15–0.30)
* `--max-fix-per-round`: Max. Anzahl x-Fixierungen pro Runde (Überfixierung vermeiden)
* `--restarts`: Multi-Start-Rundungen pro Runde
* `--perturb-y`: sehr kleine Störung auf `y` (Diversifikation im Rounding)

---

## Was ist neu 

* **Inkrementelles LP**: einmaliges Modell; spätere Änderungen via Bounds (kein Neuaufbau).
* **Zweistufiges Fixing (Schritt 2)**:

  1. `prefix_shrink` (**sicher**) setzt K auf `ceil(zLP)`;
  2. `strong_assign` (**mutiger, mit Rollback**) fixiert klare Favoriten in `x`.
* **Drei Beschleuniger** (modular in `driver/accelerators.py`):

  * `try_side_rounding` - **K+1** temporär erlauben, um UB zuerst um 1 zu senken；
  * `try_lp_guided_pack` - LP-Signal (**x***-Gewichte) steuert Packen der höchsten Farbklasse；
  * `try_graph_ub1_greedy` - rein graphisch UB-1 (Kempe/Greedy).
* **consistent compaction**: Jede neue Färbung wird **compact_colors**-neu nummeriert (0..k−1), um `allowed_colors=range(UB)` stets konsistent zu halten.

---

## Algorithmus (Schritt 0–5)

**Schritt 0 – Initialisierung**

1. Greedy-Clique - **LB** (und Symmetriebrechung im LP).
2. Initiale zulässige Lösung via `--init-heuristic` (**DSATUR** oder **Smallest-Last**), anschließend `compact_colors` - **UB** und **K = UB**.

**Schritt 1 – LP lösen (inkrementell)**

* Assignment-LP mit Precedence + optionalen Clique-Cuts.
* `IncrementalLP.solve()` liefert `zLP`, `x*`, `y*`.

**Schritt 2 – Variablen fixieren (zentral)**

* `prefix_shrink`: K <- `ceil(zLP)` (als y-Bounds), via `lock_prefix_K(K)`; **Try-Apply-Solve**, sonst **Rollback**.
* `strong_assign`: wähle Knoten mit großem Abstand der Top-2 in `x[v,·]` und fixiere `x[v,c*]=1`; ebenfalls **mit Rollback**.

**Schritt 3 – LP updaten & erneut lösen**

* Änderungen bleiben im selben Modell; bei Fehlschlag: **revert**.

**Schritt 4 – Ganzzahlige Verbesserung übernehmen**

* Rounding+Repair (Multi-Start, `perturb_y`), **compact + consolidate_colors** - bei weniger Farben: **UB** aktualisieren;
* **Beschleuniger**:

  * **Side-Rounding (K+1)** - leichter  (k+1)-Coloring - dann Richtung K packen；
  * **LP-guided Pack** - x*-geführt die höchste Farbklasse in 0..K−1 drücken；
  * **UB-1 Greedy** - rein graphische Verdichtung.
* Nach jeder Verbesserung ggf. **K auf UB synchronisieren**（wieder mit Try-Apply-Solve/Rollback）.

**Schritt 5 – Terminierung**

* **Theoretisch**: `ceil(zLP) >= UB` **und** `K <= UB`;
* **Praktisch**: Stagnation bei `K = ceil(zLP)` nach mehreren Runden；
* oder Zeit-/Rundenlimit；immer mit finalem `verify_coloring`.

---

## Protokolle & Interpretation

* `Round i: zLP=..., UB=..., K=...`

  * **zLP**: LP-Untergrenze;
  * **K**: derzeit erlaubte Farben im LP/Rounding (i. d. R. `ceil(zLP)`),
  * **UB**: beste **zulässige** Farbenzahl bisher.
* Zusätzliche Meldungen:

  * `[Fix] shrink K ...`（Schritt 2: prefix_shrink）
  * `[Rounding/Local] improved UB -> ...`
  * `[UB-1 side attempt] improved UB -> ...`
  * `[LP-guided pack] UB -> ...`
  * `[UB-1 greedy] UB -> ...`
  * `[Sync] K aligned to new UB: K=...`
* Stop-Gründe: `UB==LB` | `no_better_than_UB` | `stalled_at_K_eq_ceil_zLP` | `time_limit`.

---

## Erklärung zur „Störung“ (Diversifikation)

* **Rounding-Diversifikation**: Multi-Start + kleines `--perturb-y` (globales Farbranking leicht aufbrechen).
* **Kempe-Moves**: in Rounding/Repair & lokalen Pack-Schritten.
* **Strukturelle Diversifikation**: **Side-Rounding (K+1)** öffnet den Farbraum kurzzeitig, um UB zunächst um 1 zu senken.

> **Beispile:** Für schwerere Instanzen die Exploration erhöhen (größeres `--restarts`, kleineres `--strong-margin`), danach durch `consolidate_colors` und `lp_guided_pack` verdichten.

---

## Beispiel

```bash
python3 main.py --algo iterlp2 --time 60 --seed 0   --init-heuristic smallest_last   --fix-policy prefix_shrink+rounded_support   --max-fix-per-round 20   --restarts 48 --perturb-y 1e-6
[Main] algo=iterlp2 | time=60s | seed=0 | init=smallest_last | fix-policy=prefix_shrink+rounded_support | restarts=48 | perturb-y=1e-06
[Baseline/DSATUR] colors=5 | time=0.0072s | feasible=True | conflicts=0
[Init] graph: |V|=100 |E|=415 |LB|=3 |UB(init)|=6 |K|=6 via smallest_last
[Init] LB=3 | UB(init:smallest_last)=6
[LP] model built once (incremental), initial |C|=6 (K=6) with precedence & clique-cuts
[Round 1] zLP=3.333333 | ceil(zLP)=4 | K=6 | UB=6 | t=0.05s
  [Fix] shrink K to 4 (ceil(zLP)=4)
  [Side] trying K+1 rounding (K=4 → 5) to break stalemate
[LocalSearch] highest color class removed; colors now=5
[LocalSearch] highest color class removed; colors now=5
[LocalSearch] highest color class removed; colors now=5
  [UB-1 side attempt] improved UB -> 5 (using 5 colors)
  [Rounding] feasible=False | used=4 | conflicts=9
[LocalSearch] highest color class removed; colors now=5
[LocalSearch] highest color class removed; colors now=5
[LocalSearch] highest color class removed; colors now=5
  [Side] feasible but not better (used=5 ≥ UB=5)
  [Rounding] feasible=False | used=4 | conflicts=9
  [PackWindow] UB==K+1 (UB=5, K=4) → attempt LP-guided pack and UB-1 greedy
  [LP-guided pack] trying to empty highest color K=4 (only active if UB==K+1)
  [LP-guided pack] no full emptying of highest color — no UB reduction
  [UB-1 greedy] attempting to fold top color class graphically
  [UB-1 greedy] feasible but not better (used=5 >= UB=5)
  [FixPolicy] y_zero=1 | x_one=0 | x_zero=0 | strong_margin=0.25 | max_fix_per_round=20
  [Round 1 END] UB=5 | K=4 | ceil(zLP)=4 | improved=True | fixed=True
[Round 2] zLP=3.333333 | ceil(zLP)=4 | K=4 | UB=5 | t=3.95s
  [Rounding] feasible=False | used=4 | conflicts=9
  [PackWindow] UB==K+1 (UB=5, K=4) → attempt LP-guided pack and UB-1 greedy
  [LP-guided pack] trying to empty highest color K=4 (only active if UB==K+1)
  [LP-guided pack] no full emptying of highest color — no UB reduction
  [UB-1 greedy] attempting to fold top color class graphically
  [UB-1 greedy] feasible but not better (used=5 >= UB=5)
  [FixPolicy] y_zero=1 | x_one=0 | x_zero=0 | strong_margin=0.25 | max_fix_per_round=20
  [Round 2 END] UB=5 | K=4 | ceil(zLP)=4 | improved=False | fixed=False
  ..............
  ................
  [UB-1 greedy] feasible but not better (used=5 >= UB=5)
  [FixPolicy] y_zero=1 | x_one=0 | x_zero=0 | strong_margin=0.25 | max_fix_per_round=20
  [Round 8 END] UB=5 | K=4 | ceil(zLP)=4 | improved=False | fixed=False
[Round 9] zLP=3.333333 | ceil(zLP)=4 | K=4 | UB=5 | t=3.97s
  [Rounding] feasible=False | used=4 | conflicts=9
  [PackWindow] UB==K+1 (UB=5, K=4) → attempt LP-guided pack and UB-1 greedy
  [LP-guided pack] trying to empty highest color K=4 (only active if UB==K+1)
  [LP-guided pack] no full emptying of highest color — no UB reduction
  [UB-1 greedy] attempting to fold top color class graphically
  [UB-1 greedy] feasible but not better (used=5 >= UB=5)
  [FixPolicy] y_zero=1 | x_one=0 | x_zero=0 | strong_margin=0.25 | max_fix_per_round=20
  [Round 9 END] UB=5 | K=4 | ceil(zLP)=4 | improved=False | fixed=False
  [Stop] stalled at K==ceil(zLP) for 8 rounds
[Final] UB=5 LB=3 feasible=True zLP≈3.333333
[Compare] DSATUR colors=5 vs IterLP2 colors=5  → Tie (Δ=0)
[Compare] DSATUR time=0.0072s | IterLP2 time=3.9686s | stop=stalled_at_K_eq_ceil_zLP
[Main] Done. stop_reason=stalled_at_K_eq_ceil_zLP | LB=3 | UB=5 | iters=9
```

Interpretation: LP-Untergrenze erzwingt `K=4`; innerhalb des Limits wurde eine 5-Färbung konstruiert, 4-Färbung nicht gefunden - terminieren nach Stagnationsregel.

---

## Reproduzierbarkeit

* **Seeds**: `--seed` setzt alle stochastischen Komponenten (Rounding-Restarts & leichte Tie-Break-Jitter).
* **Konsistenz**: Jede übernommene Färbung wird mit `compact_colors` auf **0..UB−1** remapped; `verify_coloring` nutzt stets `allowed_colors=range(UB)`.

---


# 1-version

---
## Verzeichnisstruktur

```bash
ILP-Algorithm-for-Graph-Colouring-Problem/
├── main.py                     # Einstiegspunkt (inkl. DSATUR-Baseline-Vergleich)
├── driver/
│   └── iterate_lp.py           # Kernsteuerung: Start und Iterationsablauf
├── graph/
│   ├── clique.py               # Gierige maximale Clique (für LB & Symmetriebrechung)
│   ├── cliques_enum.py         # Zeit-/anzahlbegrenzte Aufzählung größerer maximaler Cliquen ( Cliquen-Ungleichungen)
│   ├── dsatur.py               # DSATUR (nur als Baseline, nicht Teil des Lösungsverfahrens)
│   ├── kempe.py                # Kempe-Ketten und Zwei-Farben-Tausche
│   ├── loader.py               # Erzeugen/Laden von Testgraphen (Standard: ER-Zufallsgraph)
│   ├── local_search.py         # Layer-Konsolidierung (Greedy + Kempe)
│   ├── ub1_greedy.py           # Reiner Graph-Versuch UB-1 (bei Misserfolg unverändert)
│   └── verify.py               # Zulässigkeitsprüfung (Vollständigkeit/Range/Konflikte)
├── heuristics/
│   └── round_and_repair.py     # Multi-Start-Rounding+Repair aus LP-Fraktionslösung
└── ilp/
    ├── fixing.py               # Wahl der Farbmenge K (derzeit Invariante K=0..UB-1)
    ├── lp_solve.py             # LP lösen und (z_LP, x*, y*, rc_y) extrahieren
    └── model.py                # Assignment-LP-Modell (mit Präzedenz, Clique-Fixierung, Cliquen-Ungleichungen)
```

**Abhängigkeiten**: `python>=3.9`, `networkx`, `ortools` (GLOP).


**useage**: 
```python
python3 main.py
```
---

## Mathematisches Modell (Assignment-LP als Relaxation)

* **Variablen:**

  * $x_{v,c} \in [0,1]$: Indikator, dass Knoten $v$ Farbe $c$ erhält (im LP kontinuierlich).
  * $y_c \in [0,1]$: Farbe $c$ wird überhaupt genutzt.
* **Ziel:** $\min \sum_c y_c$
* **Nebenbedingungen:**

  1. Genau eine Farbe je Knoten: $\sum_c x_{v,c} = 1,\ \forall v$
  2. Kantenkonflikte: $x_{u,c} + x_{v,c} \le 1,\ \forall (u,v)\in E,\ \forall c$
  3. Verknüpfung: $x_{v,c} \le y_c,\ \forall v,c$
  4. **Symmetriebrechung (Clique-Fixierung):** für $Q=\{v_0,\dots,v_{t-1}\}$ erzwingen $v_i$ nutzt Farbe $i$
     (implementiert als $x_{v_i,i}=1,\ x_{v_i,c\ne i}=0,\ y_i=1$).
  5. **Präzedenz:** $y_{c+1} \le y_c$ zur Vermeidung von Permutationssymmetrien.
  6. **Cliquen-Ungleichungen:** $\sum_{v\in Q} x_{v,c} \le 1$.

---

## Gesamtverfahren

### Schritt 0: LP-basierte initiale zulässige Lösung

1. Gierige maximale Clique $Q$ liefert die **untere Schranke (LB)** und wird zur **Symmetriebrechung** fixiert (Farben von $Q$).
2. LP-Aufbau mit $K = LB + \text{headroom}$ (inkl. Präzedenz und optionalen Cliquen-Ungleichungen).
3. LP lösen $\rightarrow$ $(x^*, y^*)$;
**LP-geführtes Rounding+Repair**:

   * Knotenreihenfolge per **DSATUR** (Bindungen über Sättigung/Grad/LP-Gewicht).
   * Farbauswahl primär nach $x^*_{v,c}$, global nach $y^*_c$; falls nötig **Kempe-Zwei-Farben-Tausch**.
4. Falls unzulässig, **$K$ erhöhen** und wiederholen, bis eine **erste zulässige Färbung $UB_0$** gefunden ist.

### Schritt 1 / Iteration (Iterative LP Heuristic)

Pro Runde:

1. **LP-Neubau** auf $K=\{0,\dots,UB-1\}$ $\rightarrow (z_{LP}, x^*, y^*)$.
2. **Multi-Start Rounding+Repair** (kleine Störung auf $y^*$): falls weniger Farben, **UB** und bestes Coloring aktualisieren.
3. **Fixing (konservativ):** **$K=\{0,\dots,UB-1\}$ beibehalten** (derzeit keine K-Schrumpfung; optional siehe „Erweiterung“).
4. **Lokalsuche (im Farbraum):** `consolidate_colors` (Greedy + Kempe) versucht, die höchste Farbklasse zu leeren.
5. **UB-1-Versuche:**

   * **Graph-Greedy:** `try_ub_minus_one_greedy`;
   * **LP(UB-1) + Rounding:** bei Erfolg UB ← UB−1.
6. **Stopp:** wenn `UB==LB`, langes Ausbleiben von Verbesserungen oder Runden-/Zeitlimit. In jeder Runde **explizite Zulässigkeitsprüfung**.

---

## Zentrale Implementierungsdetails und Designentscheidungen

### 1. `heuristics/round_and_repair.py` (LP-geführt + DSATUR)

* Knotenreihenfolge: **DSATUR** (höchste Sättigung zuerst; Bindungen nach Grad und $\max x^*$).
* Farbauswahl: **nicht von Nachbarn belegte** Farbe mit maximalem $x^*_{v,c}$; wenn keine frei, **Kempe-Tausch**; andernfalls temporär setzen und später reparieren.
* **Multi-Start:** $y^*$ wird mit Störung (1e-6) versehen, um Gleichläufe zu vermeiden.

### 2. `ilp/model.py` (Assignment-LP + Symmetriebrechung + optionale Cliquen-Ungleichungen)

* Clique-Fixierung per Gleichungen; **keine Variablenlöschung**, damit Indizes/Struktur stabil bleiben.
* Cliquen-Ungleichungen sind **optional** und werden zeit-/anzahlbegrenzt aus `cliques_enum.py` übernommen.

### 3. `ilp/fixing.py` (Invariante beibehalten, K an UB gebunden)

* Aktuelle Invariante **$K=0..UB-1$** bleibt erhalten – gut für Reproduzierbarkeit und Logging.
* **(Erweiterung)** K-Schrumpfung aktivierbar: $K \leftarrow \{0,\dots,\min(UB, \lceil z_{LP}\rceil)-1\}$. Beschleunigt das LP erheblich, erfordert jedoch, die Invariante auf „**K ist ein zusammenhängendes Präfix ab 0**“ zu lockern.

### 4. `graph/local_search.py` und `graph/ub1_greedy.py`

* **Layer-Konsolidierung** (Greedy + Kempe) und **UB-1 im Graphen** sind „Safety Nets“. Auf ER-Graphen nicht immer wirksam, aber als austauschbare Module vorbereitet.

### 5. Zulässigkeits-Guards und Logging

* `verify.py` prüft nach jeder Runde: Vollständigkeit, Farbgrenzen, Kantenkonflikte.
* Iterationslog speichert $(UB, LB, z_{LP}, |K|)$ sowie den Stoppgrund.

---

## Beispiel (ER-Zufallsgraph):

```
=== Final Result (Iterative LP) ===
LB (clique size): 3
UB (colors used): 5
Iterations: 50
Stopped: max_rounds
Time (s): 33.596994
[IterLP] feasible=True|used_colors=5|conflicts=0

=== Comparison: DSATUR vs Iterative-LP ===
DSATUR colors = 5 | IterLP colors = 5  -> Tie (Δ = 0)
DSATUR time   = 0.006819s | IterLP time   = 33.596994s
```