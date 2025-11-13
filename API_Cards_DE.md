# Schnittstellenkarten (API Cards)

---

## Inhalt
1. `greedy_max_clique(G)`  
2. `dsatur_coloring(G)` / `smallest_last_coloring(G)`  
3. `compact_colors(coloring)`  
4. `top_maximal_cliques(G, ...)`  
5. `IncrementalLP`‑Klasse (`solve`, `lock_prefix_K`, `fix_vertex_color`, `forbid_x`, `try_apply_and_solve`, `revert_all`)  
6. `round_and_repair_multi(G, x_frac, y_frac, current_UB, restarts, seed, perturb_y)`  
7. `verify_coloring(G, coloring, allowed_colors)`  
8. `consolidate_colors(G, coloring, passes)`  
9. `try_side_rounding(G, x_frac, y_frac, K, UB, restarts, perturb_y, seed, ...)`  
10. `try_lp_guided_pack(G, best_coloring, x_frac, K, UB, ...)`  
11. `try_graph_ub1_greedy(G, best_coloring, UB, ...)`  
12. `pick_fixings(G, x_frac, y_frac, z_LP, UB, LB, policy, strong_margin, max_fix_per_round, rounding_coloring)`  

---

## 1) `greedy_max_clique(G) -> List[int]`

### Eingabedefinition des Graphen (G)

**G: Eingabegraph (`networkx.Graph`, ungerichteter einfacher Graph)**

* **Quelle**: Wird von `main.py` über `graph/loader.py::load_demo_graph(seed)` erzeugt.
* **Verteilung**: Erdős–Rényi‑Zufallsgraph (G(n,p)). Derzeit **feste** Parameter:
  * Knotenzahl (n = 100);
  * Kantenwahrscheinlichkeit (p = 0.08);
  * Zufallsseed `seed` kommt von der Kommandozeile `--seed` (Default 0).
* **Größe & Sparsität**: Erwartete Kantenanzahl ($$\mathbb{E}[m]=p\cdot \frac{n(n-1)}{2} \approx 396$$); 
  Erwarteter Grad ($$\mathbb{E}[\deg]\approx p(n-1)= 7{,}92$$).
* **Anpassung**: Parameter (n,p) können in `graph/loader.py` über `nx.erdos_renyi_graph(n=100, p=0.08, seed=seed)` geändert werden.

---

### Funktion `GREEDY_MAX_CLIQUE(G)`

**Signatur**  
`Function GREEDY_MAX_CLIQUE(G) -> (Q: List[int], LB: int)`

**Input**  
* `G: networkx.Graph`

**Output**  
* `Q: List[int]`: per Greedy gewonnene **maximale Clique** (maximal clique, nicht garantiert global maximal);  
* `LB: int = |Q|`: Anzahl der Knoten in `Q`, **untere Schranke**.

  * **Bedeutung**: `LB` ist eine untere Schranke der chromatischen Zahl $$\chi(G)$$ — **Knoten einer Clique müssen paarweise verschieden gefärbt sein**, daher $$\chi(G) \ge |Q| = LB$$.

**Algorithmus**  
1. **Sortierung**: Alle Knoten **nach Grad absteigend** sortieren.  
2. **Aufbau**: Start mit leerer Menge; Knoten v in der Reihenfolge scannen. v nur dann hinzufügen, wenn v zu **allen** Knoten der aktuellen Menge adjazent ist.  
3. **Rückgabe**: Die resultierende Menge `Q` ist eine **maximale Clique**, und `LB = |Q|`.

> Hinweis: „große Clique“ meint hier die per Grad‑Greedy gefundene **maximale** Clique ohne zusätzliche Schwellen. Auf G(n,p)‑Graphen liefert diese Heuristik erfahrungsgemäß größere maximale Cliquen als zufällige Reihenfolgen, garantiert aber nicht die globale Maximalität.

**Komplexität**  
* Sortierung $$O(n\log n)$$; Nachbarschaftsprüfungen grob $$O(n^2)$$.

**Verwendung**  
* **Position im Ablauf**: **N1** (Compute LB – clique size).  
* **Stop/Verengung**:
  * Falls initial `UB ≤ LB`: sofortiger Optimalstopp (D0→E1);
  * Beim Verengen des Farbpräfixes (K) stets $$K \ge LB$$ wahren.
* **Einsatz im LP (a‑priori‑Nebenbedingungen / Symmetriebruch)**: In **N3** („Build LP once“) wird `Q` ins LP eingebettet, um die Relaxation zu verschärfen und Farbsymmetrien zu brechen (siehe unten „aufgerufene Funktionen“).

**Aufgerufene Funktionen (mit Quellorten)**  
* **Weitere große Cliquen (optional)**: `graph/cliques_enum.py::top_maximal_cliques(...)` → für **Clique‑Ungleichungen** im LP.  
* **Einmaliger LP‑Aufbau (inkl. a‑priori‑Fixierungen/Präzedenz)**:
  * `ilp/incremental.py::IncrementalLP.__init__(..., clique_nodes=Q, extra_cliques=..., add_precedence=True)`
  * intern `ilp/model.py::build_lp_model(...)`, u. a. mit:
    * **Clique Fixing** (Teil des Symmetriebruchs): `Q={v₀,…,v_{t−1}}` auf die Farben `0..t−1` abbilden, $$x_{v_i,i}=1$$ und $$y_i=1$$;
    * **Precedence (Präfixordnung)**: $$y_{c+1} \le y_c$$ zum Brechen von Farbenpermutationen;
    * **Clique Cuts**: Für jede Clique `Q'` und Farbe `c`: $$\sum_{v\in Q'} x_{v,c} \le 1$$.

---

## 2) `dsatur_coloring(G) -> Dict[int,int]` / `smallest_last_coloring(G) -> Dict[int,int]`

**Ort**: `graph/dsatur.py`, `graph/slo.py` (nutzen `networkx.coloring.greedy_color` und remappen am Ende auf ein zusammenhängendes Präfix `0..k-1`).

### Input
* `G: networkx.Graph` — siehe „Eingabedefinition des Graphen G“ (Erdős–Rényi G(n=100, p=0.08), Seed durch CLI).

### Output (einheitlich)  
* `col0: Dict[int,int]` — **zulässige** Ganzzahl‑Färbung: jedem Knoten `v` ist eine Farbe `col0[v]` zugewiesen, und **jede Kante** hat verschiedenfarbige Endpunkte.
  * **Farbbereich**: Beide Funktionen führen **intern** eine Verdichtung auf ein **zusammenhängendes Präfix** `0..UB-1` durch.
  * **Definition von `UB`**: `UB := |{ col0[v] }|` (Anzahl verwendeter Farben).
* **Kopplung an Hauptablauf**: In der Initialisierung wird `K ← UB` gesetzt (Farbpräfixbudget = Start‑UB) und `col0` als `best_coloring` übernommen.

### Algorithmen

**A. `dsatur_coloring(G)` (DSATUR)**  
* **Idee**: In jedem Schritt den ungefärbten Knoten mit **höchster Sättigung** (Anzahl **verschiedener** Farben in der gefärbten Nachbarschaft) auswählen und die **kleinste zulässige** Farbe vergeben; wiederholen bis alle Knoten gefärbt sind.  
* **Schritte**:
  1. Initial: alle ungefärbt, Sättigungen 0;
  2. Wiederhole:
     * Wähle v mit maximaler Sättigung (Tie‑break typ. nach Grad; deterministisch bei fixem `G`);
     * Vergib die **kleinste konfliktfreie** Farbe;
     * Aktualisiere Sättigungen der Nachbarn;
  3. Rückgabe der Ganzzahl‑Färbung.  
* **Eigenschaften**: Klassischer DSATUR; **zulässig**; Rückgabe bereits **verdichtet** (0..UB-1).  
* **Komplexität**: Naiv etwa \(O(n^2)\); bei n=100 vernachlässigbar.

**B. `smallest_last_coloring(G)` (Smallest‑Last / Degenerationsordnung)**  
* **Idee**: Erzeuge zuerst eine **Kleinstdreihenfolge** (iterativ jeweils Knoten minimalen Grades entfernen und stapeln), färbe dann **in umgekehrter Stapelreihenfolge** mit jeweils **kleinster zulässiger** Farbe.
* **Schritte**:
  1. Reihenfolge erzeugen: wiederholt Knoten minimalen Grades entfernen, auf Stack legen;
  2. Färben in Stack‑Umkehrreihenfolge;
  3. Rückgabe der Ganzzahl‑Färbung.
* **Eigenschaften**: Farbzahl \(\le\) Degenerationsgrad + 1; **zulässig**; Rückgabe bereits **verdichtet** (0..UB-1).
* **Komplexität**: \(O(n+m)\) bis \(O(m\log n)\) je nach Datenstruktur.

### Invarianten & Sonderfälle
* **Zulässigkeit**: Beide Methoden garantieren **Kantenkonfliktfreiheit**.  
* **Verdichtung**: Beide liefern **bereits** ein Präfix; ein erneutes `compact_colors` ist **idempotent**.  
* **Isolierte Knoten/Leerer Graph**: Isolierte Knoten → Farbe 0; Graph ohne Kanten → alle Farbe 0 (`UB=1`).  
* **Determinismus**: Bei gegebenem `G` deterministisch (NetworkX); `G` ist durch `seed` reproduzierbar.

### Rolle im Hauptablauf (Flow **N2**)
* `col0 ← dsatur_coloring(G)` oder `smallest_last_coloring(G)`;
* `UB := |{col0[v]}|`, `K ← UB`;
* Falls `UB ≤ LB`: **optimaler Stopp**; sonst weiter zu N3 (LP‑Aufbau).

---

## 3) `compact_colors(coloring) -> (Dict[int,int], int)`

**Zweck**: Remap von `coloring` auf ein zusammenhängendes Präfix `0..used-1`; Rückgabe (neue Färbung, `used`).  
**Input**: `coloring: Dict[int,int]`.  
**Output**: `(coloring_compact: Dict[int,int], used: int)`.  
**Vor/Nachbedingungen**: Vorher — gültige Ganzzahl‑Färbung; Nachher — Farben sind ein Präfix.  
**Rollback**: keiner (reines Remapping).

---

## 4) `top_maximal_cliques(G, max_cliques=None, min_size=None, time_limit_sec=None) -> List[List[int]]`

**Ort**: `graph/cliques_enum.py :: top_maximal_cliques`

**Signatur (mit Defaults)**  
`top_maximal_cliques(G, max_cliques: int = 50, min_size: int = 4, time_limit_sec: float = 2.0) -> List[List[int]]`

### Input
- `G: networkx.Graph` — ungerichteter einfacher Graph (Standard: `load_demo_graph(seed)` → G(n=100, p=0.08)).
- `max_cliques: int` (Default `50`) — **Hartes Limit** der zu sammelnden maximalen Cliquen.
- `min_size: int` (Default `4`) — **Mindestgröße**: sammle nur Cliquen mit `|Q'| ≥ min_size`. In der Pipeline oft `min_size = max(4, LB+1)`.
- `time_limit_sec: float` (Default `2.0`) — **Zeitlimit**: bei Überschreitung vorzeitig abbrechen und bisherige Ergebnisse zurückgeben.

### Output
- `extra_cliques: List[List[int]]` — Liste maximaler Cliquen als sortierte Knotenlisten, mit:
  - jede `Q'` maximal;  
  - `|Q'| ≥ min_size`;  
  - Anzahl ≤ `max_cliques`;  
  - **Deduplizierung** nach Knotenmengen; interne Sortierung stabilisiert das Format.

### Zweck
- **Clique‑Ungleichungen fürs LP**: Für jede `Q'` und jede Farbe `c ∈ {0,…,K−1}` füge  
  \[ \sum_{v \in Q'} x_{v,c} \le 1 \]  
  als Nebenbedingung hinzu. Stärkt die LP‑Relaxation, stabilisiert die Lösung.

### Verfahren (Implementationsnah)
- Generator `networkx.find_cliques(G)` (Bron–Kerbosch) enumeriert maximale Cliquen;
- Online‑Filterung nach Zeit/Anzahl/Mindestgröße;  
- Deduplizierung und Sortierung vor Rückgabe.

### Komplexität & Trunkierung
- Worst‑Case exponentiell;  
- Praktisch durch Limits (Zeit/Anzahl/Größe) auf **handhabbare** Laufzeiten begrenzt (hier: ms–10^1 ms).

### Kopplung mit Hauptablauf
- **N3**: `extra_cliques ← top_maximal_cliques(G, max_cliques=50, min_size=max(4, LB+1), time_limit_sec=2.0)`;  
- Übergabe an `IncrementalLP(..., clique_nodes=Q, extra_cliques=extra_cliques, add_precedence=True)`.

---

## 5) `class IncrementalLP(...)`

**Ort**: `ilp/incremental.py` (Modellbau in `ilp/model.py`; Lösen/Extraktion in `ilp/lp_solve.py`)  
**Rolle**: Einmal „Assignment‑LP“ bauen; danach nur noch **Bounds** ändern (**reversibel** via Tokens).

### Konstruktor
`IncrementalLP(G, allowed_colors: List[int], clique_nodes: List[int], extra_cliques: Optional[List[List[int]]] = None, add_precedence: bool = True)`

**Input**
- `G: networkx.Graph`
- `allowed_colors: List[int]` — z. B. `list(range(K))`
- `clique_nodes: List[int]` — `Q` aus `GREEDY_MAX_CLIQUE`
- `extra_cliques: Optional[List[List[int]]]` — zusätzliche große maximale Cliquen
- `add_precedence: bool = True` — aktiviere `y[c+1] ≤ y[c]`

**Interner Zustand**
- `V = list(G.nodes())`, `C = allowed_colors`
- Variablen `x[v,c] ∈ [0,1]`, `y[c] ∈ [0,1]` und deren **Bounds**
- LP‑Solverhandle
- **Token‑Stack** für reversible Bounds‑Änderungen

### LP‑Modell
- **Ziel**: `min Σ_c y[c]`  
- **NB**:
  1) `∑_c x[v,c] = 1` ∀v  
  2) `x[u,c] + x[v,c] ≤ y[c]` ∀(u,v)∈E, ∀c  
  3) `x[v,c] ≤ y[c]`; „leere Farben“ verbieten (z. B. `y[c] ≤ ∑_v x[v,c]`)  
  4) Clique‑Ungleichungen für `clique_nodes` und `extra_cliques`  
  5) Präzedenz `y[c+1] ≤ y[c]` (Symmetriebruch)  
- **A‑priori‑Fixierungen** (optional): Für `Q={v₀,…,v_{t−1}}` ggf. `x[v_i,i]=1`, `y[i]=1`

### Öffentliche Methoden

**1) `solve() -> Dict`**  
Löst mit aktuellen Bounds; liefert `z_LP`, `x_frac`, `y_frac`. Keine Bounds‑Änderung.

**2) `lock_prefix_K(K_target: int) -> BoundsToken`**  
Setzt für alle `c ≥ K_target`: `y[c] = 0`. Liefert Token; **erst** mit `try_apply_and_solve([token])` validieren. Bei Misserfolg via `revert_all([token])` zurück.

**3) `fix_vertex_color(v: int, c: int) -> BoundsToken`**  
Erzwingt `x[v,c]=1` und `x[v,c'≠c]=0`. Typischerweise nur für `c < K`. Validierung via `try_apply_and_solve`.

**4) `forbid_x(v: int, c: int) -> BoundsToken`**  
Erzwingt `x[v,c]=0`. Sparsam verwenden.

**5) `try_apply_and_solve(tokens: List[BoundsToken]) -> (Optional[Dict], bool)`**  
Tokens temporär anwenden → LP lösen → `ok`/`info` zurück; **kein** Auto‑Rollback (Aufrufer ruft `revert_all` bei Bedarf).

**6) `revert_all(tokens: List[BoundsToken]) -> None`**  
Alle Bounds per Token **rückgängig** machen (LIFO).

---

## 6) `round_and_repair_multi(G, x_frac, y_frac, current_UB, restarts, seed, perturb_y) -> Dict[int,int]`

**Ort**: `round_and_repair.py` (Multi‑Start) + `local_search.py` / `kempe.py` (Reparaturen/Tausche)

### Input
- `G: networkx.Graph` — ungerichtet, einfach; erlaubte Farben **fix** `0..current_UB-1`
- `x_frac: Dict[(int,int), float]` — LP‑Präferenzen pro (v,c)
- `y_frac: Dict[int, float]` — LP‑Aktivierungen pro Farbe c
- `current_UB: int` — Farbpräfixbudget (=K)
- `restarts: int` — Anzahl unabhängiger Versuche
- `seed: int` — Reproduzierbarkeit
- `perturb_y: float` — winziger Jitter zum Tie‑Breaking der `y`‑Sortierung

### Output
- `cand: Dict[int,int]` — **ganzzahlige** Färbung (ggf. noch mit Restkonflikten; der Aufrufer prüft anschließend).

### Auswahlkriterien (bestes `cand`)
1. **Primär**: verwendete Farben minimal  
2. **Sekundär**: Konflikte minimal  
3. **Tertiär**: `∑_v x_frac[(v, cand[v])]` maximal  
4. Tie‑Break deterministisch

### Top‑Level‑Ablauf (Multi‑Start)
1. Für jeden Restart: `y` leicht stören → Farbpriorität sortieren.  
2. `ROUND_AND_REPAIR` einmal ausführen.  
3. Kandidat bewerten; bestes merken.  
4. Bestes `cand` zurückgeben.

### Einzellauf `ROUND_AND_REPAIR`
- DSATUR‑artige Knotenwahl (Sättigung → Grad → `max x[v,c]` → winziger Lärm)
- Farbe wählen: konfliktfreie Farbe mit größtem `x[v,c]`, bei Leere **kleine Kempe‑Versuche**, sonst temporär „beste“ Farbe trotz Konflikt
- 2–3 Nachbesserungs‑Pässe (lokale Tausch/Swap/Kempe), optional Verdichtung

**Hinweise**  
- Farben stets `0..current_UB-1`  
- Temporäre Konflikte erlaubt, spätere Reparatur vorgesehen  
- Kempe: Tausch der zwei Farben auf der **Komponente** der zwei‑farbigen induzierten Teilgraphen  
- `perturb_y` nur Tie‑Breaker  
- Reproduzierbar: `seed + r`

---

## 7) `verify_coloring(G, coloring, allowed_colors) -> Dict[str,Any]`

**Zweck**: Zulässigkeit & Statistiken prüfen.  
**Input**: `G; coloring; allowed_colors`.  
**Output** (Report): `feasible: bool`, `num_conflicts: int`, `num_used_colors: int`, `used_colors: List[int]`, `missing_nodes`, `out_of_range_nodes`, `bad_nodes`, `conflicts_sample` (optional).

---

## 8) `consolidate_colors(G, coloring, passes) -> (Dict[int,int], bool)`

**Ort**: `graph/local_search.py :: consolidate_colors` (ruft `_try_recolor_vertex` sowie `graph/kempe.py::{kempe_chain_component, kempe_swap}`)

### Zweck (präzise)
**Nur committen, wenn die höchste Farbschicht vollständig geleert wird**: Ausgehend von `coloring` die aktuelle höchste Farbe `maxc` wählen und versuchen, **alle** Knoten mit Farbe `maxc` in die niedrigeren Farben `0..maxc-1` zu verschieben (Greedy + wenige Kempe‑Züge). Gelingt die vollständige Leerung, wird verdichtet und `reduced=True` geliefert; andernfalls **alles verwerfen** (`reduced=False`). Bis zu `passes` Runden (je Runde die dann aktuelle höchste Schicht).

### Input
- `G: networkx.Graph`
- `coloring: Dict[int,int]` (Ganzzahl‑Färbung)
- `passes: int` (Default 10)

### Output
- `coloring2: Dict[int,int]` — neue verdichtete Färbung bei Erfolg, sonst **Original**  
- `reduced: bool` — ob die Farbanzahl reduziert wurde

---

## 9) `try_side_rounding(G, x_frac, y_frac, K, UB, *, restarts, perturb_y, seed, ...) -> (int, Dict[int,int], bool)`

**Zweck**: **Ohne** LP‑Änderung einmal mit `K+1` Farben runden; liefert eine bessere zulässige Lösung → übernehmen.  
**Input**: `G; x_frac; y_frac; K; UB; restarts; perturb_y; seed`.  
**Output**: `(newUB, newCol, applied)`  
**Hinweis**: Sinnvoll typ. nur, wenn `K+1 ≤ UB`. Bei Erfolg `UB/best_coloring` aktualisieren; optional K mittels `lock_prefix_K(UB)` nachziehen.

---

## 10) `try_lp_guided_pack(G, best_coloring, x_frac, K, UB, ...) -> (int, Dict[int,int], bool)`

**LP‑geführtes Packen (nur wenn `UB == K+1`)**: Knoten der **höchsten** Farbe `UB-1` anhand der **Präferenzen aus `x*`** in die Farben `0..K-1` zurückverschieben (Greedy + Kempe). Nur wenn die oberste Schicht **vollständig** geleert wird, gilt der Versuch als erfolgreich → `UB ← UB-1`.

**Auslöser**: Nur im **kritischen Fenster** `UB == K+1`.  
**Präferenzordnung**: Für jeden Knoten `v` der höchsten Schicht Ziel‑Farben `0..K-1` **absteigend** nach `x_frac[(v,c)]` sortieren (Fehlwerte = 0.0).  
**Operation**: Für jede Ziel‑Farbe c: (i) konfliktfrei → direkt recolor; (ii) sonst **Kempe‑Komponente** auf den zwei Farben (alt/neu) tauschen; bei Misserfolg **sofort zurücktauschen** und nächste Farbe probieren.  
**Mehrpässe & Atomizität**: Wenige Pässe (z. B. 2). **Nur** wenn die oberste Schicht leer wird, wird committed (anschließend `compact_colors`); sonst **Rollback** (Rückgabe des Originals, `applied=False`).  
**Position im Ablauf**: Bei `UB == K+1` zuerst LP‑guided, dann (bei Misserfolg) der graph‑only‑Versuch. Bei Erfolg K ggf. sicher synchronisieren (`lock_prefix_K(UB)` + Solve, sonst Rollback).

---

## 11) `try_graph_ub1_greedy(G, best_coloring, UB, ...) -> (int, Dict[int,int], bool)`

**Reiner Graph‑Greedy + Kempe (UB → UB−1)**: Ohne `x*`‑Signale mit denselben Greedy/Kempe‑Primitive die oberste Schicht packen. Erfolg → `UB ← UB-1`; Misserfolg → keine Änderung.  
**Zweck**: Ergänzung zum LP‑geführten Packen (bei dessen Scheitern).  
**Vorgehen**: Für alle Knoten der Farbe `UB-1` Ziel‑Farben `0..UB-2`; zuerst konfliktfreie Recolorings, sonst Kempe‑Tausch; scheitert ein Knoten → Gesamter Versuch gilt als gescheitert (Rollback).  
**Einheitliche Behandlung**: Bei Erfolg `compact_colors`, Logging/Viz analog, danach UB/K‑Sync wie oben.

**Reichweite & Sicherheit der Kempe‑Züge**  
* **Kempe‑Bereich**: nur auf der **Komponente** des zweifarbigen induzierten Teilgraphen (aktuelle Ziel‑Zweifarbe).  
* **Sicheres Rollback**: Jeder einzelne fehlgeschlagene Tausch sofort zurück; ganzer Versuch nur bei kompletter Leerung erfolgreich.

---

## 12) `pick_fixings(G, x_frac, y_frac, z_LP, UB, LB, *, policy, strong_margin, max_fix_per_round, rounding_coloring) -> FixPlan`

Ohne Änderung des Farbbudgets (K) werden **wenige** „sehr vertrauenswürdige“ Fixierungen `x[v,c]=1` in das LP übernommen, um die Relaxation zu stabilisieren; **nur** wenn das Modell danach weiterhin zulässig ist, folgt optional ein weiterer „Runden+Reparieren“-Versuch (UB‑Senkung möglich). Bei **Infeasibility** wird **vollständig zurückgerollt**.

**Zweck**: Einen **konservativen** und **reversiblen** Fixierungsplan erzeugen.

### „Verlässlich/vertrauenswürdig“
Hohe Wahrscheinlichkeit, dass `(v,c)` in einer ganzzahligen Lösung korrekt ist **und** die nachgelagerte Runden‑Phase erleichtert. Drei komplementäre Signale:

#### 1) `prefix_shrink` (nur y‑Vorschläge)
* Idee: Ziel ist `sum y_c` zu minimieren. Ist `z_LP` nahe an ganzzahligem K, sind **Suffix‑Farben** praktisch überflüssig.
* Vorgehen: `K_target = max(LB, ceil(z_LP))`; alle `c ≥ K_target` in `plan["y_zero"]` sammeln.  
* Umsetzung: Das **echte** Sperren dieser `y_c` passiert **oben** via `lock_prefix_K(K_target)` (mit Try‑Solve & Rollback).  
* Zweck: Sichere Verengung auf das Farbpräfix, weniger Störungen durch „hohe Farbebenen“.

#### 2) `strong_assign` (starke Zuweisung anhand einer „Lücke“)
* Idee: Wenn für `v` die beste Farbe `c1` **deutlich** vor der zweitbesten `c2` liegt, bevorzugt das LP klar `c1`.
* Vorgehen (pro Knoten `v`):
  1. `c1 = argmax_c x_frac[v,c]`, `c2 = second_best`  
  2. Lücke `gap = x[v,c1] - x[v,c2]`  
  3. Falls `c1 < K` **und** `gap ≥ strong_margin` (typ. 0.15–0.25, z. B. 0.20): `(v,c1)` in `plan["x_one"]` aufnehmen  
  4. (optionale zusätzliche Filter, konservativ):
     * `x[v,c1] ≥ 0.6` (starke Stütze);  
     * **Konfliktdruck** klein (z. B. max\_{u∈N(v)} x[u,c1] ≤ 0.7).
* Zweck: Sehr starke `(v,c)` bevorzugt fixieren.

#### 3) `rounded_support` (Konsistenz mit der letzten Rundung)
* Idee: Wenn die letzte ganzzahlige Rundung `cR` für `v` gewählt hat **und** `x[v,cR]` im LP auch hoch ist, sind beide Signale **konsistent**.
* Vorgehen (pro `v`):
  1. `rounding_coloring[v]` existiert → `cR`  
  2. `cR < K` und `x[v,cR] ≥ 0.8` → `(v,cR)` in `plan["x_one"]`  
  3. (optional) zusätzliche Stabilitätsprüfung: `x[v,cR] - max_{c≠cR} x[v,c] ≥ 0.05` (vermeidet Kippfälle 0.51 vs. 0.50)
* Zweck: Fixiere vorrangig Positionen, auf die sich LP **und** Rundung „einigen“.

> **Kein `x_zero`** standardmäßig: Hartes Sperren `x[v,c]=0` kann zu aggressiv sein; besser nur selten und sehr gezielt.

### (Noch unvollständig) Priorität, Limit, y_zero
* Beispielhafte Priorität für Kandidaten `(v,c)`:
  `priority = 2*I[strong_assign] + 1*I[rounded_support] + 0.5*gap + 0.2*x[v,c] - 0.5*conflict_pressure`  
  (nur als Illustration).
* Nur `c < K` zulassen (nie auf „hohe“ Farben fixieren).
* Nach `priority` absteigend sortieren, **höchstens** `max_fix_per_round` Einträge in `plan["x_one"]`.  
* `plan["y_zero"]` kommt aus `prefix_shrink`.

### Input/Output
**Input**: siehe Signatur.  
**Output**: `FixPlan = {"y_zero": List[int], "x_one": List[Tuple[int,int]], "x_zero": List[Tuple[int,int]]}`.  
**Vor/Nach**: Oben werden nur `x_one` mit `c < K` tatsächlich via `fix_vertex_color` angewandt; Anzahl durch `max_fix_per_round` begrenzt.  
**Rollback**: Anwendung immer als Batch via `try_apply_and_solve`; bei `ok=False` → `revert_all`.

