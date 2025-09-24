
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
