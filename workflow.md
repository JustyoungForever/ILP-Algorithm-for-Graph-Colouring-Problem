
# Pseudocode

> **UB** = Anzahl der Farben der aktuell besten zulässigen Lösung; **LB** = untere Schranke (z. B. Cliquengröße); **K** = Größe des im LP/Rundungsverfahren erlaubten Farbpräfixes (Farbbereich {0,…,K−1}); **zLP** = LP-Zielfunktionswert (untere Schranke); **x***, **y*** = fraktionale LP-Lösung.

```text
Function MAIN(args)

  Input:   
    args.algo : string
      // Werte: {"dsatur","iterlp2"}
      // Zweck: wählt den Ausführungspfad. "dsatur" → Baseline-Zweig, sonst Hauptalgorithmus.
      // Verwendung: Verzweigung in Steps-2 und Steps-3.

    args.seed : int
      // Zweck: steuert die Zufälligkeit beim Laden/Erzeugen des Graphen, sichert Reproduzierbarkeit.
      // Verwendung: Step-1 beim Laden des Graphen.

    args.time : int (seconds)
      // Zweck: Gesamtzeitbudget für die iterative Lösung.
      // Verwendung: Step-3 als time_limit_sec an den Hauptalgorithmus.

    args.init_heuristic : string
      // Werte: {"dsatur","smallest_last"}
      // Zweck: nur im Hauptalgorithmuszweig; bestimmt die Konstruktion der Anfangslösung (beeinflusst initiales UB/K).
      // Verwendung: Step-3 Übergabe an den Hauptalgorithmus.

    args.fix_policy : string
      // Beispiel: "prefix_shrink+strong_assign" (Strategien mit '+' verkettet)
      // Zweck: nur im Hauptalgorithmuszweig; steuert optionale Fixierungsstrategien.
      // Verwendung: Step-3 Übergabe an den Hauptalgorithmus.

    args.strong_margin : float
      // Zweck: nur im Hauptalgorithmuszweig; Schwelle für starke Fixierung (strong assign).
      // Verwendung: Step-3 Übergabe an den Hauptalgorithmus.

    args.max_fix_per_round : int
      // Zweck: nur im Hauptalgorithmuszweig; Obergrenze der pro Runde zu fixierenden Variablen.
      // Verwendung: Step-3 Übergabe an den Hauptalgorithmus.

    args.restarts : int
      // Zweck: nur im Hauptalgorithmuszweig; Anzahl der Multi-Start-Versuche für das Runden.
      // Verwendung: Step-3 Übergabe an den Hauptalgorithmus.

    args.perturb_y : float
      // Zweck: nur im Hauptalgorithmuszweig; sehr kleine Störung auf y*, um Gleichstände in Sortierungen aufzubrechen (nur Tie-Breaking).
      // Verwendung: Step-3 Übergabe an den Hauptalgorithmus.

    args.viz_out : path string
      // Zweck: Ausgabeverzeichnis für Visualisierungen.
      // Verwendung: Steps-2 und -3 (Export der Visualisierungen).

    args.viz_layout_seed : int
      // Zweck: fester Layout-Seed, damit mehrere Bilder dasselbe Layout nutzen.
      // Verwendung: Steps-2 und -3 (Export der Visualisierungen).

  Output:
    console_summary : strukturierter Text auf stdout
      // Enthält (in Reihenfolge, nur wenn der jeweilige Zweig betreten wird):
      //  A) Parameterzeile (nur Hauptalgorithmus): [Main] algo, time, seed, init, fix-policy, restarts, perturb-y
      //  B) DSATUR-Baseline (entsteht in beiden Zweigen; im Hauptzweig als Vergleich):
      //     [Baseline/DSATUR] colors, time, feasible, conflicts
      //  C) Vergleichszeilen (nur Hauptzweig nach Abschluss):
      //     [Compare] DSATUR colors vs IterLP2 colors → verdict(Δ)
      //     [Compare] DSATUR time vs IterLP2 time | stop=...
      //  D) Abschlusszeile (nur Hauptzweig): [Main] Done. stop_reason=... | LB=... | UB=... | iters=...

    final_figure : PNG file
      // Beschreibung: finale Visualisierung (rendert die aktuell beste Färbung).
      // Pfad: im Verzeichnis args.viz_out.
      // Benennung: je nach Zweig anderes step-Feld ("Final-DSATUR" / "Final-IterLP2"), inkl. Rundenzähler usw.

  Steps:
    1) G ← load_demo_graph(seed = args.seed)
       // Zu färbenden Graphen laden/erzeugen. Nutzt nur den Seed.

    2) if args.algo == "dsatur" then
         t0 ← now()
         col ← dsatur_coloring(G)
         t_ds ← now() - t0
         UB_ds ← number_of_used_colors(col)
         rep_ds ← verify_coloring(G, col, allowed_colors = 0..UB_ds-1)
         print "[Baseline/DSATUR] colors=UB_ds | time=t_ds | feasible=rep_ds.feasible | conflicts=rep_ds.num_conflicts"
         visualize_coloring(G, col,
                            step="Final-DSATUR",
                            round_id=0,
                            out_dir=args.viz_out,
                            layout_seed=args.viz_layout_seed)
         return  

       else  // Hauptalgorithmuszweig ("iterlp2")
         // 2.1 Parameterzeile
         print "[Main] algo=iterlp2 | time=args.time | seed=args.seed | init=args.init_heuristic | fix-policy=args.fix_policy | restarts=args.restarts | perturb-y=args.perturb_y"

         // 2.2 DSATUR-Baseline (als Vergleich)
         t0 ← now()
         ds_col ← dsatur_coloring(G)
         t_ds ← now() - t0
         UB_ds ← number_of_used_colors(ds_col)
         res_ds ← verify_coloring(G, ds_col, allowed_colors = 0..UB_ds-1)
         print "[Baseline/DSATUR] colors=UB_ds | time=t_ds | feasible=rep_ds.feasible | conflicts=rep_ds.num_conflicts"

         // 2.3 Hauptalgorithmus ausführen
         t1 ← now()
         res ← run_iterative_lp_v2(
                 G,
                 time_limit_sec = args.time,
                 init_heuristic = args.init_heuristic,
                 fix_policy = args.fix_policy,
                 strong_margin = args.strong_margin,
                 max_fix_per_round = args.max_fix_per_round,
                 restarts = args.restarts,
                 perturb_y = args.perturb_y,
                 enable_visualization = true,
                 viz_out_dir = args.viz_out,
                 viz_layout_seed = args.viz_layout_seed
               )
         t_lp ← now() - t1

         // 2.4 Vergleich + Visualisierung + Abschluss
         verdict ← compare_colors(UB_ds, res.UB)  // gibt z. B. "better/tie/worse" zurück
         print "[Compare] DSATUR colors=UB_ds vs IterLP2 colors=res.UB → verdict=verdict"
         print "[Compare] DSATUR time=t_ds | IterLP2 time=t_lp | stop=res.stop_reason"
         visualize_coloring(G, res.coloring,
                            step="Final-IterLP2",
                            round_id=res.iters,
                            out_dir=args.viz_out,
                            layout_seed=args.viz_layout_seed)
         print "[Main] Done. stop_reason=res.stop_reason | LB=res.LB | UB=res.UB | iters=res.iters"
         return

End
```

---

# Pseudocode der Hauptschleife (iter_lp_v2)

---

# Einheitliche Objekte/Variablen

* `G: networkx.Graph`: Eingabegraph.
* `coloring: Dict[int,int]`: Abbildung Knoten→Farbe; Farbindizes sind ein zusammenhängendes Präfix `0..k-1` (via `compact_colors` verdichtet).
* `UB: int`: **aktuell beste** zulässige Farbanzahl (obere Schranke).
* `LB: int`: untere Schranke (aus Cliquengröße).
* `K: int`: Größe des im LP/Rundungsverfahren erlaubten **Farbpräfixes** (typisch `K = UB`, kann mit `ceil(zLP)` enger werden).
* `zLP: float`: LP-Zielfunktionswert (`∑ y_c`) = relaxierte Untergrenze der Farbanzahl.
* `x_frac (≡ x*), y_frac (≡ y*)`: fraktionale LP-Lösung; `x_frac` kann in drei äquivalenten Strukturen vorliegen (alle im Code unterstützt).
* `report_*: Dict`: einheitlicher „Zulässigkeitsreport“ aus `verify_coloring` mit Feldern `feasible/num_conflicts/num_used_colors/...`.

---

# Schlüsselbegriffe

* **report** = Rückgabedikt von `verify_coloring`.
* **best_coloring** = aktuell beste **ganzzahlige zulässige** Färbung im gesamten Ablauf.
* **compact / consolidate**: klar unterscheiden — `compact` remappt nur die Indizes; `consolidate` enthält echte lokale Bewegungen zur „Leerung der höchsten Farbschicht“ (kann UB senken).
* **lock_prefix_K**: „nur das Suffix der y-Variablen (Farben ≥ K) auf 0 setzen“, Präfix `0..K-1` bleibt offen — erlaubt genau K Farben.
* **side rounding (K+1)**: ändert **nicht** die LP-Sperre; dient nur dazu, eine bessere zulässige Lösung zu konstruieren.

## N1 — Untere Schranke berechnen (Cliquengröße)

```text
Function GREEDY_MAX_CLIQUE(G)
  Input:
    G: Graph — ungerichteter einfacher Graph
  Output:
    Q: List[int] — eine „gierig“ gefundene große Clique (nicht garantiert maximal, aber typischerweise groß)
    LB: int = |Q|
  Zweck:
    In Grad- oder heuristischer Reihenfolge Knoten hinzufügen und stets Paarweise-Adjazenz wahren; die Größe |Q| ist
    eine gültige untere Schranke für die Chromatische Zahl (Knoten einer Clique müssen paarweise verschieden gefärbt sein).
  Used-by:
    N1 liefert LB; Q kann später für a-priori-Nebenbedingungen / Symmetriebruch verwendet werden.
```

---

## N2 — UB und K initialisieren

* **Zweck**: Mit `dsatur_coloring` oder `smallest_last_coloring` eine **zulässige Anfangslösung** `col0` erzeugen (bereits farbkompakt).

```text
Function INITIAL_COLORING(G, init_heuristic)
  Input:
    G: Graph
    init_heuristic: str ∈ {"dsatur","smallest_last"}
  Output:
    col0: Dict[int,int] — initiale zulässige Färbung
    UB:  int   — Anzahl verwendeter Farben |{col0[v]}|
  Zweck:
    "dsatur": klassische DSATUR-Färbung; "smallest_last": Stapel nach kleinstem Grad.
    Beide liefern kompakte Farbindizes 0..UB-1.
  Side:
    setze K ← UB als Größe des zulässigen Farbpräfixes für LP/Runden.
```

* **Abgeleitet**: **UB = |{col0[v]}|** aus der Anfangsfärbung.
* **Synchronisation**: **K ← UB** setzen (LP und Rundung arbeiten auf `0..K−1`).
* **Frühstopp**: Falls **UB ≤ LB**, sofort als optimal beenden.

**Implementationsbezug**: `dsatur_coloring(G)` bzw. `smallest_last_coloring(G)`; anschließend `K = UB`.

---

## D0/E1 — Frühstopp (falls UB ≤ LB)

Wenn `UB ≤ LB`, sofort aktuelle Lösung zurückgeben (optimal). Der Code prüft dies nach der Initialisierung.

---

## N3 — Inkrementelles LP einmalig aufbauen

```text
Class IncrementalLP(G, allowed_colors, clique_nodes, extra_cliques, add_precedence=True)
  Input:
    G: Graph
    allowed_colors: List[int] = [0,1,…,K-1]
    clique_nodes:  List[int]  — Q aus N1 (Symmetriebruch/Anker)
    extra_cliques: Optional[List[List[int]]] — zusätzliche (nahezu) maximale Cliquen (stärkere Ungleichungen)
    add_precedence: bool — ob Präzedenz y_{c+1} ≤ y_c hinzugefügt wird
  Output :
    solver: OR-Tools-LP-Modell
    x_vars: Dict[(v:int,c:int)→Var] — Zuweisungsvariablen X∈[0,1]
    y_vars: Dict[c:int→Var]         — Farbaktivierungsvariablen Y∈[0,1]
    V, C: Kopien von Knoten- und Farbmenge
  Modell (minimize Σ_c Y[c]) mit Nebenbedingungen:
    ∑_c X[v,c] = 1;        X[u,c]+X[v,c] ≤ Y[c] für (u,v)∈E;
    X[v,c] ≤ Y[c];         Y[c] ≤ ∑_v X[v,c]  // verbietet „leere Farben“
    ∑_{v∈Q} X[v,c] ≤ 1 für jede Clique Q in {clique_nodes}∪extra_cliques
    Präzedenz: Y[c+1] ≤ Y[c]  // Symmetriebrechung
```

**Implementationsbezug**: Aufbau in `ilp/model.py`, gekapselt in `IncrementalLP.__init__`; zusätzliche Cliquen via `top_maximal_cliques`.

---

## L0/E2 — Zeitprüfung

Die Schleife ist direkt durch `time.time() - t0 <= time_limit_sec` begrenzt.

---

## N4 — LP lösen, (zLP, x*, y*) extrahieren

```text
Method IncrementalLP.solve()
  Output:
    info: Dict mit
      z_LP:  float     — optimaler Zielfunktionswert = Σ_c y_c (LP-Untergrenze)
      x_frac: Dict/Array — fraktionale x-Lösung (z. B. {(v,c):val} / {v:{c:val}} / {v:list})
      y_frac: Dict[int,float] — fraktionale y-Werte
  Zweck:
    OR-Tools aufrufen und anschließend via solve_lp_and_extract die drei Größen extrahieren.
```

---

## D1/K1/R1 — K gemäß ceil(zLP) enger setzen (sicherer Versuch + Rollback)

**K_target** ist die Zielgröße des verfügbaren Farbpräfixes `0..K_target-1` (Wahl `max(LB, ceil(zLP))`) und dient als vorübergehender Stellvertreter für **K**.
**tokens** sind „Schnappschüsse“ der alten Bounds der veränderten Variablen (Pointer, alte LB/UB), beim Präfixsperren für alle `c ≥ K_target` die `y_c`.
**try_apply_and_solve** ändert keine Parameter mehr, sondern löst auf dem modifizierten Modellsnapshot; falls nicht optimal/unzulässig, wird mit den tokens sofort zurückgerollt — `K` bleibt unverändert.

```text
Method IncrementalLP.lock_prefix_K(K_target)
  Input:
    K_target: int — Zielpräfixgröße (max(LB, ceil(zLP)))
  Output:
    tok: BoundsToken — rücksetzbarer Schnappschuss der Variablenbounds
  Effekt:
    Für alle c ≥ K_target: y_c = 0; anderes bleibt unberührt.
  
Method IncrementalLP.try_apply_and_solve([tok])
  Output:
    (ok_info, ok): (Optional[info], bool)
  Zweck:
    Nach Anwendung von tok lösen; bei Infeasibility/Nicht-Optimalität nichts behalten und aufruferseitig zurückrollen.
  
Fallback:
  Falls ok=False → inc.revert_all(tok) und K unverändert lassen (R1-Zweig).
```

---

## G1 — Multi-Start „Runden + Reparieren“ (im K-Farbraum)

Ziel: **Bei festem Farb-Budget `current_UB` im Präfix `0..current_UB-1` die fraktionale Lösung `(x_frac, y_frac)` in eine ganzzahlige zulässige Färbung überführen.**
Vorgehen: kleine Jitter `perturb_y` auf `y_frac` für mehrere globale Farbenordnungen; pro Ordnung DSATUR-ähnliche Knotenreihenfolge („Sättigung → Grad → max_c x* → kleiner Zufall“) und gieriges Runden; wenn keine Farbe passt, **Kempe-Kette/zweifarbiger Tausch** als Reparatur; das Ganze `restarts`-mal, Auswahl nach „möglichst wenige Farben → möglichst wenige Konflikte → größtmögliche Übereinstimmung mit `x_frac`“.

```text
Function ROUND_AND_REPAIR_MULTI(G, x_frac, y_frac, current_UB=K, restarts, seed, perturb_y)
  Input:
    x_frac: LP-Fraktionswerte als Präferenzen je (v,c)
    y_frac: LP-Fraktionswerte als globale Farbenpriorität
    current_UB: erlaubte Farbanzahl (=K)
    restarts: Anzahl unabhängiger Versuche
    seed: Reproduzierbarkeit
    perturb_y: sehr kleiner Jitter für Tie-Breaking bei y-Sortierung
  Output:
    cand: ganzzahlige Färbung (ggf. nachfolgend noch repariert)
  Ablauf (kurz):
    1) DSATUR-Reihenfolge der Knoten
    2) zulässige Farbe mit größtem x[v,c] wählen; wenn keine:
       kleiner Kempe-Tauschversuch; sonst temporär beste Farbe setzen und in Reparaturphase
    3) über alle Starts bestes Ergebnis wählen
```

---

## D2 — Zulässigkeitsprüfung

```text
Function VERIFY_COLORING(G, coloring, allowed_colors=range(K))
  Output: report: Dict{
    feasible: bool,
    num_conflicts: int,
    used_colors: List[int], num_used_colors: int,
    missing_nodes, out_of_range_nodes, bad_nodes, conflicts_sample, ...
  }
  Zweck:
    Globale Prüfung: Jeder Knoten hat eine ganzzahlige Farbe; Farbe liegt im erlaubten Satz; Endpunkte jeder Kante sind verschieden.
```

---

## F1 — Farben verdichten und lokale Zusammenführung

```text
Function COMPACT_COLORS(coloring)
  Output: (coloring_compact, used:int) — remappt die tatsächlich verwendeten Farben auf 0..used-1

Function CONSOLIDATE_COLORS(G, coloring, passes)
  Output: (coloring2, reduced:bool)
  Zweck:
    Kleine graphdomänige „Leerung der höchsten Farbschicht“ (Greedy + wenige Kempe-Züge); gelingt dies, sinkt used um 1.
```

---

## D3/U1/D4/E3 — UB aktualisieren / K synchronisieren / Optimalstopp

* Wenn `used < UB`: setze `UB ← used` und `best_coloring ← cand`.
* Wenn `UB < K`: nach einer UB-Verbesserung `lock_prefix_K(UB)` aufrufen, um das LP-Farbsuffix zu schließen und zu lösen; Erfolg → `K = UB`, Misserfolg → Rollback und Suche fortsetzen.
* Wenn `UB ≤ LB`: sofort optimal beenden.

---

## A0/A1/D5/U2/A2 — Beschleuniger: seitliches K+1-Runden & zusätzlicher UB−1-Versuch

```text
Function TRY_SIDE_ROUNDING(G, x_frac, y_frac, K, UB, restarts, perturb_y, seed, ...)
  Precond: nur zur Konstruktion einer besseren zulässigen Lösung; LP bleibt auf K gesperrt
  Do:
    side_colors := K+1; falls side_colors > UB → überspringen (keine Verbesserung möglich)
    im K+1-Farbraum ROUND_AND_REPAIR_MULTI → VERIFY → COMPACT → kleine CONSOLIDATE
  Output:
    (newUB:int, newCol:coloring, applied:bool)
  Update rule:
    Falls applied=True und newUB<UB → UB/best_coloring aktualisieren; bei Bedarf K per lock_prefix_K(UB) nachziehen.
```

*Hinweis*: „A2: Optional extra UB-1 try“ ist im Code als **weiterer** Aufruf desselben „UB-1 side attempt“ (über `try_side_rounding`) umgesetzt — gleicher Sinn, nur eine zusätzliche Chance.

---

## D6/P1/D7/U3/P2/D8/U4/FIX0 — Pack-Fenster (wenn UB == K+1)

```text
Function TRY_LP_GUIDED_PACK(G, best_coloring, x_frac, K, UB, ...)
  Precond: nur wenn (UB == K+1)
  Ziel:    Knoten der höchsten Farbschicht K zurück in 0..K-1 schieben, sodass UB um 1 sinkt
  Methode:
    S := {v | best_coloring[v] == UB-1}; S nach x_frac-Präferenz sortieren;
    v einzeln in Präfixfarben versuchen (Greedy + kleine Kempe-Züge); ist die höchste Schicht leer → Erfolg.
  Output:
    (newUB, newCol, ok)
```

Falls LP-geführt scheitert, folgt der **rein graphische** UB-1-Greedy-Versuch:

```text
Function TRY_GRAPH_UB1_GREEDY(G, best_coloring, UB, ...)
  Ziel: ohne LP, nur mit Greedy+Kempe alle Punkte in UB-1 Farben packen
  Output:
    (newUB, newCol, ok) — bei Erfolg gilt newUB = UB-1
```

Beide Erfolge triggern denselben „UB aktualisieren / K synchronisieren“-Pfad wie oben.

---

## FIX1/D9/T0/FIX2/D10/FIXR/R2 — Fixierungsstrategie & Rollback-Schutz (Phase 2)

Dies ist als Methode **noch nicht vollständig umgesetzt**; Logik und Syntax sind skizziert, die konkreten Operationen sind noch zu ergänzen.

1. **In sehr kleinem Umfang** werden nur **sehr sichere** (x[v,c]=1) fixiert (gemäß drei Pick-Strategien), strikt begrenzt durch `max_fix_per_round`, und **nur** auf Präfixfarben (c<K).
2. **Alles ist per Token rücksetzbar**: Ist `try_apply_and_solve` nicht erfolgreich, wird via `revert_all` vollständig zurückgerollt; Modell sowie UB/K bleiben unverändert.
3. **Wenn das LP nach Fixierung zulässig bleibt**, kann optional ein erneuter „Runden+Reparieren“-Versuch folgen; sinkt dabei **UB**, wird `lock_prefix_K(UB)` für eine **sichere K-Synchronisation** verwendet (nur bei erfolgreichem LP-Solve wirksam, sonst wird die Synchronisation zurückgenommen).

### Input

* `G`: Graph
* `inc`: Inkrementelles-LP-Handle (bietet `fix_vertex_color() / try_apply_and_solve() / revert_all()`)
* `x_frac, y_frac, zLP`: aktuelle fraktionale LP-Lösung
* `UB, K, LB`: Schranken und aktuelles Farbbudget
* `policy: str`: mit `+` verknüpfte Teilstrategien

  * `prefix_shrink`: **nur** Vorschläge zum Schließen von y-Farben (echtes Schließen in der K-Synchronisation, nicht in dieser Phase)
  * `strong_assign`: wähle `(v,c*)` mit Lücke `gap = (Top-1 − Top-2) ≥ strong_margin` als (x=1)
  * `rounded_support`: wähle `(v,c*)`, wenn mit der letzten Rundung konsistent und (x[v,c*] \ge 0.8)
* `strong_margin: float`: Schwelle (z. B. 0.2)
* `max_fix_per_round: int`: Obergrenze pro Runde (Überfixierung vermeiden)
* `rounding_coloring`: ganzzahlige Lösung aus der letzten Rundung (für Konsistenz)

### Output

* `applied: bool`: ob diese Fixierungen **erfolgreich angewandt und behalten** wurden
* `UB_out, K_out`: ggf. aktualisierte UB/K (typisch bleibt K unverändert; bei erfolgreicher anschließender Rundung kann sichere K-Synchronisation ausgelöst werden)
* `x_frac_out, y_frac_out, zLP_out`: **neue** fraktionale Lösung im Erfolgsfall; sonst unverändert
* `fixes_applied: int`: tatsächlich fixierte Anzahl in dieser Runde
* `best_coloring_out`: bei erfolgreicher nachgelagerter Rundung verbesserte ganzzahlige Lösung; sonst unverändert

### Minimaler 5-Schritte-Ablauf

```text
Methode pick and fix

1) Auswahl: fixes ← PICK_FIXINGS(G, x_frac, policy, max_fix, rounding_coloring), nur c < K behalten.
2) Versuch: tokens ← apply inc.fix_vertex_color(v,c) for (v,c) in fixes;
            (info, ok) ← inc.try_apply_and_solve(tokens);
            if not ok → inc.revert_all(tokens); return (UB, K, x_frac, y_frac, applied=False).

3) Annehmen: x_frac ← info.x_frac; y_frac ← info.y_frac.

4) Re-Rounding: (okR, cand, used) ← ROUND_AND_REPAIR_MULTI(G, x_frac, y_frac, K, …);
                if okR and used < UB → UB ← used; best ← cand.

5) K-Sync: if UB < K → (info2, ok2) ← inc.lock_prefix_K(UB).try_apply_and_solve();
           if ok2 → K ← UB; x_frac ← info2.x_frac; y_frac ← info2.y_frac;
           else → inc.revert_all()  // nur diese Synchronisation zurücknehmen
           return (UB, K, x_frac, y_frac, applied=True).
```

**Terminierungsprüfung T0:**
Wenn `ceil(zLP) ≥ UB` und `K ≤ UB`, gilt: „LP-Untergrenze hat die aktuelle zulässige Obergrenze erreicht, und das Präfix ist nicht breiter als UB“ ⇒ plausibel kein weiteres Verbesserungspotenzial; Runde beenden.

---

## E4 — Beenden (keine weitere Verbesserung)

Wenn der T0-„Halt“-Zweig genommen wird, die Zeit abgelaufen ist oder bei `K == ceil(zLP)` über mehrere Runden keine Verbesserung eintritt (im Code z. B. `stall_after_K=8`), die Endzusammenfassung ausgeben und zurückkehren.

---
