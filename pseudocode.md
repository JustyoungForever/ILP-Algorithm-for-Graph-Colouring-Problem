
```bash
python3 main.py --algo iterlp2 --time 60 --seed 0 \
  --init-heuristic smallest_last \
  --fix-policy prefix_shrink+rounded_support \
  --max-fix-per-round 20 \
  --restarts 48 --perturb-y 1e-6
```
              
---
  
# MAIN (Einstieg & Steuerung)

```text
Function MAIN(args)  // Haupteinstieg: Argumente parsen → Graph vorbereiten → 
                   ausgewählten Algorithmus ausführen → Zusammenfassung ausgeben

  G ← LOAD_GRAPH(args.seed)  
 // Graph laden oder generieren; 
      der Seed beeinflusst die Zufälligkeit und macht Experimente reproduzierbar.

  if args.algo == "dsatur" then  
  |  // Falls nur die Baseline ohne LP laufen soll –
  |       gut als Vergleich zum Hauptverfahren.
  |
  |  col ← DSATUR(G)  
  |  // Klassischer Greedy-Färber: färbt in Sättigungs-Reihenfolge und
  |                     liefert eine zulässige Lösung.
  |                     
  |  PRINT(VERIFY(G, col), used=|colors(col)|)  
  |  // Prüfe Konfliktfreiheit und gib die Farbanzahl aus (als UB-Baseline).
  else  // iterlp2  
    // Ansonste meine Hauptpipeline „iteratives LP + Runden + Reparatur“ starten.

    base ← DSATUR(G)  
    // Erst DSATUR laufen lassen, um ein zulässiges UB (Vergleichswert) zu bekommen.

    PRINT(VERIFY(G, base), used=|colors(base)|)  
    // Qualität der Baseline ausgeben, damit man spätere Verbesserungen einordnen kann.

    res  ← ITER_LP_V2(G, time_limit=args.time,      // wie lange laufen
                         init_heuristic=args.init_heuristic,        // womit Startlösung, UB und K initial bestimmen
                         fix_policy=args.fix_policy,            // aus x_* und y_* Fixierungen auswählen
                         restarts=args.restarts,    // wie viele Rounding-Versuche für die Fraktionslösung
                         perturb_y=args.perturb_y,  // wie stark y-Prioritäten stören, um Gleichstände/Replays zu brechen
                         max_fix_per_round=args.max_fix_per_round       // max. Anzahl (v,c) pro Runde fixieren, damit das LP nicht „festfährt“
                         )
    // Kernalgorithmus aufrufen. Nur die „Strategie-Parameter“ übergeben; Rechen-Details stecken in den Bibliotheksfunktionen.

    PRINT(res.summary)  
    // Am Ende UB/LB und Stoppgrund (Optimum erreicht/Stillstand/Zeitlimit) ausgeben – hilft bei der schnellen Laufkontrolle.
End
```

---

# ITERLP2 (Kern: iteratives LP v2)

```text
Algorithm ITER_LP_V2(G, time_limit, init_heuristic, fix_policy, restarts, perturb_y, max_fix_per_round)

  LB ← CLIQUE_LOWER_BOUND(G)  
  // Untere Schranke per schneller (approximativer) Cliquen-Enumeration.
  // Q bezeichnet eine solche Clique (also eine Menge von Knoten), und |Q| ist ihre Mächtigkeit/Größe (die Anzahl der Knoten in dieser Clique).

  col0 ← (init_heuristic == "smallest_last" ? SMALLEST_LAST(G) : DSATUR(G))  
  // Initiale Lösung: Smallest-Last liefert auf dünnen Graphen oft engere Obergrenzen als DSATUR.
  UB  ← |colors(COMPACT(col0))|  
  // Farben zuerst auf 0..k-1 komprimieren und dann zählen ⇒ aktuelle beste zulässige Obergrenze.
  // Das Komprimieren reduziert Folgekosten bei nachgelagerten Umnummerierungen.
  K   ← UB  
  // Das LP erhält zunächst K Farbschächte (Präfix 0..K-1), gleich der aktuellen UB.
  // Begründung: Nicht sofort strenger als die vorhandene zulässige Lösung werden, um Infeasibilität zu vermeiden.

  if UB ≤ LB then return RESULT(UB, LB, col0, stop="UB==LB")  
  // Falls UB bereits gleich LB ist, ist die Lösung optimal; kein LP-Zyklus nötig.

  INC ← BUILD_INCREMENTAL_LP(G, colors=0..UB-1, with_precedence=True, with_clique_cuts=True)  
  // Einmaliger Aufbau des LPs: Variablen x[v,c], y[c]; Ziel min Σ y[c];
  // Basisrestriktionen + Cliquen-Schnitte und Präzedenzbedingungen.
  // „Incremental“: In späteren Schritten werden nur Schranken (Bounds) angepasst (Präfix sperren, x fixieren), kein Neuaufbau.

  while time_remaining(time_limit) do  
    // Hauptschleife: läuft bis zum Zeitlimit; versucht innerhalb des Budgets UB so weit wie möglich zu senken.

    (zLP, x*, y*) ← INC.SOLVE()  
    // LP-Relaxation lösen. zLP ist eine Untergrenze für χ(G); x*, y* liefern Präferenzen („wohin runden“).

    if ceil(zLP) < K then  
      // Theoretisch werden mindestens ceil(zLP) Farben benötigt.
      // Ist das kleiner als das aktuelle zulässige K, wird der Suchraum vorsichtig verengt.
      K_target ← max(LB, ceil(zLP))  
      // K niemals unter LB absenken (sonst würde selbst die Theorie unterlaufen).
      if INC.TRY_APPLY_AND_SOLVE(LOCK_PREFIX(K_target)) == ok then  
        // Präfixsperre anwenden: y[c]=0 für c≥K_target und LP sofort erneut lösen.
        // Nur bei Erfolg übernehmen; so verhindern wir, dass das Modell in Unzulässigkeit „eingefroren“ wird.
        K ← K_target  
      else
        INC.REVERT()  
        // Bei Unzulässigkeit alle Änderungen zurücknehmen und fortfahren.

    cand ← ROUND_AND_REPAIR_MULTI(G, x*, y*, K, restarts, perturb_y)  
    // Rundung + Reparatur (Multi-Start): erst x*/y*-geführt „gute“ Färbungen versuchen,
    // bei Bedarf Kempe-Ketten und kleine Reparaturen anwenden.
    // Ziel: aus der Relaxationslösung schnell eine echte zulässige Ganzzahl-Färbung mit ≤ K Farben konstruieren.

    if VERIFY(G, cand, allowed=0..K-1) == feasible then  
      // Erfolgreich gerundet (keine Konflikte, Farben im zulässigen Bereich) ⇒ nun Farben weiter verdichten.
      
      cand ← CONSOLIDATE_COLORS(G, coloring, passes) 
      // Eingaben:
      //   G: Graph;
      //   coloring: aktuelle Ganzzahl-Färbung;
      //   passes: Anzahl der Konsolidierungsdurchläufe (Standard 10).
      // Ausgaben:
      //   new_coloring, reduced: neue Färbung + Flag, ob die Farbanzahl reduziert wurde (True/False).
      // Idee: von der höchsten Farbebene (z. B. k-1) nach unten umfärben (Zielmengen {0,…,k-2});
      // praktikabel via Greedy + Kempe-Kette.
      // Ist die höchste Ebene leergeräumt, sofort umnummerieren (verwendete Farben auf 0..k'-1 abbilden),
      // dies gilt als „erfolgreiche Reduktion“. Maximal ‚passes‘ Runden wiederholen.
      // Begründung: einzelne Restfarben enthalten oft nur wenige Knoten und lassen sich zusammenführen.

      // colors(cand): Menge der in cand tatsächlich verwendeten Farben (z. B. {0,1,3}).
      // |colors(cand)|: Anzahl der unterschiedlichen verwendeten Farben (im Beispiel 3).
      // best_coloring: aktuell beste zulässige Färbung (Abbildung v→color).

      if |colors(cand)| < UB then  
        // Echter Fortschritt: Obergrenze sinkt. Nur solche Verbesserungen zählen als substanziell.
        UB ← |colors(cand)|; 
        best_coloring ← cand;

        if K > UB and das LP unter der Einschränkung „nur die ersten UB Farben“ weiterhin eine zulässige optimale Relaxationslösung liefert then  
          // Falls das LP mit engerem Präfix noch lösbar ist, K mit der neuen UB synchronisieren.
          K ← UB
        if UB ≤ LB then return RESULT(UB, LB, best_coloring, stop="UB==LB")  
        // Treffen sich Ober- und Untergrenze, ist Optimalität konstruktiv und theoretisch belegt ⇒ sofort zurückgeben.

    else  
      // Rundung fehlgeschlagen: innerhalb des Präfixes K ist ein Soforterfolg schwer.
      // Seitliche Exploration zur Vermeidung von Sackgassen:
      if (K+1) < UB then  
        // Nur versuchen, wenn „K+1“ streng besser als aktuelle UB ist;
        // ist K+1 ≥ UB, bringt Aufweiten keinen Sinn und wird übersprungen.

        (ok1, cand1) ← TRY_SIDE_ROUNDING(G, x*, y*, target_UB=K+1, restarts, perturb_y) 
        // TRY_SIDE_ROUNDING: eine R&R-Runde mit Budget K+1 (Multi-Start).
        // Dabei restarts = max(4, restarts//2). Beispiel: bei restarts=16 spart dies gegenüber der Haupt-R&R Ressourcen.

        if ok1 and |colors(cand1)| < UB then  
          // Seitengang liefert direkt eine kleinere UB ⇒ übernehmen.
          UB ← |colors(cand1)|; best_coloring ← cand1
          if UB ≤ LB then return RESULT(UB, LB, best_coloring, stop="UB==LB")

      if (UB-1) ≥ LB then  
        // Optional gegenläufig probieren: Budget „UB-1“, sofern nicht unter LB; verletzt die Untergrenze nicht.
        (ok2, cand2) ← TRY_SIDE_ROUNDING(G, x*, y*, target_UB=UB-1, restarts, perturb_y)
        if ok2 and |colors(cand2)| < UB then
          UB ← |colors(cand2)|; best_coloring ← cand2
          if UB ≤ LB then return RESULT(UB, LB, best_coloring, stop="UB==LB")

    if UB == K+1 then  
      // Wichtiges „Fenster“: UB ist genau um 1 größer als K.
      // Das LP ist bereits sehr straff; es fehlt nur, die höchste Farbe in das Präfix zurückzupacken.
      (ok_pack, col_pack) ← TRY_LP_GUIDED_PACK(G, best_coloring, x*, K, UB)  
      // LP-geleitetes Packen: Knoten der höchsten Farbe (UB-1) bevorzugt in diejenigen Präfixfarben schieben,
      // die für sie gemäß x* am „freundlichsten“ sind.
      // TRY_LP_GUIDED_PACK – Kerngedanke:
      //   1) K sei die höchste Farbebene; alle Knoten mit color==K in {0..K-1} verlagern.
      //   2) Zielreihenfolge je Knoten v durch x_frac: c∈{0..K-1} absteigend nach x_frac[(v,c)] sortieren
      //      und in dieser Reihenfolge versuchen.
      // Konkrete Schritte:
      //   a.1) targets = sort_{c ∈ 0..K-1} x_frac[(v,c)] (absteigend)
      //   a.2) zunächst Greedy-Umfärben ohne Konflikt; falls nicht möglich:
      //        Kempe-Ketten-Tausch auf der Zweifarbkomponente {c_now=K, c_target}.
      //   b) ist die höchste Ebene leer, sofort kompakt umnummerieren (0..k-1).

      if ok_pack and |colors(col_pack)| < UB then
        UB ← |colors(col_pack)|; best_coloring ← col_pack
        if K > UB then (try sync K→UB via LOCK_PREFIX)  
        if UB ≤ LB then return RESULT(UB, LB, best_coloring, stop="UB==LB")

      (ok_g,  col_g) ← TRY_GRAPH_UB1_GREEDY(G, best_coloring, UB)  
      // Zusätzlich ein graphweites Greedy/Kempe-„UB-1“ (ohne LP-Führung),
      // um Fälle abzudecken, in denen das LP-geführte Packen scheitert.
      if ok_g and |colors(col_g)| < UB then
        UB ← |colors(col_g)|; best_coloring ← col_g
        if K > UB then (try sync K→UB via LOCK_PREFIX)
        if UB ≤ LB then return RESULT(UB, LB, best_coloring, stop="UB==LB")

    // Auf Basis der LP-Relaxation (z_LP, x_frac, y_frac) und des aktuellen Budgets UB(=K)
    // wird ein vorsichtiger Fixierungsplan erstellt, der nur „sichere“ Fixierungen vornimmt.

    /* plan = {
          "y_zero": [Farben c, die auf 0 gesetzt werden sollen],   # Präfixverkürzung: y_c=0 für c≥K_new
          "x_one":  [(v,c)-Paare],                                 # starke Zuweisung: x[v,c]=1 (andere Farben von v implizit 0)
          "x_zero": [(v,c)-Paare],                                 # (meist leer; derzeit selten genutzt)
      }
    */
    // Nach dem Anwenden von x_one wird das LP erneut gelöst; bei Erfolg folgt eine kleine R&R-Runde,
    // um zu prüfen, ob UB weiter sinkt.

    //   x*: LP-Fraktionswerte (v,c)↦[0,1], „Neigung“ von v für Farbe c;
    //   y*: LP-Fraktionswerte für Farbnutzung c↦[0,1] (primär für Statistik/optionale Strategien);
    //   zLP: aktueller LP-Zielfunktionswert/Untergrenze; ceil(zLP) liefert eine Mindestfarbanzahl;
    //   UB=K: aktuelle Präfixgröße (zulässige Farben 0..K-1);
    //   LB: Untergrenze (z. B. Cliquen-LB); Präfix darf nicht unter LB geschrumpft werden;
    //   policy: Strategiekette, z. B. "prefix_shrink+strong_assign+rounded_support";
    //   max_fix_per_round: Maximalzahl der (v,c), die pro Runde fixiert werden (gegen „Überfixierung“);
    //   support_from_round=cand: letzte ganzzahlige Rundung zur Konsistenzprüfung („mit der Rundung übereinstimmend“).

    plan ← PICK_FIXINGS(x*, y*, zLP, UB=K, LB, policy=fix_policy,
                        max_fix_per_round=max_fix_per_round, support_from_round=cand)
    // Hintergrund: Vor jeder Runde wird das LP neu gelöst; viele Knoten haben sehr ähnliche x*-Werte für mehrere Farben.
    // Ohne Fixierungen pendeln solche Knoten zwischen Alternativen (A↔B), was die Konvergenz erschwert und UB-Senkungen hemmt.
    // Daher werden Knoten fixiert, bei denen x[v,c] für eine Farbe deutlich über dem Zweitbesten liegt
    // und (in der Variante „rounded_support“) zudem mit der letzten Rundung übereinstimmt:
    //   Für jeden v in Farben 0..UB-1 die beste (c1, v1) und zweitbeste Wertung v2 bestimmen.
    //   Falls v1 − v2 ≥ strong_margin (Standard 0.25), wird (v, c1) in plan["x_one"] aufgenommen.
    // Pro Runde höchstens max_fix_per_round Einträge fixieren (Modell nicht „festfahren“);
    // falls LP unzulässig, sofort vollständig zurückrollen.
    // Ziel: Suchraumschwankungen reduzieren und nachfolgende LP-/Rundungsschritte stabilisieren.

    if plan.x_one ≠ ∅ then
      if INC.TRY_APPLY_AND_SOLVE(FIX_X_ON(plan.x_one where color<K)) == ok then
        // Nur Variablen im Präfix K fixieren; so kollidiert es nicht mit der Präfixsperre.
        cand3 ← ROUND_AND_REPAIR_MULTI(G, INC.x*, INC.y*, K, max(2, ⌊restarts/2⌋), perturb_y)
        // Nach Fixierung eine kompakte R&R-Runde (mit halbierten/reduzierten Neustarts); bei Erfolg UB aktualisieren.
        
        if VERIFY(G, cand3, allowed=0..K-1) == feasible and |colors(COMPACT(cand3))| < UB then
          UB ← |colors(COMPACT(cand3))|; best_coloring ← COMPACT(cand3)
          if K > UB then (try sync K→UB via LOCK_PREFIX)
          if UB ≤ LB then return RESULT(UB, LB, best_coloring, stop="UB==LB")
      else
        INC.REVERT()  
        // Bei Unzulässigkeit alle Fixierungen rückgängig machen und die Schleife fortsetzen.

    if ceil(zLP) ≥ UB and K ≤ UB then  
      // Abbruchkriterium 1: LP-Untergrenze ist bereits ≥ UB und K ist nicht kleiner als UB.
      // Weitere Verbesserungen sind höchst unwahrscheinlich.
      return RESULT(UB, LB, best_coloring, stop="no_better_than_UB")

    if STALLING_AT_FIXED_K() then  
      // Abbruchkriterium 2: Bei K≈ceil(zLP) gab es über mehrere Runden keine substanziellen Fortschritte
      // (kein engeres K, keine Fixierungen, keine UB-Senkung) ⇒ Stagnation.
      return RESULT(UB, LB, best_coloring, stop="stalled")

  end while

  return RESULT(UB, LB, best_coloring, stop="time_limit")  
  // Abbruchkriterium 3: Zeitlimit erreicht (reproduzierbares Laufzeitbudget).
End

```

---

# INCREMENTAL_LP (einmal modellieren, dann nur Bounds ändern)

```text
Class INCREMENTAL_LP
  State:
    solver                        // LP-Solver
    X[v,c] ∈ [0,1]                // Vertex-Farb-Variablen (LP-Relaxation, geben Fraktions-Hinweise)
    Y[c] ∈ [0,1]                  // Farb-Aktivierungsvariablen (LP-Relaxation; steuern „Farb-Slots“)
    bounds_stack                  // Für Rollback: alte Bounds der veränderten Variablen

  Method BUILD(G, allowed_colors, root_clique, extra_cliques, add_precedence)
    Variablen X, Y erstellen, Ziel min Σ_c Y[c]                  // Ziel: so wenig aktivierte Farben wie möglich (LP-Untergrenze)
    Für jedes v: Σ_c X[v,c] = 1                 // Jeder Knoten genau eine Farbe (auch fraktional erlaubt)
    Für jede Kante (u,v) und jede Farbe c: X[u,c] + X[v,c] ≤ Y[c]   // Assignment-ähnlich: gleiche Farbe ichd von Y[c] begrenzt;
                                                          // wenn das Prefix Y[c]=0 setzt, ist die Farbe im ganzen Graph verboten
    Für jedes (v,c): X[v,c] ≤ Y[c]                     // Kopplung: Knotenfarbe darf Aktivierung nicht überschreiten
    Für jede Farbe c: Y[c] ≤ Σ_v X[v,c]              // Kein „leerer“ Farbslot (Y>0 aber niemand nutzt ihn)
    Clique-Ungleichungen für root_clique und extra_cliques        // Clique-Cuts: Σ_{v∈Q} X[v,c] ≤ 1, stärken die Untergrenze
    Falls add_precedence: für c=0..K-2 Y[c+1] ≤ Y[c]        // Symmetriebrecher: Farben in Prefix-Reihenfolge aktivieren – hilft bei Konvergenz/Prefix-Sperre
    return self                            // Inkremental-Objekt: später nur Bounds ändern, mit Rollback

  Method SOLVE()
    solver.Solve() aufrufen, zLP, x_frac, y_frac extrahieren      // LP lösen, Untergrenze + Fraktionslösung bekommen
    return (zLP, x_frac, y_frac)                                  // y_frac ist oft nicht-integer: dient als Farb-Priorität

  Method LOCK_PREFIX_K(K_target)
    Token erzeugen und alle Y[c] für c ≥ K_target auf 0 setzen    // Prefix sperren: erlaubte Farben auf 0..K_target-1 einschränken
    return token                                                  // Für Try/Rollback – nur behalten, wenn das Re-Solve ok ist

  Method FIX_VERTEX_COLOR(v,c)
    Token erzeugen, X[v,c]=1 und X[v,c'≠c]=0 setzen               // Eine Knotenfarbe festnageln; stabilisiert LP/Runding danach
    return token

  Method TRY_APPLY_AND_SOLVE(token or tokens[])
    Bounds anwenden und lösen                                     // Bei Erfolg neues (zLP,x*,y*), sonst soll der Aufrufer zurückrollen
    return (info, ok)

  Method REVERT(token or tokens[])
    Alte Bounds aus dem Token wiederherstellen                    // Streng umkehrbar – sichere iterative Exploration
    Token verwerfen
End
```

---

# LP-Modell

```text
Function BUILD_ASSIGNMENT_LP(G, K, cliques, with_precedence)
  Variablen anlegen: X[v,c] ∈ [0,1], Y[c] ∈ [0,1]                 // LP-Relaxation für fraktionale Hinweise
  Ziel: minimize Σ_{c=0}^{K-1} Y[c]                                // Wenig aktivierte Farben → Untergrenze für χ(G)

  Nebenbed.: ∀v, Σ_c X[v,c] = 1                                   // Jeder Knoten genau eine Farbe
  Nebenbed.: ∀(u,v)∈E, ∀c, X[u,c] + X[v,c] ≤ Y[c]                  // Wenn Y[c]=0, dann ist die Farbe global verboten; spielt gut mit Prefix-Sperre zusammen
  Nebenbed.: ∀v,c, X[v,c] ≤ Y[c]                                   // Kopplung (hält die Relaxation enger), x ichd von y „eingeklemmt“
  Nebenbed.: ∀c,   Y[c] ≤ Σ_{v∈V} X[v,c]                           // Keine leeren Farben zulassen (verhindert künstlich kleine y)
  Clique-Cuts: ∀Q∈cliques, ∀c, Σ_{v∈Q} X[v,c] ≤ 1                  // Einige Cliquen stärken die LP-Untergrenze (optional, aber empfehlenswert)
  Falls with_precedence: ∀c=0..K-2, Y[c+1] ≤ Y[c]                   // Prefix-Aktivierungsreihenfolge – bricht Symmetrien und hilft K schrittweise zu schrumpfen

  return (solver, X, Y)
End
```

---

# ROUND_AND_REPAIR (Multi-Start & Single-Run)



```text
  |  cand ← ROUND_AND_REPAIR_MULTI(G, x*, y*, K, restarts, perturb_y)  
  |  // Zweck: Aus der LP-Fraktionslösung x*, y* mehrere „Rounding+Repair“-Versuche (Mehrfachstart) durchführen,
  |  //         um eine ganzzahlige zulässige Färbung cand mit höchstens K Farben zu erhalten.
  |  // Bedeutungen der Variablen:
  |  //   G         : Eingabegraph.
  |  //   x*        : LP-Fraktionswerte für Knoten-Farb-Paare (x*[v,c]∈[0,1]; größer ⇒ LP „bevorzugt“ v→c).
  |  //   y*        : LP-Fraktionswerte für Farbaktivierung (y*[c]∈[0,1]; größer ⇒ Farbe c stärker „aktiv“).
  |  //   K         : aktuelle Obergrenze (UB); Ziel ist eine Lösung mit ≤ K Farben.
  |  //   restarts  : Anzahl der Neustarts (unabhängige Versuche zur Steigerung von Robustheit/Qualität).
  |  //   perturb_y : Amplitude einer sehr kleinen Störung auf y* (Mikro-Störung), um Bindungen im Ranking zu brechen.

Function ROUND_AND_REPAIR_MULTI(G, x*, y*, current_UB, restarts, seed, perturb_y)
  best <- None                                    // aktuell beste zulässige Lösung (primär: minimale Farbanzahl)
  rnd <- init_rnd(seed)                           // deterministische Zufallsquelle (Reproduzierbarkeit)

  for r in 1..restarts do                         // Mehrfachstart: restarts unabhängige Versuche
    yj <- ADD_TINY_NOISE(y*, scale=perturb_y(float:1e-6), rnd=rnd)  // für alle c: yj[c]=y*[c]+U(-scale,+scale)
    // yj: y* mit winziger Zufallsstörung; bricht Gleichstände in der Farbpriorität auf,
    //     damit die Farbreihenfolge pro Versuch leicht variiert.

    cand <- ROUND_AND_REPAIR(G, x*, yj, current_UB, rnd)
    // Ein einzelner R&R-Durchlauf: aus der Fraktionslösung eine ganzzahlige Färbung mit ≤ current_UB Farben konstruieren.

      best <- cand
      // Falls cand zulässig ist und weniger Farben nutzt, best aktualisieren (beste Lösung merken).

  return (best if best≠None else infeasible)
  // Wenn alle Versuche scheitern ⇒ Unzulässigkeits-Marker; sonst die beste zulässige Lösung zurückgeben.
End

Function ROUND_AND_REPAIR(G, x*, yj, current_UB, rnd)
  order_colors <- sortiere nach yj (absteigend: größere Werte haben Priorität)
  // Farbpriorität: je „aktivierter“ (höher yj), desto früher probieren.
  // Ergebnis: eine Reihenfolge wie [Farbe0, Farbe1, Farbe2, …].

  order_vertices <- dynamische Auswahl der noch ungefärbten Knoten
  // Prioritäten der Auswahl:
  // 1) Höchste Sättigung zuerst (Sättigung = Anzahl verschiedener Farben in bereits gefärbten Nachbarn);
  //    „schwierige“ Knoten früh behandeln, um Fehlentscheidungen zu vermeiden.
  // 2) Bei gleicher Sättigung: höhere Gradzahl zuerst (mehr Nachbarn ⇒ stärker eingeschränkt).
  // 3) Bei gleicher Sättigung und Grad: vergleiche das Maximum von x* über dem Farbbereich 0..K-1
  //    (ohne vorherige Konflikt-Filterung) – wer das größere max_c x*[v,c] hat, kommt früher.
  // 4) Sind auch diese gleich, füge ein minimales Zufallsrauschen (≈1e-9) zur stabilen Tie-Break-Entscheidung hinzu.

  init coloring <- EMPTY
  // leere Ganzzahl-Färbung (z. B. coloring[v]=None)

  for v in order_vertices do
    cand_colors <- FILTER_AVAILABLE_COLORS(v, coloring, order_colors)
    // Kandidatenmenge: in der durch order_colors vorgegebenen Reihenfolge alle Farben,
    // die v konfliktfrei annehmen kann (keiner seiner gefärbten Nachbarn hat diese Farbe).

    prefer <- argmax_c x*[v,c] über cand_colors
    // Wähle in der zulässigen Kandidatenmenge cand_colors die Farbe c mit maximalem x*[v,c].
    // Motivation: Wir runden entlang der stärksten LP-Präferenz, solange dies konfliktfrei möglich ist.

    if prefer exists then
      ASSIGN(v, prefer)
      // Wenn die LP-präferierte Farbe zulässig ist, weise sie direkt zu (respektiert die LP-Hinweise).

    else
      if TRY_KEMPE_CHAIN_MOVE(G, coloring, v, order_colors) then
        continue
        // Keine zulässige Farbe vorhanden ⇒ Kempe-Kette versuchen:
        // TRY_KEMPE_CHAIN_MOVE():
        //   Auf einer zweifarbigen, alternierenden Komponente {a,b} einen globalen Farbswap durchführen
        //   (a↔b), um für v eine Farbe freizuräumen. Bei Erfolg nächster Knoten.

      else
        // Fallback: setze die Farbe argmax_c x*[v,c] auch dann, wenn vorübergehend Konflikte entstehen;
        // Reparatur wird in nachfolgenden lokalen Korrekturen erledigt.
        ASSIGN(v, argmax_c x*[v,c] über 0..current_UB-1)  // bewusst „temporär“ erlaubt
        

  coloring <- SMALL_REPAIR_PASSES(G, coloring, passes=2..3)
  // coloring: aktuelle Ganzzahl-Färbung (Abbildung v→color).
  // passes : Anzahl äußerer Schleifendurchläufe der lokalen Reparatur/Kompression.
  // Lokale Reparatur:
  //   höhergradige Knoten bevorzugt behandeln; zuerst freie (konfliktfreie) Farben (ggf. gemäß y*-Priorität) versuchen,
  //   dann Kempe-Tausch; wenn keine Verbesserung möglich, Änderungen zurückrollen.
  // Ziel: verbliebene schwache Konflikte oder überflüssige Farben beseitigen.

  feasible <- CHECK_NO_CONFLICTS
  // Endprüfung: keine gleichfarbigen Nachbarn; alle Farben im Bereich < current_UB.

  return (coloring if feasible else INFEASIBLE)
  // Zulässig ⇒ Färbung an die äußere Ebene zur UB-Prüfung zurückgeben;
  // sonst Unzulässigkeits-Marker, damit die äußere Ebene andere Neustarts/Seitenschritte versucht.
End

```

---

# drei Methode (Side / Pack / Greedy)

```text
Function TRY_SIDE_ROUNDING(G, x*, y*, target_UB, restarts, perturb_y)
  cand ← ROUND_AND_REPAIR_MULTI(G, x*, y*, K=target_UB, restarts, perturb_y)  
  // „Erlaubte Farbanzahl“ als K in R&R stecken (auch wenn sie target_UB heißt – im R&R ist das das Prefix-Budget).
  if VERIFY(G, cand, allowed=0..target_UB-1) == feasible then
    return (True, CONSOLIDATE_COLORS(G, COMPACT(cand)))  
    // Kompaktierte zulässige Lösung zurückgeben; die aufrufende Ebene prüft, ob’s besser als das aktuelle UB ist.
  else
    return (False, None)  
    // Unzulässig → diese Seitenroute abbrechen, keine Zeit verschwenden.
End


Function TRY_LP_GUIDED_PACK(G, best, x*, K, UB)
  if UB ≠ K+1 then return (False, None)  
  // Nur im „Differenz-1“-Fenster aktivieren; sonst ist Aufwand/Nutzen schlecht.

  S ← vertices with color = UB-1 in best  
  // Die Knoten der höchsten Farbschicht. Ziel: alle zurück in 0..K-1.

  order S by preference from x* (desc)  
  // Zuerst Knoten mit starker x*-Präferenz für niedrige Farben bewegen – höhere Erfolgsquote, kleinere Nebenichkungen.

  for v in S do
    try place v into prefix 0..K-1 using greedy + small local moves  
    // Erst Greedy; bei Konflikt kleine lokale Moves (z.B. Kempe), aber mit Schritt-Limit – kein übermäßiges Suchen.

  if all moved then
    return (True, COMPACT(best_after_moves))  
    // Oberste Farbe leer → UB um 1 gesenkt.
  else
    return (False, None)  
    // Nicht alles umgezogen → „Packen“ zählt nicht als Erfolg. Andere Strategien probieren.
End


Function TRY_GRAPH_UB1_GREEDY(G, best, UB)
  attempt ← greedy + limited Kempe to fit colors into 0..UB-2  
  // Reine Graph-Heuristik: ohne LP, mit Greedy/Kempe versuchen, alles in UB-1 Farben zu pressen.
  if success then return (True, COMPACT(attempt)) else return (False, None)
  // Erfolg → zulässige UB-1-Lösung; sonst zeigt’s, dass es graphbasiert aktuell auch nicht klappt.
End
```

---

# Fixier-Strategie (PICK_FIXINGS)

```text
Function PICK_FIXINGS(x*, y*, zLP, UB, LB, policy, max_fix_per_round, support_from_round)
  plan.x_one ← ∅  
  // Hier nutzen ich nur „x[v,c]=1 festnageln“. y_zero (Prefix-Schrumpfen) macht LOCK_PREFIX – saubere Aufgabentrennung.

  if "prefix_shrink" in policy then
    K_hint ← max(LB, ceil(zLP))  
    // LP-Untergrenze als Hinweis, welche hinteren Farben man schließen könnte; ichklich schließen tut LOCK_PREFIX.

  if "rounded_support" in policy and support_from_round ≠ None then
    // Nur Knoten fixieren, bei denen LP-Präferenz und die letzte Rundung übereinstimmen – reduziert LP/Rundungs-Divergenzen (entzittert).
    candidates ← { v | argmax_c x*[v,c] == support_from_round[v] and x*[v,that] ≥ 0.8 }
    // 0,8 ist eine Erfahrungs-Schwelle: höher = konservativer, niedriger = riskanter. Idee: erst die „offensichtlich richtigen“ Nägel setzen.
    plan.x_one ← take up to max_fix_per_round from candidates, highest x* first
    // Hartes Budget, um nicht zu viele Variablen auf einmal festzunageln → sonst blockiert das LP. Schrittweise ist stabiler.

  return plan  
  // Die Anwendung (mit Rollback-Schutz) übernimmt FIX_X_ON auf der oberen Ebene.
End
```

---

# Verifikation & Visualisierung (VERIFY / VIZ)

```text
Function VERIFY(G, coloring, allowed)
  if coloring == None then return {feasible:False, reason:"empty"}
  if EXISTS v with coloring[v] ∉ allowed then
    return {feasible:False, reason:"color_out_of_range"}
  for each edge (u,v) in E(G) do
    if coloring[u] == coloring[v] then
      return {feasible:False, reason:"conflict", edge:(u,v)}
  return {feasible:True, reason:"ok"}
End

Function VIZ(enabled, tag, round_id, coloring, K_or_UB)
  if not enabled then return                                // Visualisierung kann deaktiviert sein
  IMG <- DRAW_PARTIAL_COLORING(G, coloring, highlight_conflicts=True,
                               title=f"{tag} r{round_id} K={K_or_UB}")
  SAVE_IMAGE(IMG, dir=viz_out_dir, name=f"{tag}-{round_id}.png")
End
```

---

#  (Kompaktieren / Zusammenführen )

```text
Function COMPACT_COLORS(coloring)
  remap <- BUILD_ORDERED_MAP(sorted(unique(coloring.values()))) // Remapping-Tabelle für Farben bauen
  for v in VERTICES(G) do coloring[v] <- remap[coloring[v]]     // Farben auf 0..k-1 komprimieren
  return (coloring, |remap|)                                   // Neue Lösung + Farbanzahl
End

Function CONSOLIDATE_COLORS(G, coloring, passes)
  changed <- False
  repeat passes times:
    for c from HIGHEST_COLOR down to 1 do                      // Von oben nach unten in den Prefix reinpacken versuchen
      S <- {v | coloring[v]=c}
      if TRY_MOVE_SET_TO_PREFIX(G, S, allowed_colors=0..c-1)   // Greedy + etwas Kempe
        ERASE_COLOR(c); changed <- True
  return (coloring, changed)
End

```

---

