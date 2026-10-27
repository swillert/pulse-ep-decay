# Nachaggregation bestehender Validierungsdaten — 22.10.2026

**Keine neuen Simulationen.** Die folgenden Resultate wurden ausschließlich aus vorhandenen arm1-/v37-CSV neu aggregiert.

## Arm1: gleiche Gewichtung jeder eindeutigen Geometrie

56 Map-Dateien entsprechen 40 exakt unterschiedlichen geordneten Vertex-/Dreieck-Arrays. Doppelte Geometrien liefern identische σ/R²/Ursprungsversatz-Zeilen (maximale Differenz 0). Für die eindeutige Auswertung bleibt jeweils die kleinste Map-ID erhalten. Pro σ/Rauschzelle bleiben 40 Geometrien × 20 Trials = 800 Zeilen statt 56×20 = 1.120.

SD ist die Stichproben-SD der einzelnen Fehler; MAE und p95 beziehen sich auf deren Absolutbeträge. Die 95%-Intervalle des mittleren Bias entstehen aus 10.000 Bootstrap-Ziehungen von 40 vollständigen Geometrie-Clustern mit Zurücklegen; alle 20 Trials eines gezogenen Clusters bleiben zusammen. Sie ersetzen die bisherige Einzelzeilen-Resampling-Annahme für diese Auswertung.

**Geltungsbereich:** bedingter numerischer Benchmark der beobachteten Geometrien, Ursprünge und vorhandenen Zufallsrealisierungen; keine Patienten-/Populations- oder klinischen Genauigkeitsintervalle. Die ursprüngliche Simulation benutzt denselben Trial-Seed auf allen Geometrien. Diese gemeinsamen Zufallszahlen können Resultate zwischen Geometrien korrelieren; das Geometrie-Bootstrap ist ausdrücklich bedingt auf den vorhandenen 20 Realisierungen und quantifiziert deren Monte-Carlo-Unsicherheit nicht. Auch die 40 Geometrien sind nicht als unabhängige Patienten interpretiert. Neue unabhängige Seeds wären für eine zusätzliche unbedingte Simulationsunsicherheitsanalyse nötig.

Bei 10 % iid-Vertexrauschen (Einheiten mm):

| σ₀ | Gewichtung | Bias | SD | MAE | p95 absolut | Geometrie-Bootstrap 95% Bias |
|---:|---|---:|---:|---:|---:|---|
| 3 | map_weighted | +0.024 | 0.218 | 0.162 | 0.440 | — |
| 3 | unique_geometry_weighted | +0.018 | 0.232 | 0.171 | 0.489 | [-0.017; 0.056] |
| 5 | map_weighted | +0.050 | 0.283 | 0.205 | 0.660 | — |
| 5 | unique_geometry_weighted | +0.041 | 0.273 | 0.200 | 0.603 | [-0.015; 0.093] |
| 8 | map_weighted | +0.248 | 0.547 | 0.419 | 1.337 | — |
| 8 | unique_geometry_weighted | +0.213 | 0.512 | 0.383 | 1.194 | [0.108; 0.319] |
| 10 | map_weighted | +0.333 | 0.723 | 0.588 | 1.705 | — |
| 10 | unique_geometry_weighted | +0.296 | 0.676 | 0.535 | 1.529 | [0.145; 0.439] |
| 15 | map_weighted | +0.609 | 1.243 | 1.079 | 2.816 | — |
| 15 | unique_geometry_weighted | +0.588 | 1.188 | 1.019 | 2.737 | [0.291; 0.850] |
| 20 | map_weighted | +0.790 | 2.006 | 1.730 | 4.175 | — |
| 20 | unique_geometry_weighted | +0.838 | 1.945 | 1.706 | 3.869 | [0.295; 1.309] |
| 25 | map_weighted | +1.243 | 2.893 | 2.495 | 6.304 | — |
| 25 | unique_geometry_weighted | +1.391 | 2.886 | 2.549 | 6.281 | [0.635; 2.081] |
| 30 | map_weighted | +2.388 | 5.079 | 3.929 | 9.772 | — |
| 30 | unique_geometry_weighted | +2.729 | 5.425 | 4.149 | 10.427 | [1.550; 3.875] |

## v37: vollständiges Raster und endliche Stichproben

Die Tabelle `v37_threshold_grid.csv` enthält 56 konkrete Generatorbedingungen × 16 Schwellenpaare (R²/F jeweils 0,6/0,7/0,8/0,9), jeweils n=20. Angegeben sind Anzahl, Rate und zweiseitiges 95%-Wilson-Intervall für R²-Flag, F-Flag, mindestens einen Flag, beide Flags und gemeinsames Bestehen. Flags gelten strikt unter der Schwelle, Bestehen bei ≥. `v37_threshold_070.csv` ist die vollständige 0,70/0,70-Auswahl. Die Raten sind bedingte synthetische Flag-Häufigkeiten; sie werden nicht als klinische Sensitivität oder Spezifität bezeichnet.

Wilson-Intervalle gelten je Zelle unter der Annahme unabhängiger Simulationswiederholungen. Felder innerhalb derselben Trial-ID teilen den Ausgangspunkt und teilweise den zweiten Gipfel; die Zellen sind daher nicht voneinander unabhängig. Die Intervalle sind weder simultan über alle Zellen gültig noch für die Auswahl einer optimalen Schwelle adjustiert.

Ausgewählte Resultate bei R²/F=0,70:

- Einzelgipfel σ₀=5 mm, 10 % Rauschen: F-Flag 19/20 (95%; 95% Wilson 76.4–99.1%).
- Einzelgipfel σ₀=10 mm, 10 % Rauschen: F-Flag 0/20 (0%; 95% Wilson 0.0–16.1%).
- Einzelgipfel σ₀=15 mm, 10 % Rauschen: F-Flag 0/20 (0%; 95% Wilson 0.0–16.1%).
- Ring, 5 % Rauschen: gemeinsames Bestehen 20/20 (100%; 95% Wilson 83.9–100.0%).
- Flaches Feld, 5 % Rauschen: R²-Flag 20/20 (100%; 95% Wilson 83.9–100.0%).
- Flaches Feld, 10 % Rauschen: R²-Flag 18/20 (90%; 95% Wilson 69.9–97.2%).
- Flaches Feld, 15 % Rauschen: R²-Flag 17/20 (85%; 95% Wilson 64.0–94.8%).
- Flaches Feld, 25 % Rauschen: R²-Flag 18/20 (90%; 95% Wilson 69.9–97.2%).

Die v37-Definition der Zweigipfelfelder ist das Maximum zweier Gaußkomponenten (σ=7 mm), nicht deren Summe. Die Ringspalte `true_width_mm=4` bezeichnet die radiale Standardabweichung um den Ringradius 15 mm; beim flachen Feld ist eine Breite nicht definiert. `amplitude_score=35` ist bei flachen Feldern die Rauschnormierung, kein vorhandener Gipfel. Diese Charakterisierung validiert weder die Isotropie noch die klinische Lokalisationsgenauigkeit.

## Reproduktion

```sh
pulse-ep/.venv/bin/python reaggregate.py
```

Aus diesem Ordner ausführen. `--results`, `--meshes`, `--out`, `--bootstrap` und `--seed` sind explizit überschreibbar. Eingabe-Hashes, Versionen und Parameter stehen in `manifest.json`; die geometrischen Zuordnungen in `geometry_inventory.csv`.
