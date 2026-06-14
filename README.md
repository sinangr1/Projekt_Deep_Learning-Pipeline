Zebra Classification with Convolutional Neural Networks
Projektübersicht

In diesem Projekt wird ein Convolutional Neural Network (CNN) zur binären Bildklassifikation entwickelt. 
Ziel des Modells ist es, Bilder in zwei Klassen zu unterscheiden:

no_zebra
zebra

Das Projekt vergleicht ein Baseline-Modell mit einem optimierten CNN-Modell. 
Die Ergebnisse werden automatisch ausgewertet und im Ordner outputs gespeichert. 
Zusätzlich wird eine JSON-Datei erzeugt, in der die wichtigsten Kennzahlen und Modellresultate strukturiert zusammengefasst sind.

Ziel des Projekts

Das Ziel dieses Projekts ist es, ein CNN-Modell zu trainieren, zu evaluieren und anschliessend zu optimieren. 
Dabei werden insbesondere folgende Aspekte betrachtet:

Klassifikation von Zebra- und Nicht-Zebra-Bildern
Vergleich zwischen Baseline-Modell und optimiertem Modell
Analyse der Modellleistung mit Accuracy, Loss, Precision, Recall und F1-Score
Auswertung der Confusion Matrix
Speicherung der Resultate als Grafiken und JSON-Datei
Projektstruktur
Projekt_Deep_Learning-Pipeline/
│
├── 00_run_pipeline.py
├── 01_ordered_zebra_pipeline.py
│
├── models/
│   ├── baseline_cnn.keras
│   └── optimized_cnn.keras
│
├── outputs/
│   ├── baseline_accuracy.png
│   ├── baseline_loss.png
│   ├── optimized_accuracy.png
│   ├── optimized_loss.png
│   ├── confusion_matrix_baseline_test.png
│   ├── confusion_matrix_optimized_test.png
│   └── zebra_cnn_summary.json
Verwendete Klassen

Das Modell unterscheidet zwischen zwei Klassen:

[
  "no_zebra",
  "zebra"
]

Die Bilder werden mit einer Bildgrösse von 160 x 160 verarbeitet. Die Batch Size beträgt 32.

Datensatzverteilung

Die Klassenverteilung zeigt, dass der Datensatz unausgeglichen ist. Die Klasse no_zebra kommt deutlich häufiger vor als die Klasse zebra.

Datensatz	no_zebra	zebra
Training	5405	1059
Validation	1158	226
Test	1159	228

Das Verhältnis zwischen Mehrheits- und Minderheitsklasse beträgt im Training ungefähr 5.10 : 
1. Dadurch besteht die Gefahr, dass das Modell stärker zur Mehrheitsklasse no_zebra tendiert.

Baseline-Modell

Das Baseline-Modell wurde für 20 Epochen trainiert. Die wichtigsten Ergebnisse auf den Testdaten sind:

Metrik	Wert
Accuracy	0.8659
Loss	0.3028
Precision	0.8750
Recall	0.2149
Zebra F1-Score	0.3451

Die Accuracy des Baseline-Modells ist zwar relativ hoch, aber der Recall für die Klasse zebra ist sehr niedrig. 
Das bedeutet, dass viele Zebra-Bilder nicht korrekt erkannt wurden.

Confusion Matrix des Baseline-Modells
[[1152,   7],
 [ 179,  49]]

Interpretation:

1152 Bilder der Klasse no_zebra wurden korrekt erkannt.
7 Bilder der Klasse no_zebra wurden fälschlicherweise als zebra klassifiziert.
179 Zebra-Bilder wurden fälschlicherweise als no_zebra klassifiziert.
49 Zebra-Bilder wurden korrekt erkannt.

Das Baseline-Modell erkennt die Klasse no_zebra sehr gut, hat aber deutliche Schwächen bei der Erkennung der Klasse zebra.

Optimierungsentscheidung

Eine Optimierung war notwendig, weil:

die finale Validation Loss höher als die beste Validation Loss war,
der Datensatz unausgeglichen ist,
das Baseline-Modell einen sehr niedrigen Recall für die Klasse zebra hatte.

Empfohlene und verwendbare Optimierungsmassnahmen sind unter anderem:

Early Stopping
Anpassung der Lernrate
Class Weights wegen Klassenungleichgewicht
Verbesserung der Modellarchitektur
längeres oder kontrollierteres Training
Optimiertes Modell

Das optimierte Modell wurde für 40 Epochen trainiert. Die wichtigsten Ergebnisse auf den Testdaten sind:

Metrik	Wert
Accuracy	0.9712
Loss	0.0997
Precision	0.9747
Recall	0.8465
Zebra F1-Score	0.9061

Das optimierte Modell zeigt eine deutliche Verbesserung gegenüber dem Baseline-Modell. 
Besonders wichtig ist die starke Verbesserung des Recalls und des F1-Scores für die Klasse zebra.

Confusion Matrix des optimierten Modells
[[1154,   5],
 [  35, 193]]

Interpretation:

1154 Bilder der Klasse no_zebra wurden korrekt erkannt.
5 Bilder der Klasse no_zebra wurden fälschlicherweise als zebra klassifiziert.
35 Zebra-Bilder wurden fälschlicherweise als no_zebra klassifiziert.
193 Zebra-Bilder wurden korrekt erkannt.

Im Vergleich zum Baseline-Modell konnte die Anzahl falsch erkannter Zebra-Bilder stark reduziert werden.

Vergleich: Baseline vs. optimiertes Modell
Metrik	Baseline Test	Optimiertes Modell Test
Accuracy	0.8659	0.9712
Loss	0.3028	0.0997
Precision	0.8750	0.9747
Recall	0.2149	0.8465
Zebra F1-Score	0.3451	0.9061

Das optimierte Modell ist dem Baseline-Modell klar überlegen. Besonders die Erkennung der Minderheitsklasse zebra wurde stark verbessert.

Gespeicherte Ergebnisse

Die wichtigsten Ergebnisse werden im Ordner outputs gespeichert:

outputs/
├── baseline_accuracy.png
├── baseline_loss.png
├── optimized_accuracy.png
├── optimized_loss.png
├── confusion_matrix_baseline_test.png
├── confusion_matrix_optimized_test.png
└── zebra_cnn_summary.json

Die Datei zebra_cnn_summary.json enthält die wichtigsten Resultate des Projekts in strukturierter Form, darunter:

Klassennamen
Bildgrösse
Batch Size
Anzahl der Epochen
Klassenverteilung
Baseline-Ergebnisse
optimierte Ergebnisse
F1-Scores
Confusion Matrices
Pfade zu gespeicherten Modellen und Grafiken
Ausführung des Projekts

Das Projekt kann über die Datei 00_run_pipeline.py gestartet werden:

python 00_run_pipeline.py

Diese Datei lädt die Hauptpipeline und führt den Trainings, Evaluations und Exportprozess aus.

Ergebnisbewertung

Das optimierte CNN-Modell erreicht auf den Testdaten eine Accuracy von ungefähr 97.12 %. 
Besonders relevant ist jedoch der F1-Score der Klasse zebra, da der Datensatz unausgeglichen ist. 
Dieser konnte von 0.3451 im Baseline-Modell auf 0.9061 im optimierten Modell verbessert werden.

Damit zeigt das optimierte Modell eine deutlich bessere Fähigkeit, Zebra-Bilder korrekt zu erkennen.

Fazit

Das Projekt zeigt, dass ein einfaches Baseline-CNN bei unausgeglichenen Bilddaten eine hohe Accuracy erreichen kann, 
obwohl die Minderheitsklasse schlecht erkannt wird. Durch Optimierung konnte die Leistung des Modells deutlich verbessert werden.

Das optimierte Modell erreicht bessere Werte bei Accuracy, Precision, Recall und F1-Score. Besonders die Erkennung der Klasse zebra wurde stark verbessert. 
Die Ergebnisse sind im Ordner outputs sowohl visuell als PNG-Dateien als auch strukturiert in der Datei zebra_cnn_summary.json gespeichert.
