# Großes Ziel

Generalisierbarkeit von metrischen Zero-Shot Algorithmen für Textklassifikation.

* Task spezifisches Training schränkt die inter-domänen Generalisierbarkeit ein. 
* Metrische Zero-Shot bieten die Möglichkeit nicht nur den Input (Sprachmodelle/Word Vectors) sondern auch Embeddings für Label Hierarchien
* Geometrische lernverfahren sind in der Lage komplexer strukturen in Vektorraum metrisch zu lernen und auch externes Wissen einfließen zu lassen (Hierarchische / graphstrukturen)
* Vermutung1: Vortraining von eingabe-  __und__  ausgabevektoren + Ausnutzung externer Wissensquellen führen zu Modellen mit höheren Generalisierbarkeit ( auch zwischen den Domänen)
*  Vermutung2: Ein Model mit hoher Generalisierbarkeit und eine vortrainierten Breite an externem Wissen kann in Verbindung mit Methoden, die wenig Daten brauchen (Active learning/ Transfer ) angewandt werden, um domänenunabhängig bessere Ergebnisse zu erzielen.


# Schritt 1:

Auswahl der Modelle.

* Einschränkung auf Modelle, die sich für metrisches Zero-Shot lernen eignen.
* Auswertung der (klassischen) Multilabel Performance domänenbüergreifend mit Sprachmodellen.
* Explizit sollen dabei die Eingabe representationen __nicht__ finetuned werden, um Generalisierbarkeit zu erhalten (gilt das? Vgl Regular Training mit finetuned model)
* Label representation werden während des Trainings erzeugt 

# Schritt 2:
 
 Label Representationen

 * Word vectors für Label Strings
 * Hierarchische Information -> Graph Embeddings?
 * Concept Embeddings (unhabhängig von den Labels)

 # Schritt 3:

Übertragung in den N-Shot Bereich.

