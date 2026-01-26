# My Project: Diabetes Patient Data Analyse

## Projektübersicht
Dieses Projekt analysiert medizinische Patientendaten im Zusammenhang mit Diabetes.  
Ziel ist es, mithilfe von **Datenanalyse und Machine Learning** Muster zu erkennen und vorherzusagen, ob eine Person an Diabetes leidet.  
Der Workflow umfasst die Datenvorverarbeitung, explorative Datenanalyse (EDA), Modellierung und Evaluation.

## Zielsetzung
- Analyse medizinischer Patientendaten  
- Identifikation relevanter Zusammenhänge zwischen Merkmalen  
- Aufbau eines Machine-Learning-Modells zur Diabetes-Vorhersage  
- Bewertung der Modellleistung anhand gängiger Metriken  
- Verständliche Aufbereitung der Ergebnisse  

Dieses Projekt dient sowohl **Lernzwecken** als auch als Praxisbeispiel für KI- und Data-Science-Anwendungen im Gesundheitsbereich.

## Datensatz
Der Datensatz enthält medizinische Messwerte von Patient:innen.

### Enthaltene Merkmale (Features)
- Pregnancies – Anzahl der Schwangerschaften  
- Glucose – Glukosekonzentration im Blut  
- BloodPressure – Blutdruck  
- SkinThickness – Hautfaltendicke  
- Insulin – Insulinwert  
- BMI – Body-Mass-Index  
- DiabetesPedigreeFunction – genetische Diabetes-Wahrscheinlichkeit  
- Age – Alter  

### Zielvariable
- Outcome: `0` = Kein Diabetes, `1` = Diabetes

## Datenvorverarbeitung
- Überprüfung auf fehlende oder unrealistische Werte  
- Bereinigung und Transformation der Daten  
- Vorbereitung der Daten für Machine-Learning-Modelle  

Die Datenqualität ist entscheidend für die Modellleistung.

## Explorative Datenanalyse (EDA)
- Analyse der Verteilungen einzelner Merkmale  
- Untersuchung von Korrelationen zwischen Features  
- Identifikation von Ausreißern und Auffälligkeiten  

Ziel der EDA: fundierte Entscheidungen für die Modellwahl und tieferes Verständnis der Daten.

## Machine-Learning-Modellierung
- Aufteilung in Trainings- und Testdaten  
- Training von Klassifikationsmodellen  
- Optimierung der Modellparameter  
- Evaluation der Ergebnisse  

**Wichtige Erkenntnis:** Glukosewert, BMI und Alter haben den größten Einfluss auf die Diabetes-Vorhersage.

## Ergebnisse & Evaluation
- Accuracy  
- Confusion Matrix  
- Vergleich von tatsächlichen und vorhergesagten Werten  

Die Modelle zeigen gute Vorhersageleistung und liefern wertvolle Einblicke in die wichtigsten Einflussfaktoren.

## Erkenntnisse & Lernerfahrungen
- Anwendung des vollständigen Data-Science-Prozesses  
- Umgang mit medizinischen Datensätzen  
- Bedeutung von Datenqualität für Machine Learning  
- Interpretation und kritische Bewertung von Modellen  
- Verständliche Visualisierung und Kommunikation der Ergebnisse

## Verwendete Technologien
- Python  
- Pandas  
- NumPy  
- Matplotlib / Seaborn  
- Scikit-learn

## Projektstruktur (optional für Klarheit)
- `app/` → Preprocessing- und Utility-Scripts  
- `data/` → Patientendatensatz  
- `eda/` → Explorative Datenanalyse  
- `modeling/` → ML-Modellierung und Evaluation  
- `requirements.txt` → Python-Abhängigkeiten  
- `README.md` → Projektbeschreibung
