from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score
import pandas as pd

def trainiere_diabetes_modell(df):
    # Eingabemerkmale (alles außer Ziel & Altersklasse)
    X = df.drop(columns=["Diabetes", "Altersklasse"])
    y = df["Diabetes"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    modell = RandomForestClassifier(random_state=42)
    modell.fit(X_train, y_train)
    
    y_pred = modell.predict(X_test)
    
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    print("Genauigkeit:", round(precision, 2))
    print("\nBericht:\n", classification_report(y_test, y_pred))
    
    # Merkmalsliste merken, um später für Patienten-Vorhersagen zu verwenden
    modell.feature_names_in_ = X.columns.tolist()
    
    return modell

def vorhersage_patient(modell: RandomForestClassifier, patient_df: pd.DataFrame):
    # Sicherstellen, dass die Spalten in gleicher Reihenfolge wie im Modell sind
    patient_df = patient_df[modell.feature_names_in_]
    
    wahrscheinlichkeit = modell.predict_proba(patient_df)[0][1] * 100
    hat_diabetes = wahrscheinlichkeit >= 50  # Schwelle selbst gesetzt
    
    ergebnis = {
        "Risiko (%)": round(wahrscheinlichkeit, 1),
        "Klassifikation": "Diabetes-Risiko" if hat_diabetes else "Kein Risiko"
    }
    return ergebnis