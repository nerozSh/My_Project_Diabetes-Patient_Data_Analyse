from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import os
csv_path="data/diabetes.csv"
# --- KNN Modell laden ---
def lade_knn_modell(csv_datei="data/diabetes.csv", n_neighbors=5):
     # Absoluter Pfad zum Projektordner
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, csv_datei)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV-Datei '{csv_path}' nicht gefunden.")

    df = pd.read_csv(csv_path)
    # Zielspalte = "Diabetes" (deutsche Version)
    X = df.drop(columns=["Diabetes"])
    y = df["Diabetes"]

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X, y)

    return model, X.columns.tolist()

# --- Risiko-Vorhersage ---
def knn_risiko_vorhersage(input_data, model, feature_names):
    # input_data ist schon mit deutschen Spaltennamen
    df_input = pd.DataFrame([input_data])
    df_input = df_input[feature_names]  # gleiche Reihenfolge wie beim Training
    prediction = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]  # Wahrscheinlichkeit für "Diabetes = 1"
    return prediction, round(proba * 100, 1)