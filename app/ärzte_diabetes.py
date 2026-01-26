import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from modeling.prediction import trainiere_diabetes_modell, vorhersage_patient
from utils.preprocessing import lade_daten, bereinige_daten
from app.styles import css
from eda.eda_plots import analysiere_und_plotte  # Deine EDA-Funktion importieren

DATA_PATH = "data/diabetes.csv"
EDA_OUTPUT_PATH = "eda/output/"

def speichere_daten(df):
    if 'Altersklasse' in df.columns:
        df = df.drop(columns=['Altersklasse'])
    df.to_csv(DATA_PATH, index=False)

def generiere_patientenbericht(neuer_patient, ergebnis):
    bericht = f"""
Diabetes Risiko Analysebericht

- Alter: {neuer_patient['Alter']} Jahre
- Blutzucker: {neuer_patient['Blutzucker (mg/dL)']} mg/dL
- BMI: {neuer_patient['BMI']}
- Ergebnis: {ergebnis['Klassifikation']} mit Risiko von {ergebnis['Risiko (%)']}%.
- Empfehlung: Bitte regelmäßige Arztbesuche wahrnehmen und Lebensstil anpassen.
"""
    return bericht

def aktualisiere_daten_und_eda():
    try:
        # Gesamtdaten laden + bereinigen
        df_gesamt = lade_daten(DATA_PATH)
        df_gesamt = bereinige_daten(df_gesamt)
        analysiere_und_plotte(df_gesamt)
        st.success("EDA-Diagramme wurden aktualisiert!")
    except Exception as e:
        st.error(f"Fehler bei der EDA-Aktualisierung: {e}")

def main():
    st.markdown(css, unsafe_allow_html=True)
    st.title("Arzt-App: Diabetes Analyse & Patientenvorhersage")

    st.sidebar.header("Neue Patientendaten eingeben")

    neuer_patient = {
        'Schwangerschaften': st.sidebar.number_input("Schwangerschaften", 0, 20, 0),
        'Blutzucker (mg/dL)': st.sidebar.number_input("Blutzucker (mg/dL)", 0, 300, 100),
        'Blutdruck (mmHg)': st.sidebar.number_input("Blutdruck (mmHg)", 0, 200, 70),
        'Hautdicke (mm)': st.sidebar.number_input("Hautdicke (mm)", 0, 100, 20),
        'Insulin (mu U/ml)': st.sidebar.number_input("Insulin (mu U/ml)", 0, 900, 80),
        'BMI': st.sidebar.number_input("BMI", 0.0, 70.0, 25.0),
        'Familiäre Vorbelastung': st.sidebar.number_input("Familiäre Vorbelastung", 0.0, 2.0, 0.3),
        'Alter': st.sidebar.number_input("Alter", 0, 120, 30),
        #'Diabetes': 0  # Standardwert
    }

    # Daten laden und Modell trainieren
    df_raw = lade_daten(DATA_PATH)
    df_clean = bereinige_daten(df_raw)
    modell = trainiere_diabetes_modell(df_clean)

    patient_df = pd.DataFrame([neuer_patient])
    

    if 'ergebnis' not in st.session_state:
        st.session_state.ergebnis = None
    if 'bericht' not in st.session_state:
        st.session_state.bericht = ""
    if st.sidebar.button("Neuen Patienten speichern"):
        df = lade_daten(DATA_PATH)
        
        patient_df = pd.DataFrame([{
        'Schwangerschaften': neuer_patient['Schwangerschaften'],
        'Blutzucker (mg/dL)': neuer_patient['Blutzucker (mg/dL)'],
        'Blutdruck (mmHg)': neuer_patient['Blutdruck (mmHg)'],
        'Hautdicke (mm)': neuer_patient['Hautdicke (mm)'],
        'Insulin (mu U/ml)': neuer_patient['Insulin (mu U/ml)'],
        'BMI': neuer_patient['BMI'],
        'Familiäre Vorbelastung': neuer_patient['Familiäre Vorbelastung'],
        'Alter': neuer_patient['Alter'],
        'Diabetes': 0
    }])
        df = pd.concat([df, patient_df], ignore_index=True)
        speichere_daten(df)
        aktualisiere_daten_und_eda()
        st.sidebar.success("Neuer Patient gespeichert und EDA aktualisiert!")

    # Risiko berechnen
    if st.sidebar.button("Risiko für neuen Patienten berechnen"):
        ergebnis = vorhersage_patient(modell, patient_df)
        st.session_state.ergebnis = ergebnis
        bericht = generiere_patientenbericht(neuer_patient, ergebnis)
        st.session_state.bericht = bericht
        st.sidebar.success(f"Klassifikation: {ergebnis['Klassifikation']}")
        st.sidebar.info(f"Diabetes-Risiko: {ergebnis['Risiko (%)']}%")

    # Bericht anzeigen
    if st.sidebar.button("Bericht anzeigen") and st.session_state.bericht:
        st.sidebar.text_area("Patientenbericht", st.session_state.bericht, height=250)

    # Bericht herunterladen
    if st.sidebar.button("⬇ Bericht herunterladen") and st.session_state.bericht:
        st.sidebar.download_button(
            label="Bericht als Text herunterladen",
            data=st.session_state.bericht,
            file_name="bericht_patient.txt",
            mime="text/plain"
        )

    st.header("Patienten-Daten & Analysen")

    # CSV-Upload
    uploaded_file = st.file_uploader("CSV mit Patientendaten hochladen", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df, height=400)
        st.success("📁 CSV-Daten gespeichert & EDA aktualisiert!")

    # EDA-Bilder anzeigen
    st.subheader("EDA - Aktuelle Diagramme")
    bilder = [
        "01_alter_verteilung.png",
        "02_bmi_vs_blutzucker.png",
        "03_alter_vs_blutzucker.png",
        "04_diabetes_nach_altersklasse.png",
    ]
    for bild in bilder:
        bildpfad = os.path.join(EDA_OUTPUT_PATH, bild)
        if os.path.exists(bildpfad):
            st.image(bildpfad, use_container_width=True)
        else:
            st.warning(f"EDA-Bild {bild} nicht gefunden.")
   
if __name__ == "__main__":
    main()
