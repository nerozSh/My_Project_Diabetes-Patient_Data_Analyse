

import sys
import os

# Projektverzeichnis zum Python-Pfad hinzufügen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import os
import json
from styles import css
import openai
from modeling.knn_prediction import lade_knn_modell,knn_risiko_vorhersage

def frage_gpt_api(frage: str, api_key: str) -> str:
    if not api_key:
        return "⚠️ Bitte gib deinen OpenAI API-Key in der Sidebar ein."

    try:
       
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Du bist ein freundlicher medizinischer Assistent."},
                {"role": "user", "content": frage}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Fehler bei der GPT-Anfrage: {e}"

DATA_PATH = "data/diabetes.csv"
# FAQ laden 
def lade_faq(dateiname="faq_diabetes.json"):
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, dateiname)
    if not os.path.exists(full_path):
        return []
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    st.markdown(css, unsafe_allow_html=True)
    st.set_page_config(page_title="Diabetes Risiko & Chatbot", layout="wide")
    
    st.title(" Diabetes Risiko & Medizinischer Chatbot")
    st.sidebar.header("API-Key Eingabe")
    api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
    
    # --- Sidebar: Patientendaten & Risiko ---
    st.sidebar.header("Patientendaten & Risiko-Berechnung")

        # Modell laden
    try:
        model, feature_names = lade_knn_modell(DATA_PATH, n_neighbors=5)
    except FileNotFoundError as e:
        st.sidebar.error(str(e))
        return

    st.sidebar.header("Patientendaten")

    input_data = {
    "Schwangerschaften": st.sidebar.number_input("Schwangerschaften", 0, 20, 0),
    "Blutzucker (mg/dL)": st.sidebar.number_input("Blutzucker (mg/dL)", 0, 300, 100),
    "Blutdruck (mmHg)": st.sidebar.number_input("Blutdruck (mmHg)", 0, 200, 70),
    "Hautdicke (mm)": st.sidebar.number_input("Hautdicke (mm)", 0, 100, 20),
    "Insulin (mu U/ml)": st.sidebar.number_input("Insulin (mu U/ml)", 0, 900, 80),
    "BMI": st.sidebar.number_input("BMI", 0.0, 70.0, 25.0),
    "Familiäre Vorbelastung": st.sidebar.number_input("Familiäre Vorbelastung", 0.0, 2.0, 0.3),
    "Alter": st.sidebar.number_input("Alter", 0, 120, 30)
}
    if st.sidebar.button("Risiko berechnen"):
        vorhersage, risiko_prozent = knn_risiko_vorhersage(input_data, model, feature_names)
        if vorhersage == 1:
            st.sidebar.error(f"🚨 Hohes Diabetes-Risiko: {risiko_prozent}%")
        else:
            st.sidebar.success(f"👍 Niedriges Diabetes-Risiko: {risiko_prozent}%")
        bericht = f"""
Diabetes Risiko Bericht

Schwangerschaften: {input_data['Schwangerschaften']}
Blutzucker: {input_data['Blutzucker (mg/dL)']} mg/dL
Blutdruck: {input_data['Blutdruck (mmHg)']} mmHg
Hautdicke: {input_data['Hautdicke (mm)']} mm
Insulin: {input_data['Insulin (mu U/ml)']} U/ml
BMI: {input_data['BMI']}
Familiäre Vorbelastung: {input_data['Familiäre Vorbelastung']}
Alter: {input_data['Alter']} Jahre
Risikowahrscheinlichkeit: {risiko_prozent}%

Bitte konsultieren Sie einen Arzt für eine genaue Diagnose.
"""
        st.sidebar.download_button("Bericht herunterladen (TXT)", data=bericht, file_name="diabetes_bericht.txt")

    # --- Hauptbereich: 2 Spalten: Links Chatbot, Rechts FAQ ---
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(" Medizinischer Diabetes-Chatbot")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Anzeige vorheriger Nachrichten
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"**Du:** {msg['content']}")
            else:
                st.markdown(f"**Bot:** {msg['content']}")

        # Neue Frage eingeben
        prompt = st.chat_input("Stelle deine Frage zu Diabetes...")

        if prompt:
            if not api_key:
                st.error("Bitte gib deinen OpenAI API-Key in der Sidebar ein, um die GPT-Funktion nutzen zu können.")
            else:
                # FAQ zuerst durchsuchen
                faq_liste = lade_faq()
                antwort = None
                for faq in faq_liste:
                    if prompt.strip().lower() == faq["frage"].strip().lower():
                        antwort = faq["antwort"]
                        break
                
                # Falls keine FAQ-Antwort, GPT API fragen
                if not antwort:
                    antwort = frage_gpt_api(prompt, api_key)

                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": antwort})
                st.rerun()  # Seite neu laden, damit Nachricht angezeigt wird

    with col2:
        st.subheader("Häufig gestellte Fragen (FAQ)")
        faq_liste = lade_faq()
        if faq_liste:
            for faq in faq_liste:
                with st.expander(faq["frage"]):
                    st.write(faq["antwort"])
        else:
            st.info("Keine FAQ verfügbar.")

if __name__ == "__main__":
    main()