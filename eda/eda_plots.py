import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
def analysiere_und_plotte(df):
    # 1. Altersverteilung
    plt.figure(figsize=(8,5))
    plt.hist(df['Alter'], bins=15, edgecolor="black", color='skyblue')
    plt.title("Altersverteilung der Patienten")
    plt.xlabel("Alter")
    plt.ylabel("Anzahl")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eda/output/01_alter_verteilung.png")
    plt.close()
    
    # 2. BMI vs Blutzucker mit Trendlinie
    daten = df[['BMI', 'Blutzucker (mg/dL)']].dropna()
    X = daten[['BMI']].values
    y = daten['Blutzucker (mg/dL)'].values

    modell = LinearRegression()
    modell.fit(X, y)
    y_pred = modell.predict(X)

    slope, intercept, r_value, p_value, std_err = linregress(daten['BMI'], daten['Blutzucker (mg/dL)'])

    plt.figure(figsize=(8,5))
    plt.scatter(X, y, alpha=0.7, color='green', label='Datenpunkte')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression')

    plt.title(f"BMI vs. Blutzucker\nR² = {r_value**2:.3f}, p = {p_value:.3f}")
    plt.xlabel("BMI")
    plt.ylabel("Blutzucker (mg/dL)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eda/output/02_bmi_vs_blutzucker.png")
    plt.close()
    
    # 3. Alter vs Blutzucker mit Trendlinie
     # 3. Alter vs Blutzucker mit Linearer Regression, R² und p-Wert
    daten = df[['Alter', 'Blutzucker (mg/dL)']].dropna()
    X = daten[['Alter']].values  # 2D-Array für sklearn
    y = daten['Blutzucker (mg/dL)'].values

    modell = LinearRegression()
    modell.fit(X, y)
    y_pred = modell.predict(X)

    slope, intercept, r_value, p_value, std_err = linregress(daten['Alter'], daten['Blutzucker (mg/dL)'])

    plt.figure(figsize=(8,5))
    plt.scatter(X, y, alpha=0.7, color='purple', label='Datenpunkte')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression')

    plt.title(f"Alter vs. Blutzucker\nR² = {r_value**2:.3f}, p = {p_value:.3f}")
    plt.xlabel("Alter")
    plt.ylabel("Blutzucker (mg/dL)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("eda/output/03_alter_vs_blutzucker.png")
    plt.close()

    # 4. Diabetes-Verteilung nach Altersklasse
    altersverteilung = df.groupby('Altersklasse')['Diabetes'].value_counts().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8,5))
    altersverteilung.plot(kind='bar', stacked=True, color=['lightgreen', 'salmon'], ax=ax)
    ax.set_title("Diabetes-Verteilung nach Altersklasse")
    ax.set_xlabel("Altersklasse")
    ax.set_ylabel("Anzahl der Patienten")
    ax.legend(title="Diabetes")
    ax.grid(axis='y')
    plt.tight_layout()
    plt.savefig("eda/output/04_diabetes_nach_altersklasse.png")
    plt.close()
