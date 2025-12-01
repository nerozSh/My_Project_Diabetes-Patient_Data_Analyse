import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def lade_daten(dateipfad):
    df = pd.read_csv(dateipfad)
    return df

def bereinige_daten(df):
    # Falls doch noch alte englische Spalten auftauchen, entfernen
    # Nur englische Spalten entfernen, die nicht deutsch sind
    englische_spalten = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin']
    for spalte in englische_spalten:
        if spalte in df.columns:
            df.drop(columns=spalte, inplace=True)

    cols_mit_nullen = [
        'Blutzucker (mg/dL)', 'Blutdruck (mmHg)', 'Hautdicke (mm)',
        'Insulin (mu U/ml)', 'BMI'
    ]

    
    df[cols_mit_nullen] = df[cols_mit_nullen].replace(0, np.nan)

    imputer = SimpleImputer(strategy='mean')
    imputierte_daten = imputer.fit_transform(df[cols_mit_nullen])
    df_imputiert = pd.DataFrame(imputierte_daten, columns=cols_mit_nullen, index=df.index)
    df[cols_mit_nullen] = df_imputiert

    df = df.dropna(subset=['Diabetes'])
    # Altersklasse bestimmen
    def alter_klasse(alter):
        if alter < 30:
            return 'Jung'
        elif alter < 50:
            return 'Mittel'
        else:
            return 'Alt'
    
    df['Altersklasse'] = df['Alter'].apply(alter_klasse)
    
    # Outlier-Erkennung (IQR-Methode)
    for col in cols_mit_nullen + ['Schwangerschaften', 'Familiäre Vorbelastung', 'Alter']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        print(f"⚠️ {col}: {outlier_count} Ausreißer gefunden")

    return df
