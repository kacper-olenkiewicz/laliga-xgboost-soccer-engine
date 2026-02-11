import pandas as pd
import numpy as np
from xgboost import XGBClassifier

def main():
    
    df = pd.read_csv('processed_data.py')

    print("--- KREATOR TYPÓW ---")
    home_team = input("Podaj Gospodarza: ")
    away_team = input("Podaj nazwe Gości: ")

    odds_h = float(input("Kurs na Gospodarza (np. 1.50): "))
    odds_a = float(input("Kurs na Gości (np. 2.00): "))


