import pandas as pd
df = pd.read_csv('processed_data.csv')
print(df['HomeGaz'].value_counts())