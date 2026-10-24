import pandas as pd

# Membaca dataset
df = pd.read_csv('UCI_Credit_Card.csv')

# Menghapus kolom ID
df = df.drop(columns=['ID'])

# Menampilkan 5 baris pertama
print(df.head())

# Menampilkan daftar kolom setelah ID dihapus
print(df.columns)