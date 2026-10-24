# ============================================
# TA-03: PREDIKSI RISIKO GAGAL BAYAR KARTU KREDIT
# Tahap 1: Exploratory Data Analysis (EDA)
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Pengaturan tampilan
sns.set_theme(style="whitegrid")

# ============================================
# 1. MEMBACA DATASET
# ============================================

BASE_DIR = Path(__file__).resolve().parent
FILE_DATASET = BASE_DIR / "UCI_Credit_Card.csv"

df = pd.read_csv(FILE_DATASET)

print("=" * 60)
print("TA-03: ANALISIS DATA KARTU KREDIT")
print("=" * 60)

# ============================================
# 2. MELIHAT UKURAN DATA
# ============================================

print("\n1. UKURAN DATASET")
print("-" * 40)

print(f"Jumlah baris  : {df.shape[0]}")
print(f"Jumlah kolom  : {df.shape[1]}")

# ============================================
# 3. MELIHAT 5 DATA PERTAMA
# ============================================

print("\n2. LIMA DATA PERTAMA")
print("-" * 40)

print(df.head())

# ============================================
# 4. INFORMASI DATASET
# ============================================

print("\n3. INFORMASI DATASET")
print("-" * 40)

df.info()

# ============================================
# 5. MEMERIKSA NILAI KOSONG
# ============================================

print("\n4. PEMERIKSAAN NILAI KOSONG")
print("-" * 40)

missing = df.isnull().sum()

print(missing[missing > 0])

if df.isnull().sum().sum() == 0:
    print("Tidak ditemukan nilai kosong.")

# ============================================
# 6. MEMERIKSA DATA DUPLIKAT
# ============================================

print("\n5. PEMERIKSAAN DATA DUPLIKAT")
print("-" * 40)

jumlah_duplikat = df.duplicated().sum()

print(f"Jumlah baris duplikat: {jumlah_duplikat}")

# ============================================
# 7. STATISTIK DESKRIPTIF
# ============================================

print("\n6. STATISTIK DESKRIPTIF")
print("-" * 40)

print(df.describe().round(2))

# ============================================
# 8. DISTRIBUSI TARGET
# ============================================

target = "default.payment.next.month"

print("\n7. DISTRIBUSI TARGET")
print("-" * 40)

jumlah_target = df[target].value_counts().sort_index()

persentase_target = (
    df[target].value_counts(normalize=True).sort_index() * 100
)

for kelas in jumlah_target.index:
    if kelas == 0:
        keterangan = "Tidak gagal bayar"
    else:
        keterangan = "Gagal bayar"

    print(
        f"{keterangan}: {jumlah_target[kelas]} nasabah "
        f"({persentase_target[kelas]:.2f}%)"
    )

# ============================================
# 9. VISUALISASI DISTRIBUSI TARGET
# ============================================

plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=df,
    x=target,
    hue=target,
    order=[0, 1],
    palette="Set2",
    legend=False
)

ax.set_title("Distribusi Risiko Gagal Bayar Kartu Kredit")
ax.set_xlabel("Status Gagal Bayar")
ax.set_ylabel("Jumlah Nasabah")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Tidak Gagal Bayar", "Gagal Bayar"])

for bar in ax.patches:
    tinggi = bar.get_height()

    ax.annotate(
        f"{int(tinggi):,}",
        (bar.get_x() + bar.get_width() / 2, tinggi),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(BASE_DIR / "distribusi_target.png", dpi=300)
plt.show()

# ============================================
# 10. DISTRIBUSI USIA NASABAH
# ============================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="AGE",
    bins=25,
    kde=True
)

plt.title("Distribusi Usia Nasabah")
plt.xlabel("Usia")
plt.ylabel("Jumlah Nasabah")

plt.tight_layout()
plt.savefig(BASE_DIR / "distribusi_usia.png", dpi=300)
plt.show()

# ============================================
# 11. DISTRIBUSI BATAS KREDIT
# ============================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="LIMIT_BAL",
    bins=30,
    kde=True
)

plt.title("Distribusi Batas Kredit Nasabah")
plt.xlabel("Batas Kredit")
plt.ylabel("Jumlah Nasabah")

plt.tight_layout()
plt.savefig(BASE_DIR / "distribusi_batas_kredit.png", dpi=300)
plt.show()

# ============================================
# 12. HUBUNGAN RIWAYAT PEMBAYARAN
#     DENGAN STATUS GAGAL BAYAR
# ============================================

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="PAY_0",
    hue=target,
    order=sorted(df["PAY_0"].unique())
)

plt.title("Status Pembayaran Bulan Terakhir")
plt.xlabel("Status Pembayaran (PAY_0)")
plt.ylabel("Jumlah Nasabah")
plt.legend(
    title="Gagal Bayar Bulan Berikutnya",
    labels=["Tidak", "Ya"]
)

plt.tight_layout()
plt.savefig(BASE_DIR / "riwayat_pembayaran.png", dpi=300)
plt.show()

# ============================================
# 13. KORELASI ANTARFITUR NUMERIK
# ============================================

fitur_korelasi = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "default.payment.next.month"
]

plt.figure(figsize=(11, 8))

sns.heatmap(
    df[fitur_korelasi].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Korelasi Fitur dan Target")

plt.tight_layout()
plt.savefig(BASE_DIR / "korelasi_fitur.png", dpi=300)
plt.show()

# ============================================
# 14. RINGKASAN EDA
# ============================================

print("\n8. RINGKASAN EDA")
print("-" * 40)

print(f"Total data             : {len(df)}")
print(f"Total fitur awal       : {df.shape[1] - 1}")
print(f"Jumlah nasabah gagal bayar: {(df[target] == 1).sum()}")
print(f"Jumlah nasabah tidak gagal bayar: {(df[target] == 0).sum()}")
print(f"Jumlah nilai kosong    : {df.isnull().sum().sum()}")
print(f"Jumlah duplikat        : {df.duplicated().sum()}")

print("\nEDA selesai.")
print("Grafik disimpan di folder proyek.")