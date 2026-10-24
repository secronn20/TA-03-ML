from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


# =========================
# KONFIGURASI APLIKASI
# =========================
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "credit_default_model.pkl"


# =========================
# MEMUAT MODEL
# =========================
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"File model tidak ditemukan: {MODEL_PATH}"
    )

model_artifact = joblib.load(MODEL_PATH)

model = model_artifact["model"]
feature_columns = model_artifact["feature_columns"]
model_name = model_artifact.get("model_name", "Logistic Regression")


# =========================
# DEFINISI FORMULIR
# =========================
FORM_FIELDS = [
    {
        "name": "LIMIT_BAL",
        "label": "Limit Kartu Kredit (NT$)",
        "type": "number",
        "default": "50000",
        "min": "0",
        "step": "1000",
        "help": "Batas kredit yang diberikan kepada pelanggan.",
    },
    {
        "name": "SEX",
        "label": "Jenis Kelamin",
        "type": "select",
        "default": "2",
        "options": [
            ("1", "Laki-laki"),
            ("2", "Perempuan"),
        ],
    },
    {
        "name": "EDUCATION",
        "label": "Pendidikan",
        "type": "select",
        "default": "2",
        "options": [
            ("1", "Pascasarjana"),
            ("2", "Universitas"),
            ("3", "Sekolah menengah"),
            ("4", "Lainnya"),
            ("5", "Tidak diketahui (kode 5)"),
            ("6", "Tidak diketahui (kode 6)"),
            ("0", "Kode 0"),
        ],
    },
    {
        "name": "MARRIAGE",
        "label": "Status Pernikahan",
        "type": "select",
        "default": "1",
        "options": [
            ("1", "Menikah"),
            ("2", "Belum menikah"),
            ("3", "Lainnya"),
            ("0", "Kode 0"),
        ],
    },
    {
        "name": "AGE",
        "label": "Usia (tahun)",
        "type": "number",
        "default": "30",
        "min": "18",
        "max": "100",
        "step": "1",
    },
]

# Riwayat status pembayaran.
# Nilai negatif menunjukkan pembayaran lebih awal atau
# status khusus; 0 berarti status pembayaran tercatat 0.
PAYMENT_FIELDS = [
    ("PAY_0", "Status Pembayaran Bulan Terakhir", "0"),
    ("PAY_2", "Status Pembayaran 2 Bulan Sebelumnya", "0"),
    ("PAY_3", "Status Pembayaran 3 Bulan Sebelumnya", "0"),
    ("PAY_4", "Status Pembayaran 4 Bulan Sebelumnya", "0"),
    ("PAY_5", "Status Pembayaran 5 Bulan Sebelumnya", "0"),
    ("PAY_6", "Status Pembayaran 6 Bulan Sebelumnya", "0"),
]

# Jumlah tagihan selama 6 periode.
BILL_FIELDS = [
    ("BILL_AMT1", "Tagihan Bulan Terakhir (NT$)", "20000"),
    ("BILL_AMT2", "Tagihan 2 Bulan Sebelumnya (NT$)", "18000"),
    ("BILL_AMT3", "Tagihan 3 Bulan Sebelumnya (NT$)", "15000"),
    ("BILL_AMT4", "Tagihan 4 Bulan Sebelumnya (NT$)", "12000"),
    ("BILL_AMT5", "Tagihan 5 Bulan Sebelumnya (NT$)", "10000"),
    ("BILL_AMT6", "Tagihan 6 Bulan Sebelumnya (NT$)", "8000"),
]

# Jumlah pembayaran selama 6 periode.
AMOUNT_FIELDS = [
    ("PAY_AMT1", "Pembayaran Bulan Terakhir (NT$)", "2000"),
    ("PAY_AMT2", "Pembayaran 2 Bulan Sebelumnya (NT$)", "2000"),
    ("PAY_AMT3", "Pembayaran 3 Bulan Sebelumnya (NT$)", "2000"),
    ("PAY_AMT4", "Pembayaran 4 Bulan Sebelumnya (NT$)", "2000"),
    ("PAY_AMT5", "Pembayaran 5 Bulan Sebelumnya (NT$)", "2000"),
    ("PAY_AMT6", "Pembayaran 6 Bulan Sebelumnya (NT$)", "2000"),
]


# =========================
# HALAMAN UTAMA
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None
    submitted_data = {}

    if request.method == "POST":
        try:
            # Membaca seluruh input formulir.
            for field in FORM_FIELDS:
                submitted_data[field["name"]] = request.form.get(
                    field["name"], field["default"]
                )

            for name, label, default in (
                PAYMENT_FIELDS + BILL_FIELDS + AMOUNT_FIELDS
            ):
                submitted_data[name] = request.form.get(name, default)

            # Mengubah input menjadi angka.
            input_data = {}

            for column in feature_columns:
                if column not in submitted_data:
                    raise ValueError(
                        f"Input untuk fitur {column} tidak tersedia."
                    )

                value = submitted_data[column]

                if value is None or str(value).strip() == "":
                    raise ValueError(
                        f"Kolom {column} wajib diisi."
                    )

                input_data[column] = float(value)

            # Memastikan susunan kolom sama dengan saat pelatihan.
            input_df = pd.DataFrame(
                [input_data],
                columns=feature_columns
            )

            # Prediksi kelas:
            # 0 = tidak gagal bayar
            # 1 = gagal bayar
            predicted_class = int(model.predict(input_df)[0])

            # Probabilitas kelas gagal bayar.
            class_labels = list(model.classes_)
            default_class_index = class_labels.index(1)

            probability = float(
                model.predict_proba(input_df)[0][default_class_index]
            ) * 100

            if predicted_class == 1:
                prediction = {
                    "class": 1,
                    "title": "Berisiko Gagal Bayar",
                    "description": (
                        "Model memprediksi pelanggan termasuk "
                        "kelas gagal bayar."
                    ),
                    "status": "danger",
                }
            else:
                prediction = {
                    "class": 0,
                    "title": "Tidak Diprediksi Gagal Bayar",
                    "description": (
                        "Model memprediksi pelanggan termasuk "
                        "kelas tidak gagal bayar."
                    ),
                    "status": "success",
                }

        except (ValueError, TypeError, KeyError) as exc:
            error = f"Data tidak dapat diproses: {exc}"

        except Exception:
            app.logger.exception("Terjadi kesalahan saat melakukan prediksi.")
            error = (
                "Prediksi gagal diproses. Periksa input dan konfigurasi model."
            )

    return render_template(
        "index.html",
        form_fields=FORM_FIELDS,
        payment_fields=PAYMENT_FIELDS,
        bill_fields=BILL_FIELDS,
        amount_fields=AMOUNT_FIELDS,
        prediction=prediction,
        probability=probability,
        error=error,
        submitted_data=submitted_data,
        model_name=model_name,
    )


# =========================
# MENJALANKAN APLIKASI
# =========================
if __name__ == "__main__":
    app.run(debug=True)