import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Memuat Model dan Scaler yang telah disimpan dari Soal 3
model = joblib.load('best_model_lr.pkl')
scaler = joblib.load('scaler.pkl')

# 2. Pengaturan Konfigurasi Halaman Aplikasi
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

st.title("📊 Aplikasi Prediksi Retensi Pelanggan (Telco Customer Churn)")
st.markdown("""
Aplikasi web ini menggunakan model **Logistic Regression terbaik** untuk mendeteksi probabilitas pelanggan yang akan berhenti berlangganan (*Churn*) secara *real-time*.
""")

st.divider()

# 3. Pembuatan Form Input Data Pelanggan Terbagi Menjadi 3 Kolom Rapi
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Profil Demografis")
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    senior_citizen = st.selectbox("Lansia (Senior Citizen)?", ["No", "Yes"])
    partner = st.selectbox("Memiliki Pasangan?", ["No", "Yes"])
    dependents = st.selectbox("Memiliki Tanggungan?", ["No", "Yes"])

with col2:
    st.subheader("💳 Informasi Akun & Layanan")
    tenure = st.slider("Masa Berlangganan (Bulan)", min_value=0, max_value=72, value=12)
    phone_service = st.selectbox("Layanan Telepon", ["No", "Yes"])
    multiple_lines = st.selectbox("Multi-Line Telepon", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Layanan Internet", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Tagihan Tanpa Kertas (Paperless)?", ["No", "Yes"])
    payment_method = st.selectbox("Metode Pembayaran", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col3:
    st.subheader("💰 Informasi Biaya & Proteksi")
    monthly_charges = st.number_input("Biaya Bulanan ($)", min_value=0.0, max_value=200.0, value=65.0)
    total_charges = st.number_input("Total Biaya Keseluruhan ($)", min_value=0.0, max_value=10000.0, value=780.0)
    
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# 4. Prakondisi Pemetaan Data Input (Encoding Manual Sesuai Urutan Fitur Asli)
input_data = {
    'gender': 1 if gender == "Male" else 0,
    'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    'Partner': 1 if partner == "Yes" else 0,
    'Dependents': 1 if dependents == "Yes" else 0,
    'tenure': tenure,
    'PhoneService': 1 if phone_service == "Yes" else 0,
    'MultipleLines': 2 if multiple_lines == "Yes" else (0 if multiple_lines == "No" else 1),
    'InternetService': 0 if internet_service == "DSL" else (1 if internet_service == "Fiber optic" else 2),
    'OnlineSecurity': 2 if online_security == "Yes" else (0 if online_security == "No" else 1),
    'OnlineBackup': 2 if online_backup == "Yes" else (0 if online_backup == "No" else 1),
    'DeviceProtection': 2 if device_protection == "Yes" else (0 if device_protection == "No" else 1),
    'TechSupport': 2 if tech_support == "Yes" else (0 if tech_support == "No" else 1),
    'StreamingTV': 2 if streaming_tv == "Yes" else (0 if streaming_tv == "No" else 1),
    'StreamingMovies': 2 if streaming_movies == "Yes" else (0 if streaming_movies == "No" else 1),
    'Contract': 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2),
    'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
    'PaymentMethod': 2 if payment_method == "Electronic check" else (3 if payment_method == "Mailed check" else (0 if "Bank transfer" in payment_method else 1)),
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

df_input = pd.DataFrame([input_data])

# 5. Tombol Eksekusi Prediksi
st.divider()
if st.button("🚀 Jalankan Analisis Retensi Pelanggan", type="primary"):
    # Penskalaan fitur numerik menggunakan StandardScaler yang dimuat
    df_input_scaled = scaler.transform(df_input)
    
    # Prediksi menggunakan model Logistic Regression
    prediksi = model.predict(df_input_scaled)[0]
    probabilitas = model.predict_proba(df_input_scaled)[0][1]
    
    # Menampilkan Hasil kepada User
    st.subheader("📢 Hasil Analisis Keputusan:")
    if prediksi == 1:
        st.error(f"🚨 **RISIKO TINGGI (CHURN):** Pelanggan diprediksi akan BERHENTI berlangganan dengan tingkat probabilitas sebesar **{probabilitas*100:.2f}%**.")
        st.markdown("""
        **Rekomendasi Tindakan Segera (Tim Retensi):**
        * Tawarkan program migrasi ke kontrak 1 atau 2 tahun dengan diskon khusus.
        * Berikan bonus kuota gratis atau gratis layanan *Online Security/Tech Support* untuk meningkatkan *value*.
        """)
    else:
        st.success(f"✅ **AMAN (RETAIN):** Pelanggan diprediksi akan TETAP BERLANGGANAN dengan tingkat probabilitas loyalitas sebesar **{(1-probabilitas)*100:.2f}%**.")
