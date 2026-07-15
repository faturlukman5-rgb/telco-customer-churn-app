import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(page_title="Telco Churn Analytics Hub", layout="wide")


@st.cache_resource
def load_assets():
   
    try:
        model = joblib.load('best_model_lr.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()


st.sidebar.title(" Navigasi Sistem")
st.sidebar.markdown("Silakan pilih menu untuk meninjau kelengkapan proyek UAS:")
menu = st.sidebar.selectbox(
    "Pilih Halaman:",
    [
        "1.  Dashboard EDA",
        "2.  Model Demo (Prediksi)",
        "3.  Evaluasi Model",
        "4.  Interpretasi Hasil",
        "5.  Dokumentasi Proyek"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("**Oleh Kelompok UAS:**")
st.sidebar.caption("• FATUR LUKMAN DINATA       A11.2024.16076")
st.sidebar.caption("• ATHAYA HELGA RAMADHANU  A11.2024.15832")


if menu == "1.  Dashboard EDA":
    st.title(" Halaman 1: Dashboard Exploratory Data Analysis (EDA)")
    st.markdown("Menampilkan *insight* visual penting dari dataset historis *Telco Customer Churn*.")
    st.divider()
    
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        st.subheader(" Distribusi Target Variable (Churn vs Retain)")
       
        eda_target = pd.DataFrame({
            'Status': ['Retain (Tetap)', 'Churn (Keluar)'],
            'Jumlah Pelanggan': [5174, 1869]
        })
        st.bar_chart(data=eda_target, x='Status', y='Jumlah Pelanggan', color='#29b5e8')
        st.caption("Insight: Terjadi ketidakseimbangan kelas (imbalance data) di mana mayoritas pelanggan berstatus Retain.")

    with col_eda2:
        st.subheader(" Hubungan Jenis Kontrak terhadap Rasio Churn")

        eda_contract = pd.DataFrame({
            'Jenis Kontrak': ['Month-to-month', 'One year', 'Two year'],
            'Rasio Churn (%)': [42.7, 11.2, 2.8]
        })
        st.bar_chart(data=eda_contract, x='Jenis Kontrak', y='Rasio Churn (%)', color='#ff4b4b')
        st.caption("Insight: Pelanggan dengan kontrak jangka pendek (Month-to-month) memiliki kerentanan churn tertinggi.")


# =========================================================================
elif menu == "2.  Model Demo (Prediksi)":
    st.title(" Halaman 2: Model Demo & Prediksi Real-Time")
    st.markdown("Gunakan form di bawah ini untuk mensimulasikan karakteristik pelanggan dan menghitung risiko *churn*.")
    st.divider()

   
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader(" Profil Demografis")
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        senior_citizen = st.selectbox("Lansia (Senior Citizen)?", ["No", "Yes"])
        partner = st.selectbox("Memiliki Pasangan?", ["No", "Yes"])
        dependents = st.selectbox("Memiliki Tanggungan?", ["No", "Yes"])

    with col2:
        st.subheader(" Informasi Akun & Layanan")
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
        st.subheader(" Informasi Biaya & Proteksi")
        monthly_charges = st.number_input("Biaya Bulanan ($)", min_value=0.0, max_value=200.0, value=65.0)
        total_charges = st.number_input("Total Biaya Keseluruhan ($)", min_value=0.0, max_value=10000.0, value=780.0)
        
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

   
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

    st.divider()
    if st.button(" Jalankan Analisis Retensi Pelanggan", type="primary"):
        if model is not None and scaler is not None:
            df_input_scaled = scaler.transform(df_input)
            
            prediksi = model.predict(df_input_scaled)[0]
            
            
            probabilitas_all = model.predict_proba(df_input_scaled)[0]
            probabilitas_churn = probabilitas_all[1]
            probabilitas_retain = probabilitas_all[0]

            res_col1, res_col2 = st.columns([1.2, 1.0])
            
            with res_col1:
                st.subheader("Hasil Analisis Keputusan:")
                if prediksi == 1:
                    st.error(f" **RISIKO TINGGI (CHURN):** Pelanggan diprediksi akan BERHENTI berlangganan dengan tingkat probabilitas sebesar **{probabilitas_churn*100:.2f}%**.")
                    st.markdown("""
                    **Rekomendasi Tindakan Segera (Tim Retensi):**
                    * Tawarkan program migrasi ke kontrak 1 atau 2 tahun dengan diskon khusus.
                    * Berikan bonus kuota gratis atau gratis layanan *Online Security/Tech Support* untuk meningkatkan *value*.
                    """)
                else:
                    st.success(f"**AMAN (RETAIN):** Pelanggan diprediksi akan TETAP BERLANGGANAN dengan tingkat probabilitas loyalitas sebesar **{probabilitas_retain*100:.2f}%**.")
                    st.markdown("""
                    **Rekomendasi Tindakan (Tim Retensi):**
                    * Jaga kepuasan pelanggan dengan program *loyalty reward* berkala.
                    * Tawarkan layanan pelengkap (*upselling*) secara soft-marketing saat masa kontrak mendekati akhir.
                    """)
                    
            with res_col2:
                st.subheader(" Visualisasi Probabilitas")
                
                chart_data = pd.DataFrame({
                    'Status Keputusan': ['Tetap Bertahan (Retain)', 'Akan Pergi (Churn)'],
                    'Tingkat Keyakinan (%)': [probabilitas_retain * 100, probabilitas_churn * 100]
                })
                
              
                st.bar_chart(
                    data=chart_data,
                    x='Status Keputusan',
                    y='Tingkat Keyakinan (%)',
                    color='#ff4b4b' if prediksi == 1 else '#29b5e8'
                )
                st.caption("Grafik interaktif di atas merepresentasikan tingkat keyakinan matematis model terhadap keputusan pelanggan.")
        else:
            st.warning(" File model `best_model_lr.pkl` atau `scaler.pkl` tidak ditemukan. Menampilkan hasil simulasi dummy:")
            st.info(" **SIMULASI CHURN (Contoh Tanpa Model):** Probabilitas Churn 82.92%")


elif menu == "3.  Evaluasi Model":
    st.title(" Halaman 3: Laporan Evaluasi Model Machine Learning")
    st.markdown("Berikut adalah tabel perbandingan performa model yang diuji coba pada Soal 3.")
    st.divider()


    metrics_df = pd.DataFrame({
        'Model Baseline': ['Logistic Regression (Selected)', 'Random Forest Classifier'],
        'Accuracy': ['80.12%', '79.45%'],
        'Precision': ['76.50%', '78.10%'],
        'Recall (Prioritas Bisnis)': ['79.68%', '71.20%'],
        'F1-Score': ['78.06%', '74.49%']
    })
    st.table(metrics_df)
    
    st.markdown("""
    > **Mengapa Logistic Regression yang dipilih?** 
    > Karena memiliki nilai **Recall** tertinggi ($79.68\%$). Dalam kasus penanganan Churn, kita ingin meminimalkan kegagalan prediksi pelanggan yang sebenarnya ingin pindah (*False Negative*).
    """)


elif menu == "4.  Interpretasi Hasil":
    st.title("Halaman 4: Interpretasi Feature Importance & Business Insight")
    st.markdown("Menganalisis faktor apa saja yang paling berkontribusi mendorong pelanggan untuk *churn*.")
    st.divider()

  
    coef_df = pd.DataFrame({
        'Nama Fitur/Variabel': ['Contract (Month-to-month)', 'Internet Service (Fiber Optic)', 'Monthly Charges', 'Tenure (Masa Berlangganan)'],
        'Kekuatan Pengaruh terhadap Churn': [2.4, 1.8, 1.1, -2.9]
    })
    
    st.subheader(" Bobot Koefisien Fitur Utama")
    st.bar_chart(data=coef_df, x='Nama Fitur/Variabel', y='Kekuatan Pengaruh terhadap Churn', color='#ff9f43')
    
    st.markdown("""
    *   **Nilai Positif (+):** Mendorong peningkatan risiko pelanggan untuk *Churn* (Contoh: Kontrak bulanan dan Fiber Optic).
    *   **Nilai Negatif (-):** Mendorong loyalitas / pelanggan tetap bertahan (Contoh: Semakin lama *Tenure*, risiko churn semakin kecil).
    """)


elif menu == "5.  Dokumentasi Proyek":
    st.title(" Halaman 5: Dokumentasi Proyek & Tata Cara Penggunaan")
    st.divider()
    
    st.subheader(" Informasi Dataset")
    st.markdown("""
    *   **Sumber Data:** Kaggle Telco Customer Churn Dataset.
    *   **Ukuran Data:** 7,043 Baris dengan 21 Kolom Fitur.
    *   **Tipe Tugas ML:** Klasifikasi Biner (0: Retain, 1: Churn).
    """)
    
    st.subheader(" Cara Menjalankan Aplikasi Secara Lokal")
    st.code("""
# 1. Clone repository GitHub Anda
git clone <url-repository-anda>

# 2. Install dependency requirements
pip install -r requirements.txt

# 3. Jalankan server Streamlit
streamlit run app.py
    """, language="bash")
