import streamlit as st
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import PowerTransformer
import miceforest as mf
import matplotlib.pyplot as plt  # Tambahan import untuk visualisasi

# Load model and preprocessing steps
model_data = joblib.load('model.pkl')

# Ambil komponen model
model = model_data['model']
power_transformer = model_data['power_transformer']
log_cols = model_data['log_cols']
norm_cols = model_data['norm_cols']

# Judul aplikasi
st.title("✨ Credit Card Approval Classification ✨")

# Deskripsi aplikasi
st.write("""
Aplikasi ini menggunakan model yang sudah dilatih untuk memprediksi apakah seseorang akan disetujui atau ditolak dalam pengajuan kartu kredit berdasarkan data input yang disediakan.
Silakan unggah file CSV yang berisi data calon pemohon untuk mendapatkan prediksi.
""")

# Upload file CSV
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    # Baca file CSV
    df = pd.read_csv(uploaded_file)

    # Mengganti karakter khusus di nama kolom
    df.columns = df.columns.str.replace(r'[^\w\s]', '_')  # Mengganti karakter non-alfanumerik dengan underscore (_)
    df.columns = df.columns.str.replace(' ', '_')         # Mengganti spasi dengan underscore (_)

    # Tampilkan isi data
    st.write("Data yang diunggah:")
    st.write(df.head())

    # Daftar kolom yang dibutuhkan untuk model
    required_columns = ['Age', 'Annual_income', 'Children_to_family_ratio', 'Birthday_count', 'Car_Owner', 
                        'Children_employment_impact', 'EDUCATION', 'EMAIL_ID', 'Employed_days', 'GENDER', 
                        'Housing_type', 'Income_per_year_employed', 'Income_sgmt', 'Marital_status', 
                        'Phone', 'Propert_Owner', 'Tenure', 'Type_Income', 'Type_Occupation', 
                        'Unemployment_duration', 'Work_Phone', 'Age_group', 'CHILDREN', 'Family_Members', 
                        'Is_currently_employed']

    # Pastikan bahwa CSV memiliki kolom yang dibutuhkan
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Kolom berikut tidak ada di dalam CSV: {', '.join(missing_columns)}")
    else:
        # Menghapus duplikasi
        df.drop_duplicates(inplace=True)

        # Mapping untuk kolom kategori
        marital_status_mapping = {
            'Civil marriage': 'Married',
            'Married': 'Married',
            'Separated': 'Separated/Widow',
            'Widow': 'Separated/Widow',
            'Single / not married': 'Single'
        }
        df['Marital_status'] = df['Marital_status'].map(marital_status_mapping)
        df['EDUCATION'] = df['EDUCATION'].replace(['Academic degree'], 'Higher education')
        df = df.drop('Mobile_phone', axis=1)

        # Mapping untuk berbagai kolom kategori menjadi numerik
        mappings = {
            'Type_Income': {'Commercial associate': 4, 'State servant': 3, 'Working': 2, 'Pensioner': 1},
            'EDUCATION': {'Higher education': 4, 'Secondary / secondary special': 3, 'Incomplete higher': 2, 'Lower secondary': 1},
            'Marital_status': {'Married': 3, 'Separated/Widow': 2, 'Single': 1},
            'Housing_type': {'House / apartment': 6, 'Co-op apartment': 5, 'Municipal apartment': 4, 'Office apartment': 3, 'Rented apartment': 2, 'With parents': 1},
            'GENDER': {'M': 0, 'F': 1},
            'Car_Owner': {'N': 0, 'Y': 1},
            'Propert_Owner': {'N': 0, 'Y': 1},
            'Income_sgmt': {'H': 1, 'Medium': 0, 'Low': -1},
            'Age_group': {'Senior Adult': 1, 'Adult': 0, 'Young Adult': -1},
            'Type_Occupation': {'Managers': 18, 'High skill tech staff': 17, 'IT staff': 16, 'Accountants': 15, 'HR staff': 14, 'Core staff': 13, 
                                'Medicine staff': 12, 'Sales staff': 11, 'Realty agents': 10, 'Secretaries': 9, 'Private service staff': 8, 
                                'Security staff': 7, 'Drivers': 6, 'Cooking staff': 5, 'Cleaning staff': 4, 'Waiters/barmen staff': 3, 
                                'Laborers': 2, 'Low-skill Laborers': 1}
        }

        for col, mapping in mappings.items():
            df[col] = df[col].map(mapping)

        # Pastikan tipe data menjadi numerik
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Imputasi missing values menggunakan MICE
        columns_to_impute = ['Type_Occupation', 'Income_sgmt', 'GENDER', 'Annual_income', 'Birthday_count', 'Age']

        # Membuat kernel MICE untuk imputasi
        kds = mf.ImputationKernel(
            data=df,  # Data yang ingin diimputasi
            save_all_iterations_data=True,  # Menyimpan data dari setiap iterasi
            random_state=1991  # Untuk hasil yang dapat direproduksi
        )

        # Menjalankan algoritma MICE (Multiple Imputation by Chained Equations)
        kds.mice(iterations=5, n_estimators=50)

        # Menghasilkan dataset lengkap setelah imputasi
        df_imputed = kds.complete_data()

        # Mengganti nilai pada kolom yang di-imputasi
        df[columns_to_impute] = df_imputed[columns_to_impute].copy()

        # Log transformasi dan normalisasi
        skew_type_list = []
        skew_val_list = []
        kurtosis_val_list = []

        cols_for_outlier = ['CHILDREN', 'Annual_income', 'Employed_days', 'Family_Members', 'Tenure', 
                            'Unemployment_duration', 'Children_to_family_ratio', 
                            'Children_employment_impact', 'Income_per_year_employed']

        for column in cols_for_outlier:
            data = df[column].dropna(axis=0)
            skew_val = round(skew(data, nan_policy="omit"), 3)
            kurtosis_val = round(kurtosis(data, nan_policy="omit"), 3)

            skew_type = "Normal Distribution" if -0.2 < skew_val < 0.2 else (
                "Positively Skewed" if skew_val > 0.2 else "Negatively Skewed")

            skew_type_list.append(skew_type)
            skew_val_list.append(skew_val)
            kurtosis_val_list.append(kurtosis_val)

        dist = pd.DataFrame({
            "Column Name": cols_for_outlier,
            "Skewness": skew_val_list,
            "Kurtosis": kurtosis_val_list,
            "Type of Distribution": skew_type_list
        })

        exclude = ["CHILDREN", "Family_Members"]
        log_cols = sorted(list(dist[
            dist["Type of Distribution"].str.contains("Positively Skewed") & 
            ~dist["Column Name"].isin(exclude)
        ]["Column Name"].values))

        norm_cols = sorted(list(dist[
            dist["Type of Distribution"].str.contains("Normal Distribution") & 
            ~dist["Column Name"].isin(exclude)
        ]["Column Name"].values))

        # Terapkan log transformasi dan normalisasi
        if log_cols:
            df[log_cols] = power_transformer.transform(df[log_cols])

        if norm_cols:
            df[norm_cols] = (df[norm_cols] - df[norm_cols].mean()) / df[norm_cols].std()

        # Prediksi menggunakan model
        predictions = model.predict(df[required_columns])

        # Tampilkan hasil prediksi
        df['Prediction'] = predictions
        st.write("Hasil prediksi:")
        st.write(df[['EMAIL_ID', 'Prediction']])

        # Visualisasi hasil prediksi menggunakan pie chart
        counts = df['Prediction'].value_counts()
        labels = ['Approved', 'Rejected']
        sizes = [counts.get(1, 0), counts.get(0, 0)]
        colors = ['#81c784', '#e57373']  # Softer green for approved, softer red for rejected

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax.axis('equal')  # Agar pie chart berbentuk lingkaran proporsional
        st.pyplot(fig)

        # Tampilkan jumlah data yang rejected dan approved
        approved_count = counts.get(1, 0)
        rejected_count = counts.get(0, 0)

        st.write(f"Jumlah Approved: {approved_count}")
        st.write(f"Jumlah Rejected: {rejected_count}")

        # Unduh hasil prediksi
        csv = df[['EMAIL_ID', 'Prediction']].to_csv(index=False)
        st.download_button(label="Unduh hasil prediksi", data=csv, file_name='predictions.csv', mime='text/csv')
