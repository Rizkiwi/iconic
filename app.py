import os
import streamlit as st
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import PowerTransformer
import miceforest as mf
import matplotlib.pyplot as plt

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
Masukkan data calon pemohon untuk mendapatkan prediksi.
""")

# Membuat form untuk input data
with st.form("input_form"):
    # Membuat dua kolom yang dibagi rata untuk input
    col1, col2 = st.columns([1, 1])
    
    # Kolom pertama
    with col1:
        Ind_ID = st.text_input("Ind ID", value="5008827")
        GENDER = st.selectbox("Gender", options=['M', 'F'], index=0)
        Car_Owner = st.selectbox("Car Owner", options=['Y', 'N'], index=0)
        Propert_Owner = st.selectbox("Property Owner", options=['Y', 'N'], index=0)
        CHILDREN = st.number_input("Number of Children", min_value=0, value=0)
        Annual_income = st.number_input("Annual Income", value=180000.0)
        Type_Income = st.selectbox("Type of Income", options=['Commercial associate', 'State servant', 'Working', 'Pensioner'], index=3)
        EDUCATION = st.selectbox("Education Level", options=['Higher education', 'Secondary / secondary special', 'Incomplete higher', 'Lower secondary'], index=0)
        Marital_status = st.selectbox("Marital Status", options=['Married', 'Single', 'Separated/Widow'], index=0)
        Family_Members = st.number_input("Family Members", min_value=1, value=1)
        Age = st.number_input("Age", min_value=0.0, value=51.0)
        Tenure = st.number_input("Tenure (years)", min_value=0.0, value=0.0)
        Unemployment_duration = st.number_input("Unemployment Duration", min_value=0, value=0)
        
    # Kolom kedua
    with col2:
        Housing_type = st.selectbox("Housing Type", options=['House / apartment', 'Co-op apartment', 'Municipal apartment', 'Office apartment', 'Rented apartment', 'With parents'], index=0)
        Birthday_count = st.number_input("Birthday Count", value=-18772.0)
        Employed_days = st.number_input("Employed Days", value=365243)
        Mobile_phone = st.selectbox("Mobile Phone", options=['Y', 'N'], index=0)
        Work_Phone = st.selectbox("Work Phone", options=['Y', 'N'], index=1)
        Phone = st.selectbox("Phone", options=['Y', 'N'], index=1)
        EMAIL_ID = st.text_input("Email ID", value="")
        Type_Occupation = st.selectbox("Type of Occupation", options=[
            'Managers', 'High skill tech staff', 'IT staff', 'Accountants', 'HR staff', 
            'Core staff', 'Medicine staff', 'Sales staff', 'Realty agents', 'Secretaries',
            'Private service staff', 'Security staff', 'Drivers', 'Cooking staff', 
            'Cleaning staff', 'Waiters/barmen staff', 'Laborers', 'Low-skill Laborers'
        ], index=1)
        Is_currently_employed = st.selectbox("Is Currently Employed", options=['Y', 'N'], index=0)
        Children_to_family_ratio = st.number_input("Children to Family Ratio", min_value=0.0, step=0.01, value=0.0)
        Children_employment_impact = st.number_input("Children Employment Impact", min_value=0.0, step=0.01, value=0.0)
        Income_per_year_employed = st.number_input("Income per Year Employed", min_value=0.0, value=0.0)
        Income_sgmt = st.selectbox("Income Segment", options=['H', 'Medium', 'Low'], index=1)
        Age_group = st.selectbox("Age Group", options=['Senior Adult', 'Adult', 'Young Adult'], index=0)

    submitted = st.form_submit_button("Submit")


if submitted:
    # Buat dataframe dari input form
    data = {
        'Ind_ID': [Ind_ID],
        'GENDER': [GENDER],
        'Car_Owner': [Car_Owner],
        'Propert_Owner': [Propert_Owner],
        'CHILDREN': [CHILDREN],
        'Annual_income': [Annual_income],
        'Type_Income': [Type_Income],
        'EDUCATION': [EDUCATION],
        'Marital_status': [Marital_status],
        'Housing_type': [Housing_type],
        'Birthday_count': [Birthday_count],
        'Employed_days': [Employed_days],
        'Mobile_phone': [Mobile_phone],
        'Work_Phone': [Work_Phone],
        'Phone': [Phone],
        'EMAIL_ID': [EMAIL_ID],
        'Type_Occupation': [Type_Occupation],
        'Family_Members': [Family_Members],
        'Age': [Age],
        'Tenure': [Tenure],
        'Unemployment_duration': [Unemployment_duration],
        'Is_currently_employed': [Is_currently_employed],
        'Children_to_family_ratio': [Children_to_family_ratio],
        'Children_employment_impact': [Children_employment_impact],
        'Income_per_year_employed': [Income_per_year_employed],
        'Income_sgmt': [Income_sgmt],
        'Age_group': [Age_group]
    }

    df = pd.DataFrame(data)

    # Mapping kategori ke numerik
    mappings = {
        'GENDER': {'M': 0, 'F': 1},
        'Car_Owner': {'N': 0, 'Y': 1},
        'Propert_Owner': {'N': 0, 'Y': 1},
        'Mobile_phone': {'N': 0, 'Y': 1},
        'Work_Phone': {'N': 0, 'Y': 1},
        'Phone': {'N': 0, 'Y': 1},
        'Is_currently_employed': {'N': 0, 'Y': 1},
        'Type_Income': {'Commercial associate': 4, 'State servant': 3, 'Working': 2, 'Pensioner': 1},
        'EDUCATION': {'Higher education': 4, 'Secondary / secondary special': 3, 'Incomplete higher': 2, 'Lower secondary': 1},
        'Marital_status': {'Married': 3, 'Separated/Widow': 2, 'Single': 1},
        'Housing_type': {'House / apartment': 6, 'Co-op apartment': 5, 'Municipal apartment': 4, 'Office apartment': 3, 'Rented apartment': 2, 'With parents': 1},
        'Income_sgmt': {'H': 1, 'Medium': 0, 'Low': -1},
        'Age_group': {'Senior Adult': 1, 'Adult': 0, 'Young Adult': -1},
        'Type_Occupation': {
            'Managers': 18, 'High skill tech staff': 17, 'IT staff': 16, 'Accountants': 15, 'HR staff': 14, 
            'Core staff': 13, 'Medicine staff': 12, 'Sales staff': 11, 'Realty agents': 10, 'Secretaries': 9,
            'Private service staff': 8, 'Security staff': 7, 'Drivers': 6, 'Cooking staff': 5, 
            'Cleaning staff': 4, 'Waiters/barmen staff': 3, 'Laborers': 2, 'Low-skill Laborers': 1
        }
    }

    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)

    # Prediksi menggunakan model
    predictions = model.predict(df)

    # Tampilkan hasil prediksi
    df['Prediction'] = predictions
    st.write(f"Hasil prediksi untuk ID {Ind_ID}: {'**Approved**' if predictions[0] == 1 else '**Rejected**'}")

    # Save the result to CSV file
    csv_file = 'credit_predictions.csv'
    if os.path.exists(csv_file):
        # Append to existing CSV
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        # Create a new CSV
        df.to_csv(csv_file, mode='w', header=True, index=False)

    # Tampilkan data dari CSV
    st.write("Data hasil prediksi sebelumnya:")
    previous_data = pd.read_csv(csv_file)
    st.dataframe(previous_data)

    # Tambahkan fitur download CSV
    st.download_button(
        label="Download data sebagai CSV",
        data=previous_data.to_csv(index=False),
        file_name='credit_predictions.csv',
        mime='text/csv'
    )

    # Visualisasi hasil prediksi menggunakan pie chart
    counts = previous_data['Prediction'].value_counts()
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
