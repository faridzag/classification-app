import streamlit as st
import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Student Grade Prediction",
    page_icon="📚",
    layout="centered"
)

# Function to load data from h5 file
@st.cache_resource
def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        X_train = f['X_train'][:]
        y_train = f['y_train'][:]
        categories = eval(f.attrs['category_mapping'])
    return X_train, y_train, categories

# Function to train model
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Main function
def main():
    st.title("📚 Prediksi Grade Siswa")
    st.write("Prediksi nilai grade label G3 dari nilai G1 dan G2")
    
    try:
        # Load data and train model
        file_path = "data_train.h5"  # Sesuaikan dengan path file Anda
        X_train, y_train, categories = load_data(file_path)
        model = train_model(X_train, y_train)
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                g1 = st.number_input("Masukkan Nilai G1(0-20)", min_value=0, max_value=20)
            
            with col2:
                g2 = st.number_input("Masukkan Nilai G2(0-20)", min_value=0, max_value=20)
            
            submit = st.form_submit_button("Prediksi Grade")
            
            if submit:
                # Make prediction
                input_data = np.array([[g1, g2]])
                prediction = model.predict(input_data)[0]
                
                # Calculate approximate G3 value (average of G1 and G2)
                g3_approx = (g1 + g2) / 2
                
                # Display results in a nice format
                st.markdown("### Hasil")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Nilai G3", f"{g3_approx:.1f}")
                
                with col2:
                    st.metric("Kategori Grade/Ranking", categories[prediction])
                
                # Add some context based on the prediction
                if prediction == 2:  # tinggi
                    st.success("Kinerja yang Sangat Baik! Lanjutkan dan Pertahankan Nilaimu! 🌟")
                elif prediction == 1:  # sedang
                    st.info("Kerja Bagus. Ada ruang untuk perkembangan! 📈")
                else:  # rendah
                    st.warning("Pertimbangkan untuk mencari bantuan agar bisa memperbaiki nilaimu. 📚")
        
        # Add some information about the grading system
        with st.expander("ℹ️ Tentang Sistem Grading"):
            st.write("""
            - G1 dan G2 adalah nilai semester (skala: 0-20)
            - G3 rata-rata dari G1 dan G2
            - Kategori Grade:
                - High (Tinggi): ≥ 15
                - Medium (Sedang): 10-14
                - Low (Rendah): < 10
            """)
            
    except Exception as e:
        st.error(f"Error load model atau membuat prediksi: {str(e)}")
        st.write("Pastikan File h5 ada dan berada di tempat sesuai.")

if __name__ == "__main__":
    main()