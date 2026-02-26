import streamlit as st
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#  1. SETUP PREPROCESSING 
# Inisialisasi Sastrawi
@st.cache_resource 
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

stemmer = load_stemmer()

def preprocess_text(text):
    text = str(text).lower() # Case folding
    text = re.sub(r'[^a-z\s]', '', text) # Hapus karakter non-alfabet
    text = stemmer.stem(text) # Stemming
    return text

# 2. LOAD MODEL & VECTORIZER
@st.cache_resource
def load_models():
    
    model = joblib.load('model_spam.pkl')
    vectorizer = joblib.load('vectorizer_spam.pkl')
    return model, vectorizer

model, vectorizer = load_models()

#  3. UI STREAMLIT 
st.title(" Deteksi SMS Spam Berbahasa Indonesia")
st.write("Aplikasi NLP untuk mendeteksi pesan masuk: Normal, Fraud (Penipuan), atau Promo.")

# Form input dari user
user_input = st.text_area("Masukkan teks pesan (SMS/WhatsApp) di sini:", height=150)

if st.button("Analisis Pesan"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks pesan terlebih dahulu!")
    else:
        with st.spinner("Menganalisis pola teks..."):
            # Proses input
            cleaned_text = preprocess_text(user_input)
            
            # Ekstraksi fitur
            vectorized_text = vectorizer.transform([cleaned_text])
            
            # Prediksi
            prediction = model.predict(vectorized_text)[0]
            
            st.markdown("---")
            st.subheader("Hasil Analisis:")
            
            
            if str(prediction) == '0' or str(prediction).lower() == 'normal':
                st.success(" **Aman**: Pesan ini terdeteksi sebagai pesan NORMAL.")
            elif str(prediction) == '1' or str(prediction).lower() == 'fraud' or str(prediction).lower() == 'penipuan':
                st.error(" **Bahaya**: Pesan ini terdeteksi sebagai SPAM/PENIPUAN (FRAUD)!")
            elif str(prediction) == '2' or str(prediction).lower() == 'promo':
                st.warning(" **Info**: Pesan ini terdeteksi sebagai pesan PROMO.")
            else:
                st.info(f"Kategori terdeteksi: **{prediction}**")
            
            
            with st.expander("Lihat detail preprocessing"):
                st.write("**Teks Asli:**", user_input)
                st.write("**Teks Bersih (Setelah Case Folding, Regex, & Stemming):**", cleaned_text)