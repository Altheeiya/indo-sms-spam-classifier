# Indo SMS Spam Classifier

Aplikasi berbasis **Streamlit** untuk mendeteksi SMS spam berbahasa Indonesia menggunakan teknik **Natural Language Processing (NLP)** dan **Machine Learning**.

## Deskripsi

Aplikasi ini mengklasifikasikan pesan SMS/WhatsApp ke dalam 3 kategori:

- **Normal** : Pesan biasa/aman
- **Fraud** : Pesan penipuan
- **Promo** : Pesan promosi

## Preprocessing

1. **Case Folding** : Mengubah teks menjadi huruf kecil
2. **Regex Cleaning** : Menghapus karakter non-alfabet
3. **Stemming** : Menggunakan library [Sastrawi](https://github.com/har07/PySastrawi) untuk stemming Bahasa Indonesia

## Cara Menjalankan

```bash
# Clone repository
git clone https://github.com/Altheeiya/indo-sms-spam-classifier.git
cd indo-sms-spam-classifier

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi
streamlit run app.py
```

## Dependencies

- `streamlit`
- `scikit-learn`
- `Sastrawi`
- `joblib`

## Struktur File

| File                  | Deskripsi                                 |
| --------------------- | ----------------------------------------- |
| `app.py`              | Aplikasi Streamlit untuk klasifikasi SMS  |
| `Notebook.ipynb`      | Notebook eksplorasi data & training model |
| `requirements.txt`    | Daftar dependensi Python                  |
| `model_spam.pkl`      | Model klasifikasi spam yang sudah dilatih |
| `vectorizer_spam.pkl` | TF-IDF vectorizer untuk ekstraksi fitur   |
