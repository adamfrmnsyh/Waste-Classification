import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# ==================== Konfigurasi Awal ====================
st.set_page_config(page_title="Deteksi Sampah", page_icon="♻️", layout="centered")

# Load model dan label
model = tf.saved_model.load("saved_model")
with open("labels.json") as f:
    labels = json.load(f)

# Deskripsi kategori sampah
descriptions = {
    "cardboard": "📦 Kardus: Dapat didaur ulang menjadi kertas atau kemasan baru.",
    "clothes": "👕 Pakaian bekas: Bisa digunakan ulang, disumbangkan, atau didaur ulang.",
    "glass": "🍾 Kaca: Bisa didaur ulang berkali-kali tanpa mengurangi kualitas.",
    "metal": "🥫 Logam: Dapat dilebur dan dibentuk ulang menjadi produk logam lain.",
    "organic": "🌿 Organik: Seperti sisa makanan dan daun, cocok untuk kompos.",
    "paper": "📄 Kertas: Mudah didaur ulang dan ramah lingkungan.",
    "plastic": "🛍️ Plastik: Sulit terurai, perlu pengolahan atau daur ulang khusus."
}

# ==================== Fungsi Pendukung ====================
def crop_image(image):
    crop_size = min(image.width, image.height) * 0.6
    x = (image.width - crop_size) / 2
    y = (image.height - crop_size) / 2
    return image.crop((x, y, x + crop_size, y + crop_size))

def predict_image(image):
    img = crop_image(image).resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    infer = model.signatures["serving_default"]
    prediction = infer(tf.convert_to_tensor(img_array, dtype=tf.float32))
    
    output = list(prediction.values())[0].numpy()
    index = np.argmax(output)
    label = labels[index]
    probability = output[0][index]

    return label, probability, descriptions.get(label, "Deskripsi tidak tersedia.")

# ==================== Sidebar Menu ====================
with st.sidebar:
    st.title("♻️ SmartWaste Classifier")
    menu = st.radio("Navigasi", ["📖 Info Sampah", "📷 Prediksi Gambar"])

# ==================== Halaman Informasi ====================
if menu == "📖 Info Sampah":
    st.title("🔍 Informasi Kategori Sampah")
    st.markdown("Pelajari jenis-jenis sampah dan cara penanganannya.")

    for kategori, penjelasan in descriptions.items():
        st.subheader(kategori.capitalize())
        st.write(penjelasan)
        st.divider()

# ==================== Halaman Prediksi ====================
elif menu == "📷 Prediksi Gambar":
    st.title("📸 Deteksi Kategori Sampah")
    st.write("Unggah gambar sampah, kami bantu klasifikasinya!")

    uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diupload", use_column_width=True)

        with st.spinner("🔎 Menganalisis gambar..."):
            label, prob, description = predict_image(image)

        st.success("🎯 Deteksi Berhasil!")
        st.markdown(f"""
        <div style='font-size:24px'>
            🗑️ <strong>Kategori:</strong> <span style='color:#4CAF50'>{label.capitalize()}</span>  
            <br>📊 <strong>Probabilitas:</strong> {prob:.2%}
        </div>
        """, unsafe_allow_html=True)
        st.info(f"ℹ️ {description}")
