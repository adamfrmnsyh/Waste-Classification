import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# Load model
saved_model_path = 'saved_model'
model = tf.saved_model.load(saved_model_path)

# Load labels
with open('labels.json') as f:
    labels = json.load(f)

# Deskripsi sampah
descriptions = {
    "cardboard": "Sampah kardus termasuk dalam sampah kering yang dapat didaur ulang menjadi kertas atau kemasan baru.",
    "clothes": "Sampah pakaian dapat digunakan kembali, disumbangkan, atau didaur ulang menjadi bahan lain.",
    "glass": "Sampah kaca dapat didaur ulang tanpa mengurangi kualitasnya dan digunakan kembali dalam industri.",
    "metal": "Sampah logam memiliki nilai ekonomi tinggi dan dapat didaur ulang menjadi berbagai produk logam baru.",
    "organic": "Sampah organik seperti sisa makanan dan daun dapat diolah menjadi kompos.",
    "paper": "Sampah kertas dapat didaur ulang menjadi kertas baru atau produk berbahan dasar kertas.",
    "plastic": "Sampah plastik sulit terurai, namun dapat didaur ulang menjadi barang plastik baru."
}

# Fungsi crop
def crop_image(image):
    crop_size = min(image.width, image.height) * 0.6
    crop_x = (image.width - crop_size) / 2
    crop_y = (image.height - crop_size) / 2
    return image.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))

# Fungsi prediksi
def predict_image(image):
    cropped_image = crop_image(image)
    size = 150
    img = cropped_image.resize((size, size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    infer = model.signatures["serving_default"]
    prediction = infer(tf.convert_to_tensor(img_array, dtype=tf.float32))

    output_name = list(prediction.keys())[0]
    prediction_array = prediction[output_name].numpy()

    max_index = np.argmax(prediction_array)
    predicted_label = labels[max_index]
    prediction_probability = prediction_array[0][max_index]

    descriptions = {
        "cardboard": "Sampah dari kardus, bisa didaur ulang.",
        "clothes": "Pakaian bekas, bisa disumbangkan atau didaur ulang.",
        "glass": "Botol atau benda kaca, bisa digunakan kembali atau didaur ulang.",
        "metal": "Kaleng atau logam lainnya, bisa didaur ulang.",
        "organic": "Sampah makanan atau daun, bisa dibuat kompos.",
        "paper": "Kertas, mudah untuk didaur ulang.",
        "plastic": "Sampah plastik, sulit terurai, sebaiknya dikurangi penggunaannya."
    }

    description = descriptions.get(predicted_label, "Deskripsi tidak tersedia.")
    return predicted_label, prediction_probability, description



# Sidebar
with st.sidebar:
    st.header("♻️ Klasifikasi Sampah")
    menu = st.radio("Pilih Menu", ["Informasi Sampah", "Prediksi Sampah"])

# Halaman Informasi Sampah
if menu == "Informasi Sampah":
    st.title("🧾 Informasi Kategori Sampah")

    for kategori, penjelasan in descriptions.items():
        st.subheader(kategori.capitalize())
        st.write(penjelasan)
        st.divider()

# Halaman Prediksi
elif menu == "Prediksi Sampah":
    st.title("📷 Upload Gambar Sampah")
    st.write("Unggah gambar untuk mengetahui kategori sampahnya.")

    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", width=700)

        with st.spinner("Sedang memproses gambar..."):
            label, prob, description = predict_image(image)

        st.success(f"✅ Kategori Sampah: {label.capitalize()} (Probabilitas: {prob:.2f})")
        st.info(f"🧠 Deskripsi: {description}")
