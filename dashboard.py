import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import streamlit.components.v1 as components

# Load model
saved_model_path = 'saved_model'
model = tf.saved_model.load(saved_model_path)

# Load labels
with open('labels.json') as f:
    labels = json.load(f)

# Deskripsi sampah
sampah_deskripsi = {
    "cardboard": (
        "📦 <b>Kardus</b> termasuk sampah kering yang mudah didaur ulang.<br>"
        "<i>Cara Pengolahan:</i> Sobek menjadi lembaran kecil dan kompres agar mudah dikirim ke tempat daur ulang.<br>"
        "<i>Reuse Idea:</i> Bisa digunakan ulang sebagai organizer, alas tanam, atau kerajinan DIY seperti rak mini atau mainan anak."
    ),

    "clothes": (
        "👕 <b>Pakaian bekas</b> sering kali masih bisa dimanfaatkan kembali.<br>"
        "<i>Cara Pengolahan:</i> Pilah pakaian layak untuk disumbangkan. Gunakan pakaian tak layak sebagai kain lap atau bahan kerajinan.<br>"
        "<i>Reuse Idea:</i> Ubah jeans bekas jadi tas, atau kaus lama jadi keset, bantal duduk, hingga boneka kain."
    ),

    "glass": (
        "🍾 <b>Kaca</b> adalah bahan anorganik yang dapat didaur ulang tanpa kehilangan kualitas.<br>"
        "<i>Cara Pengolahan:</i> Cuci bersih dan pisahkan dari bahan lain sebelum dibawa ke bank sampah atau fasilitas daur ulang.<br>"
        "<i>Reuse Idea:</i> Botol kaca bisa dijadikan vas bunga, tempat lilin, atau bahkan lampu hias DIY."
    ),

    "metal": (
        "🛠️ <b>Logam</b> seperti aluminium, baja, dan besi sangat bernilai dan mudah didaur ulang.<br>"
        "<i>Cara Pengolahan:</i> Bersihkan dari sisa makanan/minyak, pisahkan jenis logam, dan kumpulkan untuk dijual atau disetor ke pengepul.<br>"
        "<i>Fakta Menarik:</i> Kaleng aluminium dapat kembali menjadi kaleng baru hanya dalam waktu 60 hari!"
    ),

    "organic": (
        "🌿 <b>Sampah organik</b> berasal dari sisa makanan, kulit buah, sayuran, dan daun kering.<br>"
        "<i>Cara Pengolahan:</i> Campurkan dalam komposter, aduk secara berkala agar proses pengomposan cepat dan optimal.<br>"
        "<i>Manfaat:</i> Kompos bermanfaat untuk menyuburkan tanah, mengurangi sampah di TPA, dan mendukung pertanian ramah lingkungan."
    ),

    "paper": (
        "📄 <b>Kertas</b> adalah sampah yang sangat mudah diolah kembali menjadi produk baru.<br>"
        "<i>Cara Pengolahan:</i> Kertas bekas bisa direndam, diblender, lalu dicetak ulang sebagai kertas baru.<br>"
        "<i>Reuse Idea:</i> Gunakan sebagai catatan, pembungkus kado, bahan scrapbook, atau media seni."
    ),

    "plastic": (
        "🧴 <b>Plastik</b> adalah sampah yang sulit terurai dan menjadi ancaman serius bagi lingkungan.<br>"
        "<i>Cara Pengolahan:</i> Pilah berdasarkan jenis, cuci bersih, lalu setorkan ke bank sampah atau pusat daur ulang plastik.<br>"
        "<i>Fakta Menarik:</i> Plastik PET bisa dilelehkan menjadi benang tekstil, dan plastik HDPE dapat diubah menjadi paving block!"
    )
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

    return predicted_label, prediction_probability

# Sidebar
with st.sidebar:
    st.image("logo.png", width=200)
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #dff0d8;
            padding: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("♻️ Klasifikasi Sampah")
    menu = st.radio("Pilih Menu", ["Informasi Sampah", "Prediksi Sampah"])

# Halaman Informasi
if menu == "Informasi Sampah":
    st.title(" ‼ 𝑺𝒆𝒍𝒂𝒎𝒂𝒕 𝑫𝒂𝒕𝒂𝒏𝒈 𝒅𝒊 𝑾𝒆𝒃𝒔𝒊𝒕𝒆 𝑲𝒍𝒂𝒔𝒊𝒇𝒊𝒌𝒂𝒔𝒊 𝑺𝒂𝒎𝒑𝒂𝒉 ‼ ")
    st.title("🧾 Informasi Kategori Sampah")
    for kategori, konten in sampah_deskripsi.items():
        components.html(f"""
        <div style='background:#f1f8e9;padding:20px;margin-bottom:30px;border-radius:20px;'>
            <h4 style='margin:0 0 10px;text-transform:capitalize;'>{kategori}</h4>
            <p style='margin:0;font-size:20px;line-height:1.6;'>{konten}</p>
        </div>
""", height=300, scrolling=True)


# Halaman Prediksi
elif menu == "Prediksi Sampah":
    st.title("📷 Upload Foto Sampah Kamu")
    st.write("Unggah gambar untuk mengetahui kategori sampahnya.")

    uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", width=600)

        with st.spinner("Sedang memproses gambar..."):
            label, prob = predict_image(image)

        deskripsi = sampah_deskripsi.get(label, "Deskripsi tidak tersedia.")
        st.markdown(f"""
            <div style='background:#e8f5e9;padding:20px;border-radius:10px;'>
                <h3>✅ Kategori Sampah: {label.capitalize()} ({prob:.2f})</h3>
                <p>{deskripsi}</p>
            </div>
        """, unsafe_allow_html=True)
