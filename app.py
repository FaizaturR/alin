import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import joblib
import trafilatura

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from gensim.models import Word2Vec


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Klasifikasi Berita",
    page_icon="📰",
    layout="centered"
)


# =========================================================
# LOAD STOPWORD DAN STEMMER
# =========================================================

@st.cache_resource
def load_preprocessing():
    # Download stopword Bahasa Indonesia
    nltk.download("stopwords", quiet=True)

    stop_words = set(stopwords.words("indonesian"))

    # Membuat stemmer Sastrawi
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    return stop_words, stemmer


stop_words, stemmer = load_preprocessing()


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_models():

    # Load model Skip-Gram
    skipgram_model = Word2Vec.load(
        "skipgram_final.model"
    )

    # Load model Naive Bayes
    nb_model = joblib.load(
        "naive_bayes_skipgram.pkl"
    )

    return skipgram_model, nb_model


skipgram_model, nb_model = load_models()


# =========================================================
# FUNGSI PREPROCESSING
# =========================================================

def preprocessing(teks):

    # 1. Lowercase
    teks = teks.lower()

    # 2. Hapus angka
    teks = re.sub(
        r"\d+",
        " ",
        teks
    )

    # 3. Hapus tanda baca
    teks = re.sub(
        r"[^\w\s]",
        " ",
        teks
    )

    # 4. Hapus spasi berlebih
    teks = re.sub(
        r"\s+",
        " ",
        teks
    ).strip()

    # 5. Tokenisasi
    tokens = teks.split()

    # 6. Stopword removal
    tokens = [
        kata
        for kata in tokens
        if kata not in stop_words
    ]

    # 7. Stemming
    tokens = [
        stemmer.stem(kata)
        for kata in tokens
    ]

    return tokens


# =========================================================
# FUNGSI DOCUMENT VECTOR / VSM
# =========================================================

def document_vector(tokens, model):

    vectors = []

    for word in tokens:

        if word in model.wv:
            vectors.append(
                model.wv[word]
            )

    # Jika tidak ada kata yang terdapat
    # di vocabulary Skip-Gram
    if len(vectors) == 0:

        return np.zeros(
            model.vector_size
        )

    # Rata-rata vector semua kata
    return np.mean(
        vectors,
        axis=0
    )


# =========================================================
# FUNGSI MENGAMBIL ARTIKEL DARI URL
# =========================================================

def extract_article(url):

    try:

        downloaded = trafilatura.fetch_url(url)

        if downloaded is None:
            return None

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False
        )

        return text

    except Exception:
        return None


# =========================================================
# TAMPILAN APLIKASI
# =========================================================

st.title("📰 Klasifikasi Berita")

st.write(
    "Klasifikasi berita menggunakan "
    "Skip-Gram dan Naive Bayes."
)

st.write(
    "Masukkan URL berita dari kategori "
    "**Sport** atau **Finance**."
)


# =========================================================
# INPUT URL
# =========================================================

url = st.text_input(
    "Masukkan URL berita:",
    placeholder="https://sport.detik.com/..."
)


# =========================================================
# TOMBOL KLASIFIKASI
# =========================================================

if st.button(
    "🔍 Klasifikasi Berita",
    use_container_width=True
):

    if not url:

        st.warning(
            "Silakan masukkan URL berita terlebih dahulu."
        )

    else:

        # -------------------------------------------------
        # Mengambil artikel
        # -------------------------------------------------

        with st.spinner(
            "Mengambil isi berita..."
        ):

            artikel = extract_article(url)


        if artikel is None:

            st.error(
                "Gagal mengambil isi berita dari URL tersebut."
            )

            st.info(
                "Pastikan URL merupakan halaman artikel "
                "berita yang dapat diakses."
            )

        else:

            # -------------------------------------------------
            # Menampilkan isi artikel
            # -------------------------------------------------

            st.success(
                "Artikel berhasil diambil."
            )

            with st.expander(
                "📄 Lihat isi berita"
            ):

                st.write(artikel)


            # -------------------------------------------------
            # PREPROCESSING
            # -------------------------------------------------

            tokens = preprocessing(
                artikel
            )


            if len(tokens) == 0:

                st.error(
                    "Tidak ada teks yang dapat diproses."
                )

            else:

                # -------------------------------------------------
                # DOCUMENT VECTOR / VSM
                # -------------------------------------------------

                vector = document_vector(
                    tokens,
                    skipgram_model
                )


                # Ubah menjadi bentuk 2D
                X_input = np.array(
                    [vector]
                )


                # -------------------------------------------------
                # PREDIKSI
                # -------------------------------------------------

                prediction = nb_model.predict(
                    X_input
                )[0]


                # -------------------------------------------------
                # HASIL
                # -------------------------------------------------

                st.subheader(
                    "🎯 Hasil Klasifikasi"
                )

                if str(prediction).lower() == "sport":

                    st.success(
                        f"Kategori berita: **{prediction.upper()}**"
                    )

                elif str(prediction).lower() == "finance":

                    st.info(
                        f"Kategori berita: **{prediction.upper()}**"
                    )

                else:

                    st.success(
                        f"Kategori berita: **{prediction}**"
                    )


                # -------------------------------------------------
                # PROBABILITAS
                # -------------------------------------------------

                if hasattr(
                    nb_model,
                    "predict_proba"
                ):

                    probabilities = (
                        nb_model.predict_proba(
                            X_input
                        )[0]
                    )

                    classes = nb_model.classes_

                    st.subheader(
                        "📊 Probabilitas Klasifikasi"
                    )

                    probability_data = pd.DataFrame({
                        "Kategori": classes,
                        "Probabilitas": [
                            f"{prob * 100:.2f}%"
                            for prob in probabilities
                        ]
                    })

                    st.table(
                        probability_data
                    )