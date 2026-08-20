import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import base64
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor

# ----------------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Klasifikasi Kelapa Lokal",
    page_icon="🥥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CLASS_NAMES = ["Mature", "Potential", "Premature"]

CLASS_INFO = {
    "Mature": {
        "label_id": "Matang",
        "desc": "Buah kelapa sudah matang penuh dan siap panen. Tempurung keras, sabut mulai mengering, dan kandungan air relatif berkurang dibanding fase sebelumnya.",
        "color": "#6B4226",
        "emoji": "🟤",
    },
    "Potential": {
        "label_id": "Berpotensi / Menuju Matang",
        "desc": "Buah kelapa berada pada fase transisi menuju kematangan. Masih dapat dibiarkan di pohon beberapa waktu sebelum panen optimal.",
        "color": "#C98A2C",
        "emoji": "🟠",
    },
    "Premature": {
        "label_id": "Belum Matang / Muda",
        "desc": "Buah kelapa masih muda. Sabut lunak dan kadar air tinggi, belum ideal untuk dipanen sebagai kelapa tua.",
        "color": "#2F7D3A",
        "emoji": "🟢",
    },
}

# Ambang batas kepercayaan untuk deteksi gambar di luar domain (bukan kelapa)
# CATATAN KETERBATASAN: model ini HANYA dilatih untuk 3 kelas kematangan kelapa,
# tidak pernah dilatih dengan contoh "bukan kelapa". Threshold di bawah ini adalah
# heuristik berbasis confidence/entropy/margin softmax -- BUKAN detektor objek
# sesungguhnya, sehingga tidak akan selalu akurat (bisa false positive/negative).
OOD_CONFIDENCE_THRESHOLD = 0.70   # sebelumnya 0.55 -- dinaikkan agar lebih ketat
OOD_ENTROPY_THRESHOLD = 0.70      # sebelumnya 0.85 -- diturunkan agar lebih sensitif
OOD_MARGIN_THRESHOLD = 0.20       # selisih minimum antara kelas ke-1 dan ke-2

# ----------------------------------------------------------------------------
# CLIP Pembanding -- HANYA sebagai info tambahan / cross-check, TIDAK memblokir
# atau menimpa hasil dari model ripeness Anda. Hasil utama tetap murni dari
# model_ripeness. CLIP di sini membandingkan per-KELAS (Mature/Potential/
# Premature), bukan cuma "kelapa vs bukan kelapa" generik -- supaya
# pembandingannya selaras dengan apa yang diprediksi model Anda.
# ----------------------------------------------------------------------------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Satu/lebih prompt teks untuk tiap kelas yang PERSIS sama dengan kelas model ripeness
CLIP_CLASS_PROMPTS = {
    "Mature": [
        "a photo of a mature, fully ripe coconut fruit",
        "a photo of a hard-shelled ripe coconut",
    ],
    "Potential": [
        "a photo of a coconut fruit that is partially ripe, transitioning to maturity",
        "a photo of a coconut in the middle stage of ripening",
    ],
    "Premature": [
        "a photo of a premature, unripe young coconut fruit",
        "a photo of a young green coconut with soft husk",
    ],
}
# Prompt tambahan HANYA untuk info "kemiripan objek dengan kelapa secara umum"
CLIP_NOT_COCONUT_PROMPTS = [
    "a photo of an oil palm fruit bunch",
    "a photo of a random object, not a coconut",
    "a photo of a person",
    "a photo of an animal",
    "a photo of food that is not a coconut",
    "a blank or blurry photo",
]

CLIP_ALL_PROMPTS = (
    CLIP_CLASS_PROMPTS["Mature"]
    + CLIP_CLASS_PROMPTS["Potential"]
    + CLIP_CLASS_PROMPTS["Premature"]
    + CLIP_NOT_COCONUT_PROMPTS
)
_N_MATURE = len(CLIP_CLASS_PROMPTS["Mature"])
_N_POTENTIAL = len(CLIP_CLASS_PROMPTS["Potential"])
_N_PREMATURE = len(CLIP_CLASS_PROMPTS["Premature"])


@st.cache_resource(show_spinner=False)
def load_clip():
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model.eval()
    return clip_model, clip_processor


def clip_compare(image: Image.Image, clip_model, clip_processor):
    """CLIP HANYA untuk pembanding informasional -- tidak memblokir apapun.

    Mengembalikan dict:
      - per_class: {"Mature": skor, "Potential": skor, "Premature": skor} (skor CLIP per kelas)
      - not_coconut_score: skor gabungan prompt "bukan kelapa"
      - clip_top_class: kelas dengan skor CLIP tertinggi (di antara 3 kelas kelapa saja)
      - agrees_with_model: diisi belakangan (dibandingkan dengan top_class model)
      - detail: skor mentah tiap prompt (untuk expander debug)
    """
    inputs = clip_processor(
        text=CLIP_ALL_PROMPTS,
        images=image.convert("RGB"),
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).squeeze(0).numpy()

    i0 = 0
    mature_score = float(probs[i0 : i0 + _N_MATURE].sum()); i0 += _N_MATURE
    potential_score = float(probs[i0 : i0 + _N_POTENTIAL].sum()); i0 += _N_POTENTIAL
    premature_score = float(probs[i0 : i0 + _N_PREMATURE].sum()); i0 += _N_PREMATURE
    not_coconut_score = float(probs[i0:].sum())

    per_class = {"Mature": mature_score, "Potential": potential_score, "Premature": premature_score}
    clip_top_class = max(per_class, key=per_class.get)
    detail = {prompt: float(p) for prompt, p in zip(CLIP_ALL_PROMPTS, probs)}

    return {
        "per_class": per_class,
        "not_coconut_score": not_coconut_score,
        "clip_top_class": clip_top_class,
        "detail": detail,
    }

# ----------------------------------------------------------------------------
# Gaya visual kustom (tema kelapa: hijau tropis, krem, coklat sabut)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Fraunces:wght@600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #f4f1e6 0%, #e9e3cf 45%, #dfd8bd 100%);
    }

    .hero {
        background: linear-gradient(135deg, #1f5c2e 0%, #2f7d3a 55%, #6fae4a 100%);
        border-radius: 22px;
        padding: 34px 30px;
        color: #fbf7ea;
        margin-bottom: 22px;
        box-shadow: 0 12px 30px rgba(31,92,46,0.25);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "";
        position: absolute;
        right: -40px;
        top: -40px;
        width: 160px;
        height: 160px;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
    }
    .hero h1 {
        font-family: 'Fraunces', serif;
        font-size: 1.9rem;
        margin-bottom: 6px;
        line-height: 1.25;
    }
    .hero p {
        font-size: 0.92rem;
        opacity: 0.92;
        margin-bottom: 0;
    }
    .badge-row {
        margin-top: 14px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
    }

    .card {
        background: #fffdf7;
        border-radius: 18px;
        padding: 24px;
        border: 1px solid #e7ddc4;
        box-shadow: 0 6px 18px rgba(90, 70, 30, 0.08);
        margin-bottom: 20px;
    }

    .result-title {
        font-family: 'Fraunces', serif;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 3px;
        font-weight: 500;
    }

    .ood-warning {
        background: #fdecea;
        border: 1px solid #f4b5ac;
        color: #8a2c1f;
        border-radius: 14px;
        padding: 16px 18px;
        margin-top: 14px;
        font-size: 0.9rem;
    }
    .ood-warning b { color: #6e1e14; }

    .footer-note {
        text-align: center;
        font-size: 0.78rem;
        color: #8a7f5f;
        margin-top: 30px;
        padding-bottom: 10px;
    }

    .identity-card {
        display: flex;
        align-items: center;
        gap: 16px;
        background: #fffdf7;
        border: 1px solid #e7ddc4;
        border-radius: 16px;
        padding: 14px 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(90, 70, 30, 0.06);
    }
    .uny-logo {
        width: 56px;
        height: 56px;
        object-fit: contain;
        flex-shrink: 0;
    }
    .identity-text {
        line-height: 1.35;
    }
    .identity-uni {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-size: 1.02rem;
        color: #1f5c2e;
    }
    .identity-prodi {
        font-size: 0.8rem;
        color: #6b5f3e;
        margin-bottom: 4px;
    }
    .identity-name {
        font-size: 0.88rem;
        font-weight: 600;
        color: #3a3423;
        letter-spacing: 0.02em;
    }
    .identity-nim {
        font-size: 0.78rem;
        color: #8a7f5f;
    }
    @media (max-width: 480px) {
        .identity-card { flex-direction: column; text-align: center; }
    }

    div[data-testid="stFileUploader"] {
        background: #fffdf7;
        border-radius: 16px;
        padding: 14px;
        border: 1.5px dashed #b7a878;
    }

    .stButton>button {
        background: linear-gradient(135deg, #2f7d3a, #1f5c2e);
        color: white;
        border-radius: 999px;
        border: none;
        padding: 10px 26px;
        font-weight: 600;
        box-shadow: 0 6px 14px rgba(47,125,58,0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #35913f, #256b31);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Identitas Peneliti / Kampus
# ----------------------------------------------------------------------------
def _logo_base64():
    logo_path = Path(__file__).parent / "logo_uny.png"
    if logo_path.exists():
        return base64.b64encode(logo_path.read_bytes()).decode()
    return None

_logo_b64 = _logo_base64()
_logo_html = (
    f'<img src="data:image/png;base64,{_logo_b64}" alt="Logo UNY" class="uny-logo"/>'
    if _logo_b64 else ""
)

st.markdown(
    f"""
    <div class="identity-card">
        {_logo_html}
        <div class="identity-text">
            <div class="identity-uni">Universitas Negeri Yogyakarta</div>
            <div class="identity-prodi">Program Studi Ilmu Teknik &middot; Program Pascasarjana</div>
            <div class="identity-name">USMAN</div>
            <div class="identity-nim">NIM: 24052050028</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Header / Hero
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🥥 Klasifikasi Kematangan Kelapa Lokal</h1>
        <p>Model Hybrid GA‑PSO &amp; Simulated Annealing dengan Attention Mechanism untuk Optimasi CNN
        pada Klasifikasi Buah Kelapa Lokal — unggah foto kelapa untuk memprediksi tingkat kematangannya.</p>
        <div class="badge-row">
            <span class="badge">🧬 GA‑PSO Search</span>
            <span class="badge">🌡️ Simulated Annealing</span>
            <span class="badge">🎯 Attention-based CNN</span>
            <span class="badge">📊 3 Kelas Kematangan</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Definisi arsitektur model (ResNet-18 backbone + classifier head kustom)
# Struktur ini direkonstruksi agar cocok dengan state_dict pada file .pth
# ----------------------------------------------------------------------------
class CoconutClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        backbone = models.resnet18(weights=None)
        # Buang fc bawaan resnet18, pakai backbone sebagai feature extractor
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.fc = nn.Sequential(
            nn.Dropout(0.4),                 # fc.0
            nn.Linear(512, 512),             # fc.1
            nn.ReLU(inplace=True),           # fc.2
            nn.BatchNorm1d(512),             # fc.3
            nn.Dropout(0.4),                 # fc.4
            nn.Linear(512, num_classes),     # fc.5
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


@st.cache_resource(show_spinner=False)
def load_model():
    model = CoconutClassifier(num_classes=len(CLASS_NAMES))
    state_dict = torch.load("model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict(model, image: Image.Image):
    tensor = preprocess(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
    return probs


def compute_entropy_normalized(probs: np.ndarray) -> float:
    eps = 1e-9
    entropy = -np.sum(probs * np.log(probs + eps))
    max_entropy = np.log(len(probs))
    return float(entropy / max_entropy)


# ----------------------------------------------------------------------------
# Sumber gambar: Upload File atau Scan/Foto Langsung
# ----------------------------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("#### 📸 Pilih Sumber Gambar")

mode = st.radio(
    "Sumber gambar",
    ["📤 Upload File", "📷 Scan / Foto Langsung"],
    horizontal=True,
    label_visibility="collapsed",
)

uploaded_file = None
if mode == "📤 Upload File":
    uploaded_file = st.file_uploader(
        "Format yang didukung: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
else:
    st.caption("Arahkan kamera ke buah kelapa, lalu klik tombol ambil foto di bawah.")
    uploaded_file = st.camera_input("Ambil foto kelapa", label_visibility="collapsed")

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        caption_src = "Foto hasil scan kamera" if mode == "📷 Scan / Foto Langsung" else "Gambar yang diunggah"
        st.image(image, caption=caption_src, use_container_width=True)

    with col2:
        # ------------------------------------------------------------------
        # 0) Hitung dulu kedua-duanya (model + CLIP) sebelum menampilkan apapun,
        #    supaya urutan tampilan bisa diatur: CLIP tampil sebagai banner
        #    PALING ATAS kalau mencurigakan, baru hasil model di bawahnya.
        # ------------------------------------------------------------------
        with st.spinner("Menganalisis tingkat kematangan..."):
            model = load_model()
            probs = predict(model, image)

        top_idx = int(np.argmax(probs))
        top_class = CLASS_NAMES[top_idx]
        top_conf = float(probs[top_idx])
        entropy_norm = compute_entropy_normalized(probs)
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1])

        ripeness_uncertain = (
            top_conf < OOD_CONFIDENCE_THRESHOLD
            or entropy_norm > OOD_ENTROPY_THRESHOLD
            or margin < OOD_MARGIN_THRESHOLD
        )

        with st.spinner("Memuat pembanding (CLIP)..."):
            clip_model, clip_processor = load_clip()
            clip_result = clip_compare(image, clip_model, clip_processor)

        clip_top_class = clip_result["clip_top_class"]
        clip_top_score = clip_result["per_class"][clip_top_class]
        not_coconut_score = clip_result["not_coconut_score"]
        clip_strongly_not_coconut = not_coconut_score > 0.5
        agrees = clip_top_class == top_class

        # ------------------------------------------------------------------
        # 1) BANNER PALING ATAS -- kalau CLIP sangat yakin ini bukan kelapa,
        #    munculkan peringatan mencolok DULU, sebelum hasil model.
        #    (Hasil model tetap tampil utuh di bawah, tidak dihapus/ditimpa.)
        # ------------------------------------------------------------------
        if clip_strongly_not_coconut:
            st.markdown(
                f"""
                <div class="ood-warning">
                    🚫 <b>Peringatan: gambar ini kemungkinan besar BUKAN kelapa.</b><br>
                    Sistem pembanding (CLIP) memberi skor <b>{not_coconut_score*100:.1f}%</b>
                    bahwa gambar ini bukan buah kelapa sama sekali. Hasil klasifikasi kematangan
                    di bawah ini tetap ditampilkan (murni dari model Anda), tapi
                    <b>kemungkinan besar tidak relevan</b> untuk gambar ini.
                    Silakan unggah foto buah kelapa yang jelas untuk hasil yang bermakna.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ------------------------------------------------------------------
        # 2) HASIL UTAMA -- murni dari model ripeness Anda, SELALU tampil
        #    apa adanya, tidak diblokir/ditimpa oleh apapun.
        # ------------------------------------------------------------------
        info = CLASS_INFO[top_class]
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("##### 🥥 Hasil Model Anda")
        st.markdown(
            f"<div class='result-title' style='color:{info['color']}'>{info['emoji']} {top_class} ({info['label_id']})</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Tingkat keyakinan:** {top_conf*100:.1f}%")
        st.progress(min(max(top_conf, 0.0), 1.0))
        st.markdown(info["desc"])
        st.markdown("</div>", unsafe_allow_html=True)

        if ripeness_uncertain and not clip_strongly_not_coconut:
            st.markdown(
                """
                <div class="ood-warning">
                    ⚠️ <b>Model kurang yakin pada gambar ini.</b><br>
                    Tingkat keyakinan antar kelas kematangan berdekatan. Coba foto dengan
                    pencahayaan lebih baik atau sudut yang lebih jelas untuk hasil yang lebih akurat.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ------------------------------------------------------------------
        # 3) PEMBANDING -- CLIP, detail cross-check per-kelas. Kalau CLIP
        #    sudah sangat yakin bukan kelapa (banner di atas sudah muncul),
        #    perbandingan per-kelas Mature/Potential/Premature jadi kurang
        #    relevan (karena skornya kecil semua), jadi tidak ditampilkan lagi
        #    di sini supaya tidak membingungkan -- cukup expander detail saja.
        # ------------------------------------------------------------------
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("##### 🔍 Pembanding (CLIP) — bukan hasil resmi")
        st.caption(
            "Panel ini hanya cross-check independen menggunakan CLIP. "
            "Hasil resmi tetap yang di atas, dari model Anda."
        )
        if clip_strongly_not_coconut:
            st.markdown(
                f"🚫 CLIP menilai skor **{not_coconut_score*100:.1f}%** kemungkinan gambar ini "
                f"bukan kelapa sama sekali (lihat banner peringatan di atas)."
            )
        elif agrees:
            st.markdown(
                f"✅ **CLIP sependapat dengan model:** juga cenderung melihat gambar ini "
                f"sebagai **{clip_top_class}** (skor CLIP: {clip_top_score*100:.1f}%)."
            )
        else:
            st.markdown(
                f"ℹ️ **CLIP punya pandangan berbeda:** model Anda bilang **{top_class}**, "
                f"tapi CLIP paling condong ke **{clip_top_class}** (skor CLIP: {clip_top_score*100:.1f}%). "
                f"Ini normal — CLIP tidak dilatih khusus untuk tugas ini, jadi bisa saja kurang tepat."
            )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🔬 Detail skor CLIP (semua prompt yang diuji)"):
            for prompt, score in sorted(clip_result["detail"].items(), key=lambda x: -x[1]):
                is_coconut_prompt = prompt not in CLIP_NOT_COCONUT_PROMPTS
                tag = "🥥" if is_coconut_prompt else "❌"
                st.markdown(f"{tag} `{prompt}` — {score*100:.1f}%")


    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 📊 Distribusi Probabilitas Semua Kelas")
    order = np.argsort(-probs)
    for idx in order:
        cname = CLASS_NAMES[idx]
        cinfo = CLASS_INFO[cname]
        p = float(probs[idx])
        st.markdown(
            f"<div class='prob-bar-label'><span>{cinfo['emoji']} {cname} ({cinfo['label_id']})</span><span>{p*100:.1f}%</span></div>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(p, 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👆 Unggah foto atau scan langsung buah kelapa untuk memulai klasifikasi kematangan.")

# ----------------------------------------------------------------------------
# Tentang model
# ----------------------------------------------------------------------------
with st.expander("ℹ️ Tentang Model Ini"):
    st.markdown(
        """
**Judul Penelitian:**
*Model Hybrid Berbasis GA-PSO dan Simulated Annealing (SA) dengan Attention Mechanism
untuk Optimasi CNN dalam Klasifikasi Buah Kelapa Lokal*

**Arsitektur:** Backbone ResNet-18 dengan classifier head kustom
(Dropout → Linear(512,512) → ReLU → BatchNorm → Dropout → Linear(512,3)).

**Proses optimasi:** Kombinasi Genetic Algorithm (GA), Particle Swarm Optimization (PSO),
dan Simulated Annealing (SA) digunakan selama tahap pelatihan untuk mencari konfigurasi
hyperparameter/arsitektur terbaik secara multi-objektif (akurasi vs. efisiensi/beban komputasi),
dengan mekanisme attention untuk menekankan fitur visual penting pada citra kelapa.
Checkpoint (`.pth`) yang digunakan pada aplikasi ini adalah bobot model terbaik hasil proses tersebut.

**Kelas keluaran:**
- 🟤 **Mature** — Matang
- 🟠 **Potential** — Berpotensi / menuju matang
- 🟢 **Premature** — Belum matang / muda

**Catatan:** Aplikasi ini menyertakan deteksi sederhana berbasis tingkat keyakinan (confidence)
dan entropi prediksi untuk memberi peringatan apabila gambar yang diunggah kemungkinan
besar bukan gambar kelapa, sehingga hasil prediksi tidak dapat diandalkan.
        """
    )

st.markdown(
    """
    <div class='footer-note'>
        Disertasi: Model Hybrid Berbasis GA-PSO dan Simulated Annealing (SA) dengan Attention Mechanism
        untuk Optimasi CNN dalam Klasifikasi Buah Kelapa Lokal<br>
        USMAN &middot; NIM 24052050028 &middot; Program Studi Ilmu Teknik, Program Pascasarjana &middot;
        Universitas Negeri Yogyakarta
    </div>
    """,
    unsafe_allow_html=True,
)
