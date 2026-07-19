"""
preprocessing.py
-----------------
Pipeline pre-processing citra daun stroberi, dipindahkan langsung dari
notebook pelatihan model (Implementasi_Klasifikasi_Penyakit_Daun_Stroberi_CNN).

Berisi setiap tahapan berikut:
resize -> koreksi pencahayaan (gamma, khusus low-light) -> CLAHE (kontras)
-> reduksi noise (median + gaussian) -> dehazing / Dark Channel Prior
(khusus foggy) -> normalisasi piksel 0-1

Fungsi tersedia:
  - baseline_pipeline       : resize + normalize saja (untuk Model A)
  - preprocess_pipeline     : pipeline lengkap (untuk Model B, sama seperti training)
  - detect_condition_auto   : deteksi kondisi otomatis (normal/foggy/lowlight)
  - run_inference_pipeline  : inferensi end-to-end satu model (backward-compat)
  - run_dual_inference      : inferensi end-to-end dua model sekaligus (A & B)
"""

import cv2
import numpy as np

IMG_SIZE = (224, 224)


# ---------------------------------------------------------------------
# 1. Resize
# ---------------------------------------------------------------------
def resize_image(img_bgr, size=IMG_SIZE):
    """Menyeragamkan dimensi citra menjadi 224x224 piksel."""
    return cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------
# 2. Koreksi pencahayaan — Gamma Correction (untuk kondisi low-light)
# ---------------------------------------------------------------------
def gamma_correction(img_bgr, gamma=1.6):
    """
    Koreksi pencahayaan menggunakan transformasi non-linear (gamma correction).
    gamma > 1 -> mencerahkan citra gelap (low-light)
    Rumus: I_out = 255 * (I_in / 255) ** (1/gamma)
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img_bgr, table)


# ---------------------------------------------------------------------
# 3. Peningkatan kontras — CLAHE
# ---------------------------------------------------------------------
def apply_clahe(img_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Meningkatkan kontras lokal citra menggunakan CLAHE pada kanal L (ruang warna LAB)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------
# 4. Reduksi noise — Median Filter dan Gaussian Filter
# ---------------------------------------------------------------------
def reduce_noise(img_bgr, median_ksize=3, gaussian_ksize=(3, 3)):
    """Median filter (noise impuls) dilanjutkan Gaussian filter (noise acak)."""
    denoised = cv2.medianBlur(img_bgr, median_ksize)
    denoised = cv2.GaussianBlur(denoised, gaussian_ksize, 0)
    return denoised


# ---------------------------------------------------------------------
# 5. Dehazing — Dark Channel Prior (untuk kondisi foggy)
# ---------------------------------------------------------------------
def _dark_channel(im, patch_size=15):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(dc, kernel)


def _atmospheric_light(im, dark):
    h, w = im.shape[:2]
    imsz = h * w
    numpx = max(int(imsz / 1000), 1)
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz, 3)
    indices = darkvec.argsort()[imsz - numpx:]
    atmsum = np.zeros((1, 3))
    for ind in indices:
        atmsum += imvec[ind]
    return atmsum / numpx


def _transmission_estimate(im, A, patch_size=15, omega=0.95):
    im3 = np.empty(im.shape, im.dtype)
    for c in range(3):
        im3[:, :, c] = im[:, :, c] / (A[0, c] + 1e-6)
    return 1 - omega * _dark_channel(im3, patch_size)


def _recover(im, t, A, t0=0.1):
    res = np.empty(im.shape, im.dtype)
    t = np.maximum(t, t0)
    for c in range(3):
        res[:, :, c] = (im[:, :, c] - A[0, c]) / t + A[0, c]
    return res


def dehaze_dcp(img_bgr):
    """Menghilangkan efek kabut (foggy) menggunakan metode Dark Channel Prior."""
    I = img_bgr.astype("float64") / 255.0
    dark = _dark_channel(I, 15)
    A = _atmospheric_light(I, dark)
    te = _transmission_estimate(I, A, 15)
    J = _recover(I, te, A, 0.1)
    J = np.clip(J, 0, 1)
    return (J * 255).astype("uint8")


# ---------------------------------------------------------------------
# 6. Normalisasi piksel
# ---------------------------------------------------------------------
def normalize_pixels(img_bgr):
    """Menormalisasi nilai piksel citra ke rentang 0-1."""
    return img_bgr.astype("float32") / 255.0


# ---------------------------------------------------------------------
# Deteksi kondisi otomatis (normal / foggy / lowlight)
# ---------------------------------------------------------------------
def detect_condition_auto(img_bgr, dark_thresh=90, haze_thresh=0.35):
    """Mendeteksi kondisi citra secara otomatis: lowlight, foggy, atau normal."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    I = img_bgr.astype("float64") / 255.0
    dark_ch = _dark_channel(I, 15)
    mean_dark_channel = np.mean(dark_ch)

    if mean_brightness < dark_thresh:
        return "lowlight"
    elif mean_dark_channel > haze_thresh:
        return "foggy"
    else:
        return "normal"


# ---------------------------------------------------------------------
# Pipeline lengkap (dipakai untuk training Model B, disediakan untuk
# konsistensi apabila dibutuhkan ulang)
# ---------------------------------------------------------------------
def preprocess_pipeline(img_bgr, condition, return_steps=False):
    steps = {}
    steps["1_original"] = img_bgr.copy()

    img = resize_image(img_bgr)
    steps["2_resize"] = img.copy()

    if condition == "lowlight":
        img = gamma_correction(img)
        steps["3_koreksi_pencahayaan"] = img.copy()
    else:
        steps["3_tanpa_koreksi_pencahayaan"] = img.copy()

    img = apply_clahe(img)
    steps["4_clahe_kontras"] = img.copy()

    img = reduce_noise(img)
    steps["5_reduksi_noise"] = img.copy()

    if condition == "foggy":
        img = dehaze_dcp(img)
        steps["6_dehazing"] = img.copy()
    else:
        steps["6_tanpa_dehazing"] = img.copy()

    img_norm = normalize_pixels(img)
    steps["7_normalisasi"] = (img_norm * 255).astype("uint8")

    if return_steps:
        return img_norm, steps
    return img_norm


# ---------------------------------------------------------------------
# Baseline pipeline — Model A (resize + normalize saja)
# ---------------------------------------------------------------------
def baseline_pipeline(img_bgr):
    """
    Pipeline minimal untuk Model A (dilatih tanpa augmentasi kondisi):
      resize 224x224 -> normalisasi piksel 0-1

    Return:
        img_norm      : float32 array (224,224,3) siap masuk model
        img_resized   : uint8 BGR array (224,224,3) untuk ditampilkan
    """
    img_resized = resize_image(img_bgr)
    img_norm = normalize_pixels(img_resized)
    return img_norm, img_resized


# ---------------------------------------------------------------------
# Inferensi end-to-end (backward-compatible, satu model saja)
# ---------------------------------------------------------------------
def run_inference_pipeline(img_bgr, model):
    """
    Menjalankan pipeline preprocessing (deteksi kondisi otomatis) lalu
    prediksi memakai model yang sudah dimuat (Model B / model final).

    Return dict berisi:
        condition        : "normal" | "foggy" | "lowlight"
        image_resized     : citra setelah resize (BGR, uint8) — "before"
        image_processed   : citra setelah pipeline lengkap (BGR, uint8) — "after"
        label             : "sehat" | "sakit"
        prob_sakit        : float (0-1)
        confidence        : float (0-1) -- probabilitas kelas yang dipilih
        stages_applied    : list nama tahap yang benar-benar dijalankan
    """
    img_resized = resize_image(img_bgr)
    condition = detect_condition_auto(img_resized)

    stages_applied = ["Resize (224x224)"]
    processed = img_resized.copy()

    if condition == "lowlight":
        processed = gamma_correction(processed)
        stages_applied.append("Koreksi pencahayaan (Gamma)")

    processed = apply_clahe(processed)
    stages_applied.append("Peningkatan kontras (CLAHE)")

    processed = reduce_noise(processed)
    stages_applied.append("Reduksi noise (Median + Gaussian)")

    if condition == "foggy":
        processed = dehaze_dcp(processed)
        stages_applied.append("Dehazing (Dark Channel Prior)")

    stages_applied.append("Normalisasi piksel (0-1)")
    processed_norm = normalize_pixels(processed)

    pred_prob = float(model.predict(np.expand_dims(processed_norm, axis=0), verbose=0)[0][0])
    label = "sakit" if pred_prob >= 0.5 else "sehat"
    confidence = pred_prob if label == "sakit" else 1 - pred_prob

    return {
        "condition": condition,
        "image_resized": img_resized,
        "image_processed": processed,
        "label": label,
        "prob_sakit": pred_prob,
        "confidence": confidence,
        "stages_applied": stages_applied,
    }


# ---------------------------------------------------------------------
# Inferensi dua model sekaligus (A = baseline, B = pipeline lengkap)
# ---------------------------------------------------------------------
def run_dual_inference(img_bgr, model_a, model_b):
    """
    Menjalankan dua jalur preprocessing sekaligus untuk satu citra:
      - Jalur A : baseline_pipeline (resize + normalize)
      - Jalur B : preprocess_pipeline (pipeline lengkap, deteksi kondisi otomatis)

    Parameter:
        img_bgr  : numpy array BGR uint8 (citra asli dari upload)
        model_a  : model Keras tanpa pipeline (Model A)
        model_b  : model Keras dengan pipeline (Model B)

    Return dict berisi:
        condition          : "normal" | "foggy" | "lowlight" (dari jalur B)
        image_original     : citra asli setelah resize (BGR uint8) — untuk jalur A
        image_baseline     : sama dengan image_original (alias, untuk kejelasan UI)
        image_processed    : citra setelah pipeline lengkap B (BGR uint8)
        stages_applied     : list tahap yang dijalankan jalur B
        model_a            : dict {"label", "prob_sakit", "confidence"}
        model_b            : dict {"label", "prob_sakit", "confidence"}
    """
    # --- Jalur A: baseline ---
    a_norm, img_resized = baseline_pipeline(img_bgr)
    pred_a = float(model_a.predict(np.expand_dims(a_norm, axis=0), verbose=0)[0][0])
    label_a = "sakit" if pred_a >= 0.5 else "sehat"
    conf_a = pred_a if label_a == "sakit" else 1 - pred_a

    # --- Jalur B: pipeline lengkap ---
    condition = detect_condition_auto(img_resized)

    stages_applied = ["Resize (224x224)"]
    processed = img_resized.copy()

    if condition == "lowlight":
        processed = gamma_correction(processed)
        stages_applied.append("Koreksi pencahayaan (Gamma)")

    processed = apply_clahe(processed)
    stages_applied.append("Peningkatan kontras (CLAHE)")

    processed = reduce_noise(processed)
    stages_applied.append("Reduksi noise (Median + Gaussian)")

    if condition == "foggy":
        processed = dehaze_dcp(processed)
        stages_applied.append("Dehazing (Dark Channel Prior)")

    stages_applied.append("Normalisasi piksel (0-1)")
    b_norm = normalize_pixels(processed)

    pred_b = float(model_b.predict(np.expand_dims(b_norm, axis=0), verbose=0)[0][0])
    label_b = "sakit" if pred_b >= 0.5 else "sehat"
    conf_b = pred_b if label_b == "sakit" else 1 - pred_b

    return {
        "condition": condition,
        "image_original": img_resized,
        "image_baseline": img_resized,      # alias — jalur A tidak mengubah gambar
        "image_processed": processed,
        "stages_applied": stages_applied,
        "model_a": {
            "label": label_a,
            "prob_sakit": pred_a,
            "confidence": conf_a,
        },
        "model_b": {
            "label": label_b,
            "prob_sakit": pred_b,
            "confidence": conf_b,
        },
    }
