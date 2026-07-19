# Spesimen Daun Stroberi — Web App Deployment

Web app Streamlit untuk klasifikasi penyakit daun stroberi pada kondisi
foggy dan low-light menggunakan pipeline pre-processing citra dan CNN.

## Struktur folder

```
strawberry-app/
├── app.py              # Aplikasi Streamlit (UI)
├── preprocessing.py     # Pipeline pre-processing citra (dipindah dari notebook)
├── requirements.txt
├── README.md
└── model/
    ├── model_final_mobilenetv2_dengan_pipeline.keras   # ⬅️ taruh di sini
    └── metadata_model.json                              # opsional
```

## Langkah menjalankan

1. **Ambil model hasil training dari notebook.**
   Di akhir notebook (`Bagian 7 — Deployment`), model final sudah otomatis
   tersimpan di Google Drive:
   ```
   Strawberry/hasil/model/model_final_mobilenetv2_dengan_pipeline.keras
   Strawberry/hasil/model/metadata_model.json
   ```
   Download kedua file itu dari Google Drive, lalu taruh di folder
   `model/` pada project ini (nama file harus persis sama).

2. **Install dependency** (disarankan pakai virtual environment):
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py
   ```
   Browser akan otomatis terbuka di `http://localhost:8501`.

## Apa yang ditampilkan aplikasi

1. **Upload** — unggah foto daun stroberi (jpg/jpeg/png).
2. **Kondisi & tahapan** — sistem mendeteksi otomatis kondisi citra
   (normal / berkabut / minim cahaya) dan menampilkan tahapan
   pre-processing yang benar-benar dijalankan (gamma correction hanya
   untuk low-light, dehazing hanya untuk foggy).
3. **Before/After** — slider interaktif membandingkan citra sebelum dan
   sesudah pipeline pre-processing.
4. **Diagnosis** — label SEHAT/SAKIT beserta tingkat keyakinan model.
5. **Metadata model** — jika `metadata_model.json` tersedia, akurasi model
   dari hasil training ditampilkan di footer.

## Catatan

- Jika file model belum ditaruh di `model/`, aplikasi akan menampilkan
  pesan instruksi alih-alih error.
- Model dimuat sekali dan di-cache (`@st.cache_resource`) supaya tidak
  reload setiap kali upload gambar baru.
- Desain (font Fraunces/Work Sans/JetBrains Mono, palet warna) sepenuhnya
  bisa disesuaikan lewat blok CSS di bagian atas `app.py`.
