# NorgesGruppen Training Pipeline — HPC README

## Hurtigstart (4-timers deadline)

```bash
# 1. Pakk opp og naviger til mappen
unzip training_pipeline.zip
cd training_pipeline/

# 2. Kopier/symlink data til forventet struktur (se under)
# 3. Kjør pipeline sekvensielt:
python 01_data_exploration.py       # ~1 min  — datakontroll
python 02_prepare_detection_data.py # ~15 min  — COCO + SKU110K → YOLO-format
python 03_train_detector.py         # ~2.5 t   — YOLOv8x, 80 epoker, 1280×1280
python 04_prepare_classification_data.py # ~15 min — crop GT-bboxes + produkt-bilder
python 05_train_classifier.py       # ~40 min  — EfficientNet-B3, 20 epoker
python 06_build_submission.py       # ~1 min   — bygger submission_onnx.zip
```

Totalt: ~3.5–4 timer på H100 NVL.

---

## Forventet katalogstruktur

Skriptene forventer at data ligger relativt til mappen der du kjører:

```
./
├── data/
│   ├── raw/
│   │   ├── coco_dataset/train/
│   │   │   ├── annotations.json
│   │   │   └── images/
│   │   ├── extra_data/SKU110K_fixed/
│   │   │   ├── annotations/
│   │   │   │   ├── annotations_train.csv
│   │   │   │   └── annotations_val.csv
│   │   │   └── images/
│   │   └── product_images/
│   │       └── <category_id or name>/
│   └── augmented_data/aug_product_images/
│       └── <category_id or name>/
└── submission/
    └── run.py     ← allerede inkludert i denne zip-en
```

---

## Filbeskrivelser

| Fil | Hva den gjør |
|-----|-------------|
| `01_data_exploration.py` | Statistikk over alle datasett — kjør først for å verifisere data |
| `02_prepare_detection_data.py` | Konverterer COCO + SKU110K til YOLOv8-format. SKU110K → klasse 0. |
| `03_train_detector.py` | Trener YOLOv8x (1280×1280, batch=16, 80 epoker) → eksporterer `detector.onnx` |
| `04_prepare_classification_data.py` | Cropper GT-bboxes + kopierer product_images + aug_product_images |
| `05_train_classifier.py` | Trener EfficientNet-B3 (224×224, 20 epoker) → eksporterer `classifier.onnx` |
| `06_build_submission.py` | Packer `submission/` til `submission_onnx.zip` klar for opplasting |
| `submission/run.py` | Inferenskode (kun onnxruntime) — kopieres til `submission/` |

---

## Outputs

Etter vellykket kjøring:

```
submission/
├── run.py
├── detector.onnx          (~140 MB)
├── classifier.onnx        (~46 MB)
├── idx_to_category_id.json
└── classifier_idx_to_category_id.json

submission_onnx.zip        ← Last ned og last opp til konkurransen
```

---

## Tidsbesparende tips

**Hvis du er tidspress:**

- Reduser detektorepoker: `EPOCHS = 50` i `03_train_detector.py`
- Reduser oppløsning: `IMG_SIZE = 640` i `03_train_detector.py` (~ 4× raskere, men lavere mAP)
  - Husk å oppdatere `_IMG_SIZE = 640` i `submission/run.py` tilsvarende
- Reduser SKU110K: `SKU_MAX_TRAIN = 2000` i `02_prepare_detection_data.py`

**Batch-størrelser:**
- YOLOv8x @ 1280 på H100 (94 GB): batch=16 er trygt, batch=32 muligens OOM
- EfficientNet-B3 @ 224 på H100: batch=128/256 er fint

---

## Arkitektur

```
Bilde → YOLOv8x (1280×1280) → Bboxes
                                  ↓
              Crop hvert bbox → EfficientNet-B3 (224×224) → category_id
```

- **Deteksjon (70% av score):** YOLOv8x trent på COCO (356 klasser) + SKU110K (klasse 0)
- **Klassifikasjon (30% av score):** EfficientNet-B3 trent på COCO-crops + produktbilder

---

## Innsendingskrav (oppsummert)

- `run.py` MÅ ligge på rot i zip (ikke i undermappe)
- Maks 10 Python-filer, maks 3 vektfiler, maks 420 MB totalt
- `run.py` bruker KUN `onnxruntime`, `numpy`, `cv2`, `pathlib`, `json`, `argparse`
- Score-threshold: 0.05, COCO bbox-format `[x, y, width, height]`
- ONNX opset 17, GPU: CUDA

---

## Nedlasting av submission

Etter `python 06_build_submission.py`:

```bash
# I terminalen lokalt (JupyterHub → Download)
# eller via scp:
scp user@hpc:/path/to/submission_onnx.zip ./
```
