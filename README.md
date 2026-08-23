# Medical Diagnosis — Pneumonia Detection from Chest X-Rays

A CNN-based image classifier that predicts **Normal vs Disease (Pneumonia)** from
chest X-ray images, using transfer learning (MobileNetV2), with a deployed
Streamlit demo and Grad-CAM explainability.

## 1. Get the real dataset

The sample `dataset/` folder in this repo is a tiny placeholder (a handful of
images) — replace it before training:

1. Download **Chest X-Ray Images (Pneumonia)** from Kaggle:
   `https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia`
2. Extract it so you have:
   ```
   dataset/
     train/Normal, train/Disease
     val/Normal,   val/Disease
     test/Normal,  test/Disease
   ```
   (Rename the Kaggle `PNEUMONIA` folder to `Disease` to match this repo's class names.)

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Train

```bash
python train.py --data_dir dataset --epochs 15
```

This will:
- Use a frozen **MobileNetV2** backbone (transfer learning) + a custom classification head
- Apply **class weighting** to correct for pneumonia/normal class imbalance
- Use **EarlyStopping** and **ReduceLROnPlateau** to avoid overfitting
- Save the trained model to `model/medical_model.h5`
- Save a classification report (precision/recall/F1), confusion matrix, and ROC-AUC to `outputs/`
- Save training accuracy/loss curves to `outputs/training_curves.png`

## 4. Run inference from the command line

```bash
python predict.py --image path/to/xray.jpg
```

## 5. Run the demo app locally

```bash
streamlit run streamlit_app.py
```

Opens a browser UI where you can upload an X-ray, see the prediction + confidence,
and a **Grad-CAM heatmap** showing which regions of the image the model focused on.

## 6. Deploy for free (pick one)

**Option A — Streamlit Community Cloud (easiest)**
1. Push this repo to GitHub (include the trained `model/medical_model.h5` — use Git LFS if it's large).
2. Go to https://share.streamlit.io, connect your GitHub repo, and point it at `streamlit_app.py`.
3. You get a public URL you can put directly in your resume/portfolio.

**Option B — Hugging Face Spaces**
1. Create a new Space, choose the "Streamlit" SDK.
2. Upload these files (or push via git) including the trained model.
3. Space auto-builds and gives you a public URL.

**Option C — Docker + Render/Railway**
1. `docker build -t medical-diagnosis .`
2. `docker run -p 8501:8501 medical-diagnosis`
3. Push the image / repo to Render or Railway for a hosted public URL.

## Model Architecture

```
MobileNetV2 (ImageNet pretrained, frozen) 
  -> GlobalAveragePooling2D
  -> Dense(128, relu)
  -> Dropout(0.5)
  -> Dense(1, sigmoid)
```

## Notes / Limitations

- This is an educational/portfolio project — **not** a validated medical device.
- Performance depends heavily on dataset size and quality; report the real
  precision/recall/F1/ROC-AUC from `outputs/classification_report.txt` after training,
  not just accuracy (pneumonia datasets are typically class-imbalanced).
