# AI Animal Detection — Streamlit

This project creates a website where a user uploads an image and the saved TensorFlow model predicts one of these classes:
- CATS
- DOGS
- ELEPHENTS
- LIONS
- TIGURES

## Important: add your trained model files

Copy these two files from Kaggle's `/kaggle/working/` folder into the same folder as `app.py`:
- `animal_detection_model.keras`
- `animal_classes.json`

The model files are not included in this ZIP because they must be downloaded from your Kaggle notebook.

## Test on your computer

1. Install Python 3.10 or 3.11.
2. Open a terminal in this folder.
3. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Run:

   ```bash
   streamlit run app.py
   ```

## Publish a shareable website link with Streamlit Community Cloud

1. Download `animal_detection_model.keras` and `animal_classes.json` from the Kaggle notebook's Output files.
2. Create a GitHub repository and upload `app.py`, `requirements.txt`, `animal_detection_model.keras`, and `animal_classes.json`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository, branch, and `app.py`.
5. Click **Deploy**. Streamlit will create a public website URL that you can send to your teacher.

If GitHub rejects the model file due to size, use Git LFS or another supported model-hosting method and update `MODEL_PATH` loading accordingly.

## Dataset warning

The current model was trained with only 10 images (2 per class). It is a working demonstration, not a reliable general-purpose classifier. Predictions on new images may be wrong. Collect substantially more varied images per class and evaluate on a separate test set before presenting accuracy claims.
