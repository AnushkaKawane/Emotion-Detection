# 🌸 Social Sentiment Analytics Dashboard

An interactive, pastel-themed dashboard exploring emotion, engagement, and
trends across platforms, countries, and time — built on a social media
sentiment dataset.

## Project structure
```
sentiment-dashboard/
├── data/
│   └── sentimentdataset.csv
├── data_prep.py          # cleaning + emotion-label grouping
├── app.py                # Streamlit dashboard
├── train_model.py        # optional TF-IDF + Logistic Regression classifier
├── requirements.txt
└── README.md
```

## Setup in VS Code

1. Open this folder in VS Code.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
3. Select the venv interpreter: `Ctrl+Shift+P` → **Python: Select Interpreter**.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the dashboard:
   ```bash
   streamlit run app.py
   ```
   It opens automatically in your browser at `http://localhost:8501` and
   hot-reloads whenever you save changes to `app.py`.

## (Optional) Add live ML predictions

1. Train the classifier:
   ```bash
   python train_model.py
   ```
   This saves `emotion_classifier.joblib`.
2. In `app.py`, load it inside the "Try It" tab with `joblib.load(...)`
   and replace the keyword-based demo with `pipeline.predict([user_text])`.

## Notes on the data

The raw `Sentiment` column has 191 unique fine-grained labels (Joy,
Elation, Ecstasy, Overjoyed, etc.). `data_prep.py` maps them into 9 core
emotion groups (Joy, Love, Optimism, Surprise, Sadness, Anger, Fear,
Calm, Neutral) so charts stay readable — edit `EMOTION_GROUPS` in that
file if you want a different taxonomy.
