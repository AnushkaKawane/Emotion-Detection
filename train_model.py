"""
train_model.py
Trains a TF-IDF + Logistic Regression classifier on Text -> Emotion group,
and saves it so app.py can load it for live predictions.

Run:  python train_model.py
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from data_prep import load_and_clean_data


def main():
    df = load_and_clean_data("data/sentimentdataset.csv")

    X = df["Text"]
    y = df["Emotion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(pipeline, "emotion_classifier.joblib")
    print("Saved model to emotion_classifier.joblib")


if __name__ == "__main__":
    main()
