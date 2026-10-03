"""
data_prep.py
Loads and cleans the sentiment dataset, and maps the 191 raw fine-grained
sentiment labels down into a small set of core emotion groups so the
dashboard stays readable.
"""

import pandas as pd
import re

# ---------------------------------------------------------------------
# Emotion grouping: the raw dataset has ~191 unique labels (Joy, Elation,
# Overjoyed, Ecstasy, ... all mean roughly "happy"). We bucket them into
# 9 core groups for cleaner visuals. Anything not explicitly listed
# falls back to "Neutral / Other".
# ---------------------------------------------------------------------
EMOTION_GROUPS = {
    "Joy": [
        "joy", "happiness", "happy", "elation", "ecstasy", "euphoria",
        "overjoyed", "enjoyment", "amusement", "playful", "playfuljoy",
        "cheerful", "delight", "festivejoy", "joyfulreunion", "joy in baking",
        "whimsy", "mischievous", "zest", "vibrancy", "radiance", "dazzle",
        "colorful", "spark", "energy", "thrill", "thrilling journey",
        "adrenaline", "excitement", "enthusiasm",
    ],
    "Love": [
        "love", "affection", "adoration", "romance", "tenderness",
        "compassion", "compassionate", "kindness", "kind", "friendship",
        "connection", "empathetic", "sympathy", "touched", "heartwarming",
        "charm", "elegance", "admiration", "appreciation", "gratitude",
        "grateful", "blessed",
    ],
    "Optimism": [
        "hope", "hopeful", "optimism", "positivity", "positive",
        "motivation", "inspiration", "inspired", "determination",
        "confidence", "confident", "empowerment", "triumph", "success",
        "accomplishment", "pride", "proud", "fulfillment", "satisfaction",
        "relief", "renewed effort", "resilience", "breakthrough",
        "rejuvenation", "reverence",
    ],
    "Surprise": [
        "surprise", "amazement", "awe", "wonder", "wonderment", "marvel",
        "curiosity", "intrigue", "mesmerizing", "hypnotic", "captivation",
        "enchantment", "celestial wonder", "grandeur", "immersion",
        "adventure", "exploration", "journey", "innerjourney",
        "envisioning history", "creativity", "creative inspiration",
        "artisticburst", "imagination", "runway creativity",
        "culinary adventure", "culinaryodyssey", "winter magic",
        "nature's beauty", "ocean's freedom", "freedom", "free-spirited",
    ],
    "Sadness": [
        "sad", "sadness", "grief", "sorrow", "heartbreak", "heartache",
        "loss", "lostlove", "melancholy", "despair", "desolation",
        "loneliness", "isolation", "solitude", "numbness", "devastated",
        "disappointment", "disappointed", "regret", "yearning",
        "nostalgia", "bittersweet", "pensive", "reflection",
        "contemplation", "solace", "shame", "helplessness", "suffering",
    ],
    "Anger": [
        "anger", "frustration", "frustrated", "resentment", "bitterness",
        "bitter", "betrayal", "disgust", "hate", "jealousy", "jealous",
        "envy", "envious", "dismissive", "indifference", "ambivalence",
    ],
    "Fear": [
        "fear", "fearful", "anxiety", "apprehensive", "arousal",
        "intimidation", "suspense", "confusion", "miscalculation",
        "obstacle", "pressure", "challenge", "overwhelmed", "exhaustion",
        "boredom", "darkness", "ruins", "emotionalstorm", "bad",
        "embarrassed",
    ],
    "Calm": [
        "calmness", "tranquility", "serenity", "harmony", "mindfulness",
        "coziness", "peaceful", "acceptance", "whispers of the past",
    ],
}

# Build a fast lookup: lowercase raw label -> group name
_LABEL_TO_GROUP = {}
for group, labels in EMOTION_GROUPS.items():
    for lbl in labels:
        _LABEL_TO_GROUP[lbl.lower()] = group


def map_emotion(raw_label: str) -> str:
    """Map a raw sentiment label to one of the core emotion groups."""
    if not isinstance(raw_label, str):
        return "Neutral"
    key = raw_label.strip().lower()
    if key in ("neutral", "emotion"):
        return "Neutral"
    return _LABEL_TO_GROUP.get(key, "Neutral")


def _clean_hashtags(raw: str):
    """Turn a '#Nature #Park' style string into a list of tags."""
    if not isinstance(raw, str):
        return []
    return re.findall(r"#(\w+)", raw)


def load_and_clean_data(path: str = "data/sentimentdataset.csv") -> pd.DataFrame:
    """Load the raw CSV and return a cleaned, analysis-ready DataFrame."""
    df = pd.read_csv(path)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Drop the stray unnamed index columns if present
    for col in ["Unnamed: 0"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Strip whitespace from every string/object column
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # Parse timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Numeric cleanup
    df["Retweets"] = pd.to_numeric(df["Retweets"], errors="coerce").fillna(0)
    df["Likes"] = pd.to_numeric(df["Likes"], errors="coerce").fillna(0)
    df["Engagement"] = df["Retweets"] + df["Likes"]

    # Core emotion grouping
    df["Emotion"] = df["Sentiment"].apply(map_emotion)

    # Hashtags as a list column + exploded tag for counts
    df["HashtagList"] = df["Hashtags"].apply(_clean_hashtags)

    # Day of week name, useful for time trend charts
    df["DayOfWeek"] = df["Timestamp"].dt.day_name()

    return df


def explode_hashtags(df: pd.DataFrame) -> pd.DataFrame:
    """Return a long-format DataFrame: one row per (post, hashtag)."""
    exploded = df.explode("HashtagList").rename(columns={"HashtagList": "Hashtag"})
    return exploded.dropna(subset=["Hashtag"])


if __name__ == "__main__":
    data = load_and_clean_data()
    print(data.shape)
    print(data["Emotion"].value_counts())
