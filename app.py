"""
app.py
Interactive social sentiment analytics dashboard — pastel themed.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

from data_prep import load_and_clean_data, explode_hashtags

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Social Sentiment Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Pastel palette + custom CSS
# ----------------------------------------------------------------------
PASTEL = {
    "Joy": "#FFE29A",
    "Love": "#FFB6C1",
    "Optimism": "#B8E6D5",
    "Surprise": "#C9B8FF",
    "Sadness": "#A8D0E6",
    "Anger": "#FFB3A7",
    "Fear": "#D9C2FF",
    "Calm": "#BEE7E8",
    "Neutral": "#E5E1DA",
}
PASTEL_SEQUENCE = list(PASTEL.values())

CUSTOM_CSS = f"""
<style>
.stApp {{
    background: linear-gradient(160deg, #FDF6F0 0%, #F3F0FA 50%, #EAF6F6 100%);
}}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #FBEFF2 0%, #EFE9FB 100%);
}}
h1, h2, h3 {{
    color: #6B5B95 !important;
    font-family: 'Trebuchet MS', sans-serif;
}}
div[data-testid="stMetric"] {{
    background: #FFFFFFAA;
    border-radius: 16px;
    padding: 14px 10px;
    box-shadow: 0 2px 10px rgba(150, 130, 200, 0.15);
    border: 1px solid #EDE6F7;
}}
div[data-testid="stMetricValue"] {{
    color: #7B6BA8;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px 10px 0 0;
    background-color: #F5EFFA;
    padding: 8px 16px;
}}
.stTabs [aria-selected="true"] {{
    background-color: #E3D8F7 !important;
    color: #6B5B95 !important;
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Load data (cached so it's fast on filter changes)
# ----------------------------------------------------------------------
@st.cache_data
def get_data():
    return load_and_clean_data("data/sentimentdataset.csv")


df = get_data()

# ----------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------
st.sidebar.markdown("## 🎛️ Filters")

platforms = st.sidebar.multiselect(
    "Platform", options=sorted(df["Platform"].unique()),
    default=sorted(df["Platform"].unique())
)

countries = st.sidebar.multiselect(
    "Country", options=sorted(df["Country"].unique()),
    default=[]
)

emotions = st.sidebar.multiselect(
    "Emotion group", options=sorted(df["Emotion"].unique()),
    default=sorted(df["Emotion"].unique())
)

year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
year_range = st.sidebar.slider(
    "Year range", min_value=year_min, max_value=year_max,
    value=(year_min, year_max)
)

st.sidebar.markdown("---")

# Apply filters
filtered = df[
    df["Platform"].isin(platforms)
    & df["Emotion"].isin(emotions)
    & df["Year"].between(year_range[0], year_range[1])
]
if countries:
    filtered = filtered[filtered["Country"].isin(countries)]

# ----------------------------------------------------------------------
# Header + KPI row
# ----------------------------------------------------------------------
st.title("🌸 Social Sentiment Analytics Dashboard")
st.caption("Explore emotion, engagement, and trends across platforms, countries, and time.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Posts", f"{len(filtered):,}")
k2.metric("Avg. Likes", f"{filtered['Likes'].mean():.1f}" if len(filtered) else "–")
k3.metric("Avg. Retweets", f"{filtered['Retweets'].mean():.1f}" if len(filtered) else "–")
top_emotion = filtered["Emotion"].mode()[0] if len(filtered) else "–"
k4.metric("Top Emotion", top_emotion)


if filtered.empty:
    st.warning("No posts match the current filters — try widening your selection.")
    st.stop()

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["💗 Emotions", "📈 Engagement", "🌍 Geography & Time", "🏷️ Hashtags", "🔮 Try It"]
)

# --- Tab 1: Emotions ---------------------------------------------------
with tab1:
    c1, c2 = st.columns([1.2, 1])

    with c1:
        counts = filtered["Emotion"].value_counts().reset_index()
        counts.columns = ["Emotion", "Count"]
        fig = px.bar(
            counts, x="Count", y="Emotion", orientation="h",
            color="Emotion", color_discrete_map=PASTEL,
            title="Emotion distribution",
        )
        fig.update_layout(
            showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.pie(
            counts, names="Emotion", values="Count", hole=0.5,
            color="Emotion", color_discrete_map=PASTEL,
            title="Emotion share",
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Emotion mix by platform")
    cross = pd.crosstab(filtered["Platform"], filtered["Emotion"])
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
    fig3 = px.bar(
        cross_pct, barmode="stack",
        color_discrete_map=PASTEL,
        labels={"value": "% of posts", "Platform": "Platform"},
    )
    fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

# --- Tab 2: Engagement ---------------------------------------------------
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        fig4 = px.box(
            filtered, x="Emotion", y="Engagement", color="Emotion",
            color_discrete_map=PASTEL, title="Engagement (Likes+Retweets) by emotion",
            points=False,
        )
        fig4.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

    with c2:
        fig5 = px.scatter(
            filtered, x="Likes", y="Retweets", color="Emotion",
            color_discrete_map=PASTEL, hover_data=["Platform", "Country"],
            title="Likes vs. Retweets", opacity=0.75,
        )
        fig5.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("#### Average engagement by platform")
    plat_eng = filtered.groupby("Platform", as_index=False)["Engagement"].mean()
    fig6 = px.bar(
        plat_eng, x="Platform", y="Engagement", color="Platform",
        color_discrete_sequence=PASTEL_SEQUENCE,
    )
    fig6.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig6, use_container_width=True)

# --- Tab 3: Geography & Time --------------------------------------------
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        top_countries = filtered["Country"].value_counts().head(12).reset_index()
        top_countries.columns = ["Country", "Posts"]
        fig7 = px.bar(
            top_countries, x="Posts", y="Country", orientation="h",
            color="Posts", color_continuous_scale=["#EAF6F6", "#B8E6D5", "#6B5B95"],
            title="Top countries by post volume",
        )
        fig7.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False,
        )
        st.plotly_chart(fig7, use_container_width=True)

    with c2:
        yearly = filtered.groupby(["Year", "Emotion"]).size().reset_index(name="Count")
        fig8 = px.line(
            yearly, x="Year", y="Count", color="Emotion",
            color_discrete_map=PASTEL, markers=True,
            title="Emotion trend over years",
        )
        fig8.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig8, use_container_width=True)

    st.markdown("#### Posting activity by hour of day")
    hourly = filtered.groupby("Hour").size().reset_index(name="Posts")
    fig9 = px.area(
        hourly, x="Hour", y="Posts",
        color_discrete_sequence=["#C9B8FF"],
    )
    fig9.update_traces(line_color="#6B5B95", fillcolor="rgba(201,184,255,0.4)")
    fig9.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig9, use_container_width=True)

# --- Tab 4: Hashtags -----------------------------------------------------
with tab4:
    exploded = explode_hashtags(filtered)
    if exploded.empty:
        st.info("No hashtags found for the current filters.")
    else:
        top_tags = exploded["Hashtag"].value_counts().head(15).reset_index()
        top_tags.columns = ["Hashtag", "Count"]
        fig10 = px.bar(
            top_tags, x="Count", y="Hashtag", orientation="h",
            color="Count", color_continuous_scale=["#FFE29A", "#FFB6C1", "#C9B8FF"],
            title="Top 15 hashtags",
        )
        fig10.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False,
        )
        st.plotly_chart(fig10, use_container_width=True)

        st.markdown("#### Emotion breakdown for top hashtag")
        chosen_tag = st.selectbox("Pick a hashtag", top_tags["Hashtag"])
        tag_df = exploded[exploded["Hashtag"] == chosen_tag]
        tag_counts = tag_df["Emotion"].value_counts().reset_index()
        tag_counts.columns = ["Emotion", "Count"]
        fig11 = px.pie(
            tag_counts, names="Emotion", values="Count", hole=0.45,
            color="Emotion", color_discrete_map=PASTEL,
        )
        fig11.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig11, use_container_width=True)

# --- Tab 5: Try It (simple live rule-based demo) --------------------------
with tab5:
    st.markdown("#### Type a sentence and see it plotted against the dataset's mood")
    st.caption(
        "This is a lightweight keyword-based demo. Swap in a trained "
        "TF-IDF + Logistic Regression (or transformer) model here for "
        "real predictions — see train_model.py."
    )
    user_text = st.text_area("Your text", placeholder="I had such a wonderful day at the park!")
    if st.button("Analyze mood"):
        if user_text.strip():
            text_low = user_text.lower()
            scores = {}
            for group, keywords in {
                "Joy": ["happy", "great", "wonderful", "excited", "fun", "joy"],
                "Sadness": ["sad", "cry", "down", "miss", "lonely", "grief"],
                "Anger": ["angry", "hate", "furious", "annoyed", "mad"],
                "Fear": ["scared", "afraid", "anxious", "nervous", "worried"],
                "Love": ["love", "adore", "grateful", "thankful"],
                "Calm": ["calm", "peaceful", "relaxed", "serene"],
            }.items():
                scores[group] = sum(text_low.count(k) for k in keywords)
            best = max(scores, key=scores.get)
            if scores[best] == 0:
                best = "Neutral"
            st.success(f"Detected mood: **{best}**")
        else:
            st.warning("Type something first!")

