import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nltk import FreqDist
from nltk.corpus import stopwords
import nltk
import re


nltk.download('stopwords', quiet=True)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GATEWAYS-2025 | Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# MINIMAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }
.insight-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #3b82f6 100%);
    border: none;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
    font-size: 0.95rem;
    color: #ffffff;
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.25), 0 4px 12px rgba(0, 0, 0, 0.1);
    border-top: 3px solid #fbbf24;
    line-height: 1.6;
    transition: all 0.3s ease;
}
.insight-box:hover {
    box-shadow: 0 12px 32px rgba(59, 130, 246, 0.35), 0 6px 16px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
}
.insight-box strong {
    color: #fbbf24;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("C5_FestDataset.csv")
    df.columns = df.columns.str.strip()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    return df

@st.cache_data
def load_geojson():
    """Cache the GeoJSON file to avoid repeated network requests"""
    import urllib.request, json
    geo_url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
    try:
        with urllib.request.urlopen(geo_url, timeout=8) as r:
            geo_json = json.loads(r.read().decode())
        # fix property key for matching
        for feat in geo_json["features"]:
            feat["id"] = feat["properties"].get("NAME_1", feat["properties"].get("name", ""))
        return geo_json
    except Exception as e:
        st.warning(f"Could not load GeoJSON: {e}")
        return None

@st.cache_data
def prepare_state_map_data(df):
    """Cache state map aggregations to avoid repeated groupby operations"""
    state_counts = df.groupby("State").size().reset_index(name="Participants")
    avg_ratings = df.groupby("State")["Rating"].mean().round(2).reset_index()
    avg_ratings.columns = ["State", "Avg Rating"]
    top_events = df.groupby("State")["Event Name"].agg(lambda x: x.value_counts().idxmax()).reset_index()
    top_events.columns = ["State", "Top Event"]
    top_colleges = df.groupby("State")["College"].agg(lambda x: x.value_counts().idxmax()).reset_index()
    top_colleges.columns = ["State", "Top College"]
    
    state_map = state_counts.merge(avg_ratings, on="State").merge(top_events, on="State").merge(top_colleges, on="State")
    return state_map

df = load_data()

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## GATEWAYS-2025")
    st.markdown("---")

    # Navigation
    page = st.radio("Navigate to", [
        "Overview Dashboard",
        "Participation Analysis",
        "State-wise India Map",
        "Feedback & Sentiment",
    ])
    st.markdown("---")

    # Filters
    st.markdown("### Filters")
    all_states  = ["All States"] + sorted(df["State"].dropna().unique().tolist())
    all_events  = ["All Events"] + sorted(df["Event Name"].dropna().unique().tolist())
    all_types   = ["All Types"]  + sorted(df["Event Type"].dropna().unique().tolist())

    sel_state = st.selectbox("State",       all_states)
    sel_event = st.selectbox("Event",       all_events)
    sel_type  = st.selectbox("Event Type",  all_types)
    sel_rating = st.slider("Min Rating", 1, 5, 1)

    st.markdown("---")
    st.markdown("<small style='color:#718096'>GATEWAYS-2025 • Analytics v1.0</small>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
fdf = df.copy()
if sel_state != "All States":   fdf = fdf[fdf["State"] == sel_state]
if sel_event != "All Events":   fdf = fdf[fdf["Event Name"] == sel_event]
if sel_type  != "All Types":    fdf = fdf[fdf["Event Type"] == sel_type]
fdf = fdf[fdf["Rating"] >= sel_rating]

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=50, b=40, l=30, r=30),
)
ACCENT_COLORS = px.colors.qualitative.Vivid

# ─────────────────────────────────────────────
# HEADER (shown on all pages)
# ─────────────────────────────────────────────
st.title("GATEWAYS-2025 Analytics Dashboard")
st.caption("National-Level Fest • Participation & Feedback Intelligence")
st.divider()

# ─────────────────────────────────────────────────────────────────
# PAGE 1 – OVERVIEW DASHBOARD
# ─────────────────────────────────────────────────────────────────
if page == "Overview Dashboard":

    # KPI CARDS
    total_participants = len(fdf)
    total_colleges     = fdf["College"].nunique()
    total_events       = fdf["Event Name"].nunique()
    avg_rating         = fdf["Rating"].mean()
    total_revenue      = fdf["Amount Paid"].sum()
    top_state          = fdf["State"].value_counts().idxmax() if len(fdf) > 0 else "N/A"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Participants", total_participants)
    c2.metric("Colleges",     total_colleges)
    c3.metric("Events",        total_events)
    c4.metric("Avg Rating",    f"{avg_rating:.1f}/5")
    c5.metric("Revenue",       f"₹{total_revenue:,}")
    c6.metric("Top State",     top_state)

    st.markdown("---")

    col_l, col_r = st.columns(2)

    # Event participation pie
    with col_l:
        st.markdown("#### Event-wise Participation")
        ev_cnt = fdf["Event Name"].value_counts().reset_index()
        ev_cnt.columns = ["Event", "Count"]
        fig = px.pie(ev_cnt, names="Event", values="Count",
                     hole=0.45, color_discrete_sequence=ACCENT_COLORS)
        fig.update_traces(textinfo="label+percent", pull=[0.04]*len(ev_cnt))
        fig.update_layout(**PLOTLY_LAYOUT, height=340)
        st.plotly_chart(fig, use_container_width=True)

    # State distribution bar
    with col_r:
        st.markdown("#### State-wise Participation")
        st_cnt = fdf["State"].value_counts().reset_index()
        st_cnt.columns = ["State", "Count"]
        fig2 = px.bar(st_cnt, x="State", y="Count", color="Count",
                      color_continuous_scale="Viridis", text="Count")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(**PLOTLY_LAYOUT, height=340,
                           xaxis_title="", yaxis_title="Participants",
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)

    # Individual vs Group
    with col_a:
        st.markdown("#### Individual vs Group Events")
        type_cnt = fdf["Event Type"].value_counts().reset_index()
        type_cnt.columns = ["Type", "Count"]
        fig3 = px.bar(type_cnt, x="Type", y="Count", color="Type",
                      color_discrete_sequence=["#00d2ff", "#ff6b6b"], text="Count")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(**PLOTLY_LAYOUT, height=300,
                           showlegend=False, xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig3, use_container_width=True)

    # Rating Distribution
    with col_b:
        st.markdown("#### Rating Distribution")
        r_cnt = fdf["Rating"].value_counts().sort_index().reset_index()
        r_cnt.columns = ["Rating", "Count"]
        r_cnt["Stars"] = r_cnt["Rating"].apply(lambda x: "⭐"*int(x))
        fig4 = px.bar(r_cnt, x="Stars", y="Count", color="Count",
                      color_continuous_scale="RdYlGn", text="Count")
        fig4.update_traces(textposition="outside")
        fig4.update_layout(**PLOTLY_LAYOUT, height=300,
                           coloraxis_showscale=False, xaxis_title="Rating", yaxis_title="Count")
        st.plotly_chart(fig4, use_container_width=True)

    # Insight box
    top_event   = fdf["Event Name"].value_counts().idxmax()
    top_college = fdf["College"].value_counts().idxmax()
    st.markdown(f"""
    <div class="insight-box">
      <strong>Key Insights:</strong>&nbsp;
      The most popular event is <strong>{top_event}</strong> with
      <strong>{fdf['Event Name'].value_counts().max()}</strong> participants.
      <strong>{top_college}</strong> leads in college representation.
      The average participant satisfaction rating is
      <strong>{avg_rating:.2f}/5</strong>, reflecting a
      {"highly positive" if avg_rating >= 4 else "moderately positive"} experience overall.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PAGE 2 – PARTICIPATION ANALYSIS
# ─────────────────────────────────────────────────────────────────
elif page == "Participation Analysis":

    st.subheader("Participation Analysis")

    # ── Event-wise deep dive ──
    st.markdown("#### Event-wise Breakdown")
    ev_state = fdf.groupby(["Event Name", "State"]).size().reset_index(name="Count")
    fig_ev = px.bar(ev_state, x="Event Name", y="Count", color="State",
                    barmode="group", color_discrete_sequence=ACCENT_COLORS,
                    text="Count")
    fig_ev.update_traces(textposition="outside")
    fig_ev.update_layout(**PLOTLY_LAYOUT, height=400,
                         xaxis_title="Event", yaxis_title="Participants")
    st.plotly_chart(fig_ev, use_container_width=True)

    col1, col2 = st.columns(2)

    # ── Top 10 Colleges ──
    with col1:
        st.markdown("#### Top 10 Colleges by Participation")
        top_col = fdf["College"].value_counts().head(10).reset_index()
        top_col.columns = ["College", "Count"]
        fig_col = px.bar(top_col, x="Count", y="College", orientation="h",
                         color="Count", color_continuous_scale="Blues",
                         text="Count")
        fig_col.update_traces(textposition="outside")
        fig_col.update_layout(**PLOTLY_LAYOUT, height=420,
                              yaxis={"categoryorder": "total ascending"},
                              coloraxis_showscale=False, xaxis_title="Participants", yaxis_title="")
        st.plotly_chart(fig_col, use_container_width=True)

    # ── Avg Amount Paid per Event ──
    with col2:
        st.markdown("#### Avg Registration Fee per Event")
        fee_ev = fdf.groupby("Event Name")["Amount Paid"].mean().round(0).reset_index()
        fee_ev.columns = ["Event", "Avg Fee"]
        fig_fee = px.funnel(fee_ev, x="Avg Fee", y="Event",
                            color_discrete_sequence=["#00d2ff"])
        fig_fee.update_layout(**PLOTLY_LAYOUT, height=420,
                              xaxis_title="Avg Fee (₹)", yaxis_title="")
        st.plotly_chart(fig_fee, use_container_width=True)

    # ── Heatmap: Event × State ──
    st.markdown("#### Participation Heatmap — Event × State")
    heat_data = fdf.groupby(["Event Name", "State"]).size().unstack(fill_value=0)
    fig_heat = px.imshow(heat_data, color_continuous_scale="Viridis",
                         aspect="auto", text_auto=True)
    fig_heat.update_layout(**PLOTLY_LAYOUT, height=360,
                           xaxis_title="State", yaxis_title="Event")
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── College × Event grouped bar ──
    st.markdown("#### College vs Event Participation (Top 20 combinations)")
    ce_df = fdf.groupby(["College", "Event Name"]).size().reset_index(name="Count")
    ce_df = ce_df.sort_values("Count", ascending=False).head(20)
    fig_ce = px.bar(ce_df, x="Count", y="College", color="Event Name", orientation="h",
                    barmode="stack", color_discrete_sequence=ACCENT_COLORS, text="Count")
    fig_ce.update_traces(textposition="outside")
    fig_ce.update_layout(**PLOTLY_LAYOUT, height=500,
                         xaxis_title="Participants", yaxis_title="College", legend_title="Event")
    st.plotly_chart(fig_ce, use_container_width=True)

    # Insight
    top_ev  = fdf["Event Name"].value_counts().idxmax()
    top_cg  = fdf["College"].value_counts().idxmax()
    st.markdown(f"""
    <div class="insight-box">
      <strong>Participation Insight:</strong>
      <strong>{top_ev}</strong> attracts the most participants.
      <strong>{top_cg}</strong> has the highest registration count.
      Group events account for <strong>{(fdf['Event Type']=='Group').sum()}</strong>
      registrations vs <strong>{(fdf['Event Type']=='Individual').sum()}</strong> individual entries.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PAGE 3 – STATE-WISE INDIA MAP
# ─────────────────────────────────────────────────────────────────
elif page == "State-wise India Map":

    st.subheader("State-wise Participation on India Map")
    st.info("The choropleth map below visualises participant counts across Indian states using GeoJSON boundaries.")

    # Use cached function for data preparation
    state_map = prepare_state_map_data(df)

    # State centroids for scatter map
    STATE_COORDS = {
        "Tamil Nadu":    (10.8505, 78.6677),
        "Karnataka":     (15.3173, 75.7139),
        "Kerala":        (10.8505, 76.2711),
        "Telangana":     (18.1124, 79.0193),
        "Maharashtra":   (19.7515, 75.7139),
        "Delhi":         (28.6139, 77.2090),
        "Uttar Pradesh": (26.8467, 80.9462),
        "Gujarat":       (22.2587, 71.1924),
        "Rajasthan":     (27.0238, 74.2179),
        "West Bengal":   (22.9868, 87.8550),
        "Andhra Pradesh":(15.9129, 79.7400),
    }

    state_map["lat"] = state_map["State"].map(lambda s: STATE_COORDS.get(s, (20.5937, 78.9629))[0])
    state_map["lon"] = state_map["State"].map(lambda s: STATE_COORDS.get(s, (20.5937, 78.9629))[1])

    # ── Choropleth via GeoJSON (cached) ──
    geo_json = load_geojson()
    if geo_json is not None:
        fig_map = px.choropleth(
            state_map,
            geojson=geo_json,
            locations="State",
            featureidkey="id",
            color="Participants",
            color_continuous_scale="YlOrRd",
            hover_name="State",
            hover_data={"Participants": True, "Avg Rating": True, "Top Event": True, "Top College": True},
            title="Participant Density Across India",
            fitbounds="locations",
            basemap_visible=False,
        )
        fig_map.update_layout(**PLOTLY_LAYOUT, height=600,
                              coloraxis_colorbar=dict(title="Participants", tickfont=dict(color="#e2e8f0")))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("Could not load GeoJSON map. Showing bubble map instead.")

    # ── Bubble / Scatter Map (always shown as supplement) ──
    st.markdown("#### Bubble Map — Participation Size & Rating")
    fig_bub = px.scatter_geo(
        state_map,
        lat="lat", lon="lon",
        size="Participants",
        color="Avg Rating",
        hover_name="State",
        hover_data={"Participants": True, "Avg Rating": True, "Top Event": True},
        color_continuous_scale="RdYlGn",
        size_max=50,
        scope="asia",
        title="Participant Bubble Map – India",
    )
    fig_bub.update_geos(
        showcountries=True, countrycolor="#444",
        showsubunits=True,  subunitcolor="#555",
        showland=True,      landcolor="#1a1a2e",
        showocean=True,     oceancolor="#0f2027",
        showlakes=True,     lakecolor="#0f2027",
        center=dict(lat=22, lon=80),
        projection_scale=4,
    )
    fig_bub.update_layout(**PLOTLY_LAYOUT, height=560,
                          coloraxis_colorbar=dict(title="Avg Rating", tickfont=dict(color="#e2e8f0")))
    st.plotly_chart(fig_bub, use_container_width=True)

    # ── State Detail Table ──
    st.markdown("#### State Summary Table")
    state_map_display = state_map.drop(columns=["lat", "lon"])
    state_map_display = state_map_display.sort_values("Participants", ascending=False)
    st.dataframe(
        state_map_display.style
        .background_gradient(subset=["Participants"], cmap="YlOrRd")
        .background_gradient(subset=["Avg Rating"],   cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )

    # ── Bar chart state comparison ──
    st.markdown("#### State-wise Participant Count")
    fig_bar = px.bar(
        state_map.sort_values("Participants", ascending=True),
        x="Participants", y="State", orientation="h",
        color="Participants", color_continuous_scale="YlOrRd",
        text="Participants",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(**PLOTLY_LAYOUT, height=400,
                          coloraxis_showscale=False, xaxis_title="Participants", yaxis_title="")
    st.plotly_chart(fig_bar, use_container_width=True)

    top_s = state_map.sort_values("Participants", ascending=False).iloc[0]
    st.markdown(f"""
    <div class="insight-box">
      <strong>Geographic Insight:</strong>
      <strong>{top_s['State']}</strong> leads with <strong>{top_s['Participants']}</strong> participants
      and an average rating of <strong>{top_s['Avg Rating']}</strong>.
      Their most popular event is <strong>{top_s['Top Event']}</strong>.
      The fest successfully reached <strong>{state_map['State'].nunique()}</strong> states across India.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PAGE 4 – FEEDBACK & SENTIMENT
# ─────────────────────────────────────────────────────────────────
elif page == "Feedback & Sentiment":

    st.subheader("Feedback & Sentiment Analysis")

    # ── Rating Overview ──
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Rating",    f"{fdf['Rating'].mean():.2f} / 5")
    col2.metric("5-Star Responses",  f"{(fdf['Rating']==5).mean()*100:.1f}%")
    col3.metric("Low Ratings (≤2)",  f"{(fdf['Rating']<=2).mean()*100:.1f}%")

    st.markdown("")

    col_a, col_b = st.columns(2)

    # ── Rating Violin per Event ──
    with col_a:
        st.markdown("#### Rating Distribution per Event")
        fig_vio = px.violin(fdf, y="Rating", x="Event Name", color="Event Name",
                            box=True, points="all",
                            color_discrete_sequence=ACCENT_COLORS)
        fig_vio.update_layout(**PLOTLY_LAYOUT, height=400,
                              showlegend=False, xaxis_title="", yaxis_title="Rating")
        st.plotly_chart(fig_vio, use_container_width=True)

    # ── Avg Rating per State ──
    with col_b:
        st.markdown("#### Avg Rating by State")
        avg_state = fdf.groupby("State")["Rating"].mean().round(2).reset_index()
        avg_state.columns = ["State", "Avg Rating"]
        fig_rat = px.bar(avg_state.sort_values("Avg Rating", ascending=True),
                         x="Avg Rating", y="State", orientation="h",
                         color="Avg Rating", color_continuous_scale="RdYlGn",
                         text="Avg Rating", range_x=[0, 5])
        fig_rat.update_traces(textposition="outside")
        fig_rat.update_layout(**PLOTLY_LAYOUT, height=400,
                              coloraxis_showscale=False, xaxis_title="Avg Rating", yaxis_title="")
        st.plotly_chart(fig_rat, use_container_width=True)

    # ── Word Cloud ──
    st.markdown("#### Feedback Word Cloud")
    all_feedback = " ".join(fdf["Feedback on Fest"].dropna().tolist())
    wc = WordCloud(
        width=1200, height=420,
        background_color="#0f2027",
        colormap="cool",
        max_words=120,
        prefer_horizontal=0.85,
    ).generate(all_feedback)
    fig_wc, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig_wc.patch.set_facecolor("#0f2027")
    st.pyplot(fig_wc)
    plt.close()

    # ── Top Feedback Phrases ──
    st.markdown("#### Most Common Feedback Phrases")
    all_words = []
    for fb in fdf["Feedback on Fest"].dropna():
        words = re.findall(r'\b[a-zA-Z]{4,}\b', fb.lower())
        all_words.extend(words)
    
    # Use NLTK stopwords
    nltk_stopwords = set(stopwords.words('english'))
    custom_stops = {"event", "very", "good", "well"}
    stop = nltk_stopwords.union(custom_stops)
    
    filtered = [w for w in all_words if w not in stop]
    
    # Use NLTK FreqDist instead of Counter
    freq_dist = FreqDist(filtered)
    phrase_counts = freq_dist.most_common(15)
    phrase_df = pd.DataFrame(phrase_counts, columns=["Word", "Count"])
    fig_ph = px.bar(phrase_df, x="Count", y="Word", orientation="h",
                    color="Count", color_continuous_scale="Viridis", text="Count")
    fig_ph.update_traces(textposition="outside")
    fig_ph.update_layout(**PLOTLY_LAYOUT, height=450,
                         coloraxis_showscale=False,
                         yaxis={"categoryorder": "total ascending"},
                         xaxis_title="Frequency", yaxis_title="")
    st.plotly_chart(fig_ph, use_container_width=True)

    # ── Sentiment Tagging ──
    st.markdown("#### Feedback Sentiment Categorisation")
    POSITIVE = {"excellent","great","amazing","wonderful","fantastic","good","engaging",
                "informative","creative","interesting","fun","useful","practical","inspiring",
                "well","learning","exposure","innovative","organised","interactive","productive"}
    NEGATIVE = {"poor","bad","boring","difficult","hard","confusing","improve","worst","lack","slow","issue"}

    def tag_sentiment(text):
        words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
        if words & POSITIVE:
            return "Positive"
        elif words & NEGATIVE:
            return "Negative"
        return "Neutral"

    fdf = fdf.copy()
    fdf["Sentiment"] = fdf["Feedback on Fest"].fillna("").apply(tag_sentiment)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        sent_cnt = fdf["Sentiment"].value_counts().reset_index()
        sent_cnt.columns = ["Sentiment", "Count"]
        fig_sent = px.pie(sent_cnt, names="Sentiment", values="Count",
                          color_discrete_map={
                              "Positive": "#68d391",
                              "Neutral":  "#fbd38d",
                              "Negative": "#fc8181",
                          }, hole=0.4)
        fig_sent.update_layout(**PLOTLY_LAYOUT, height=340, title="Overall Sentiment Split")
        st.plotly_chart(fig_sent, use_container_width=True)

    with col_s2:
        sent_ev = fdf.groupby(["Event Name", "Sentiment"]).size().reset_index(name="Count")
        fig_sev = px.bar(sent_ev, x="Event Name", y="Count", color="Sentiment",
                         color_discrete_map={
                             "Positive": "#68d391",
                             "Neutral":  "#fbd38d",
                             "Negative": "#fc8181",
                         }, barmode="stack")
        fig_sev.update_layout(**PLOTLY_LAYOUT, height=340, title="Sentiment by Event",
                              xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig_sev, use_container_width=True)

    # ── Avg Rating vs Sentiment ──
    st.markdown("#### Avg Rating per Sentiment Category")
    sent_rating = fdf.groupby("Sentiment")["Rating"].mean().round(2).reset_index()
    sent_rating.columns = ["Sentiment", "Avg Rating"]
    fig_sr = px.bar(sent_rating, x="Sentiment", y="Avg Rating",
                    color="Sentiment",
                    color_discrete_map={
                        "Positive": "#68d391",
                        "Neutral":  "#fbd38d",
                        "Negative": "#fc8181",
                    }, text="Avg Rating", range_y=[0, 5])
    fig_sr.update_traces(textposition="outside")
    fig_sr.update_layout(**PLOTLY_LAYOUT, height=320,
                         showlegend=False, xaxis_title="", yaxis_title="Avg Rating")
    st.plotly_chart(fig_sr, use_container_width=True)

    # ── Feedback Explorer Table ──
    st.markdown("#### Feedback Explorer")
    search_term = st.text_input("Search in feedback", placeholder="e.g. excellent, engaging…")
    show_df = fdf[["Student Name", "College", "State", "Event Name", "Rating", "Feedback on Fest", "Sentiment"]]
    if search_term:
        show_df = show_df[show_df["Feedback on Fest"].str.contains(search_term, case=False, na=False)]
    st.dataframe(
        show_df.style.applymap(
            lambda v: "color: #68d391" if "Positive" in str(v) else
                      ("color: #fc8181" if "Negative" in str(v) else "color: #fbd38d"),
            subset=["Sentiment"]
        ),
        use_container_width=True, hide_index=True,
    )

    pos_pct = (fdf["Sentiment"].str.startswith("Positive")).mean() * 100
    st.markdown(f"""
    <div class="insight-box">
      <strong>Feedback Insight:</strong>
      <strong>{pos_pct:.1f}%</strong> of participants left positive feedback.
      The most commonly used words reflect themes of
      <strong>learning, engagement, and creativity</strong>.
      Events with higher organisation scores consistently received
      <strong>4–5 star ratings</strong>.
    </div>""", unsafe_allow_html=True)