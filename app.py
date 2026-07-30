import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


## Page config

st.set_page_config(
    page_title="Vitalis - Life Expectancy Studio",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)


## Custom styling (dark navy theme, fonts, hide default Streamlit chrome)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@300;400;600&display=swap');

/* hide Streamlit's top bar, menu, deploy button and footer */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stHeader"] {display: none;}

/* main app background - deep navy gradient */
.stApp {
    background:
        radial-gradient(1100px 500px at 15% -10%, #12315e 0%, rgba(18,49,94,0) 55%),
        linear-gradient(160deg, #050f22 0%, #0a1930 55%, #0b2545 100%);
    color: #e6ecf5;
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: #eaf2ff; }

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07122a 0%, #0a1c3a 100%);
    border-right: 1px solid rgba(80,140,220,0.18);
}
[data-testid="stSidebar"] * { color: #d5e2f5; }

/* hero header card */
.hero {
    background: linear-gradient(120deg, rgba(34,211,238,0.12), rgba(129,140,248,0.05));
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 20px;
    padding: 26px 32px;
    margin-bottom: 20px;
}
.hero h1 {
    margin: 0; font-size: 2.5rem; letter-spacing: 1px;
    background: linear-gradient(90deg, #67e8f9, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #9fb3d1; margin: 0.5rem 0 0 0; font-size: 1.05rem; }

/* prediction result card */
.result {
    background: linear-gradient(135deg, rgba(56,189,248,0.16), rgba(129,140,248,0.08));
    border: 1px solid rgba(103,232,249,0.35);
    border-radius: 22px;
    padding: 30px 24px;
    text-align: center;
    box-shadow: 0 12px 45px rgba(6,20,44,0.7);
}
.result .lbl {
    color: #9fb3d1; text-transform: uppercase;
    letter-spacing: 4px; font-size: 0.78rem;
}
.result .num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4rem; font-weight: 700; line-height: 1.1;
    color: #67e8f9; margin: 6px 0;
}
.result .unit { color: #9fb3d1; font-size: 1rem; }

/* generic panel */
.panel {
    background: rgba(13,32,64,0.55);
    border: 1px solid rgba(80,140,220,0.18);
    border-radius: 16px; padding: 18px 20px;
}
.section {
    font-family: 'Space Grotesk', sans-serif;
    color: #67e8f9; font-size: 1.3rem;
    margin: 22px 0 8px 0;
}

/* buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #22d3ee, #3b82f6);
    color: #04101f; font-weight: 700; border: none;
    border-radius: 12px; padding: 0.6rem 1.2rem; width: 100%;
    font-family: 'Space Grotesk', sans-serif; letter-spacing: 0.5px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 26px rgba(56,189,248,0.45);
    color: #04101f;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


## Load model and data

@st.cache_resource
def load_model():
    return joblib.load("life_expectancy_model.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv("Life Expectancy Data.csv")
    df.columns = df.columns.str.strip()
    return df


model = load_model()
raw = load_data()

## plot palette
NAVY = "#0a1930"
ACCENT = "#38bdf8"
ACCENT2 = "#67e8f9"
GRID = "#1e3a5f"
TEXT = "#cfe0f5"


def style_dark(fig, ax):
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(ACCENT2)
    ax.grid(color=GRID, alpha=0.4, linewidth=0.6)


def plot_importance():
    imp = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    imp = imp.sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.barh(imp.index, imp.values, color=ACCENT)
    ax.set_title("What drives the prediction (feature importance)")
    style_dark(fig, ax)
    fig.tight_layout()
    return fig


def plot_distribution():
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.hist(raw["Life expectancy"].dropna(), bins=30,
            color=ACCENT2, edgecolor=NAVY, alpha=0.9)
    ax.set_title("Distribution of life expectancy in the data")
    ax.set_xlabel("Life expectancy (years)")
    ax.set_ylabel("Countries")
    style_dark(fig, ax)
    fig.tight_layout()
    return fig


def plot_relationship():
    d = raw[["Schooling", "Life expectancy"]].dropna()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.scatter(d["Schooling"], d["Life expectancy"],
               s=12, alpha=0.5, color=ACCENT)
    ax.set_title("Schooling vs life expectancy")
    ax.set_xlabel("Schooling (years)")
    ax.set_ylabel("Life expectancy (years)")
    style_dark(fig, ax)
    fig.tight_layout()
    return fig



## Sidebar - user inputs

st.sidebar.markdown("## Indicators")
st.sidebar.caption("Set a country's profile, then estimate.")

status_selected = st.sidebar.selectbox("Country status", ["Developing", "Developed"])
schooling_selected = st.sidebar.slider("Schooling (years)", 0.0, 21.0, 12.4)
income_selected = st.sidebar.slider("Income composition (0-1)", 0.0, 1.0, 0.68)
adult_mortality_selected = st.sidebar.slider("Adult mortality (per 1000)", 0.0, 750.0, 138.0)
bmi_selected = st.sidebar.slider("Average BMI", 1.0, 80.0, 43.5)
hiv_selected = st.sidebar.slider("HIV/AIDS (per 1000)", 0.1, 51.0, 0.1)
immunization_selected = st.sidebar.slider("Immunization avg (%)", 0.0, 100.0, 93.0)
alcohol_selected = st.sidebar.slider("Alcohol (litres pc)", 0.0, 18.0, 3.96)
thinness_selected = st.sidebar.slider("Thinness 1-19 yrs (%)", 0.0, 28.0, 3.3)
total_exp_selected = st.sidebar.slider("Total health expenditure (%)", 0.0, 15.0, 5.73)
gdp_selected = st.sidebar.number_input("GDP per capita (USD)", 0.0, 120000.0, 1753.0)
pct_exp_selected = st.sidebar.number_input("Percentage expenditure", 0.0, 20000.0, 155.0)
measles_selected = st.sidebar.number_input("Measles (cases)", 0, 215000, 15)
under_five_selected = st.sidebar.number_input("Under-five deaths (per 1000)", 0, 2500, 3)

predict_clicked = st.sidebar.button("Estimate life expectancy")


## Main area - header

st.markdown(
    """
    <div class="hero">
        <h1>Vitalis</h1>
        <p>A machine-learning studio that estimates a country's life expectancy
        from its health, economic and social indicators.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

## session state
if "pred" not in st.session_state:
    st.session_state.pred = None
if "history" not in st.session_state:
    st.session_state.history = []


## Prediction

if predict_clicked:
    df_input = pd.DataFrame({
        'Adult Mortality': [adult_mortality_selected],
        'Alcohol': [alcohol_selected],
        'percentage expenditure': [pct_exp_selected],
        'Measles': [measles_selected],
        'BMI': [bmi_selected],
        'under-five deaths': [under_five_selected],
        'Total expenditure': [total_exp_selected],
        'HIV/AIDS': [hiv_selected],
        'GDP': [gdp_selected],
        'thinness  1-19 years': [thinness_selected],
        'Income composition of resources': [income_selected],
        'Schooling': [schooling_selected],
        'Immunization avg': [immunization_selected],
        'Status': [status_selected],
    })
    df_input = pd.get_dummies(df_input, columns=['Status'])
    df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)
    st.session_state.pred = model.predict(df_input)[0]

    ## Save this scenario (inputs + prediction) so the user can download it later
    record = {
        'Status': status_selected,
        'Schooling': schooling_selected,
        'Income composition': income_selected,
        'Adult Mortality': adult_mortality_selected,
        'BMI': bmi_selected,
        'HIV/AIDS': hiv_selected,
        'Immunization avg': immunization_selected,
        'Alcohol': alcohol_selected,
        'Thinness 1-19': thinness_selected,
        'Total expenditure': total_exp_selected,
        'GDP': gdp_selected,
        'Percentage expenditure': pct_exp_selected,
        'Measles': measles_selected,
        'Under-five deaths': under_five_selected,
        'Predicted Life Expectancy': round(float(st.session_state.pred), 1),
    }
    st.session_state.history.append(record)


## Result + model info

left, right = st.columns([1, 1])

with left:
    if st.session_state.pred is None:
        st.markdown(
            """
            <div class="result">
                <div class="lbl">Awaiting input</div>
                <div class="num">--</div>
                <div class="unit">Set the indicators in the sidebar, then click Estimate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result">
                <div class="lbl">Predicted life expectancy</div>
                <div class="num">{st.session_state.pred:.1f}</div>
                <div class="unit">years</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    st.markdown('<div class="section">Model at a glance</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel">
            <b>Model:</b> tuned Random Forest Regressor<br>
            <b>Accuracy:</b> R&sup2; &asymp; 0.96 &nbsp; | &nbsp; MAE &asymp; 1.2 years<br>
            <b>Trained on:</b> Life Expectancy (WHO) dataset, ~2,300 country-year records
        </div>
        """,
        unsafe_allow_html=True,
    )


## Saved scenarios (download as CSV)

st.markdown('<div class="section">Saved scenarios</div>', unsafe_allow_html=True)

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    csv = history_df.to_csv(index=False).encode("utf-8")
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="life_expectancy_scenarios.csv",
            mime="text/csv",
        )
    with c2:
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
else:
    st.caption("No scenarios saved yet. Click Estimate in the sidebar to add one.")


## Data insights (graphs of the training data)

st.markdown('<div class="section">Data insights</div>', unsafe_allow_html=True)

g1, g2, g3 = st.columns(3)
with g1:
    st.pyplot(plot_importance())
with g2:
    st.pyplot(plot_distribution())
with g3:
    st.pyplot(plot_relationship())

st.caption("Data source: Life Expectancy (WHO) - Kaggle.")
