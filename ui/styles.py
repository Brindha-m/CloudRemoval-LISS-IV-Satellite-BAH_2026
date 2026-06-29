"""Custom CSS for the Streamlit dashboard."""

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #42A5F5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #90CAF9;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .info-card {
        background: #1A1F2E;
        border-left: 4px solid #1E88E5;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
    .warn-card {
        background: #2A1F1A;
        border-left: 4px solid #FF9800;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
    .success-card {
        background: #1A2A1F;
        border-left: 4px solid #4CAF50;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #B0BEC5;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #151B28 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 16px;
    }
    .viz-title {
        text-align: center;
        font-size: 1.35rem;
        font-weight: 600;
        color: #E3F2FD;
        margin: 0.5rem 0 1.2rem 0;
    }
    .viz-pair {
        border: 1px solid #263238;
        border-radius: 8px;
        padding: 0.5rem;
        margin-bottom: 1rem;
        background: #12151C;
    }
    .viz-label {
        text-align: center;
        font-size: 0.9rem;
        color: #90CAF9;
        margin-top: 0.35rem;
    }
    .app-title {
        font-size: 0.95rem;
        font-weight: 700;
        line-height: 1.35;
        color: #E3F2FD;
        margin: 0 0 0.5rem 0;
    }
    .app-subtitle {
        font-size: 0.78rem;
        line-height: 1.4;
        color: #90CAF9;
        margin: 0 0 0.25rem 0;
    }
    .app-footer {
        text-align: center;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #64B5F6;
        margin: 0.5rem 0 0 0;
    }
</style>
"""


def inject_styles() -> None:
    import streamlit as st

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
