import io
import requests
import numpy as np
import pandas as pd
import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt

# Optional (interactive alternative)
import plotly.express as px

sns.set_theme(style="whitegrid")


DIAMONDS_FALLBACK_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"


@st.cache_data(show_spinner=False)
def load_diamonds() -> pd.DataFrame:
    """
    Streamlit Cloud friendly loader with fallbacks:
    1) seaborn.load_dataset (uses internet)
    2) GitHub raw CSV fallback
    """
    # 1) seaborn built-in loader (downloads from seaborn-data repo)
    try:
        df = sns.load_dataset("diamonds")
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass

    # 2) Fallback: direct CSV download (robust)
    try:
        r = requests.get(DIAMONDS_FALLBACK_URL, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        return df
    except Exception as e:
        raise RuntimeError(
            "Diamonds dataset could not be loaded. "
            "Please upload a CSV file in the sidebar (or check internet access)."
        ) from e


def make_seaborn_plot(
    df: pd.DataFrame,
    clarity_order: list,
    palette_name: str,
    point_alpha: float,
    size_min: int,
    size_max: int,
    fig_w: float,
    fig_h: float,
    log_x: bool,
    log_y: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.despine(fig, left=True, bottom=True)

    sns.scatterplot(
        x="carat",
        y="price",
        hue="clarity",
        size="depth",
        palette=palette_name,
        hue_order=clarity_order,
        sizes=(size_min, size_max),
        linewidth=0,
        alpha=point_alpha,
        data=df,
        ax=ax,
    )

    ax.set_title("Diamonds: Carat vs Price (hue=clarity, size=depth)")
    ax.set_xlabel("Carat")
    ax.set_ylabel("Price")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.grid(True, which="major", linestyle="--", alpha=0.3)
    return fig


def make_plotly_plot(df: pd.DataFrame, clarity_order: list, log_x: bool, log_y: bool):
    fig = px.scatter(
        df,
        x="carat",
        y="price",
        color="clarity",
        size="depth",
        category_orders={"clarity": clarity_order},
        hover_data=["cut", "color", "table", "depth", "x", "y", "z"],
        title="Diamonds: Carat vs Price (interactive)",
    )

    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")

    fig.update_layout(
        height=650,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def main():
    st.set_page_config(page_title="Diamonds Scatter (Streamlit Cloud)", layout="wide")
    st.title("💎 Diamonds Scatter — Streamlit Cloud Ready")
    st.caption("Seaborn/Matplotlib + optional Plotly. Filters + PNG export + offline-safe loader.")

    with st.sidebar:
        st.header("Settings")

        # Data source
        st.subheader("Data")
        use_upload = st.toggle("Upload my own CSV instead", value=False)

        uploaded = None
        if use_upload:
            uploaded = st.file_uploader("Upload CSV", type=["csv"])

        st.divider()

        # Plot engine
        st.subheader("Visualization")
        engine = st.radio("Chart engine", ["Seaborn (Matplotlib)", "Plotly (interactive)"], index=0)

        # Filters
        st.subheader("Filters")

        # Palette + plot params
        palette_name = st.selectbox(
            "Palette (Seaborn)",
            ["ch:r=-.2,d=.3_r", "viridis", "magma", "coolwarm", "Spectral", "Set2"],
            index=0,
            help="Used for Seaborn mode only.",
        )

        point_alpha = st.slider("Point alpha", 0.1, 1.0, 0.85, 0.05)

        size_min, size_max = st.slider("Point size range", 1, 20, (2, 10), 1)

        fig_w = st.slider("Figure width", 4.0, 14.0, 7.0, 0.5)
        fig_h = st.slider("Figure height", 4.0, 14.0, 7.0, 0.5)

        log_x = st.toggle("Log scale X (carat)", value=False)
        log_y = st.toggle("Log scale Y (price)", value=False)

    # Load data
    df = None
    if use_upload:
        if uploaded is None:
            st.info("Upload a CSV to continue, or turn off 'Upload my own CSV instead'.")
            st.stop()
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()
    else:
        try:
            with st.spinner("Loading diamonds dataset..."):
                df = load_diamonds()
        except Exception as e:
            st.warning(str(e))
            st.info("You can upload a CSV from the sidebar.")
            st.stop()

    # Basic validation / required columns
    required = {"carat", "price", "clarity", "depth"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Dataset missing required columns: {sorted(missing)}")
        st.stop()

    # Ensure types
    df = df.copy()
    df["carat"] = pd.to_numeric(df["carat"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["clarity"] = df["clarity"].astype(str)

    df = df.dropna(subset=["carat", "price", "depth", "clarity"])

    # Standard clarity ranking (your original)
    clarity_ranking = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
    # If dataset has different/extra categories, keep ranking first then append others
    present = [c for c in clarity_ranking if c in set(df["clarity"])]
    extras = sorted(list(set(df["clarity"]) - set(present)))
    clarity_order = present + extras

    # Filter controls based on data ranges
    left, right = st.columns([1, 1])

    with left:
        carat_min, carat_max = float(df["carat"].min()), float(df["carat"].max())
        carat_range = st.slider("Carat range", carat_min, carat_max, (carat_min, carat_max))

    with right:
        price_min, price_max = float(df["price"].min()), float(df["price"].max())
        price_range = st.slider("Price range", price_min, price_max, (price_min, price_max))

    selected_clarity = st.multiselect("Clarity", options=clarity_order, default=clarity_order)

    filtered = df[
        (df["carat"] >= carat_range[0]) & (df["carat"] <= carat_range[1]) &
        (df["price"] >= price_range[0]) & (df["price"] <= price_range[1]) &
        (df["clarity"].isin(selected_clarity))
    ].copy()

    top_kpis = st.columns(4)
    top_kpis[0].metric("Rows (filtered)", f"{len(filtered):,}")
    top_kpis[1].metric("Carat (median)", f"{filtered['carat'].median():.3f}")
    top_kpis[2].metric("Price (median)", f"{filtered['price'].median():,.0f}")
    top_kpis[3].metric("Depth (median)", f"{filtered['depth'].median():.2f}")

    if filtered.empty:
        st.warning("No rows after filtering. Adjust filters.")
        st.stop()

    # Charts
    if engine == "Seaborn (Matplotlib)":
        fig = make_seaborn_plot(
            filtered,
            clarity_order=clarity_order,
            palette_name=palette_name,
            point_alpha=point_alpha,
            size_min=size_min,
            size_max=size_max,
            fig_w=fig_w,
            fig_h=fig_h,
            log_x=log_x,
            log_y=log_y,
        )
        st.pyplot(fig, use_container_width=True)

        png_bytes = fig_to_png_bytes(fig)
        st.download_button(
            "⬇️ Download chart as PNG",
            data=png_bytes,
            file_name="diamonds_scatter.png",
            mime="image/png",
        )
    else:
        fig = make_plotly_plot(filtered, clarity_order=clarity_order, log_x=log_x, log_y=log_y)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Preview data")
    st.dataframe(filtered.head(50), use_container_width=True)

    st.subheader("Summary by clarity")
    summary = (
        filtered.groupby("clarity")[["carat", "price", "depth"]]
        .agg(["count", "mean", "median", "std"])
        .round(3)
    )
    st.dataframe(summary, use_container_width=True)


if __name__ == "__main__":
    main()
