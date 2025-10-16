import io
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
import joblib

st.set_page_config(page_title="EEG Eye State (Segment Viewer)", layout="wide")

st.title("EEG Eye State — Segment Viewer")

with st.sidebar:
    st.header("1) Load Files")
    model_file = st.file_uploader("Model (.pkl)", type=["pkl", "joblib"])
    data_file  = st.file_uploader("Dataset (.arff)", type=["arff"])

@st.cache_data(show_spinner=False)
def load_arff_from_bytes(raw: bytes):
    """
    Streamlit uploads are bytes; ARFF wants text.
    Decode bytes -> text, then read via StringIO.
    """
    # try utf-8, then latin-1 as a safe fallback for older ARFFs
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    data = arff.loadarff(io.StringIO(text))
    df = pd.DataFrame(data[0])
    # normalize target column name & convert to int
    label_col = ("eyeDetection" if "eyeDetection" in df.columns
                 else "eyeState" if "eyeState" in df.columns
                 else None)
    if label_col is None:
        raise ValueError("Couldn't find 'eyeDetection' or 'eyeState' column in the dataset.")
    df[label_col] = df[label_col].apply(
        lambda x: int(x.decode("utf-8")) if isinstance(x, (bytes, bytearray)) else int(x)
    )
    # split features/labels
    features = df.drop(columns=[label_col], errors="ignore").select_dtypes(include=[np.number])
    df_clean = features.copy()
    df_clean[label_col] = df[label_col].values
    df_clean = df_clean.dropna().reset_index(drop=True)
    return df_clean, list(features.columns), label_col

@st.cache_resource(show_spinner=False)
def load_model_from_bytes(raw: bytes):
    """
    joblib can load from a binary buffer.
    """
    return joblib.load(io.BytesIO(raw))

df = feature_names = label_col = model = None

# Use .getvalue() to get raw bytes from the uploader for caching
if data_file is not None:
    try:
        df, feature_names, label_col = load_arff_from_bytes(data_file.getvalue())
        st.sidebar.success(f"Data loaded: {len(df)} rows, {len(feature_names)} features.")
    except Exception as e:
        st.sidebar.error(f"Data load error: {e}")

if model_file is not None:
    try:
        model = load_model_from_bytes(model_file.getvalue())
        st.sidebar.success("Model loaded.")
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")

if data_file is not None and model_file is not None:
    st.header("2) Select a Segment")
if df is None:
    st.info("Upload your ARFF dataset to continue.")
else:
    n = len(df)
    c1, c2, c3 = st.columns(3)
    with c1:
        start_idx = st.number_input("Start index", min_value=0, max_value=max(0, n-1), value=0, step=1)
    with c2:
        seg_len = st.number_input("Segment length", min_value=1, max_value=n, value=min(500, n), step=1)
    with c3:
        use_truth = st.checkbox("Shade by ground-truth labels (if present)", value=True)

    end_idx = int(min(n, start_idx + seg_len))
    seg = df.iloc[int(start_idx):end_idx].copy()

    if seg.empty:
        st.warning("Selected segment is empty. Adjust start/length.")
    else:
        st.subheader("Segment summary")
        st.write(f"Rows: **{len(seg)}** &nbsp;&nbsp; Range: **{start_idx}–{end_idx-1}**")

        # Predictions if model is available
        preds = None
        if model is not None:
            X = seg[feature_names].values
            try:
                preds = model.predict(X).astype(int)
            except Exception as e:
                st.error(f"Prediction error: {e}")

        # Plot channel-0 with markers for predictions
        fig, ax = plt.subplots(figsize=(12, 3))
        signal = seg[feature_names[0]].values
        ax.plot(signal, linewidth=1)

        xs = np.arange(len(signal))

        # Shade truth if requested
        if use_truth and label_col in seg.columns:
            y = seg[label_col].astype(int).values
            for i, yv in enumerate(y):
                color = "green" if yv == 0 else "red"
                ax.axvspan(i-0.5, i+0.5, alpha=0.12, color=color, linewidth=0)

        # Overlay prediction markers
        if preds is not None:
            open_idx = xs[preds == 0]
            closed_idx = xs[preds == 1]
            ax.scatter(open_idx, signal[open_idx], s=20, label="Pred: Open")
            ax.scatter(closed_idx, signal[closed_idx], s=20, marker="x", label="Pred: Closed")

        ax.set_title("EEG Channel 0 — Selected Segment")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # Badges / counts
        cols = st.columns(3)
        if preds is not None:
            open_count = int((preds == 0).sum())
            closed_count = int((preds == 1).sum())
            cols[0].metric("Pred Open", open_count)
            cols[1].metric("Pred Closed", closed_count)
        else:
            cols[0].write("Upload a model to see predictions.")
