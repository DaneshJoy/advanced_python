
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.io import arff
import gradio as gr

OPEN_ICON = "open_eye.png"
CLOSED_ICON = "closed_eye.png"

class EEGStreamer:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.df = None
        self.idx = 0
        self.running = False
        self.window = 200  # number of points to show on the rolling plot
        self.sleep = 0.05  # stream speed (seconds between samples)

    # ---- model/data ----
    def load_model(self, model_file):
        if model_file is None:
            raise gr.Error("Please upload or select a trained model (.pkl).")
        path = model_file.name if hasattr(model_file, "name") else str(model_file)
        self.model = joblib.load(path)
        return "✅ Model loaded."

    def load_data(self, data_file):
        if data_file is None:
            raise gr.Error("Please upload/select the EEG ARFF dataset.")
        path = data_file.name if hasattr(data_file, "name") else str(data_file)
        data = arff.loadarff(path)
        df = pd.DataFrame(data[0])
        # normalize target column name & convert to int
        label_col = "eyeDetection" if "eyeDetection" in df.columns else "eyeState" if "eyeState" in df.columns else None
        if label_col is None:
            raise gr.Error("Couldn't find 'eyeDetection' or 'eyeState' column in the dataset.")
        # bytes -> int
        df[label_col] = df[label_col].apply(lambda x: int(x.decode("utf-8")) if isinstance(x, bytes) else int(x))
        # separate features/labels
        features = df.drop(columns=[label_col], errors="ignore").select_dtypes(include=[np.number])
        df_clean = features.copy()
        df_clean[label_col] = df[label_col].values
        df_clean = df_clean.dropna().reset_index(drop=True)
        self.df = df_clean
        self.feature_names = list(features.columns)
        self.idx = 0
        return f"📄 Data loaded: {len(self.df)} rows, {len(self.feature_names)} features."

    def _plot_segment(self, signal, labels, preds):
        fig, ax = plt.subplots(figsize=(10,3))
        ax.plot(signal, linewidth=1)
        # shade by label (true): green=open(0), red=closed(1)
        for i, y in enumerate(labels):
            color = "green" if y == 0 else "red"
            ax.axvspan(i-0.5, i+0.5, alpha=0.12, color=color)
        # overlay prediction markers
        xs = np.arange(len(preds))
        ax.scatter(xs, signal, s=10, label="signal")
        # optional: mark mismatches
        mism = labels != preds
        ax.scatter(xs[mism], signal[mism], s=25, marker="x")
        ax.set_ylabel("EEG ch0")
        ax.set_title("Streaming EEG (shaded = true label, X = miscls)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # ---- streaming ----
    def stream(self):
        if self.model is None:
            raise gr.Error("Load a model first.")
        if self.df is None:
            raise gr.Error("Load the dataset first.")
        self.running = True
        # roll over when reaching the end
        while self.running:
            if self.idx >= len(self.df):
                self.idx = 0
            row = self.df.iloc[self.idx]
            x = row[self.feature_names].values.reshape(1, -1)
            y_true = int(row.iloc[-1])
            y_pred = int(self.model.predict(x)[0])

            # rolling window slice (last self.window points, use channel 0 for plotting)
            lo = max(0, self.idx - self.window + 1)
            seg = self.df.iloc[lo:self.idx+1]
            signal = seg[self.feature_names[0]].values
            labels = seg.iloc[:,-1].astype(int).values

            # to align preds with signal length, fill with last prediction for window
            preds = np.zeros_like(labels)
            preds[:] = y_pred  # simple: current pred across window (fast); replace with history if desired

            fig = self._plot_segment(signal, labels, preds)

            eye_img = OPEN_ICON if y_pred == 0 else CLOSED_ICON
            status = "👁️ OPEN" if y_pred == 0 else "😴 CLOSED"
            i_text = f"sample: {self.idx} / {len(self.df)}"
            self.idx += 1
            yield fig, eye_img, status, i_text
            plt.close(fig)
            time.sleep(self.sleep)

    def stop(self):
        self.running = False
        return "⏹️ Streaming stopped."

streamer = EEGStreamer()

with gr.Blocks(title="EEG Eye State Streamer") as demo:
    gr.Markdown("""# EEG Eye State Streamer
Stream a dataset through a trained model.""")

    with gr.Row():
        model_file = gr.File(label="Model (.pkl)")
        data_file = gr.File(label="Dataset (.arff)")

    with gr.Row():
        load_model_btn = gr.Button("Load Model", variant="primary")
        load_data_btn  = gr.Button("Load Data", variant="secondary")

    model_status = gr.Markdown("")
    data_status  = gr.Markdown("")

    with gr.Row():
        start_btn = gr.Button("▶️ Start Streaming", variant="primary")
        stop_btn  = gr.Button("⏹️ Stop", variant="stop")

    with gr.Row():
        plot = gr.Plot(label="EEG Signal (rolling)")
        eye = gr.Image(type="filepath", label="Eye State", interactive=False)
    with gr.Row():
        state_text = gr.Markdown("")
        idx_text = gr.Markdown("")

    load_model_btn.click(streamer.load_model, inputs=[model_file], outputs=[model_status])
    load_data_btn.click(streamer.load_data, inputs=[data_file], outputs=[data_status])
    start_btn.click(streamer.stream, inputs=[], outputs=[plot, eye, state_text, idx_text], api_name="stream", queue=True)
    stop_btn.click(streamer.stop, inputs=[], outputs=[state_text])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
