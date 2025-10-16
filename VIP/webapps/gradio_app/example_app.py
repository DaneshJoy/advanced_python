import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

def load_and_plot(file, x_col, y_col):
    if file is None:
        return "Please upload a CSV file.", None

    try:
        # Read the uploaded CSV
        df = pd.read_csv(file.name)

        # Create a simple plot
        fig, ax = plt.subplots()
        ax.plot(df[x_col], df[y_col], marker='o')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        return f"✅ Loaded {len(df)} rows from CSV.", fig

    except Exception as e:
        return f"❌ Error: {e}", None


# Define UI
with gr.Blocks(title="CSV Plotter") as demo:
    gr.Markdown("## 📊 Simple CSV Plotter")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ⚙️ Upload and Options")

                file_input = gr.File(label="Upload CSV File", file_types=[".csv"])

                # Dynamic column selectors (populated after upload)
                x_col = gr.Dropdown(label="X-axis Column", interactive=True)
                y_col = gr.Dropdown(label="Y-axis Column", interactive=True)

                # Button to trigger plotting
                plot_button = gr.Button("Generate Plot")

        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Status", interactive=False)
            output_plot = gr.Plot(label="Plot Output")

    # When file uploaded → update dropdowns
    def update_dropdowns(file):
        if file is None:
            return gr.update(choices=[]), gr.update(choices=[])
        df = pd.read_csv(file.name)
        cols = list(df.columns)
        return gr.update(choices=cols, value=cols[0]), gr.update(choices=cols, value=cols[1])

    file_input.change(update_dropdowns, inputs=file_input, outputs=[x_col, y_col])
    plot_button.click(load_and_plot, inputs=[file_input, x_col, y_col], outputs=[output_text, output_plot])


# Run app
if __name__ == "__main__":
    demo.launch()

