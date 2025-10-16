import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# --- Page Configuration ---
st.set_page_config(page_title="Example Streamlit App", layout="wide")

# --- Main Page ---
st.title("📊 Simple Streamlit Starter App")
st.write("This app demonstrates how to use sidebar, file upload, and data display in Streamlit.")

st.sidebar.title("Navigation Menu")

# File uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Sidebar options

# Plot options
st.sidebar.subheader("Plot Settings")
x_col = st.sidebar.text_input("X-axis column (default: date)", value="date")
y_cols_input = st.sidebar.text_input("Y-axis columns (comma-separated)", value="sales,visitors")
agg = st.sidebar.selectbox("Aggregation (for non-numeric X)", ["None", "mean", "sum", "median"])
group_col = st.sidebar.text_input("Optional group column (e.g., region)", value="")

# DataFrame options
st.sidebar.write("Choose a display option:")
show_head = st.sidebar.checkbox("Show first 5 rows")
show_info = st.sidebar.checkbox("Show basic info")

if uploaded_file is not None:
    # Read the uploaded CSV
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Resolve columns
        y_cols = [c.strip() for c in y_cols_input.split(",") if c.strip()]

        # Basic validations
        missing = [c for c in [x_col] + y_cols if c not in df.columns]
        if missing:
            st.error(f"Column(s) not found: {missing}")
        else:
            plot_df = df.copy()

            # If x is a date-like column, try to parse it
            if pd.api.types.is_string_dtype(plot_df[x_col]):
                try:
                    plot_df[x_col] = pd.to_datetime(plot_df[x_col])
                except Exception:
                    pass

            # Aggregate if requested
            if agg != "None":
                if group_col and group_col in plot_df.columns:
                    plot_df = (
                        plot_df.groupby([x_col, group_col])[y_cols]
                        .agg(agg)
                        .reset_index()
                    )
                else:
                    plot_df = plot_df.groupby(x_col)[y_cols].agg(agg).reset_index()

            # ---- Plot ----
            st.subheader("Line Plot")
            if group_col and group_col in plot_df.columns:
                # One line per group (faceted on legend)
                fig, ax = plt.subplots()
                for g, gdf in plot_df.groupby(group_col):
                    # Sort by x for nicer lines
                    gdf = gdf.sort_values(by=x_col)
                    # Plot each y on same axes
                    for y in y_cols:
                        ax.plot(gdf[x_col], gdf[y], label=f"{g} • {y}")
                ax.set_xlabel(x_col)
                ax.set_ylabel("value")
                ax.legend(loc="best")
                st.pyplot(fig, clear_figure=True)
            else:
                # Single set of lines
                fig, ax = plt.subplots()
                plot_df = plot_df.sort_values(by=x_col)
                for y in y_cols:
                    ax.plot(plot_df[x_col], plot_df[y], label=y)
                ax.set_xlabel(x_col)
                ax.set_ylabel("value")
                ax.legend(loc="best")
                st.pyplot(fig, clear_figure=True)

            # Extra: quick filters
            with st.expander("🔎 Filter data"):
                cols = st.multiselect("Select columns to view", options=list(df.columns), default=[x_col] + y_cols)
                st.dataframe(df[cols].head(50), use_container_width=True)
                
        if show_head:
            st.subheader("Preview (first 5 rows)")
            st.dataframe(df.head())

        if show_info:
            st.subheader("Dataset Information")
            st.write("Shape:", df.shape)
            st.write("Columns:", list(df.columns))
            st.write(df.describe())

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("👈 Please upload a CSV file from the sidebar to get started.")