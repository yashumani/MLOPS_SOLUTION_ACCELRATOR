# app_ui.py
import streamlit as st
import pandas as pd
import subprocess
import sys
import os
from pathlib import Path
import time
import requests
import json
import re
import io
import mlflow
import base64

# --- Page Configuration ---
st.set_page_config(page_title="MLOps Model Garden", layout="wide", initial_sidebar_state="expanded")

# --- UI Styling ---
st.markdown("""
<style>
    .stButton>button {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration using Absolute Paths for Robustness ---
SCRIPT_DIR = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000"
MLFLOW_UI_URL = "http://127.0.0.1:5000"
ARTIFACTS_PATH = SCRIPT_DIR / "artifacts"
DATA_PATH = SCRIPT_DIR / "data"

# --- Helper Functions ---
def run_and_log_process(command, log_placeholder):
    """Runs a subprocess and streams its output to a Streamlit container."""
    log_lines = st.session_state.get('log_lines', [])
    log_lines.append("\n" + "="*80)
    log_lines.append(f"          RUNNING COMMAND: {' '.join(command)}")
    log_lines.append("="*80 + "\n")
    log_placeholder.code("\n".join(log_lines), language="log")

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1, universal_newlines=True,
            cwd=SCRIPT_DIR 
        )
        for line in iter(process.stdout.readline, ''):
            log_lines.append(line.strip())
            log_placeholder.code("\n".join(log_lines), language="log")
            time.sleep(0.01)
        process.wait()
        st.session_state.log_lines = log_lines
        return process.returncode
    except Exception as e:
        st.session_state.log_lines.append(f"Failed to execute command: {e}")
        log_placeholder.code("\n".join(st.session_state.log_lines), language="log")
        return 1

def handle_file_upload():
    """This function is called immediately when the file_uploader state changes."""
    if st.session_state.file_uploader is not None:
        try:
            uploaded_file = st.session_state.file_uploader
            DATA_PATH.mkdir(exist_ok=True)
            saved_filepath = DATA_PATH / uploaded_file.name
            with open(saved_filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            df_cols = pd.read_csv(saved_filepath, nrows=0).columns.tolist()
            st.session_state.column_options = df_cols
            st.session_state.saved_filepath = str(saved_filepath)
            
        except Exception as e:
            st.error(f"Could not read columns from file: {e}")
            st.session_state.column_options = []
            st.session_state.saved_filepath = None
    else:
        st.session_state.column_options = []
        st.session_state.saved_filepath = None

def check_api_status():
    """Checks the FastAPI server status and returns loaded models."""
    try:
        response = requests.get(f"{API_URL}/available_models", timeout=2)
        if response.status_code == 200:
            return response.json().get("available_model_aliases", [])
    except requests.exceptions.ConnectionError:
        return None
    return []

# --- Session State Initialization ---
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False
if 'log_lines' not in st.session_state:
    st.session_state.log_lines = ["Pipeline logs will appear here after a run is started."]

# =====================================================================================
# Main UI Layout
# =====================================================================================

st.title("MLOps Model Garden & Experiment Platform 🧪")
st.markdown("An end-to-end platform to **run experiments**, **compare results**, and **serve predictions**.")

tab_run, tab_predict, tab_results, tab_eda = st.tabs([
    "▶️ **Run Experiments**", 
    "🔮 **Make a Prediction**", 
    "📊 **View Experiment Results**", 
    "📈 **View EDA Reports**"
])

# ========================= TAB 1: RUN EXPERIMENTS =========================
with tab_run:
    st.header("Run a New Set of Experiments")
    
    # --- FIX: File uploader is now OUTSIDE the form ---
    uploaded_file = st.file_uploader(
        "1. Upload Dataset (CSV)", 
        type=["csv"], 
        key="file_uploader",
        on_change=handle_file_upload
    )

    with st.form("pipeline_run_form"):
        st.write("Configure and launch a new training run.")
        
        if st.session_state.get('saved_filepath'):
            st.success(f"Dataset selected: `{Path(st.session_state.saved_filepath).name}`. Please configure the options below.")
        else:
            st.warning("Please upload a CSV file to configure a run.")

        col1, col2 = st.columns(2)
        with col1:
            task_type = st.selectbox("2. Select ML Task Type", ("classification", "regression", "clustering"))
        with col2:
            target_column = st.selectbox(
                "3. Select Target Column",
                options=st.session_state.get('column_options', []),
                index=None,
                placeholder="Select a column...",
                disabled=not st.session_state.get('column_options', [])
            )

        n_trials = st.slider("4. Number of Optuna Trials (per model)", 1, 100, 5)

        submitted = st.form_submit_button("🚀 Launch Experiments", type="primary", use_container_width=True)

    st.subheader("Pipeline Execution Log")
    with st.expander("Click to view the full live log", expanded=True):
        log_placeholder = st.code("", language="log")
        if st.button("Clear Logs"):
            st.session_state.log_lines = []
            st.rerun()
    log_placeholder.code("\n".join(st.session_state.log_lines))

    if submitted:
        if not st.session_state.get('saved_filepath') or not target_column:
            st.error("❌ **Error:** Please upload a dataset AND select a target column before launching.")
        else:
            st.session_state.log_lines = []
            st.session_state.pipeline_running = True
            st.rerun()
    
    if st.session_state.pipeline_running:
        with st.status("Running All Experiments...", expanded=True) as status:
            run_command = [
                sys.executable, "run_experiments.py",
                "--input", st.session_state.saved_filepath,
                "--target", target_column,
                "--task_type", task_type,
                "--n_trials_optuna", str(n_trials)
            ]
            return_code = run_and_log_process(run_command, log_placeholder)
            if return_code == 0:
                status.update(label="✅ All Experiments Finished Successfully!", state="complete", expanded=False)
                st.balloons()
            else:
                status.update(label="❌ Experimental Pipeline Failed.", state="error", expanded=True)
        st.session_state.pipeline_running = False
        st.info("Run complete. Check other tabs for results.")
        time.sleep(1)
        st.rerun()

# ========================= TAB 2: MAKE A PREDICTION (API) =========================
with tab_predict:
    st.header("Get a Live Prediction from the API")
    st.info("For this tab to work, you must have the **FastAPI server running** in a separate terminal.", icon="ℹ️")
    st.code("uvicorn model_serving_api:app --reload --port 8000", language="bash")
    
    if st.button("🔄 Check API Status & Refresh Models"):
        st.rerun()

    api_status_placeholder = st.empty()
    available_models = check_api_status()
    if available_models is None:
        api_status_placeholder.error("API server is not running.")
    elif not available_models:
        api_status_placeholder.warning("API is running, but no models are loaded.")
    else:
        api_status_placeholder.success(f"API is running. {len(available_models)} models available across all recipes.")
        st.markdown("---")
        
        model_to_use = st.selectbox("Select a Model Alias to Use", available_models)
        
        recipe_name = Path(model_to_use).parts[0] if '/' in model_to_use or '\\' in model_to_use else ""
        if recipe_name:
            st.info(f"This model was trained with the **'{recipe_name}'** recipe.")
        
        manifest_path = ARTIFACTS_PATH / recipe_name / "prep_manifest.json" if recipe_name else ARTIFACTS_PATH / "prep_manifest.json"

        if manifest_path.exists():
            with open(manifest_path, 'r') as f: manifest = json.load(f)
            raw_data_path = SCRIPT_DIR / manifest.get("input_file")
            
            if raw_data_path.exists():
                df_raw_for_inputs = pd.read_csv(raw_data_path)
                
                raw_feature_names = manifest.get("numerical_features_in_X", []) + manifest.get("categorical_features_in_raw_X", [])
                raw_feature_names = [feat for feat in raw_feature_names if feat in df_raw_for_inputs.columns]

                with st.form("prediction_form"):
                    st.write("**Enter Feature Values for Prediction:**")
                    features_payload = {}
                    for feature in sorted(raw_feature_names):
                        if feature in manifest.get("categorical_features_in_raw_X", []):
                            unique_vals = df_raw_for_inputs[feature].dropna().unique().tolist()
                            if 1 < len(unique_vals) < 50:
                                features_payload[feature] = st.selectbox(f"Value for '{feature}'", sorted([str(val) for val in unique_vals]))
                            else:
                                features_payload[feature] = st.text_input(f"Value for '{feature}' (High Cardinality)", value=str(df_raw_for_inputs[feature].mode()[0]))
                        elif feature in manifest.get("numerical_features_in_X", []):
                            min_val, max_val, mean_val = float(df_raw_for_inputs[feature].min()), float(df_raw_for_inputs[feature].max()), float(df_raw_for_inputs[feature].mean())
                            features_payload[feature] = st.number_input(f"Value for '{feature}'", min_value=min_val, max_value=max_val, value=mean_val, help=f"Valid range: {min_val:.2f} to {max_val:.2f}")
                    
                    predict_button = st.form_submit_button("Get Prediction", type="primary")

                if predict_button:
                    api_payload = {"model_alias": model_to_use, "features": features_payload}
                    with st.spinner("Sending request to API..."):
                        try:
                            prediction_response = requests.post(f"{API_URL}/predict", json=api_payload, timeout=10)
                            if prediction_response.status_code == 200:
                                result = prediction_response.json()
                                st.metric(label=f"Prediction from '{result['model_alias_used']}'", value=f"{result['prediction']:.4f}")
                            else:
                                st.error(f"API Error (HTTP {prediction_response.status_code}):"); st.json(prediction_response.json())
                        except Exception as e:
                            st.error(f"Failed to connect to API: {e}")
            else:
                st.warning(f"Could not find the original dataset file '{raw_data_path}' referenced in the manifest.")
        else:
            st.warning(f"`prep_manifest.json` not found for recipe '{recipe_name}'.")

# ========================= TAB 3: VIEW RESULTS =========================
with tab_results:
    st.header("Compare Experiment Results")
    if st.button("🔄 Refresh Results", key="refresh_results_tab"): st.rerun()

    all_results = []
    recipe_dirs = [d for d in ARTIFACTS_PATH.iterdir() if d.is_dir() and (d / "final_results.json").exists()]

    if not recipe_dirs:
        st.warning("No experiment results found. Please run the pipeline from the 'Run Experiments' tab first.")
    else:
        for recipe_dir in recipe_dirs:
            with open(recipe_dir / "final_results.json", 'r') as f:
                results_data = json.load(f)
                for result in results_data:
                    result['recipe'] = recipe_dir.name
                    all_results.append(result)

        st.subheader("🏆 Overall Best Performing Combination")
        df_all_results = pd.DataFrame([res for res in all_results if res.get('model_alias') != 'OVERALL_BEST_MODEL'])
        if not df_all_results.empty:
            best_combo = df_all_results.loc[df_all_results['score'].idxmax()]
            col1, col2, col3 = st.columns(3)
            col1.metric("Best Recipe", best_combo['recipe'])
            col2.metric("Best Model", best_combo['model_alias'])
            col3.metric(f"Score ({best_combo['primary_metric']})", f"{best_combo['score']:.4f}")
        
        st.subheader("Combined Model Leaderboard")
        if not df_all_results.empty:
            try:
                client = mlflow.tracking.MlflowClient()
                first_run_id = df_all_results['mlflow_run_id'].iloc[0]
                exp_id = client.get_run(first_run_id).info.experiment_id
                df_all_results['MLflow Link'] = df_all_results['mlflow_run_id'].apply(lambda x: f"{MLFLOW_UI_URL}/#/experiments/{exp_id}/runs/{x}")
                st.dataframe(df_all_results, use_container_width=True, hide_index=True,
                             column_config={"mlflow_run_id": None, "score": st.column_config.NumberColumn(format="%.4f"), "MLflow Link": st.column_config.LinkColumn("View Run", display_text="Open ↗️")})
            except Exception as e:
                st.warning(f"Could not build MLflow links. Is MLflow server running? Error: {e}")
                st.dataframe(df_all_results[['recipe', 'model_alias', 'primary_metric', 'score']], use_container_width=True, hide_index=True)
        else:
            st.info("No model results found in any recipe directories.")

# ========================= TAB 4: EDA REPORTS =========================
with tab_eda:
    st.header("Exploratory Data Analysis (EDA) Reports")
    st.info("These reports are generated for each recipe during a pipeline run.")
    
    recipe_dirs_with_eda = [d.name for d in ARTIFACTS_PATH.iterdir() if d.is_dir() and (d / "eda_report.html").exists()]
    if not recipe_dirs_with_eda:
        st.warning("No EDA reports found. Run the experimental pipeline to generate them.")
    else:
        selected_recipe = st.selectbox("Select a Recipe to View its EDA Report", recipe_dirs_with_eda)
        
        eda_html_path = ARTIFACTS_PATH / selected_recipe / "eda_report.html"
        if eda_html_path.exists():
            with open(eda_html_path, 'r', encoding='utf-8') as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)
        
        eda_pdf_path = ARTIFACTS_PATH / selected_recipe / "custom_eda_report.pdf"
        if eda_pdf_path.exists():
            with open(eda_pdf_path, "rb") as pdf_file:
                st.download_button(label="⬇️ Download Custom Plots PDF", data=pdf_file.read(), file_name=f"{selected_recipe}_custom_eda.pdf", mime="application/pdf")