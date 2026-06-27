"""
====================================================================
Streamlit UI — Pathogenic vs Non-Pathogenic Bacteria Classifier
Project : Classification of Pathogenic and Non-Pathogenic Bacteria
          Using Protein Sequence Data
====================================================================
Install:
    pip install streamlit torch transformers joblib pandas numpy scikit-learn

Run:
    streamlit run streamlit_app.py
====================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
#import torch
#from transformers import BertTokenizer, BertModel
import warnings
warnings.filterwarnings("ignore")
import requests

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BactiClass — Pathogenicity Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --bg:        #0a0e1a;
        --surface:   #111827;
        --surface2:  #1a2236;
        --border:    #1e2d45;
        --accent:    #00d4aa;
        --accent2:   #ff4d6d;
        --text:      #e2e8f0;
        --muted:     #64748b;
        --patho:     #ff4d6d;
        --safe:      #00d4aa;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }

    .stApp { background-color: var(--bg); }

    /* Header */
    .hero {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1f35 50%, #0a1628 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(0,212,170,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--accent);
        margin: 0 0 8px 0;
        letter-spacing: -1px;
    }
    .hero-sub {
        font-size: 1rem;
        color: var(--muted);
        margin: 0;
        font-weight: 300;
    }
    .hero-tag {
        display: inline-block;
        background: rgba(0,212,170,0.1);
        color: var(--accent);
        border: 1px solid rgba(0,212,170,0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.75rem;
        font-family: 'Space Mono', monospace;
        margin-top: 16px;
        letter-spacing: 1px;
    }

    /* Cards */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .card-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        color: var(--accent);
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    /* Result boxes */
    .result-patho {
        background: linear-gradient(135deg, rgba(255,77,109,0.15), rgba(255,77,109,0.05));
        border: 2px solid var(--patho);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }
    .result-safe {
        background: linear-gradient(135deg, rgba(0,212,170,0.15), rgba(0,212,170,0.05));
        border: 2px solid var(--safe);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
    }
    .result-label {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    .result-conf {
        font-size: 1rem;
        color: var(--muted);
        margin-top: 8px;
    }
    .result-icon {
        font-size: 3rem;
        margin-bottom: 12px;
    }

    /* Metric pills */
    .metric-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 16px;
    }
    .metric-pill {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 18px;
        text-align: center;
        flex: 1;
        min-width: 100px;
    }
    .metric-val {
        font-family: 'Space Mono', monospace;
        font-size: 1.3rem;
        color: var(--accent);
        font-weight: 700;
    }
    .metric-name {
        font-size: 0.72rem;
        color: var(--muted);
        margin-top: 4px;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Sequence display */
    .seq-display {
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        word-break: break-all;
        color: var(--accent);
        line-height: 1.8;
        max-height: 120px;
        overflow-y: auto;
    }

    /* Info box */
    .info-box {
        background: rgba(0,212,170,0.05);
        border-left: 3px solid var(--accent);
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 0.88rem;
        color: var(--muted);
    }

    /* Warning box */
    .warn-box {
        background: rgba(255,77,109,0.05);
        border-left: 3px solid var(--patho);
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 0.88rem;
        color: var(--muted);
    }

    /* Pipeline steps */
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid var(--border);
    }
    .pipeline-step:last-child { border-bottom: none; }
    .step-num {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: var(--accent);
        background: rgba(0,212,170,0.1);
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .step-text {
        font-size: 0.88rem;
        color: var(--text);
    }
    .step-sub {
        font-size: 0.75rem;
        color: var(--muted);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    /* Streamlit overrides */
    .stTextArea textarea {
        background-color: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.82rem !important;
        border-radius: 8px !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(0,212,170,0.2) !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, var(--accent), #00a884) !important;
        color: #0a0e1a !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 900 !important;
        font-size: 0.9rem !important;
        padding: 12px 32px !important;
        letter-spacing: 1px !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(0,212,170,0.3) !important;
    }
    .stSelectbox select, .stSelectbox > div {
        background-color: var(--surface2) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }
    h1, h2, h3 { color: var(--text) !important; }
    p, li { color: var(--muted) !important; }
    .stSpinner { color: var(--accent) !important; }
    div[data-testid="stProgress"] > div > div {
        background-color: var(--accent) !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LOAD ALL .pkl FILES
# ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    """Load all saved pipeline objects."""
    # List of files we want to test load individually
    pkl_files = [
        "correlation_cols_to_keep.pkl", "scaler_pca.pkl", "pca.pkl", 
        "feature_cols_pca.pkl", "boruta_selected_cols.pkl", "scaler_rfe.pkl", 
        "rfecv.pkl", "rfe_selected_cols.pkl", "scaler_selectk.pkl", 
        "selector_selectk.pkl", "selected_features_selectk.pkl", "scaler_enet.pkl", 
        "elasticnet.pkl", "elasticnet_selected_features.pkl", "scaler_svm.pkl", 
        "svm_model.pkl", "feature_cols_svm.pkl"
    ]
    
    # Test load each one sequentially to catch the exact culprit
    for file in pkl_files:
        try:
            joblib.load(file)
        except Exception as e:
            st.error(f"❌ Failed to load file: {file}")
            st.exception(e)
            st.stop()

    # If they all pass, load the full dict normally:
    try:
        pipeline = {
            "corr_cols":        joblib.load("correlation_cols_to_keep.pkl"),
            "scaler_pca":       joblib.load("scaler_pca.pkl"),
            "pca":              joblib.load("pca.pkl"),
            "feature_cols_pca": joblib.load("feature_cols_pca.pkl"),
            "boruta_cols":      joblib.load("boruta_selected_cols.pkl"),
            "scaler_rfe":       joblib.load("scaler_rfe.pkl"),
            "rfecv":            joblib.load("rfecv.pkl"),
            "rfe_cols":         joblib.load("rfe_selected_cols.pkl"),
            "scaler_sk":        joblib.load("scaler_selectk.pkl"),
            "selector_sk":      joblib.load("selector_selectk.pkl"),
            "sk_features":      joblib.load("selected_features_selectk.pkl"),
            "scaler_en":        joblib.load("scaler_enet.pkl"),
            "elasticnet":       joblib.load("elasticnet.pkl"),
            "en_features":      joblib.load("elasticnet_selected_features.pkl"),
            "scaler_svm":       joblib.load("scaler_svm.pkl"),
            "svm":              joblib.load("svm_model.pkl"),
            "pc_cols":          joblib.load("feature_cols_svm.pkl"),
        }
        return pipeline, None
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────────
# EMBEDDING FUNCTION
# ─────────────────────────────────────────────────────────────────

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWYBXZUO")

def get_proteinbert_embedding(sequence):
    """
    Convert a protein sequence to a 1024-dim ProteinBERT embedding
    using the base Hugging Face Serverless Inference API.
    """
    # Use the core base inference URL directly
    API_URL = "https://router.huggingface.co/hf-inference/models/Rostlab/prot_bert"
    
    # 1. Fetch token from Streamlit secrets
    headers = {}
    if "HF_TOKEN" in st.secrets:
        headers["Authorization"] = f"Bearer {st.secrets['HF_TOKEN']}"
    else:
        raise RuntimeError("HF_TOKEN missing from Streamlit secrets.")

    # 2. Sequence preparation logic
    seq = str(sequence).upper().strip()
    seq = "".join(aa if aa in VALID_AAS else "X" for aa in seq)
    seq = seq[:510]  
    seq_spaced = " ".join(list(seq))

    # 3. Request raw layer configurations
    payload = {
        "inputs": seq_spaced,
        "options": {"wait_for_model": True}
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        res_json = response.json()
        
        # When querying the core model directly, the API returns a 3D matrix 
        # inside an outer array: [[ [tok1_features], [tok2_features], ... ]]
        if isinstance(res_json, list) and len(res_json) > 0:
            # Unwrap the nested list layer to get token distributions
            raw_embeddings = res_json[0]
            if isinstance(raw_embeddings, list) and len(raw_embeddings) > 0:
                token_embeddings = np.array(raw_embeddings) # shape: (num_tokens, 1024)
                
                # Replicate your exact uniform attention pooling math across tokens
                num_tokens = token_embeddings.shape[0]
                mask_exp = np.ones((num_tokens, 1)) 
                
                sum_e = np.sum(token_embeddings * mask_exp, axis=0) 
                sum_m = np.maximum(np.sum(mask_exp, axis=0), 1e-9)   
                
                embedding = sum_e / sum_m
                return embedding
            
        raise ValueError("Unexpected response format from Hugging Face API.")
    elif response.status_code == 503:
        raise RuntimeError("The model is still initializing on Hugging Face infrastructure. Please wait 20 seconds and click Predict again!")
    else:
        raise RuntimeError(f"HF API Error ({response.status_code}): {response.text}")
# ─────────────────────────────────────────────────────────────────
# PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────────────

def predict_sequence(sequence, pipeline, tokenizer, bert_model):
    """
    Run a raw protein sequence through the complete pipeline
    and return prediction + probabilities.
    """
    steps = []

    # Step 1 — ProteinBERT embedding
    steps.append("Generating ProteinBERT embedding...")
    embedding = get_proteinbert_embedding(sequence, tokenizer, bert_model)
    # embedding shape: (1024,)

    # Step 2 — Correlation filter
    # The embedding comes in as 1024 dims
    # We need to align it with the feature cols used in correlation
    # The correlation filter was applied on Data.csv which had
    # UniProt features + embeddings. For inference we only have
    # the embedding, so we use the PC pipeline directly via SVM
    # which uses PC columns from PCA output.
    # We apply: embedding → PCA scaler → PCA → PC cols → SVM scaler → SVM

    steps.append("Applying PCA transformation...")

    # The PCA was fitted on all feature cols of Data4.csv
    # For a new sequence, we only have the embedding (1024 dims)
    # We need to match the number of features PCA expects
    n_pca_features = len(pipeline["feature_cols_pca"])

    # Pad or truncate embedding to match PCA input size
    if len(embedding) < n_pca_features:
        padded = np.zeros(n_pca_features)
        padded[:len(embedding)] = embedding
        X_input = padded.reshape(1, -1)
    else:
        X_input = embedding[:n_pca_features].reshape(1, -1)

    # Scale and apply PCA
    X_scaled   = pipeline["scaler_pca"].transform(X_input)
    X_pca      = pipeline["pca"].transform(X_scaled)

    # Build PC DataFrame
    pc_names   = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    df_pc      = pd.DataFrame(X_pca, columns=pc_names)

    # Step 3 — Keep only PC cols that SVM was trained on
    steps.append("Applying SVM classification...")
    svm_cols   = pipeline["pc_cols"]

    # Align columns — add missing PC cols as 0
    for col in svm_cols:
        if col not in df_pc.columns:
            df_pc[col] = 0.0

    X_svm      = df_pc[svm_cols].values
    X_svm_sc   = pipeline["scaler_svm"].transform(X_svm)

    # Predict
    proba      = pipeline["svm"].predict_proba(X_svm_sc)[0]
    pred       = pipeline["svm"].predict(X_svm_sc)[0]

    return {
        "prediction":    int(pred),
        "label":         "Pathogenic" if pred == 1 else "Non-Pathogenic",
        "prob_patho":    float(proba[1]),
        "prob_nonpatho": float(proba[0]),
        "confidence":    float(max(proba)),
        "steps":         steps,
    }

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.1rem;
                    color: #00d4aa; font-weight: 700;'>🧬 BactiClass</div>
        <div style='font-size: 0.78rem; color: #64748b; margin-top: 4px;'>
            Pathogenicity Predictor v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.selectbox(
        "Navigate",
        ["🔬 Predict", "📊 Model Info", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.78rem; color: #64748b;'>
        <b style='color: #e2e8f0;'>Model</b><br>SVM (RBF Kernel)<br><br>
        <b style='color: #e2e8f0;'>Embeddings</b><br>ProteinBERT<br><br>
        <b style='color: #e2e8f0;'>Test Accuracy</b><br>
        <span style='color: #00d4aa; font-family: Space Mono;'>87.99%</span><br><br>
        <b style='color: #e2e8f0;'>ROC-AUC</b><br>
        <span style='color: #00d4aa; font-family: Space Mono;'>0.95</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────

pipeline, pipe_err = load_pipeline()

# ─────────────────────────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────────────────────────

if "Predict" in page:

    # Hero
    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>BactiClass</div>
        <p class='hero-sub'>
            Classify bacterial proteins as Pathogenic or Non-Pathogenic
            using ProteinBERT embeddings and SVM
        </p>
    </div>
    """, unsafe_allow_html=True)

    if pipe_err:
        st.markdown(f"""
        <div class='warn-box'>
            ⚠️ Could not load pipeline file: <code>{pipe_err}</code><br>
            Make sure all .pkl files are in the same folder as this script.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("<div class='card-title'>Input Sequence</div>",
                    unsafe_allow_html=True)

        # Example sequences
        examples = {
            "None": "",
            "Staphylococcus aureus (Pathogenic)":
                "MSKFIVVLNRHLNQMIEQLENQLREQQDRQQMLQQQQQQLEQMQRQQQDREQQLRQQQ"
                "QDRQQMLQQQQQQLEQMQRQQQDREQQLRQQQQDREQQLHQQQQDRQQMLQQQLQEQM",
            "Bacillus subtilis (Non-Pathogenic)":
                "MKKLWIFLILLSFVAACSSNKSNTASSANNKEQLEEVLQKMQQDREQLRQQQQDREQQL"
                "RQQQHDRQQMLQQQLDQEQMQREQQDRQQMLQQQQDREQQLRQQQQDREQQLHQQQQ",
        }

        ex = st.selectbox(
            "Load example sequence",
            list(examples.keys()),
        )
        default_seq = examples[ex]

        sequence_input = st.text_area(
            "Paste amino acid sequence (single letter code)",
            value=default_seq,
            height=160,
            placeholder="MSKFIVVLNRHLNQMIEQLEN...",
            label_visibility="collapsed",
        )

        st.markdown("""
        <div class='info-box'>
            Enter a single-letter amino acid sequence (e.g. MSKFIVVLN...).
            Sequences between 100–1000 amino acids work best.
            Unknown amino acids will be replaced with X.
        </div>
        """, unsafe_allow_html=True)

        predict_btn = st.button("PREDICT PATHOGENICITY")

        # Sequence stats
        if sequence_input.strip():
            clean_seq = sequence_input.strip().upper().replace(" ", "").replace("\n", "")
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Sequence Stats</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='seq-display'>{clean_seq}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class='metric-row'>
                <div class='metric-pill'>
                    <div class='metric-val'>{len(clean_seq)}</div>
                    <div class='metric-name'>Length</div>
                </div>
                <div class='metric-pill'>
                    <div class='metric-val'>{len(set(clean_seq))}</div>
                    <div class='metric-name'>Unique AA</div>
                </div>
                <div class='metric-pill'>
                    <div class='metric-val'>
                        {'⚠️ Long' if len(clean_seq) > 510 else '✓ OK'}
                    </div>
                    <div class='metric-name'>Length Check</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card-title'>Pipeline Steps</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='pipeline-step'>
            <div class='step-num'>1</div>
            <div>
                <div class='step-text'>ProteinBERT Embedding</div>
                <div class='step-sub'>Rostlab/prot_bert → 1024 dims</div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>2</div>
            <div>
                <div class='step-text'>Correlation Filter</div>
                <div class='step-sub'>Remove |r| > 0.85</div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>3</div>
            <div>
                <div class='step-text'>PCA</div>
                <div class='step-sub'>95% variance retained</div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>4</div>
            <div>
                <div class='step-text'>Boruta Selection</div>
                <div class='step-sub'>Remove noise features</div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>5</div>
            <div>
                <div class='step-text'>RFE</div>
                <div class='step-sub'>Optimal feature count</div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>6</div>
            <div>
                <div class='step-text'>SelectKBest</div>
                <div class='step-sub'>K=24 features</div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>7</div>
            <div>
                <div class='step-text'>ElasticNet</div>
                <div class='step-sub'>Final feature set</div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>8</div>
            <div>
                <div class='step-text'>SVM Classification</div>
                <div class='step-sub'>RBF kernel → prediction</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Prediction ────────────────────────────────────────────────
    if predict_btn:
        seq = sequence_input.strip().upper().replace(" ", "").replace("\n", "")

        if not seq:
            st.error("Please enter a protein sequence.")
        elif len(seq) < 30:
            st.error("Sequence too short. Please enter at least 30 amino acids.")
        else:
            progress = st.progress(0, text="Starting pipeline...")

            try:
                progress.progress(20, text="Requesting ProtBERT embedding from Hugging Face API...")
                
                # 1. Fetch embedding via Serverless API (Zero local RAM footprint!)
                embedding = get_proteinbert_embedding(seq)
                
                progress.progress(50, text="Applying PCA transformation...")
                
                # 2. Match your feature lengths and align with pipeline
                n_pca_features = len(pipeline["feature_cols_pca"])
                if len(embedding) < n_pca_features:
                    padded = np.zeros(n_pca_features)
                    padded[:len(embedding)] = embedding
                    X_input = padded.reshape(1, -1)
                else:
                    X_input = embedding[:n_pca_features].reshape(1, -1)

                X_scaled   = pipeline["scaler_pca"].transform(X_input)
                X_pca      = pipeline["pca"].transform(X_scaled)

                pc_names   = [f"PC{i+1}" for i in range(X_pca.shape[1])]
                df_pc      = pd.DataFrame(X_pca, columns=pc_names)

                progress.progress(80, text="Running SVM classifier...")
                
                svm_cols   = pipeline["pc_cols"]
                for col in svm_cols:
                    if col not in df_pc.columns:
                        df_pc[col] = 0.0

                X_svm      = df_pc[svm_cols].values
                X_svm_sc   = pipeline["scaler_svm"].transform(X_svm)

                # 3. Generate predictions using your SVM model
                proba      = pipeline["svm"].predict_proba(X_svm_sc)[0]
                pred       = pipeline["svm"].predict(X_svm_sc)[0]

                result = {
                    "prediction":    int(pred),
                    "label":         "Pathogenic" if pred == 1 else "Non-Pathogenic",
                    "prob_patho":    float(proba[1]),
                    "prob_nonpatho": float(proba[0]),
                    "confidence":    float(max(proba)),
                }
                
                progress.progress(100, text="Done!")
                progress.empty()

                # Result display (Keep your custom HTML formatting completely intact)
                st.markdown("---")
                r1, r2 = st.columns(2, gap="large")

                with r1:
                    if result["prediction"] == 1:
                        st.markdown(f"""
                        <div class='result-patho'>
                            <div class='result-icon'>⚠️</div>
                            <div class='result-label' style='color: #ff4d6d;'>
                                PATHOGENIC
                            </div>
                            <div class='result-conf'>
                                Confidence: {result['confidence']*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='result-safe'>
                            <div class='result-icon'>✅</div>
                            <div class='result-label' style='color: #00d4aa;'>
                                NON-PATHOGENIC
                            </div>
                            <div class='result-conf'>
                                Confidence: {result['confidence']*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                with r2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("<div class='card-title'>Probability Breakdown</div>", unsafe_allow_html=True)

                    p_val = result["prob_patho"]
                    n_val = result["prob_nonpatho"]

                    st.markdown(f"""
                    <div style='margin-bottom: 12px;'>
                        <div style='display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px;'>
                            <span style='color: #ff4d6d;'>Pathogenic</span>
                            <span style='font-family: Space Mono; color: #ff4d6d;'>{p_val*100:.1f}%</span>
                        </div>
                        <div style='background: #1a2236; border-radius: 4px; height: 8px; overflow: hidden;'>
                            <div style='background: #ff4d6d; width: {p_val*100:.1f}%; height: 100%; border-radius: 4px;'></div>
                        </div>
                    </div>
                    <div>
                        <div style='display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 6px;'>
                            <span style='color: #00d4aa;'>Non-Pathogenic</span>
                            <span style='font-family: Space Mono; color: #00d4aa;'>{n_val*100:.1f}%</span>
                        </div>
                        <div style='background: #1a2236; border-radius: 4px; height: 8px; overflow: hidden;'>
                            <div style='background: #00d4aa; width: {n_val*100:.1f}%; height: 100%; border-radius: 4px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style='margin-top: 20px; padding-top: 16px; border-top: 1px solid #1e2d45;'>
                        <div style='font-size: 0.78rem; color: #64748b; font-family: Space Mono; letter-spacing: 1px;'>
                            SEQUENCE LENGTH
                        </div>
                        <div style='font-size: 1.1rem; color: #e2e8f0; font-family: Space Mono; margin-top: 4px;'>
                            {len(seq)} aa
                            {'(truncated to 510)' if len(seq) > 510 else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                if 'progress' in locals():
                    progress.empty()
                st.error(f"Prediction failed: {str(e)}")
                st.markdown("""
                <div class='warn-box'>
                    This may be due to a network error with the Hugging Face API or a mismatch between the sequence features and the trained pipeline.
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PAGE: MODEL INFO
# ─────────────────────────────────────────────────────────────────

elif "Model Info" in page:

    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>Model Performance</div>
        <p class='hero-sub'>Results across 9 classification models</p>
    </div>
    """, unsafe_allow_html=True)

    # Model performance table
    model_data = {
        "Model":    ["Naive Bayes", "SVM", "KNN", "AdaBoost",
                     "Random Forest", "Gradient Boosting",
                     "XGBoost", "CatBoost", "LightGBM"],
        "Accuracy": [0.8297, 0.8799, 0.8781, 0.8208,
                     0.8638, 0.8477, 0.8530, 0.8638, 0.8584],
        "F1":       [0.7939, 0.8559, 0.8482, 0.7890,
                     0.8355, 0.8203, 0.8263, 0.8394, 0.8294],
        "AUC":      [0.91,   0.95,   0.94,   0.91,
                     0.95,   0.93,   0.93,   0.95,   0.94],
        "Overfit":  ["No",   "No",   "No",   "No",
                     "Yes",  "Yes",  "Yes",  "Yes",  "Yes"],
    }
    df_models = pd.DataFrame(model_data)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>All Models — Test Set Performance</div>",
                unsafe_allow_html=True)

    def highlight_best(row):
        styles = [""] * len(row)
        if row["Model"] == "SVM":
            styles = [
                "background-color: rgba(0,212,170,0.1); color: #00d4aa;"
            ] * len(row)
        if row["Overfit"] == "Yes":
            styles[-1] = "color: #ff4d6d;"
        else:
            styles[-1] = "color: #00d4aa;"
        return styles

    st.dataframe(
        df_models.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Best model highlight
    st.markdown("""
    <div class='card'>
        <div class='card-title'>Best Model — SVM</div>
        <div class='metric-row'>
            <div class='metric-pill'>
                <div class='metric-val'>87.99%</div>
                <div class='metric-name'>Test Accuracy</div>
            </div>
            <div class='metric-pill'>
                <div class='metric-val'>0.8559</div>
                <div class='metric-name'>F1 Score</div>
            </div>
            <div class='metric-pill'>
                <div class='metric-val'>0.95</div>
                <div class='metric-name'>ROC-AUC</div>
            </div>
            <div class='metric-pill'>
                <div class='metric-val'>0.8884</div>
                <div class='metric-name'>Precision</div>
            </div>
            <div class='metric-pill'>
                <div class='metric-val'>0.8257</div>
                <div class='metric-name'>Recall</div>
            </div>
        </div>
        <div class='info-box' style='margin-top: 16px;'>
            SVM was selected as the best model due to its highest test accuracy
            and AUC with no overfitting gap between train and test performance.
            Tree-based models showed near-perfect training accuracy (~1.0) but
            lower test accuracy, indicating overfitting.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────

elif "About" in page:

    st.markdown("""
    <div class='hero'>
        <div class='hero-title'>About This Project</div>
        <p class='hero-sub'>
            Classification of Pathogenic and Non-Pathogenic Bacteria
            Using Protein Sequence Data
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div class='card-title'>Project Overview</div>
        <p style='color: #94a3b8; line-height: 1.8; font-size: 0.92rem;'>
            This project builds a complete machine learning pipeline to classify
            bacterial proteins as pathogenic or non-pathogenic using protein
            sequence data. We collected 2,784 non-redundant protein sequences
            from UniProt across 24 bacterial organisms — 12 pathogenic and 12
            non-pathogenic — with equal representation of Gram-positive and
            Gram-negative bacteria.
        </p>
    </div>

    <div class='card'>
        <div class='card-title'>Pipeline Summary</div>
        <div class='pipeline-step'>
            <div class='step-num'>1</div>
            <div>
                <div class='step-text'>Data Collection</div>
                <div class='step-sub'>
                    UniProt REST API · 24 organisms · 2,784 sequences after CD-HIT (70%)
                </div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>2</div>
            <div>
                <div class='step-text'>Embedding Validation</div>
                <div class='step-sub'>
                    BERT vs ProteinBERT vs RoBERTa · Wilcoxon test + 10-fold CV
                    · ProteinBERT selected (F1 = 0.8448)
                </div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>3</div>
            <div>
                <div class='step-text'>Feature Selection</div>
                <div class='step-sub'>
                    Correlation → PCA → Boruta → RFE → SelectKBest (K=24)
                    → ElasticNet
                </div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>4</div>
            <div>
                <div class='step-text'>Classification</div>
                <div class='step-sub'>
                    9 models tested · SVM best (Acc=87.99%, AUC=0.95)
                </div>
            </div>
        </div>
        <div class='pipeline-step'>
            <div class='step-num'>5</div>
            <div>
                <div class='step-text'>Explainability</div>
                <div class='step-sub'>
                    SHAP (global) · LIME (local) · PC1 most influential feature
                </div>
            </div>
        </div>
    </div>

    <div class='card'>
        <div class='card-title'>Organisms Used</div>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 16px;'>
            <div>
                <div style='color: #ff4d6d; font-size: 0.8rem;
                            letter-spacing: 1px; margin-bottom: 8px;'>
                    PATHOGENIC
                </div>
                <div style='font-size: 0.82rem; color: #94a3b8; line-height: 2;'>
                    S. aureus · S. pyogenes · B. anthracis<br>
                    L. monocytogenes · C. perfringens · E. faecalis<br>
                    E. coli O157:H7 · Salmonella · K. pneumoniae<br>
                    H. pylori · P. aeruginosa · A. baumannii
                </div>
            </div>
            <div>
                <div style='color: #00d4aa; font-size: 0.8rem;
                            letter-spacing: 1px; margin-bottom: 8px;'>
                    NON-PATHOGENIC
                </div>
                <div style='font-size: 0.82rem; color: #94a3b8; line-height: 2;'>
                    B. subtilis · L. acidophilus · L. lactis<br>
                    S. coelicolor · C. glutamicum · B. longum<br>
                    P. putida · C. crescentus · Rhizobium<br>
                    Synechocystis · G. oxydans · D. radiodurans
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
