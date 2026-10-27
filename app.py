
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, rdFingerprintGenerator

st.set_page_config(
    page_title="DeepMol",
    page_icon="🧬",
    layout="wide"
)

# ---------- Load project files ----------

@st.cache_resource
def load_model():
    return joblib.load("deepmol_rf_model.pkl")

@st.cache_data
def load_candidates():
    return pd.read_csv("deepmol_ranked_candidates.csv")

rf_model = load_model()
candidates = load_candidates()

generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)

# ---------- Prediction function ----------

def predict_solubility(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None, None

    fingerprint = generator.GetFingerprint(mol)
    array = np.zeros(2048, dtype=np.float32)

    DataStructs.ConvertToNumpyArray(fingerprint, array)

    prediction = float(
        rf_model.predict(array.reshape(1, -1))[0]
    )

    return prediction, mol, Descriptors.MolWt(mol)


# ---------- Dashboard ----------

st.title("🧬 DeepMol")
st.write(
    "Molecular Property Prediction and "
    "Goal-Directed Candidate Exploration"
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    ["Overview", "Predict Solubility", "Candidate Explorer"]
)

if page == "Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Ranked Candidates", len(candidates))
    col2.metric("Random Forest R²", "0.7175")
    col3.metric("GNN R²", "0.7008")

    st.subheader("Project Pipeline")

    st.markdown("""
    1. Load and preprocess the ESOL dataset.
    2. Convert molecular structures into Morgan fingerprints.
    3. Train Random Forest and Graph Convolutional Network models.
    4. Generate molecular candidates using RDKit transformations.
    5. Check chemical validity and filter candidates.
    6. Rank candidates using predicted solubility.
    """)

    st.subheader("Model Performance")

    performance = pd.DataFrame({
        "Model": ["Random Forest", "GCN"],
        "MAE": [0.8027, 0.8736],
        "RMSE": [1.0939, 1.2012],
        "R²": [0.7175, 0.6594]
    })

    st.dataframe(performance, hide_index=True)

    st.bar_chart(
        performance.set_index("Model")[["MAE", "RMSE"]]
    )
    
    # ---------- Candidate Analytics ----------

    st.divider()
    st.header("Candidate Analytics")

    # Convert columns to numeric in case CSV values are strings
    candidates["Predicted_Solubility"] = pd.to_numeric(
        candidates["Predicted_Solubility"],
        errors="coerce"
    )

    candidates["Molecular_Weight"] = pd.to_numeric(
        candidates["Molecular_Weight"],
        errors="coerce"
    )

    valid_candidates = candidates.dropna(
        subset=["Predicted_Solubility", "Molecular_Weight"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Solubility Distribution")

        solubility_bins = pd.cut(
            valid_candidates["Predicted_Solubility"],
            bins=10
        )

        solubility_counts = (
            solubility_bins.value_counts(sort=False)
            .rename_axis("Solubility Range")
            .reset_index(name="Number of Molecules")
        )

        solubility_counts["Solubility Range"] = (
            solubility_counts["Solubility Range"].astype(str)
        )

        st.bar_chart(
            solubility_counts.set_index("Solubility Range")
        )

    with col2:
        st.subheader("Molecular Weight Distribution")

        weight_bins = pd.cut(
            valid_candidates["Molecular_Weight"],
            bins=10
        )

        weight_counts = (
            weight_bins.value_counts(sort=False)
            .rename_axis("Weight Range")
            .reset_index(name="Number of Molecules")
        )

        weight_counts["Weight Range"] = (
            weight_counts["Weight Range"].astype(str)
        )

        st.bar_chart(
            weight_counts.set_index("Weight Range")
        )

    st.subheader("Candidate Summary")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Valid candidate records",
        len(valid_candidates)
    )

    metric2.metric(
        "Average predicted solubility",
        f"{valid_candidates['Predicted_Solubility'].mean():.3f}"
    )

    metric3.metric(
        "Average molecular weight",
        f"{valid_candidates['Molecular_Weight'].mean():.2f} g/mol"
    )

    # ---------- Molecular Diversity Analysis ----------

    st.divider()
    st.header("Molecular Diversity Analysis")

    st.write(
        "Tanimoto similarity compares molecular fingerprints. "
        "Lower average similarity among sampled candidates "
        "indicates greater structural diversity."
    )

    diversity_smiles = (
        valid_candidates["SMILES"]
        .drop_duplicates()
        .head(100)
        .tolist()
    )

    diversity_fps = []

    for smiles in diversity_smiles:
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            diversity_fps.append(
                generator.GetFingerprint(mol)
            )

    similarities = []

    for i in range(len(diversity_fps)):
        for j in range(i + 1, len(diversity_fps)):
            similarities.append(
                DataStructs.TanimotoSimilarity(
                    diversity_fps[i],
                    diversity_fps[j]
                )
            )

    if similarities:
        avg_similarity = float(np.mean(similarities))
        min_similarity = float(np.min(similarities))
        max_similarity = float(np.max(similarities))

        d1, d2, d3 = st.columns(3)

        d1.metric(
            "Average Pairwise Similarity",
            f"{avg_similarity:.3f}"
        )

        d2.metric(
            "Minimum Similarity",
            f"{min_similarity:.3f}"
        )

        d3.metric(
            "Maximum Similarity",
            f"{max_similarity:.3f}"
        )

        st.caption(
            f"Analysis uses {len(diversity_fps)} unique valid "
            "molecules from the first 100 candidate records. "
            "It is a sample, not the full candidate set."
        )

    else:
        st.warning(
            "Not enough valid molecules to calculate diversity."
        )


    st.caption(
        "Metrics are from the recorded experiments. "
        "The two models used separately created random splits, "
        "so this comparison is preliminary."
    )

elif page == "Predict Solubility":

    st.header("Molecular Solubility Prediction")

    smiles = st.text_input(
        "Enter molecular SMILES",
        value="CCO",
        help="Example: CCO represents ethanol."
    )

    if st.button("Predict Solubility"):

        prediction, mol, mw = predict_solubility(smiles)

        if mol is None:
            st.error("Invalid SMILES. Please try another structure.")

        else:
            left, right = st.columns(2)

            with left:
                st.image(
                    Draw.MolToImage(mol, size=(400, 300)),
                    caption="Input molecular structure"
                )

            with right:
                st.metric(
                    "Predicted log solubility (mol/L)",
                    f"{prediction:.4f}"
                )

                st.metric(
                    "Molecular Weight",
                    f"{mw:.2f} g/mol"
                )

                st.info(
                    "This is a model prediction, not an "
                    "experimentally measured value."
                )

elif page == "Candidate Explorer":

    st.header("Molecular Candidate Explorer")

    min_solubility = st.slider(
        "Minimum predicted log solubility",
        min_value=-10.0,
        max_value=2.0,
        value=-2.0,
        step=0.1
    )

    max_weight = st.slider(
        "Maximum molecular weight",
        min_value=100,
        max_value=800,
        value=400,
        step=10
    )

    filtered = candidates[
        (candidates["Predicted_Solubility"] >= min_solubility)
        & (candidates["Molecular_Weight"] < max_weight)
    ].copy()

    st.write(f"Candidates found: {len(filtered)}")
    st.dataframe(filtered, use_container_width=True)

    if not filtered.empty:

        selected_smiles = st.selectbox(
            "Select a molecule to inspect",
            filtered["SMILES"].tolist()
        )

        mol = Chem.MolFromSmiles(selected_smiles)

        if mol is not None:

            left, right = st.columns(2)

            with left:
                st.image(
                    Draw.MolToImage(mol, size=(400, 300)),
                    caption="Selected molecule"
                )

            with right:
                selected = filtered[
                    filtered["SMILES"] == selected_smiles
                ].iloc[0]

                st.metric(
                    "Predicted Solubility",
                    f"{selected['Predicted_Solubility']:.4f}"
                )

                st.metric(
                    "Molecular Weight",
                    f"{selected['Molecular_Weight']:.2f} g/mol"
                )

                st.code(selected_smiles)

    st.download_button(
        "Download Candidate Results",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="deepmol_filtered_candidates.csv",
        mime="text/csv"
    )

st.divider()

st.caption(
    "DeepMol is an educational computational prototype. "
    "Generated candidates and predicted properties require "
    "independent chemical and experimental validation."
)
