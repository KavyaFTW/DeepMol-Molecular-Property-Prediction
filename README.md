# DeepMol: Molecular Property Prediction and Goal-Directed Candidate Exploration

## Overview
DeepMol is a deep learning and machine learning project for predicting molecular solubility and exploring potential candidate molecules. It combines molecular fingerprints, graph-based learning, and cheminformatics to analyze chemical structures.

## Objectives
- Predict molecular solubility from SMILES representations.
- Compare Random Forest and Graph Convolutional Network (GCN) models.
- Generate and rank candidate molecules using predicted properties.
- Analyze molecular weight, predicted solubility, and molecular diversity.
- Provide an interactive dashboard using Streamlit.

## Technologies Used
- Python
- RDKit
- Scikit-learn
- PyTorch
- Streamlit
- Pandas and NumPy
- Matplotlib

## Dataset
The project uses the ESOL (Estimated SOLubility) dataset, which contains molecular structures represented as SMILES strings and their corresponding experimental solubility values.

## Methodology
1. **Data preprocessing:** Load molecular structures and prepare molecular representations.
2. **Random Forest:** Convert molecules into Morgan fingerprints and train a regression model.
3. **GCN:** Represent molecules as graphs and train a graph-based regression model.
4. **Candidate exploration:** Apply predefined chemical transformations to generate candidate structures and filter them using predicted properties.
5. **Molecular diversity:** Calculate pairwise Tanimoto similarity using molecular fingerprints.
6. **Dashboard:** Explore predictions, model metrics, and candidate molecules through a Streamlit interface.

## Model Performance

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Random Forest | 0.8027 | 1.0939 | 0.7175 |
| Graph Convolutional Network | 0.8736 | 1.2012 | 0.6594 |

Lower MAE and RMSE indicate better prediction accuracy, while a higher R² score indicates that the model explains more of the variation in the target values.

## Dashboard Features
- Predict solubility from a SMILES string.
- View molecular structure and molecular weight.
- Explore ranked candidate molecules.
- Filter and download candidate data.
- Compare model performance.
- Analyze molecular properties and structural diversity.

## Installation and Usage

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/DeepMol-Molecular-Property-Prediction.git
cd DeepMol-Molecular-Property-Prediction
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the dashboard:

```bash
python -m streamlit run app.py
```

## Limitations
- Predicted solubility values are estimates and should not be treated as experimental measurements.
- Candidate generation uses predefined chemical transformations; it is not a trained neural molecular-generation model.
- Generated candidates require further chemical validity, synthesizability, safety, and experimental evaluation.
- Model performance depends on the dataset, preprocessing, and train-test split.

## Future Improvements
- Explore additional molecular datasets and properties.
- Improve graph-based model performance.
- Add more rigorous chemical validity and synthesizability filters.
- Evaluate candidates using additional predictive models.

## Disclaimer
This project is intended for academic and research purposes. Predictions and generated candidates require independent validation before practical use.
