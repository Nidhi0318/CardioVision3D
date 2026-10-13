# CardioVision3D ❤️

### AI-Powered Cardiovascular Risk Prediction and 3D Heart Visualization

CardioVision3D is a machine learning-based application developed to explore cardiovascular disease indicators using clinical, laboratory, ECG, demographic, and echocardiographic data. It combines predictive modeling with interactive 3D heart visualization to make model outputs easier to explore and understand.

**Project:** CardioVision3D
**Domain:** Healthcare AI · Machine Learning · Explainable AI · 3D Visualization
**Dataset:** Z-Alizadeh Sani cardiovascular dataset
**Purpose:** Research and educational use

---

## 🚀 Key Features

* **Machine Learning:** Train and evaluate models using cardiovascular patient data.
* **Multi-Target Prediction:** Separate prediction pipelines for `Cath`, `LAD`, `LCX`, and `RCA`.
* **Data Preprocessing:** Handle missing values, categorical features, and numerical features.
* **Interactive 3D Heart:** Explore a schematic visualization of the heart and its major coronary arteries.
* **Model Evaluation:** Review model performance using available evaluation metrics.
* **Explainable AI:** Integrate SHAP and LIME to investigate which input features influence predictions.
* **Interactive Dashboard:** Explore patient assessments, model results, and visualizations through Streamlit.

*Note: SHAP/LIME explanations and prediction-linked vessel coloring should be described as implemented features only after they have been integrated and tested in the application.*

## 🧠 How It Works

1. **Data Collection:** Load the cardiovascular dataset.
2. **Preprocessing:** Clean and prepare the input features for model training.
3. **Model Training:** Train machine learning models for the supported prediction targets.
4. **Prediction:** Accept eligible clinical inputs and generate model predictions.
5. **Visualization:** Display relevant results and explore the illustrative 3D heart.
6. **Interpretation:** Use evaluation metrics and, when integrated, explainability methods to examine model behavior.

### Prediction Targets

| Target | Description                                     |
| ------ | ----------------------------------------------- |
| `Cath` | Angiographic category recorded in the dataset   |
| `LAD`  | Left Anterior Descending coronary artery target |
| `LCX`  | Left Circumflex coronary artery target          |
| `RCA`  | Right Coronary Artery target                    |

The exact meaning of each predicted class depends on the dataset's original label encoding and preprocessing configuration.

**Important:** The four target columns must be excluded from the input features when training each prediction model to prevent target leakage.

## 🛠️ Technology Stack

* **Programming Language:** Python
* **Machine Learning:** scikit-learn and compatible ML libraries
* **Data Processing:** pandas, NumPy
* **Web Application:** Streamlit
* **Visualization:** Plotly
* **Explainable AI:** SHAP and LIME, where implemented
* **Testing:** pytest

## 📁 Project Structure

```text
CardioVision3D/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── raw/
│       └── extention of Z-Alizadeh sani dataset.xlsx
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── explain.py
│   └── visualization.py
│
├── models/
├── results/
│   └── metrics.json
│
└── tests/
```

*The structure above reflects the starter project. Your actual repository may contain additional files or folders.*

## ⚙️ Installation and Setup

### Prerequisites

* Python 3.10 or a compatible version supported by the dependencies
* Git
* Visual Studio Code or another Python IDE

### 1. Clone the repository

```bash
git clone https://github.com/Nidhi0318/CardioVision3D.git
cd CardioVision3D
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the models

If the repository's training module is configured as an executable module, run:

```bash
python -m src.train
```

This should generate the model files and evaluation results according to the project's configuration.

### 5. Launch the application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to access the application.

## 📊 Model Evaluation

Model performance should be assessed using appropriate metrics for each prediction target.

Potential evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC, where appropriate
* Confusion matrix

The actual results should be taken from the project's generated evaluation output, such as `results/metrics.json`.

**Do not interpret model performance as evidence of clinical effectiveness.** Results depend on dataset quality, preprocessing, class distribution, model selection, and evaluation methodology.

## 🔍 Explainable AI

CardioVision3D is designed to support model interpretability through techniques such as:

* **SHAP:** Estimates feature contributions to model predictions.
* **LIME:** Explains an individual prediction using a locally interpretable approximation.

When implemented, these explanations can help users investigate which features influenced a model's output. Feature importance and explanations describe model behavior; they do not establish causation or clinical significance.

## ❤️ Interactive 3D Heart Visualization

The application includes an illustrative 3D heart visualization built with Plotly.

The visualization is intended to help users explore the general locations of the LAD, LCX, and RCA coronary arteries. If prediction-linked vessel coloring is enabled, the display should clearly identify the relevant model output and class mapping.

**The 3D model is schematic and is not an anatomically accurate reconstruction or a medical imaging tool.**

## 🧪 Testing

Run the project's test suite from the repository root:

```bash
python -m pytest tests/ -v
```

Check that the required dependencies are installed and that any expected model files or dataset paths are configured before running tests.

## 🔒 Data Privacy and Responsible Use

* Use appropriately authorized datasets.
* Avoid uploading identifiable patient information to public repositories.
* Do not commit credentials, API keys, passwords, or private configuration files.
* Follow the dataset's applicable licensing and usage conditions.
* Validate preprocessing and evaluation procedures before drawing conclusions.

## ⚠️ Medical Disclaimer

CardioVision3D is an experimental research and educational project. It has not been established as a clinically validated diagnostic or treatment system.

Predictions may be inaccurate or biased and must not be used as a substitute for professional medical advice, diagnosis, or treatment. Do not use this application to make real-world clinical decisions.

## 🔮 Future Enhancements

* Improved explainability with SHAP and LIME
* Prediction-linked coronary artery visualization
* Enhanced model comparison and evaluation dashboards
* External validation on suitable independent datasets
* Improved input validation and automated testing
* Secure deployment with appropriate privacy protections

## 👩‍💻 Author and Project

**GitHub Repository:**
https://github.com/Nidhi0318/CardioVision3D

Developed as a machine learning and visualization project exploring the use of AI techniques in cardiovascular research.

---

*CardioVision3D — Exploring cardiovascular data through machine learning and interactive visualization.*
