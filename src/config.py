from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "extention of Z-Alizadeh sani dataset.xlsx"
SHEET_NAME = "Sheet 1 - Table 1"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"

TARGETS = {
    "cad": "Cath",
    "lad": "LAD",
    "lcx": "LCX",
    "rca": "RCA",
}

TARGET_LABELS = {
    "cad": {"positive": "CAD", "negative": "Normal", "name": "Coronary Artery Disease (Overall)"},
    "lad": {"positive": "Stenotic", "negative": "Normal", "name": "Left Anterior Descending (LAD)"},
    "lcx": {"positive": "Stenotic", "negative": "Normal", "name": "Left Circumflex (LCX)"},
    "rca": {"positive": "Stenotic", "negative": "Normal", "name": "Right Coronary Artery (RCA)"},
}

LEAKAGE_COLUMNS = ["Cath", "LAD", "LCX", "RCA"]

# Clinical feature groupings for structured UI presentation
FEATURE_GROUPS = {
    "Demographics & Vitals": ["Age", "Weight", "Length", "Sex", "BMI", "BP", "PR"],
    "Medical History & Risk Factors": [
        "DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Obesity", "CRF", "CVA",
        "Airway disease", "Thyroid Disease", "CHF", "DLP", "Weak Peripheral Pulse"
    ],
    "Symptoms & Examination": [
        "Typical Chest Pain", "Function Class", "Lung rales", "Systolic Murmur",
        "Diastolic Murmur", "Dyspnea", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang"
    ],
    "Electrocardiography (ECG)": [
        "Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression", "BBB"
    ],
    "Laboratory Biomarkers": [
        "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT"
    ],
    "Echocardiography": [
        "EF-TTE", "Region RWMA", "VHD"
    ]
}

# Plain English descriptions for complex medical features
FEATURE_DESCRIPTIONS = {
    "Age": "Patient age in years",
    "Weight": "Weight in kilograms",
    "Length": "Height in centimeters",
    "Sex": "Biological sex (Male / Female)",
    "BMI": "Body Mass Index (kg/m²)",
    "DM": "Diabetes Mellitus history (0 = No, 1 = Yes)",
    "HTN": "Hypertension history (0 = No, 1 = Yes)",
    "Current Smoker": "Active smoker status (0 = No, 1 = Yes)",
    "EX-Smoker": "Former smoker status (0 = No, 1 = Yes)",
    "FH": "Family History of premature coronary artery disease (0 = No, 1 = Yes)",
    "BP": "Blood Pressure (systolic mm Hg)",
    "PR": "Pulse Rate (beats per minute)",
    "Typical Chest Pain": "Substernal chest discomfort precipitated by exertion/stress (0 = No, 1 = Yes)",
    "Function Class": "NYHA heart failure functional classification (1 to 4)",
    "Q Wave": "Pathologic Q-wave on ECG indicating prior myocardial infarction",
    "St Elevation": "ST-segment elevation on ECG indicating acute myocardial injury",
    "St Depression": "ST-segment depression indicating myocardial ischemia",
    "Tinversion": "T-wave inversion on ECG suggesting ischemia or strain",
    "LVH": "Left Ventricular Hypertrophy on ECG",
    "Poor R Progression": "Abnormal R-wave growth across precordial leads V1-V4",
    "BBB": "Bundle Branch Block conduction delay (None, LBBB, RBBB)",
    "FBS": "Fasting Blood Sugar (mg/dL)",
    "CR": "Serum Creatinine level (mg/dL)",
    "TG": "Serum Triglycerides (mg/dL)",
    "LDL": "Low-Density Lipoprotein cholesterol ('bad' cholesterol, mg/dL)",
    "HDL": "High-Density Lipoprotein cholesterol ('good' cholesterol, mg/dL)",
    "BUN": "Blood Urea Nitrogen (mg/dL)",
    "ESR": "Erythrocyte Sedimentation Rate (inflammatory marker, mm/hr)",
    "HB": "Hemoglobin level (g/dL)",
    "EF-TTE": "Ejection Fraction by Transthoracic Echocardiography (%)",
    "Region RWMA": "Regional Wall Motion Abnormality on echocardiography (0-4 segments)",
    "VHD": "Valvular Heart Disease (None, Mild, Moderate, Severe)",
}

MANDATORY_DISCLAIMER = (
    "CardioVision3D is an experimental educational and research prototype. "
    "It is not clinically validated and must not be used to diagnose, exclude, or treat cardiovascular disease. "
    "Predictions are generated from a limited dataset and may be inaccurate. "
    "They are not medical diagnoses or validated clinical risk estimates. "
    "Consult a qualified healthcare professional for medical assessment."
)
