# CardioVision3D — Clinical Scope, Ethics & Limitations

## 1. System Overview & Intended Use
CardioVision3D is an experimental decision-support research prototype designed for non-invasive risk stratification of coronary artery disease (CAD) and regional epicardial stenosis.

**Intended Users:**
- Cardiovascular clinicians and medical informatics researchers.
- Cardiology fellows and trainees studying epicardial perfusion territories.
- AI healthcare researchers evaluating transparent, explainable decision support.

## 2. Mandatory Medical Disclaimer
> **Notice:** CardioVision3D is an experimental educational and research prototype. It is not clinically validated and must not be used to diagnose, exclude, or treat cardiovascular disease. Predictions may be inaccurate and are not validated clinical risk estimates. Consult a qualified healthcare professional for medical assessment.

## 3. Dataset Attribution
The system uses the **Z-Alizadeh Sani Cardiovascular Cohort**:
- **Patients:** 303 patients referred for coronary catheterization.
- **Predictors:** 55 clinical variables spanning demographics, symptoms, 12-lead ECG, blood biomarkers, and echocardiography.
- **Ground Truth:** Invasive coronary angiography (Cath, LAD, LCX, RCA).
- **Citation:** Alizadeh Sani et al., *A data mining approach for diagnosis of coronary artery disease*, Computer Methods and Programs in Biomedicine, 2013.

## 4. Known Clinical Limitations
1. **Sample Size:** 303 patient records is a small clinical cohort. While 5-fold cross-validation and held-out test evaluation demonstrate strong performance, multi-center generalization across diverse demographic populations has not been established.
2. **Referral Pre-Test Bias:** Because all patients in the cohort were clinically indicated for cardiac catheterization, CAD prevalence (71.3%) is significantly higher than in general outpatient screening populations.
3. **Statistical Attribution vs. Biological Causation:** SHAP values represent statistical feature weights within this specific model; they do not establish biological causation.

## 5. Privacy & Data Security
- **100% Offline Local Execution:** All machine learning inferences, SHAP calculations, and 3D visual transformations run locally.
- **Zero Cloud Transmission:** No patient health data is transmitted to external servers or third-party APIs.
- **Strict Leakage Prevention:** Outcome columns (`Cath`, `LAD`, `LCX`, `RCA`) are strictly excluded from prediction inputs.
