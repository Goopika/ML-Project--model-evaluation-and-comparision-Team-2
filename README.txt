MODEL EVALUATION & COMPARISON STUDIO — UPGRADE

CONTENTS
- app.py: upgraded Streamlit application
- requirements.txt: required Python packages

IMPORTANT
Keep a backup of your current app.py before replacing it. This upgrade uses your existing app.py as the starting point and preserves the built-in datasets, custom tabular upload, classification/regression workflows, model comparison, confusion matrix, and CSV comparison report. It adds stronger validation, dashboard metrics, macro F1, training time, per-class report, binary ROC curve, prediction form, test prediction CSV, and PDF report.

RUN IN VS CODE (Windows)
1. Copy app.py and requirements.txt into your project folder (the folder that contains creditcard.csv, if using that dataset).
2. Open VS Code terminal in that folder.
3. Activate your existing virtual environment if you have one:
   .venv\Scripts\activate
4. Install packages:
   python -m pip install -r requirements.txt
5. Run the syntax check:
   python -m py_compile app.py
6. Start the app:
   python -m streamlit run app.py
7. Streamlit will show a local URL, usually http://localhost:8501.

DATASET NOTES
- Credit Card Fraud expects creditcard.csv beside app.py with a target column named Class.
- If the file is elsewhere, choose Upload Your Own Dataset in the sidebar.
- Supported upload formats: CSV, XLSX/XLS, JSON, TSV/TXT, and Parquet.
- Student Performance downloads its built-in sample from UCI and needs internet access.
- For very large datasets, select a row limit to reduce training time. Metrics from a sampled dataset may differ from metrics on the full dataset.

HOW TO USE
1. Select or upload a dataset.
2. Choose the target column.
3. Choose Classification or Regression.
4. Select the test split and row limit.
5. Click Train and Compare Models.
6. Use the tabs for leaderboard, detailed evaluation, single-record prediction, and reports.

TROUBLESHOOTING
- If a package is missing, run: python -m pip install -r requirements.txt
- If Streamlit is not recognized, run: python -m streamlit run app.py
- If Excel/Parquet uploads fail, ensure openpyxl/pyarrow were installed from requirements.txt.
- Always verify the target column before interpreting results. Credit Card Fraud is imbalanced; do not judge it by accuracy alone.
