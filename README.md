# User Intent Classification

This repository contains a complete machine‑learning project that classifies customer‑support messages into 27 intent categories. The data comes from the **Bitext Customer Support LLM Chatbot Training Dataset**.

## Overview

The project loads the dataset, cleans the raw text, encodes the labels, splits the data into training and test sets (80/20 stratified), extracts TF‑IDF features, trains several classic models (Logistic Regression, Random Forest, Decision Tree), evaluates them, and saves a few visualisation charts (class distribution, accuracy bar chart, confusion matrix, over‑fitting analysis, cross‑validation, etc.).

## Getting Started

1. **Clone the repository** (you already have it on GitHub).
2. **Create a virtual environment** (optional but recommended).
3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the pipeline**:
   ```bash
   python user_intent_classification.py
   ```
   This will generate the PNG charts and a LaTeX report (`project_report.tex`).

## Files

- `user_intent_classification.py` – Main script that performs data loading, preprocessing, model training and evaluation.
- `generate_report.py` – Helper script that builds the LaTeX report from the results.
- `project_report.tex` – LaTeX source for the final PDF report.
- `requirements.txt` – List of Python dependencies.
- `README.md` – You are reading it!

## Results Summary (simple)

- Logistic Regression achieved the highest accuracy among the three models.
- The confusion matrix and accuracy bar chart are saved as `confusion_matrix.png` and `accuracy_bar_chart.png`.
- Over‑fitting analysis shows a small gap between train and test accuracy, indicating a well‑generalised model.

## How to Contribute

If you want to improve the project, feel free to fork the repo, make changes, and submit a pull request. Typical contributions include:
- Adding new models (e.g., SVM, XGBoost).
- Trying deep‑learning approaches such as Transformers.
- Extending the dataset to more languages.

## License

This project is released under the MIT License.

---
*Generated automatically for the User Intent Classification project.*
