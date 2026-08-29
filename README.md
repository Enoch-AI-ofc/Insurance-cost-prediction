# Medical Insurance Cost Predictor

**Goal:** An end-to-end Machine Learning regression project that predicts the medical insurance costs of individuals based on their demographic and lifestyle data.

This project serves as a practical implementation of data preprocessing, feature engineering, and polynomial regression with regularization.

## The Dataset
The model is trained on a public medical insurance dataset containing 1,338 records. 
Features included:
* **Age:** Age of the primary beneficiary
* **Sex:** Male / Female
* **BMI:** Body mass index (providing an understanding of body, weights that are relatively high or low relative to height)
* **Children:** Number of children covered by health insurance / Number of dependents
* **Smoker:** Smoking status (Yes / No)
* **Region:** The beneficiary's residential area in the US (northeast, southeast, southwest, northwest)
* **Charges:** Individual medical costs billed by health insurance (Target Variable)

## Model Architecture
* **Algorithm:** Polynomial Regression with Ridge (L2) Regularization.
* **Why Ridge?** Medical data often has complex, non-linear interactions (e.g., the compounding cost effect of high BMI *and* smoking). Polynomial features capture these relationships, while Ridge regularization penalizes excessive weights to prevent overfitting.
* **Pipeline:** The data flows through a Scikit-Learn `Pipeline` utilizing `StandardScaler`, `PolynomialFeatures(degree=2)`, and `Ridge(alpha=1.0)`.
* **Performance:** The model achieves an $R^2$ score of ~0.84, successfully explaining 84% of the variance in medical costs.

## Project Structure
\`\`\`text
insurance-cost-predictor/
├── data/
│   ├── raw/                 # Raw insurance.csv dataset
│   └── processed/           # Cleaned data ready for modeling
├── notebooks/
│   └── 01_end_to_end_model.ipynb  # EDA, training, and evaluation
├── models/
│   └── final_ridge_model.pkl      # Serialized Scikit-Learn Pipeline
├── src/
│   └── predict.py                 # Script for terminal predictions
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
\`\`\`

## How to Run Locally

**1. Clone the repository and install dependencies:**
\`\`\`bash
git clone https://github.com/yourusername/insurance-cost-predictor.git
cd insurance-cost-predictor
pip install -r requirements.txt
\`\`\`

**2. Make a prediction:**
Run the prediction script to test the model against a hardcoded patient profile.
\`\`\`bash
python src/predict.py
\`\`\`

**3. Explore the training process:**
Launch Jupyter to view the exploratory data analysis and model training steps.
\`\`\`bash
jupyter notebook notebooks/01_end_to_end_model.ipynb
\`\`\`# Insurance cost prediction
build a predictive model that estimates individual medical insurance charges.

