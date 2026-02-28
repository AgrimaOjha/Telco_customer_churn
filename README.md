# 📉 Predicting Customer Churn: A Machine Learning & Gen AI Approach
### *For the Telecommunications Industry*

> **Department of CS and AI — Newton School of Technology**
> Supervisors: Kartik Gupta · Arun Chauhan

---

## 📌 Project Overview

This project designs and implements an **AI-driven customer churn prediction system** for the telecommunications industry. Using a real-world Telco dataset of 7,043 customers, we build a complete pipeline from raw data to a deployed, interactive retention tool.

- **Milestone 1:** Classical ML pipeline to predict churn risk, uncover key behavioral drivers, and surface insights through an interactive Streamlit UI — *no LLMs*.
- **Milestone 2:** Extension with a **Generative AI retention layer** that autonomously generates personalized, structured intervention strategies for at-risk customers.

The core insight motivating this work: *replacing a lost customer costs exponentially more than retaining one.* Identifying churners before they leave is essential for long-term profitability.

---

## 👥 Team

| Name | Role |
| :--- | :--- |
| Agrima Ojha | UI Development & Deployment |
| Suhani Gupta | Data Preprocessing & GenAI Retention Agent |
| Khyati Batra | Model Evaluation & Reporting |
| Raghav Kaushal | ML Pipeline & Feature Engineering |

---

## 🗂️ Repository Structure

📦 Telco_customer_churn
 ┣ 📂 .devcontainer/
 ┃ ┗ 📄 devcontainer.json             # Dev container configuration
 ┣ 📂 data/
 ┃ ┗ 📄 Customer_Churn_Dataset.csv    # IBM Telco dataset (Kaggle)
 ┣ 📄 app.py                          # Streamlit application
 ┣ 📄 churn_model.pkl                 # Trained XGBoost model
 ┣ 📄 model_columns.pkl               # Feature columns for inference
 ┣ 📄 telco_xgboost.ipynb             # Full ML pipeline notebook
 ┣ 📄 requirements.txt
 ┗ 📄 README.md

---

## 🧰 Technology Stack

| Component | Technology |
| :--- | :--- |
| **Dataset** | IBM Telco Customer Churn — Kaggle (7,043 records) |
| **ML Models** | Logistic Regression, XGBoost (via Scikit-Learn) |
| **Imbalance Handling** | SMOTE (`imbalanced-learn`) |
| **Feature Engineering** | Custom derived metric: *Charge per Tenure* |
| **Gen AI Layer (M2)** | Open-source LLMs / Free-tier APIs |
| **UI & Deployment** | Streamlit (hosted on Streamlit Cloud) |

---

## 📊 Dataset

**Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
**File:** `data/Customer_Churn_Dataset.csv`
**Size:** 7,043 customer records · **Target Variable:** `Churn` (Yes = 1, No = 0)

| Category | Features |
| :--- | :--- |
| **Demographic** | Gender, Senior Citizen status, Partners, Dependents |
| **Service Details** | Phone, Internet type, Online Security, Tech Support, Streaming |
| **Account Info** | Tenure, Contract type, Payment method, Paperless Billing |
| **Billing** | Monthly Charges, Total Charges |

---

## 🔬 Methodology

### Stage 1 — Exploratory Data Analysis

EDA on `Customer_Churn_Dataset.csv` revealed four key early warning signals:

- **Contract Vulnerability** — Month-to-Month customers churn at the highest rates; Two-Year contracts show the lowest.
- **The Loyalty Gap** — Customers with lower tenure are significantly more likely to leave.
- **Price Sensitivity** — Higher monthly charges correlate directly with increased churn probability.
- **Infrastructure Impact** — Fiber Optic internet users churn at notably higher rates than DSL users.

---

### Stage 2 — Data Engineering & Feature Creation

**Step 1 — Cleaning:** Removed missing values in the `TotalCharges` column.

**Step 2 — Transformation:** Encoded categorical variables, mapped target to binary (`Yes → 1`, `No → 0`), and applied feature scaling.

**Step 3 — Train-Test Split:** Strict 80:20 split for training and validation.

**Feature Engineering Highlight:**
> A new derived metric — **Charge per Tenure** (`MonthlyCharges / Tenure`) — was engineered to combine billing weight with customer loyalty into a single high-signal feature. This ranked as the **4th most important predictor** in the final model.

---

### Stage 3 — Handling Class Imbalance

The dataset is heavily imbalanced — the vast majority of customers do not churn. Training naively on this data creates an *accuracy illusion*: a model predicting "No Churn" for everyone achieves high accuracy while completely failing to identify actual churners.

**Solution: SMOTE** (Synthetic Minority Oversampling Technique) was applied to synthetically balance the churn and non-churn classes before retraining, resulting in significantly improved recall and F1-score.

---

### Stage 4 — Progressive Modeling

| Step | Model | Purpose |
| :--- | :--- | :--- |
| **1 — Baseline** | Logistic Regression (raw data) | Set performance floor; exposed the accuracy illusion |
| **2 — Corrected** | Logistic Regression + SMOTE | Forced model to learn churn patterns; improved recall |
| **3 — Optimized** | **XGBoost + SMOTE** | Best overall performance across all metrics |

**Why XGBoost?**
- Robust against overfitting compared to simpler models.
- Handles non-linear relationships in customer behavior.
- Automatically captures complex feature interactions (e.g., high charges × low tenure).

The final trained model is saved at `model/churn_model.pkl` and feature columns at `model/model_columns.pkl` for use in the Streamlit app.

---

### Stage 5 — Top Predictive Drivers (XGBoost Feature Importance)

| Rank | Feature |
| :--- | :--- |
| 🥇 1 | Internet Service: Fiber Optic |
| 🥈 2 | Contract Type: Two Year *(protective factor)* |
| 🥉 3 | Contract Type: One Year |
| 4 | Engineered Feature: Charge per Tenure |
| 5 | Internet Service: No |

---

## 🚀 Milestones & Deliverables

### Milestone 1 — ML-Based Churn Prediction *(Mid-Sem · 25%)*

**Objective:** Predict customers at risk using historical behavioral data — classical ML pipelines *without LLMs*.

- [x] Problem understanding & business context
- [x] Data preprocessing pipeline (cleaning → encoding → SMOTE balancing)
- [x] Full ML pipeline in `notebook/telco_xgboost.ipynb`
- [x] Trained XGBoost model saved to `model/churn_model.pkl`
- [x] Working Streamlit application (`app/app.py`)
- [x] Model evaluation report (Accuracy, Precision, Recall, F1, Confusion Matrix)

### Milestone 2 — Gen AI Retention Assistant *(End-Sem · 30%)*

**Objective:** Extend the system with a Gen AI layer that reasons about individual churn risk and generates structured, personalized retention recommendations.

- [ ] Publicly deployed application on Streamlit Cloud *(link required)*
- [ ] Retention report generation per customer profile
- [ ] Risk Meter UI with churn probability percentage
- [ ] Risk tier segmentation (Low / Medium / High)
- [ ] Executive summary with per-customer feature importance breakdown
- [ ] Demo Video (max 5 mins)

---

## 🖥️ Application — Dashboard Capabilities

The Streamlit app (`app/app.py`) loads `churn_model.pkl` and `model_columns.pkl` to serve real-time predictions. Retention agents and non-technical stakeholders can input a customer profile and instantly receive:

| Output | Description |
| :--- | :--- |
| **Risk Meter** | Visual gauge showing exact churn probability (%) |
| **Prediction Status** | Binary classification — *Retention Likely* or *Churn Risk* |
| **Risk Level** | Customer segmented into Low, Medium, or High risk tier |
| **Executive Summary** | Feature importance breakdown driving that individual's score |
| **Retention Strategy** *(M2)* | AI-generated, personalized intervention recommendations |

**Input fields:** Tenure · Monthly Charges · Total Charges · Contract Type · Internet Service · Paperless Billing

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/AgrimaOjha/Telco_customer_churn.git
cd Telco_customer_churn

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app/app.py
```
Streamlit link: [https://telcocustomerchurn-h2itkknxncwmjgfhmhlnf9.streamlit.app/](https://telcocustomerchurn-genaiproject.streamlit.app/)
---

## 📈 Evaluation Criteria

| Phase | Weight | Criteria |
| :--- | :--- | :--- |
| **Mid-Sem** | 25% | ML technique application, Feature Engineering quality, UI usability, Evaluation metrics |
| **End-Sem** | 30% | Gen AI output quality, Risk segmentation, Deployment success, Output clarity |

---

## 📋 Constraints & Requirements

- **Team Size:** 4 Students
- **API Budget:** Free Tier Only (open-source models / free APIs)
- **UI Framework:** Streamlit
- **Hosting:** Mandatory — Streamlit Cloud
