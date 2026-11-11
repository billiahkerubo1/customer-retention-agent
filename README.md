# CUSTOMER RETENTION AGENT

An AI-powered customer retention system that combines machine learning, explainable AI, retrieval-augmented generation (RAG), and agentic tool orchestration to support proactive customer retention decisions.

## 1. Project Overview

Customer churn presents a significant challenge for subscription-based businesses, particularly telecommunications providers. Identifying customers who are likely to leave is only the first step. Businesses also need to understand the factors associated with churn and determine which retention interventions are appropriate.

This project develops an end-to-end customer retention solution that combines a predictive machine learning model with an AI agent.

The system is designed to:

- Predict customer churn probability using historical customer and account information.
- Identify the factors influencing individual churn predictions.
- Retrieve relevant retention policies and operational guidance.
- Evaluate potential policy eligibility using deterministic business rules.
- Generate evidence-backed retention recommendations.
- Require human approval before customer-facing retention actions.

The project demonstrates how predictive machine learning and agentic AI can work together within a business decision-support workflow.

## 2. System Architecture

The solution consists of five main components.

### A. Data Processing and Feature Engineering

The project uses the IBM Telco Customer Churn dataset.

Data preparation includes:

- Data inspection and validation.
- Handling missing values in `TotalCharges`.
- Converting numerical and categorical variables into appropriate formats.
- Feature engineering, including tenure groups, average monthly spend and service counts.
- Stratified training and testing splits.

Preprocessing is implemented using Scikit-learn pipelines to ensure that transformations are fitted on training data.

### B. Churn Prediction Model

Several classification algorithms were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

XGBoost was selected as the final model based on its overall evaluation results.

The classification threshold was adjusted from the default 0.50 to 0.35 to improve the identification of customers at risk of churning.

### C. Explainable AI

SHAP (SHapley Additive exPlanations) is used to explain model predictions.

The explanation component identifies features that contribute positively or negatively to an individual prediction.

This helps the retention agent distinguish between:

- The predicted likelihood of churn.
- The features influencing the prediction.
- Additional customer information requiring investigation.

SHAP values represent model contributions, not proof of causation or the customer's actual motivation.

### D. Retrieval-Augmented Generation (RAG)

A policy knowledge base contains illustrative retention strategies covering:

- Early-tenure onboarding support.
- Contract review and education.
- Technical support review.

The retrieval component uses:

- TF-IDF vectorization.
- Unigram and bigram representations.
- Cosine similarity.
- Top-k policy chunk retrieval.

Retrieved policy content is supplied to the AI agent as supporting context.

The current implementation uses lightweight lexical retrieval rather than a production vector database.

### E. Agentic Tool Orchestration

The retention agent uses Google's Gemini API and function calling to coordinate the available tools.

Registered tools include:

| Tool | Responsibility |
|---|---|
| `get_customer_profile` | Retrieves customer account information. |
| `predict_customer_churn` | Obtains the model's churn prediction. |
| `explain_customer_churn` | Retrieves SHAP-based prediction explanations. |
| `search_retention_policies` | Retrieves relevant policy knowledge through RAG. |
| `assess_customer_retention_eligibility` | Evaluates potential policy eligibility using deterministic rules. |

The agent decides which tools are required to investigate a customer request.

A typical investigation follows this workflow:

1. Retrieve the customer's profile.
2. Obtain the churn prediction.
3. Investigate the factors influencing the prediction.
4. Retrieve relevant retention policies.
5. Evaluate potential eligibility against deterministic business rules.
6. Consolidate the evidence into a retention recommendation.
7. Identify missing information and approval requirements.

The agent is instructed not to invent discounts, compensation, customer circumstances or policy permissions.

Human approval is required before customer-facing retention actions.

## 3. Model Performance

The final XGBoost model was evaluated on a held-out test split.

| Metric | Result |
|---|---:|
| Accuracy | 77.79% |
| Precision | 56.59% |
| Recall | 70.05% |
| F1-score | 62.60% |
| ROC-AUC | 84.20% |
| PR-AUC | 65.03% |

### Classification threshold

The selected operating threshold is 0.35.

This means customers with predicted churn probabilities at or above 35% are flagged as potential churn risks.

At this threshold, the test-set confusion matrix was:

| | Predicted stayed | Predicted churn |
|---|---:|---:|
| Actual stayed | 683 | 145 |
| Actual churn | 85 | 214 |

The model identified 214 of the 299 actual churners in the test set, corresponding to a recall of approximately 70%.

The threshold reflects a preference for identifying more potential churners, while accepting additional false positives.

### Feature importance

Global SHAP analysis identified several influential model features:

- Month-to-month contracts.
- Customer tenure.
- Absence of online security.
- Fiber optic internet service.
- Total charges.
- Monthly charges.
- Average monthly spend.
- Absence of technical support.

These are predictive associations within the dataset, not established causes of customer churn.

## 4. Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| Model explainability | SHAP |
| API development | FastAPI, Uvicorn |
| Agent orchestration | Google Gemini API, function calling |
| Retrieval-augmented generation | TF-IDF, cosine similarity |
| Testing | Pytest |
| Environment management | Python virtual environments, python-dotenv |
| Version control | Git, GitHub |
| Containerization | Docker (planned) |

## 5. Project Structure

```text
customer-retention-agent/
│
├── artifacts/
│   └── models/
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── knowledge_base/
│
├── docs/
│
├── notebooks/
│
├── reports/
│   └── figures/
│
├── scripts/
│
├── src/
│   ├── api/
│   ├── agent/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── rag/
│   └── tools/
│
├── tests/
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## 6. Getting Started

### Prerequisites

- Python 3.13
- Git
- A Google Gemini API key
- Docker Desktop (required for the planned containerization stage)

### Clone the repository

```bash
git clone https://github.com/username/customer-retention-agent.git

cd customer-retention-agent
```

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file in the project root using `.env.example` as a reference.

Add your Gemini API key:

```env
APP_ENV=development
DATABASE_URL=
GEMINI_API_KEY=your_actual_gemini_api_key
```

Keep your actual `.env` file private. Never commit API keys or other credentials to GitHub.

### Run the prediction API

```powershell
python -m uvicorn src.api.main:app --reload
```

The API will be available at:

- API base URL: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`

Use the interactive documentation to inspect the available endpoints and test prediction requests.

### Run the retention agent

Open another terminal, activate the virtual environment, and execute:

```powershell
python -m src.agent.retention_agent
```

Follow the command-line prompts to investigate a customer.

The agent requires a valid Gemini API key and access to the configured Gemini model.

## 7. Running Tests

The project includes automated tests for the API, feature engineering, model functionality, customer tools, policy retrieval, eligibility assessment and agent orchestration.

Run the automated test suite:

```powershell
python -m pytest tests/ -v --ignore=src/agent/test_gemini.py
```

The live Gemini integration test is excluded from this command because it depends on an external API, network availability and API quota.

## 8. Engineering and Responsible AI Considerations

Several safeguards are incorporated into the system:

**Prediction versus certainty**

A churn probability is a model estimate, not confirmation that a customer will leave.

**Explainability versus causality**

SHAP explains how model features contribute to predictions. It does not establish why a customer is dissatisfied.

**Retrieved information versus eligibility**

A relevant policy retrieval result does not automatically establish that a customer qualifies for a particular intervention.

**Deterministic business rules**

Eligibility checks are implemented separately from LLM-generated reasoning to reduce the risk of unsupported policy decisions.

**Human oversight**

The agent cannot independently authorize discounts, credits, refunds or other commercial concessions.

**Failure handling and traceability**

The agent includes structured logging, trace IDs, tool-execution records, bounded tool-call rounds, request timeouts and retry handling.

**Data considerations**

The IBM dataset is a historical snapshot. It does not independently establish future churn over a specific prediction horizon. Production deployment would require appropriately dated customer snapshots, temporal validation and ongoing performance monitoring.

## 9. Future Improvements

Planned enhancements include:

1. Containerization with Docker.
2. Deployment of the FastAPI application.
3. Improved RAG retrieval using semantic and hybrid search.
4. Cross-encoder reranking of retrieved policy chunks.
5. Retrieval evaluation using metrics such as Recall@K, Precision@K and MRR.
6. Improved observability with structured traces, latency tracking and error monitoring.
7. Model monitoring for prediction drift and performance degradation.
8. Time-based model validation using customer snapshots.
9. Retention outcome tracking and controlled intervention experiments.
10. Human approval workflows for proposed customer-facing actions.

## 10. Project Objective

The broader objective is to demonstrate how traditional machine learning can be extended into a practical AI-powered decision-support system.

Rather than stopping at churn prediction, the system connects predictive insights to customer investigation, policy retrieval, deterministic eligibility checks and human-reviewed recommendations.

The result is an end-to-end prototype illustrating the engineering considerations involved in building explainable, tool-using and policy-aware AI systems.




