# Customer Retention Agent

An AI-powered customer retention decision-support system combining machine learning, explainable AI, retrieval-augmented generation (RAG), and agentic tool orchestration.

The system moves beyond predicting customer churn by connecting predictive insights to customer investigation, policy retrieval, deterministic eligibility assessment, and human-reviewed retention recommendations.

## 1. Project Overview

Customer churn is a significant challenge for subscription-based businesses. Identifying customers who are likely to leave is only one part of the problem. Businesses also need to understand the factors associated with churn, investigate individual customer circumstances, and determine which retention interventions are appropriate.

This project develops an end-to-end customer retention prototype that combines a predictive machine learning pipeline with an AI-powered decision-support agent.

The system is designed to:

- Predict customer churn probability using historical customer and account information.
- Explain individual churn predictions using SHAP.
- Retrieve relevant retention policies and operational guidance.
- Evaluate potential policy eligibility using deterministic business rules.
- Coordinate tools through an LLM-powered agent.
- Generate evidence-backed retention recommendations.
- Require human approval before customer-facing retention actions.

The project demonstrates how traditional machine learning can be extended into a practical AI application through APIs, retrieval systems, tool calling, validation layers, and operational safeguards.

## 2. System Architecture

The solution consists of five major components.

### A. Data Processing and Feature Engineering

The project uses the IBM Telco Customer Churn dataset.

Data preparation includes:

- Data inspection and validation.
- Handling missing values in `TotalCharges`.
- Converting numerical and categorical variables into appropriate formats.
- Feature engineering, including tenure groups, average monthly spend, and service counts.
- Stratified training and testing splits.

Scikit-learn preprocessing pipelines ensure transformations are fitted on training data to reduce data leakage.

### B. Churn Prediction Model

Several classification algorithms were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

XGBoost was selected as the final model based on its evaluation results.

The classification threshold was adjusted from the default 0.50 to 0.35 to improve the identification of customers at risk of churning.

The final model and associated metadata are stored as versioned project artifacts:

- `artifacts/models/churn_model.joblib`
- `artifacts/models/model_metadata.json`

### C. Explainable AI

SHAP (SHapley Additive exPlanations) is used to interpret model predictions.

The explanation component identifies features that contribute positively or negatively to an individual prediction.

This allows the retention agent to distinguish between:

- The predicted likelihood of churn.
- The features influencing the prediction.
- Additional customer information requiring investigation.

SHAP values represent model contributions, not proof of causation or the customer's actual motivation.

### D. Retrieval-Augmented Generation (RAG)

The project includes a policy knowledge base containing illustrative retention strategies covering:

- Early-tenure onboarding support.
- Contract review and education.
- Technical support review.

The retrieval pipeline is designed to identify relevant policy content using multiple retrieval signals.

The current implementation includes:

- BM25 lexical retrieval.
- Semantic retrieval using sentence embeddings.
- Reciprocal Rank Fusion (RRF) to combine retrieval rankings.
- Cross-encoder reranking to improve the ordering of retrieved policy chunks.

Retrieved policy content is supplied to the AI agent as supporting context.

The retriever also supports identifying situations where no sufficiently relevant policy should be returned.

Retrieval relevance does not establish customer eligibility. Eligibility is assessed separately using deterministic business rules.

### E. Agentic Tool Orchestration

The retention agent uses Google's Gemini API and function calling to coordinate available tools.

Registered tools include:

| Tool | Responsibility |
|---|---|
| `get_customer_profile` | Retrieves customer account information. |
| `predict_customer_churn` | Obtains the model's churn prediction. |
| `explain_customer_churn` | Retrieves SHAP-based prediction explanations. |
| `search_retention_policies` | Retrieves relevant policy knowledge through RAG. |
| `assess_customer_retention_eligibility` | Evaluates potential policy eligibility using deterministic rules. |

The agent determines which tools are required to investigate a customer request.

A typical investigation follows this workflow:

1. Retrieve the customer's profile.
2. Obtain the churn prediction.
3. Investigate the factors influencing the prediction.
4. Retrieve relevant retention policies.
5. Evaluate potential eligibility against deterministic business rules.
6. Consolidate available evidence into a retention recommendation.
7. Identify missing information, restrictions, and approval requirements.

The agent is instructed not to invent discounts, compensation, customer circumstances, or policy permissions.

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

Customers with predicted churn probabilities at or above 35% are flagged as potential churn risks.

At this threshold, the test-set confusion matrix was:

| | Predicted stayed | Predicted churn |
|---|---:|---:|
| Actual stayed | 683 | 145 |
| Actual churn | 85 | 214 |

The model identified 214 of the 299 actual churners in the test set, corresponding to approximately 70% recall.

The threshold reflects a preference for identifying more potential churners while accepting additional false positives.

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

These represent predictive associations within the dataset, not established causes of customer churn.

## 4. RAG Evaluation

The retrieval component was evaluated using a collection of test queries covering relevant policy retrieval and no-match scenarios.

Evaluation metrics included:

- Recall@K.
- Precision@K.
- Mean Reciprocal Rank (MRR).
- No-match accuracy.

Recorded evaluation results included:

| Metric | Result |
|---|---:|
| Recall@K | 1.00 |
| MRR | 1.00 |
| Precision@3 | 0.3611 |
| Precision among returned results | 1.00 |
| No-match accuracy | 1.00 |

These results reflect the evaluated test-query collection and should not be interpreted as guaranteed production retrieval performance.

## 5. API and Application Architecture

The project includes a FastAPI application for exposing churn prediction functionality.

The API provides an application interface for accessing the trained prediction model and related functionality.

The Gemini retention agent currently operates through a separate execution path and coordinates its tools independently.

This separation allows the predictive service and agent orchestration layer to be developed, tested, and deployed independently.

### API endpoints

The API includes:

- `/health` - Application health check.
- `/predict` - Customer churn prediction.

Interactive API documentation is available through FastAPI's Swagger UI.

## 6. Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python 3.13 |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| Model explainability | SHAP |
| API development | FastAPI, Uvicorn |
| Agent orchestration | Google Gemini API, function calling |
| Lexical retrieval | BM25 |
| Semantic retrieval | Sentence Transformers |
| Retrieval fusion | Reciprocal Rank Fusion |
| Reranking | Cross-encoder |
| Testing | Pytest |
| Environment management | Python virtual environments, python-dotenv |
| Containerization | Docker |
| Cloud build | AWS CodeBuild |
| Container registry | Amazon Elastic Container Registry (ECR) |
| Planned application hosting | Amazon ECS / AWS Fargate |
| Version control | Git, GitHub |

## 7. Project Structure

```text
customer-retention-agent/
│
├── artifacts/
│   └── models/
│       ├── churn_model.joblib
│       └── model_metadata.json
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
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
├── buildspec.yml
├── .dockerignore
├── .env.example
├── .gitignore
└── README.md
```

## 8. Getting Started

### Prerequisites

- Python 3.13
- Git
- A Google Gemini API key for the retention agent
- Access to the project model artifacts and required datasets

### Clone the repository

```powershell
git clone https://github.com/billiahkerubo1/customer-retention-agent.git

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

python -m pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file in the project root using `.env.example` as a reference.

Add your Gemini API key:

```env
APP_ENV=development
GEMINI_API_KEY=your_actual_gemini_api_key
```

Keep your actual `.env` file private. Never commit API keys or other credentials to GitHub.

### Run the prediction API

```powershell
python -m uvicorn src.api.main:app --reload
```

The API will be available at:

- API base URL: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs
- OpenAPI schema: http://127.0.0.1:8000/openapi.json

Use the interactive documentation to inspect available endpoints and test prediction requests.

### Run the retention agent

Open another terminal, activate the virtual environment, and execute:

```powershell
python -m src.agent.retention_agent
```

Follow the command-line prompts to investigate a customer.

The agent requires a valid Gemini API key and access to the configured Gemini model.

## 9. Running Tests

The project includes automated tests covering:

- API functionality.
- Feature engineering.
- Model functionality.
- Customer tools.
- Policy retrieval.
- Eligibility assessment.
- Agent orchestration.

Run the automated test suite:

```powershell
python -m pytest tests/ -v --ignore=src/agent/test_gemini.py
```

The live Gemini integration test is excluded because it depends on an external API, network availability, and API quota.

## 10. Containerization and AWS Deployment

The project includes a Docker configuration for packaging the FastAPI prediction application.

The Docker image uses a lean production dependency file, `requirements-docker.txt`, to install only the packages required by the API.

The container is configured to start the FastAPI application using Uvicorn.

### Deployment architecture

The intended deployment workflow is:

```text
GitHub Repository
       |
       v
AWS CodeBuild
       |
       v
Docker Image Build
       |
       v
Amazon ECR
       |
       v
Amazon ECS / AWS Fargate
       |
       v
FastAPI Prediction Service
```

The AWS infrastructure preparation currently includes:

- An Amazon ECR repository.
- A CodeBuild project connected to GitHub.
- A build specification for building and pushing the Docker image.
- An IAM service role with ECR push permissions.

The first cloud build is pending resolution of an AWS CodeBuild account queue restriction. The image has not yet been confirmed as uploaded to ECR, and the application has not yet been deployed to ECS.

The current container deployment target is the FastAPI prediction service, not the complete Gemini retention agent.

## 11. Engineering and Responsible AI Considerations

Several safeguards are incorporated into the system.

### Prediction versus certainty

A churn probability is a model estimate, not confirmation that a customer will leave.

### Explainability versus causality

SHAP explains how model features contribute to predictions. It does not establish why a customer is dissatisfied.

### Retrieved information versus eligibility

A relevant policy retrieval result does not automatically establish that a customer qualifies for a particular intervention.

### Deterministic business rules

Eligibility checks are implemented separately from LLM-generated reasoning to reduce unsupported policy decisions.

### Human oversight

The agent cannot independently authorize discounts, credits, refunds, or other commercial concessions.

### Failure handling and traceability

The agent includes:

- Structured logging.
- Trace IDs.
- Tool-execution records.
- Bounded tool-call rounds.
- Request timeouts.
- Retry handling.

These mechanisms help investigate failures, understand tool execution, and prevent uncontrolled agent loops.

### Data considerations

The IBM dataset is a historical snapshot. It does not independently establish future churn over a specific prediction horizon.

Production deployment would require appropriately dated customer snapshots, temporal validation, and ongoing performance monitoring.

## 12. Future Improvements

Planned enhancements include:

1. Expand retrieval evaluation across larger and more diverse query sets.
2. Improve observability with structured traces, latency tracking, and error monitoring.
3. Introduce model monitoring for prediction drift and performance degradation.
4. Implement time-based model validation using customer snapshots.
5. Track retention outcomes and evaluate interventions through controlled experiments.
6. Refine retention strategies using feedback from business stakeholders and operational outcomes.

## 13. Project Objective

The broader objective is to demonstrate how traditional machine learning can be extended into a practical AI-powered decision-support system.

Rather than stopping at churn prediction, this project connects predictive insights to customer investigation, policy retrieval, deterministic eligibility assessment, and human-reviewed recommendations.

It illustrates the engineering considerations involved in building explainable, tool-using, retrieval-augmented, and policy-aware AI systems.

The emphasis is on moving from an isolated predictive model towards a reliable application that connects machine learning outputs to operational business decisions.