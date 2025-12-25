# Customer Churn Prediction & MLOps Pipeline

This project implements an end-to-end machine learning pipeline to predict customer churn. It covers the full lifecycle from data engineering and automated hyperparameter tuning to **production-grade deployment using FastAPI**.

---

## 🚀 Key Features
* **Production API:** Developed a **FastAPI** wrapper to serve the model as a service for real-time predictions.
* **Automated Hyperparameter Tuning:** Utilized **Optuna** for Bayesian optimization to maximize the F1-score.
* **Experiment Tracking:** Integrated **MLFlow** to log parameters, metrics, and model versions, ensuring full transparency in the experimentation phase.
* **Data Validation:** Implemented **Pydantic** schemas to ensure data integrity for API inputs.

## 🛠 Tech Stack
* **Deployment:** FastAPI, Uvicorn
* **MLOps:** MLFlow, Optuna
* **ML Libraries:** Scikit-Learn, XGBoost, Pandas, NumPy
* **Data Validation:** Pydantic

## 🌐 API Implementation
The model is served via a POST endpoint that accepts customer data in JSON format and returns a churn probability.
* **Endpoint:** `/predict`
* **Interactive Docs:** Automatically generated via Swagger UI at `/docs`.



## 📂 Project Structure
```text
├── app/
│   ├── main.py          # FastAPI application
│   └── model.pkl        # Serialized best model
├── notebooks/
│   └── training.ipynb   # Exploration, Optuna, and MLFlow logic
├── README.md            # Project documentation
```

🛠 Setup & Usage
Install Dependencies:

```Bash
uvicorn app.main:app --reload
```

Launch MLFlow Dashboard: To view experiment logs:

```Bash


mlflow ui
```
## Notes
This project was created as a learning exercise using AI-assisted tools such as ChatGPT, Gemini, and Claude. All code was reviewed, tested, and modified by me.

