# 🧠 Mental Health Prediction Challenge

This project predicts depression likelihood based on survey data using a machine learning pipeline. It includes complete lifecycle management from data preprocessing to model deployment with FastAPI and Docker.

---

## 📁 Project Structure

```
Kaggle-Mental-Health-Prediction-Challenge/
├── .github/workflows           # GitHub Actions workflows
├── data/                       # Raw datasets
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── eda/                        # Jupyter notebooks for EDA
├── mlruns/                     # MLflow tracking directory
├── models/                     # Saved model artifacts (e.g., SVC.pkl)
├── src/                        # Source code
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   └── model_building.py
├── static/css/                # CSS for frontend styling
├── templates/                 # HTML templates for FastAPI
├── main.py                    # FastAPI app
├── Dockerfile                 # Docker instructions
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview (this file)
├── .gitignore
└── .dockerignore
```

---

## 📊 Data Collection

The project uses survey data files (`train.csv`, `test.csv`) stored in the `/data` directory. These datasets include features related to sleep patterns, stress, and demographics.

## 📈 Exploratory Data Analysis (EDA)

Performed using:

* `pandas`
* `matplotlib`
* `seaborn`

Insights were drawn on distributions, outliers, missing values, and correlations.

## 🧹 Data Preprocessing

Done using:

* `pandas`
* `scikit-learn`

Steps include:

* Label encoding / one-hot encoding
* Imputation for missing values
* Train-test split
* Feature scaling using `StandardScaler`

## 🧠 Model Building

We use `scikit-learn` to train an SVC (Support Vector Classifier). Model training, evaluation, and saving are done via the script in `src/model_building.py`.

## 📋 Model Tracking with MLflow

We use MLflow for model tracking and experiment management:

* Launched on `localhost:5000`
* Stores runs in `mlruns/`

## 🌐 FastAPI Backend

The backend is built with `FastAPI`:

* Serves a simple web UI
* Receives user input via HTML forms
* Predicts depression risk using trained model
* Runs on port `8000`

## 🎨 Frontend

The web frontend is built with:

* `HTML`
* `CSS`
* `Bootstrap`

Located in `/templates` and `/static/css` folders.

## 🐳 Docker Containerization

A Dockerfile is included to containerize the app. This makes it portable and easy to deploy anywhere.

## 🔁 CI/CD with GitHub Actions

GitHub Actions handles:

* ✅ Testing: Runs model pipeline & ensures model is saved.
* 📇 Build: Builds Docker image.
* 🚀 Push: Pushes image to DockerHub: [`pranavreddy123/depression-prediction`](https://hub.docker.com/repository/docker/pranavreddy123/depression-prediction)

## ☁️ Deployment on AWS EC2

### EC2 Setup:

```bash
# SSH into your EC2
ssh -i "your-key.pem" ubuntu@your-ec2-public-dns

# Install Docker
sudo apt update && sudo apt install docker.io -y
sudo usermod -aG docker ubuntu
newgrp docker

# Pull and run Docker image
docker pull pranavreddy123/depression-prediction:latest

# Run the container
docker run -d -p 8000:8000 pranavreddy123/depression-prediction:latest
```

App will be available at: `http://<EC2_PUBLIC_IP>:8000`

---

## 🛠️ Run Locally

```bash
# Clone the repo
git clone https://github.com/ka1817/Kaggle-Mental-Health-Prediction-Challenge.git
cd Kaggle-Mental-Health-Prediction-Challenge

# Install dependencies
pip install -r requirements.txt

# Run MLflow server (optional)
mlflow ui --port 5000

# Train model
python src/model_building.py

# Start app
uvicorn main:app --reload --port 8000
```

---

## 📆 requirements.txt

```
fastapi
uvicorn[standard]
pandas
scikit-learn
jinja2
python-multipart
mlflow
matplotlib
seaborn
```

---

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Developed by

Pranav Reddy
