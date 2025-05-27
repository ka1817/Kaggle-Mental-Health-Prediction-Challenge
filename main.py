from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import os
import uvicorn

app = FastAPI()

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja templates
templates = Jinja2Templates(directory="templates")

# Load model (SVC)
model_path = os.path.join("models", "SVC.pkl")
model = joblib.load(model_path)

# Feature order expected by the model
FEATURES = [
    "Gender", "Age", "Academic Pressure", "Study Satisfaction", "Sleep Duration",
    "Dietary Habits", "Have you ever had suicidal thoughts ?", "Study Hours",
    "Financial Stress", "Family History of Mental Illness"
]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Gender: str = Form(...),
    Age: int = Form(...),
    Academic_Pressure: float = Form(...),
    Study_Satisfaction: float = Form(...),
    Sleep_Duration: str = Form(...),
    Dietary_Habits: str = Form(...),
    Suicidal_Thoughts: str = Form(...),
    Study_Hours: int = Form(...),
    Financial_Stress: int = Form(...),
    Family_History: str = Form(...)
):
    # Construct a DataFrame with correct column names
    input_data = pd.DataFrame([[
        Gender, Age, Academic_Pressure, Study_Satisfaction, Sleep_Duration,
        Dietary_Habits, Suicidal_Thoughts, Study_Hours, Financial_Stress, Family_History
    ]], columns=FEATURES)

    prediction = model.predict(input_data)[0]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction
    })

if __name__=='__main__':
    uvicorn.run(app,host='0.0.0.0',port=2003)