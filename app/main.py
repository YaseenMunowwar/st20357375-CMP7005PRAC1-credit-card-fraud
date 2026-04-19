import logging
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from model_utils import load_models, predict_transaction

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Credit Card Fraud Detection App")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Load models once when app starts
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Credit_Card_Dataset_2025_Sept_Combined.csv"
MODELS_DIR = BASE_DIR / "app" / "models"
MODELS = load_models(MODELS_DIR)

# Load dataset once for overview page
# Using the standard naming from the previous steps
df_main = pd.read_csv(DATA_PATH)

# App summary values
dataset_shape = df_main.shape
fraud_count = int(df_main["Target"].sum())
non_fraud_count = int((df_main["Target"] == 0).sum())
missing_summary = df_main.isnull().sum()
missing_summary = missing_summary[missing_summary > 0].to_dict()
column_types = df_main.dtypes.astype(str).to_dict()
sample_rows = df_main.head(5).to_dict(orient="records")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Home"
        }
    )


@app.get("/overview", response_class=HTMLResponse)
def overview(request: Request):
    return templates.TemplateResponse(
        "data_overview.html",
        {
            "request": request,
            "title": "Data Overview",
            "dataset_shape": dataset_shape,
            "fraud_count": fraud_count,
            "non_fraud_count": non_fraud_count,
            "missing_summary": missing_summary,
            "column_types": column_types,
            "sample_rows": sample_rows
        }
    )


@app.get("/eda", response_class=HTMLResponse)
def eda(request: Request):
    return templates.TemplateResponse(
        "eda.html",
        {
            "request": request,
            "title": "EDA"
        }
    )


@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse(
        "prediction.html",
        {
            "request": request,
            "title": "Prediction"
        }
    )


@app.post("/predict", response_class=HTMLResponse)
def predict_result(
    request: Request,
    model_name: str = Form(...),
    GENDER: str = Form(...),
    CAR: str = Form(...),
    REALITY: str = Form(...),
    NO_OF_CHILD: int = Form(...),
    FAMILY_TYPE: str = Form(...),
    HOUSE_TYPE: str = Form(...),
    WORK_PHONE: int = Form(...),
    PHONE: int = Form(...),
    E_MAIL: int = Form(...),
    FAMILY_SIZE: float = Form(...),
    BEGIN_MONTH: int = Form(...),
    AGE: int = Form(...),
    YEARS_EMPLOYED: float = Form(...),
    INCOME: float = Form(...),
    INCOME_TYPE: str = Form(...),
    EDUCATION_TYPE: str = Form(...)
):
    input_data = {
        "GENDER": GENDER,
        "CAR": CAR,
        "REALITY": REALITY,
        "NO_OF_CHILD": NO_OF_CHILD,
        "FAMILY_TYPE": FAMILY_TYPE,
        "HOUSE_TYPE": HOUSE_TYPE,
        "WORK_PHONE": WORK_PHONE,
        "PHONE": PHONE,
        "E_MAIL": E_MAIL,
        "FAMILY SIZE": FAMILY_SIZE,
        "BEGIN_MONTH": BEGIN_MONTH,
        "AGE": AGE,
        "YEARS_EMPLOYED": YEARS_EMPLOYED,
        "INCOME": INCOME,
        "INCOME_TYPE": INCOME_TYPE,
        "EDUCATION_TYPE": EDUCATION_TYPE,
    }

    prediction_label, explanation = predict_transaction(
        model_name=model_name,
        input_data=input_data,
        models=MODELS
    )

    return templates.TemplateResponse(
        "prediction_result.html",
        {
            "request": request,
            "title": "Prediction Result",
            "model_name": model_name,
            "prediction_label": prediction_label,
            "explanation": explanation,
            "input_data": input_data
        }
    )
