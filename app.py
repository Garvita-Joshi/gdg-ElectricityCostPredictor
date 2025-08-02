from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the trained pipeline
model = joblib.load("model.pkl")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    site_area: float = Form(...),
    structure_type: str = Form(...),
    water_consumption: float = Form(...),
    recycling_rate: float = Form(...),
    utilisation_rate: float = Form(...),
    air_qality_index: float = Form(...),   # keep typo to match dataset
    issue_reolution_time: float = Form(...), # keep typo to match dataset
    resident_count: float = Form(...)
):
    try:
        # Build DataFrame with EXACT column names from your dataset
        input_data = pd.DataFrame([{
            "site area": site_area,
            "structure type": structure_type,
            "water consumption": water_consumption,
            "recycling rate": recycling_rate,
            "utilisation rate": utilisation_rate,
            "air qality index": air_qality_index,  
            "issue reolution time": issue_reolution_time,  
            "resident count": resident_count,
        }])

        # Prediction
        prediction = model.predict(input_data)[0]

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"Predicted Electricity Cost: {prediction:.2f}"
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": f"Error: {str(e)}"
        })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
