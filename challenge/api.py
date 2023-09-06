from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from challenge.model import DelayModel

app = FastAPI()
model = DelayModel()

# List of valid airlines
VALID_AIRLINES = [
    "Aerolineas Argentinas",
    "Aeromexico",
    "Air Canada",
    "Air France",
    "Alitalia",
    "American Airlines",
    "Austral",
    "Avianca",
    "British Airways",
    "Copa Air",
    "Delta Air",
    "Gol Trans",
    "Grupo LATAM",
    "Iberia",
    "JetSmart SPA",
    "K.L.M.",
    "Lacsa",
    "Latin American Wings",
    "Oceanair Linhas Aereas",
    "Plus Ultra Lineas Aereas",
    "Qantas Airways",
    "Sky Airline",
    "United Airlines"
]


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Exception handler for RequestValidationError.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()},
    )


@app.get("/health", status_code=status.HTTP_200_OK)
async def get_health() -> dict:
    """
    Endpoint to check the health of the API.
    """
    return {"status": "OK"}


@app.post("/predict", status_code=status.HTTP_200_OK)
async def post_predict(data: dict) -> dict:
    """
    Endpoint to predict flight delays.

    Parameters:
    - data: dict containing a list of flights data.

    Returns:
    - dict containing a list of predictions.
    """
    try:
        global model

        flights = data.get("flights", [])
        predictions_list = []

        if not flights:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="No flights data provided."
            )

        for flight in flights:
            month = flight.get("MES")
            flight_type = flight.get("TIPOVUELO")
            airline = flight.get("OPERA")

            # Validate flight data
            if (
                month is None or not (1 <= month <= 12)
                or flight_type not in ["I", "N"]
                or airline is None
                or airline not in VALID_AIRLINES
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid flight data."
                )

            df = pd.DataFrame([flight])
            data_process = model.preprocess(df)
            prediction = model.predict(data_process)
            predictions_list.append(prediction[0])

        return {"predict": predictions_list}

    except HTTPException as e:
        print(f"HTTP Exception: {e.status_code}, Detail: {e.detail}")
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        print(f"Exception: {str(e)}")
        return {"error": str(e)}
