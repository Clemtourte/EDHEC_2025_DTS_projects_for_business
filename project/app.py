from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import uvicorn
import re
import os

app = FastAPI(
    title="Car Price Prediction API",
    description="Predict car prices using a tuned Random Forest model",
    version="1.0.0"
)
model_dir = os.path.join(os.path.dirname(__file__), "model_artifacts")
# Load the necessary files
with open(os.path.join(model_dir, "tuned_rf_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(model_dir, "preprocessing_pipeline.pkl"), "rb") as f:
    preprocessing_pipeline = pickle.load(f)

with open(os.path.join(model_dir, "feature_selector.pkl"), "rb") as f:
    feature_selector = pickle.load(f)

# Define allowed categories from training data
VALID_DOORS = ["2", "4", "5", "6"]
VALID_FUEL_TYPES = ["Petrol", "Diesel", "CNG", "LPG"]
VALID_WHEEL = ["Left", "Right", "Left wheel", "Right wheel"]

class CarFeatures(BaseModel):
    prod_year: int = Field(..., ge=1900, le=2024, example=2004)
    mileage: float = Field(..., ge=0, example=214000)
    manufacturer: str = Field(..., example="MERCEDES-BENZ")
    model: str = Field(..., example="E 320")
    engine_volume: float = Field(..., ge=0, example=3.2)
    cylinders: int = Field(..., ge=1, example=6)
    fuel_type: str = Field(..., example="Petrol")
    gear_box_type: str = Field(..., example="Automatic")
    drive_wheels: str = Field(..., example="Rear")
    category: str = Field(..., example="Sedan")
    leather_interior: bool = Field(..., example=True)
    color: str = Field(..., example="Grey")
    airbags: int = Field(..., ge=0, example=8)
    turbo: bool = Field(..., example=False)
    levy: float = Field(..., ge=0, example=15053)
    doors: str = Field(..., example="4")
    wheel: str = Field(..., example="Left")

def validate_door_format(door: str):
    """Ensure doors contains only digits"""
    if not re.match(r'^\d+$', door):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid door format: '{door}'. Must contain only numbers (e.g. '4')"
        )
    return door

@app.post("/predict/")
def predict_price(car: CarFeatures):
    try:
        car_data = car.dict()
        
        # Validate inputs
        car_data["doors"] = validate_door_format(car_data["doors"])
        
        if car_data["doors"] not in VALID_DOORS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid doors: '{car_data['doors']}'. Valid options: {VALID_DOORS}"
            )
            
        if car_data["fuel_type"] not in VALID_FUEL_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid fuel_type: '{car_data['fuel_type']}'. Valid options: {VALID_FUEL_TYPES}"
            )
            
        if car_data["wheel"] not in VALID_WHEEL:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid wheel: '{car_data['wheel']}'. Valid options: {VALID_WHEEL}"
            )

        # Map to DataFrame with correct feature names
        input_dict = {
            "Prod. year": car_data["prod_year"],
            "Mileage": car_data["mileage"],
            "Manufacturer": car_data["manufacturer"],
            "Model": car_data["model"],
            "Engine volume": car_data["engine_volume"],
            "Cylinders": car_data["cylinders"],
            "Fuel type": car_data["fuel_type"],
            "Gear box type": car_data["gear_box_type"],
            "Drive wheels": car_data["drive_wheels"],
            "Category": car_data["category"],
            "Leather interior": 1 if car_data["leather_interior"] else 0,
            "Color": car_data["color"],
            "Airbags": car_data["airbags"],
            "Turbo": 1 if car_data["turbo"] else 0,
            "Levy": car_data["levy"],
            "Doors": car_data["doors"],
            "Wheel": 0 if "Left" in car_data["wheel"] else 1  # Map to 0/1
        }

        # Add derived features
        input_dict["Car_Age"] = 2024 - input_dict["Prod. year"]
        input_dict["Engine_Age_Ratio"] = input_dict["Engine volume"] / max(1, input_dict["Car_Age"])
        input_dict["Cylinders_per_liter"] = input_dict["Cylinders"] / max(0.1, input_dict["Engine volume"])

        # Create DataFrame
        input_df = pd.DataFrame([input_dict])

        # Debug: Print input features
        print("\nInput Features:")
        print(input_df.to_string(index=False))

        # Process data
        X_processed = preprocessing_pipeline.transform(input_df)
        X_selected = feature_selector.transform(X_processed)
        
        # Predict
        prediction = model.predict(X_selected)[0]
        prediction = max(0, round(prediction, 2))

        # Debug: Print prediction
        print(f"\nPredicted Price: ${prediction:,.2f}")

        return {"predicted_price": prediction}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)