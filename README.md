# Car Price Prediction API
### Project Overview
This project focuses on developing a machine learning model to predict the price of used cars based on various features such as mileage, engine volume, fuel type, manufacturer, and more. The final model is integrated into an API, enabling users to get car price predictions by sending requests to a FastAPI web service.

### Dataset
The dataset used in this project was obtained from Kaggle and contains various features related to car sales. It underwent preprocessing to clean the data, handle missing values, and extract meaningful features.
https://www.kaggle.com/datasets/sidharth178/car-prices-dataset

### Machine Learning Approach
#### Data Preprocessing:
- Cleaning categorical and numerical data
- Handling missing values
- Feature engineering (e.g., car age, engine age ratio)

#### Model Training & Selection:
- Baseline Linear Regression model
- Comparison of multiple models: Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, KNN
- Random Forest was selected as the best model based on RMSE and R².

#### Hyperparameter Tuning:
- GridSearchCV was used to fine-tune Random Forest hyperparameters.
- Final model achieved an RMSE of $9,022.34, improving from the initial $11,922.02.

#### API Implementation
The trained model was deployed as an API using FastAPI and hosted on Render. The API allows users to input car details and receive a price prediction.

##### Endpoints
- GET /: Returns a welcome message and API documentation link.
- POST /predict/: Accepts car features as JSON input and returns a predicted price.

##### Example API Request (go in the docs and click on try it out)
{
    "prod_year": 2018,
    "mileage": 30000,
    "manufacturer": "Toyota",
    "model": "Corolla",
    "engine_volume": 1.8,
    "cylinders": 4,
    "fuel_type": "Petrol",
    "gear_box_type": "Automatic",
    "drive_wheels": "Front",
    "category": "Sedan",
    "leather_interior": false,
    "color": "White",
    "airbags": 6,
    "turbo": false,
    "levy": 500,
    "doors": "4",
    "wheel": "Left"
}

##### Example API Response
{
    "predicted_price": 13500.75
}

### Project Structure
|-- requirements.txt             # Dependencies for the project\
|-- .gitignore\
|-- README.md\
|-- project/
>    |-- app.py                  # API Implementation using FastAPI\
>    |-- baseline_model.py        # Baseline model (Linear Regression)\
>    |-- comp_models.py           # Model comparison script\
>    |-- hyperparam_tuning.py     # Hyperparameter tuning script\
>    |-- clean_data.ipynb         # data cleaning notebook\
>    |-- run_project.ipynb        # notebook including a run of all files to have a direct visual output\
>    |-- model_artifacts/         # Saved models, preprocessing pipeline, and artifacts\


### How to Run Locally
#### Clone the Repository

git clone https://github.com/Clemtourte/EDHEC_2025_DTS_projects_for_business/
cd project/

#### Set Up a Virtual Environment (pyenv used in our case with WSL, but you can very well use virtualenv as shown below)
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate      # Windows

#### Install Dependencies
pip install -r requirements.txt

#### Run the API
uvicorn app:app --host 0.0.0.0 --port 8000

#### Access the API Documentation Open http://127.0.0.1:8000/docs in your browser.

### Deployment
The API is deployed on Render. You can test it using:
- API URL: https://edhec-2025-dts-projects-for-business.onrender.com/
- Swagger Docs: https://edhec-2025-dts-projects-for-business.onrender.com/docs
