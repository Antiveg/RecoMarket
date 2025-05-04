from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

model_filename = './SVD/model_surprise.pkl'
trainset_filename = './SVD/trainset_surprise.pkl'

origins = [
    "http://localhost:5000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

def load_model_and_data():
    try:
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
        with open(trainset_filename, 'rb') as f:
            trainset = pickle.load(f)
        return model, trainset
    except Exception as e:
        raise RuntimeError(f"Error loading model or dataset: {e}")

model, trainset = load_model_and_data()

def get_recommendations(user_id: int, top_n=5):
    try:
        all_products = set(trainset.all_items())
        interacted_products = set([iid for (iid, _) in trainset.ur[trainset.to_inner_uid(user_id)]])
        candidates = list(all_products - interacted_products)
        predictions = [model.predict(user_id, trainset.to_raw_iid(iid)) for iid in candidates]
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_predictions = predictions[:top_n]
        recommended_product_ids = [pred.iid for pred in top_predictions]
        return recommended_product_ids
    except KeyError:
        raise HTTPException(status_code=404, detail=f"User ID {user_id} not found in dataset.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {e}")

@app.get("/recommendations/{user_id}")
def recommend_for_user(user_id: int, top_n: int = 5):
    recommendations = get_recommendations(user_id, top_n)
    return {"user_id": user_id, "recommendation_ids": recommendations}