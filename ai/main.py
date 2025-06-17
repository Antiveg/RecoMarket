from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from hybrid import hybrid_recommendation

app = FastAPI()

origins = [
    "http://localhost:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.get("/recommendations/{user_id}")
def recommend_for_user(user_id: int, top_n: int = 5, collabweight: float = 0.5, contentweight: float = 0.5):
    try:
        recommendations = hybrid_recommendation(user_id, top_n, weight_svd=collabweight, weight_nn=contentweight)
        response_data = [(int(item), float(score)) for item, score in recommendations]
        return {"user_id": user_id, "recommendations": response_data}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"User ID {user_id} not found in dataset.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {e}")