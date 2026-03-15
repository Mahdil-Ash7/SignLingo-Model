from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/predict")
def predict():
    # return dummy prediction
    return {"prediction": "A"}