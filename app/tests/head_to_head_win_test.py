# app/tests/head_to_head_win_test.py
from fastapi.testclient import TestClient
from app.main import app  # assuming this points to your FastAPI app
import time 
import json

client = TestClient(app)

def test_head_to_head_win_default():
    response = client.get("/predict/mlb/head-to-head-win-prediction")
    assert response.status_code == 200
    data = response.json()  
    assert isinstance(data, list)



def test_head_to_head_win_with_n():
    response = client.get("/predict/mlb/head-to-head-win-prediction?n=1")

    assert response.status_code == 200
    data = response.json()  
    assert isinstance(data, list)
    #assert len(data) == 1

