"""
FastAPI Unit tests
"""
import pytest
from fastapi.testclient import TestClient

from api_server import app


@pytest.fixture
def client():
    api_client = TestClient(app)
    return api_client


def test_get_welcome(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Greetings!"}


def test_get_wrong_url(client):
    r = client.get("/wrong_url")
    assert r.status_code != 200


def test_post_case_1(client):
    r = client.post("/", json={
        "age": 32,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Some-college",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": "United-States",
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": '>50K'}


def test_post_case_2(client):
    r = client.post("/", json={
        "age": 50,
        "workclass": "Private",
        "fnlgt": 77516,
        "education": "Masters",
        "education_num": 9,
        "marital_status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 50000,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}


def test_post_wrong_input(client):
    r = client.post("/", json={
        "age": 50,
        "workclass": "Private",
        "education": "FALSE",
        "education_num": 150,
        "marital_status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 50000,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    })
    assert r.status_code == 422
