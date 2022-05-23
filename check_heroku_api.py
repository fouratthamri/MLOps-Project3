"""
Test Heroku API
"""
import requests


input_data = {
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
    }
r = requests.post('https://classifier-api-heroku.herokuapp.com/', json=input_data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
