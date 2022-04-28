"""
Heroku Api test script
"""
import requests


data = {
    "age": 24,
    "workclass": "Private",
    "education": "Some-college",
    "maritalStatus": "Divorced",
    "occupation": "Tech-support",
    "relationship": "Husband",
    "race": "Black",
    "sex": "Male",
    "hoursPerWeek": 40,
    "nativeCountry": "United-States"
    }
r = requests.post('https://classifier-api-heroku.herokuapp.com/', json=data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
