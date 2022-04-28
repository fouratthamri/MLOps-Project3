# ML-DevOPs Project 3

Udacity project num 3

# How to run


## Steps


* Data cleaning `python main.py --action basic_cleaning`

* Model training  `python main.py --action train_test_model`

* Evaluate performance `python main.py --action check_score`

* To run all the steps `python main.py --action all` or `python main.py`

## Testing
* Run all the tests `pytest`



### Run API locally

If testing FastAPi serving on local is needed, execute `uvicorn api_server:app --reload`

### Test API Deployment on Heroku

Check Heroku deployed APi using `python check_heroku_api.py`


