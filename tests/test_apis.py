import os
import pytest
from fastapi.testclient import TestClient

from app import app

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
client = TestClient(app)


def send_test_file(filename: str, target: str):
    filepath = os.path.join(BASE_DIR, 'static', 'dataset', filename)
    response = client.post("/create", files={'csv_file': open(filepath)},
                           params={'target': target})
    return response


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World!"}


def test_creation_of_model():
    model_file = os.path.join(BASE_DIR, 'static', 'models', 'mymodel.pkl')
    response = send_test_file('iris.csv', 'Species')
    assert response.status_code == 200
    assert os.path.exists(model_file)


def test_creation_with_nodata():
    response = send_test_file('empty.csv', 'Species')
    assert response.status_code == 400


def test_creation_with_only_columns():
    response = send_test_file('withonlycolumns.csv', 'Species')
    assert response.status_code == 400


def test_creation_wrong_column():
    response = send_test_file('iris.csv', 'Test')
    assert response.status_code == 400


def test_creation_with_non_numeric_datatype():
    output = {
        "detail": "Found Non Numeric Column. "
                  "Please provide data with numeric column for classification"
    }
    response = send_test_file('nonnumeric.csv', 'Species')

    assert response.status_code == 400
    assert response.json() == output


test_data = [
    ('4.6,3.1,1.5,0.2', "setosa"),
    ('7,3.2,4.7,1.4', "versicolor"),
    ('5,2.3,3.3,1', "versicolor"),
    ('6.3,2.9,5.6,1.8', "virginica"),
    ('6.9,3.1,5.1,2.3', "virginica"),
]


@pytest.mark.parametrize("input_line,output", test_data)
def test_proper_predicition(input_line, output):
    create_response = send_test_file('iris.csv', 'Species')
    assert create_response.status_code == 200

    predict_response = client.post("/predict", params={'input_line': input_line})

    assert predict_response.status_code == 200
    assert predict_response.json() == output


def test_prediction_before_create():
    output = {
        "detail": "Please Create Model using /create api then you call this API to predict"

    }
    model_file = os.path.join(BASE_DIR, 'static', 'models', 'mymodel.pkl')
    if os.path.exists(model_file):
        os.remove(model_file)

    predict_response = client.post("/predict", params={'input_line': '4.6,3.1,1.5,0.2'})
    assert predict_response.status_code == 404
    assert predict_response.json() == output


test_data = [
    ('4.6,3.1,,0.2',),
    ('7,test,4.7,1.4',),
    ('5,2.3,3.3',),
    ('6.3,2.9,',),
    ('6.9',),
]


@pytest.mark.parametrize("input_line", test_data)
def test_prediction_with_missing_value(input_line):
    output = {
        "detail": "Please Provide proper input line"

    }

    create_response = send_test_file('iris.csv', 'Species')
    assert create_response.status_code == 200
    predict_response = client.post("/predict", params={'input_line': input_line})
    assert predict_response.status_code == 400
    assert predict_response.json() == output
