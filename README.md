# Databot

Databot is a FastAPI project as it mostly resembles Flask

## Ideology
To keep code as pythonic as possible.
Created a context manager for creating API to make code as simple as possible

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install -r requirements.txt
```

## Usage

```python
python app.py
```
- open http://127.0.0.1:8000/docs/
- test or access each API

or 

- create - http://127.0.0.1:8000/create
- predict - http://127.0.0.1:8000/predict


## Choosing FastAPI over Flask
- Default OpenAPI scheme: http://127.0.0.1:8000/docs.
- Default Type Checkings. 
- support of ASYNC for faster application.
- Provision to test API without postman: http://127.0.0.1:8000/docs > URL > Try it Out!
- Default Pytest APIs to test application easily.

## Enchancement
- Jenkins Pipeline and Dockerization of the Application
- Coverage tests
- Improvement to be placed in reading larger files and faster training.
- Improvements in predict API specific error handling and message to user.
- Possible to use gherkins for much more behavioural testing
