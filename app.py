import os
import pickle
import uvicorn
from fastapi import FastAPI, File, UploadFile, status
from fastapi import HTTPException
from utils.create import ModelCreator
from io import BytesIO

app = FastAPI()


@app.get("/")
def read_root():
    return {"msg": "Hello World!"}


@app.post("/create")
def create_model(target: str, csv_file: UploadFile = File('')):
    csv_file = BytesIO(csv_file.file.read())
    with ModelCreator(csv_file, target=target) as model:
        model.train()
        model.save_model("mymodel.pkl")

    return {"message": "Created Model Successfully"}


@app.post("/predict")
def predict_model(input_line: str):
    filename = os.path.join("static", 'models', 'mymodel.pkl')

    if not os.path.exists(filename):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Please Create Model using /create api then you call this API to predict")

    with open(filename, 'rb') as fid:
        try:
            classifier = pickle.load(fid)
            input_line = [float(value) for value in input_line.split(',')]
            y_pred = classifier.predict([input_line])
            return y_pred[0]
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Please Provide proper input line")


if __name__ == '__main__':
    uvicorn.run('app:app', host='127.0.0.1', port=8000, reload=True)
