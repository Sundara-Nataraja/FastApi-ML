import pickle
import os
from typing import Optional
from fastapi import HTTPException, status
import pandas as pd
from pandas.errors import EmptyDataError
from pandas.api.types import is_numeric_dtype
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


class ModelCreator(object):

    def __init__(self, csv_file, target):
        self.csv_file = csv_file
        self.target = target
        self.__dataset = None
        self.classifier = SVC(kernel='linear', random_state=0)

    @property
    def data(self):
        return self.__dataset

    @property
    def X(self):
        return self.__dataset.loc[:, self.__dataset.columns != self.target]

    @property
    def y(self):
        return self.__dataset.loc[:, self.__dataset.columns == self.target].values.flatten()

    def train(self):
        self.classifier.fit(self.X, self.y)

    def save_model(self, filename: str) -> Optional[bool]:
        # raises NotFitted Exception if classifer is not trained
        check_is_fitted(self.classifier, msg="Please Train data before saving the model!")
        with open(os.path.join("static", 'models', filename), 'wb') as fid:
            pickle.dump(self.classifier, fid)
        return True

    def __enter__(self):
        try:
            self.__dataset = pd.read_csv(self.csv_file)
            target_column = self.target.strip()
            if target_column not in self.__dataset.columns:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f"Column Name {target_column} does not present."
                                           f"Choose from {list(self.__dataset.columns)}")
            if not all(is_numeric_dtype(ctypes) for ctypes in self.X.dtypes):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                    detail="Found Non Numeric Column. "
                                           "Please provide data with numeric column for classification")

        except EmptyDataError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please Provide CSV with data")
        if len(self.__dataset) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Please Provide CSV with data")

        return self

    def __exit__(self, extype: Exception, exvalue: int, extraceback: str) -> Optional[bool]:
        if extype == NotFittedError:
            raise NotFittedError("Please Train the data before save to model")

        return True

