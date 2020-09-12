import pickle as pickle
from datetime import datetime
from abc import ABCMeta, abstractmethod
import pandas as pd
from sklearn.linear_model import LogisticRegression


class BaseModel(metaclass=ABCMeta):
    """
    Base class for models

    The class has a save and load method for serializing model objects.
    It enforces implementation of a fit and predict method and a model name attribute.
    """

    def __init__(self):
        self.model_initiated_dt = datetime.utcnow()

    @property
    @classmethod
    @abstractmethod
    def MODEL_NAME(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def __str__(self):
        return f"Model: {self.MODEL_NAME},  initiated at {self.model_initiated_dt}"

    def save(self, **kwargs):
        """Serialize model to file or variable"""
        serialize_dict = self.__dict__

        if "fname" in kwargs.keys():
            fname = kwargs["fname"]
            with open(fname, "wb") as f:
                pickle.dump(serialize_dict, f)
        else:
            pickled = pickle.dumps(serialize_dict)
            return pickled

    def load(self, serialized):
        """Deserialize model from file or variable"""
        assert isinstance(serialized, str) or isinstance(serialized, bytes), (
            "serialized must be a string (filepath) or a "
            "bytes object with the serialized model"
        )
        model = self.__class__()

        if isinstance(serialized, str):
            with open(serialized, "rb") as f:
                serialize_dict = pickle.load(f)
        else:
            serialize_dict = pickle.loads(serialized)

        # Set attributes of model
        model.__dict__ = serialize_dict

        return model


class LogRegWrapper(BaseModel):
    MODEL_NAME = "Logistic regression wrapper"

    def __init__(self, features=None, params={}):
        super().__init__()
        self.features = features
        self.params = params
        self.model = LogisticRegression(**params)

    def fit(self, X, y):
        self.model.fit(X[self.features], y)
        return self

    def predict(self, X):
        predictions = self.model.predict(X[self.features])
        predictions_df = pd.DataFrame(predictions, columns=["prediction"])
        return predictions_df


class FeatureSplitModel(BaseModel):
    MODEL_NAME = "Feature split meta model"

    def __init__(self, features=None, group_column=None, group_model_dict=None):
        super().__init__()
        self.features = features
        self.group_model_dict = group_model_dict
        self.group_column = group_column

    def fit(self, X, y):
        for group in X[self.group_column].unique():
            mask = X[self.group_column] == group
            self.group_model_dict[group].fit(X[mask], y[mask])
        return self

    def predict(self, X):
        x_columns = X.columns.tolist()
        for group in X[self.group_column].unique():
            mask = X[self.group_column] == group
            predictions_df = self.group_model_dict[group].predict(X[mask])
            X = X.append(
                pd.DataFrame(columns=predictions_df.columns.tolist(), dtype=float)
            )
            X.loc[mask, predictions_df.columns.tolist()] = predictions_df.values

        prediction_columns = [col for col in X.columns.tolist() if col not in x_columns]
        return X[prediction_columns]
