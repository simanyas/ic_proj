from typing import Any, List, Optional, Union
import datetime

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[str]


class DataInputSchema(BaseModel):
    predict_image: Optional[str]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
