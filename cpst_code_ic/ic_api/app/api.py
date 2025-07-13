import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
#from ic_model.config.core import TRAINED_MODEL_DIR, PRED_DIR
#from ic_model.predict import make_prediction
#from mp_model import __version__ as model_version
from tensorflow import keras
from app import __version__, schemas
from app.config import settings
from keras.layers import TextVectorization
import tensorflow as tf
from typing import Dict, List
import re
import os

# Project Directories
PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
DATASET_DIR_APP = ROOT / "datasets" / "flickr8k"
reload_path = str(ROOT / "ic__model_output_v0.0.1.keras")
reloaded_model = keras.models.load_model(reload_path)

#reloaded_model = keras.models.load_model("ic__model_output_v0.0.1.keras")
model_version = "0.0.1"
# Vocabulary size
VOCAB_SIZE = 10000
# Fixed length allowed for any sequence
SEQ_LENGTH = 25
api_router = APIRouter()
# Desired image dimensions
IMAGE_SIZE = (299, 299)

strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
strip_chars = strip_chars.replace("<", "")
strip_chars = strip_chars.replace(">", "")

def preprocess_training_data():
    print("Extracting data")
    command_3 = 'unzip -qq ' + str(DATASET_DIR_APP) + "/" + 'Flickr8k_Dataset.zip -d ' + str(DATASET_DIR_APP)
    command_4 = 'unzip -qq ' + str(DATASET_DIR_APP) + "/" + 'Flickr8k_text.zip -d ' + str(DATASET_DIR_APP)
    print("unzip dataset")
    os.system(command_3)
    print("unzip text")
    os.system(command_4)
    #print("remove zip files")
    #os.system(command_5)
    print("Done extracting data")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

vectorization = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH,
        standardize=custom_standardization,
    )
def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1
valid_images = list(os.listdir(str(DATASET_DIR_APP)))


def generate_caption(test_image=""):
    # Select a random image from the validation dataset
    if test_image == "":
        sample_img = np.random.choice(valid_images)
    else:
        sample_img = test_image

    # Read the image from the disk
    sample_img = decode_and_resize(str(DATASET_DIR_APP) + "/" + "Flickr8k_Dataset" + sample_img)
    img = sample_img.numpy().clip(0, 255).astype(np.uint8)

    # Pass the image to the CNN
    img = tf.expand_dims(sample_img, 0)
    img = reloaded_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = reloaded_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = reloaded_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    print("Predicted Caption: ", decoded_caption)
    return decoded_caption

def make_prediction(test_image) -> dict:
    predicted_caption = generate_caption(test_image)
    errors = False
    results = {"predictions": predicted_caption, "version": _version, "errors": errors}
    print("Results:", results, predicted_caption)
    return results

@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()

example_input = {
    "inputs": [
        {
            "predict_image": ""
        }
    ]
}

@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    Mask prediction with the mp_model
    """
    results = make_prediction("")

    if results["errors"] is not False:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results