import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any
import re

from PIL import Image
import cv2
import numpy as np
import gradio as gr
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
#from ic_model.config.core import TRAINED_MODEL_DIR, PRED_DIR
#from ic_model.predict import make_prediction
#from mp_model import __version__ as model_version
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import save_model, load_model
from app import __version__, schemas
from app.config import settings
from keras.layers import TextVectorization
from keras.applications import efficientnet
import tensorflow as tf
from typing import Dict, List
import re
import os
import huggingface_hub
from huggingface_hub import from_pretrained_keras
from keras import saving
from google import genai
from google.genai import types
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TextStreamer
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch

# Path to the images
IMAGES_PATH = "Flicker8k_Dataset"
# Initialize the Gemini client with your API key
client = genai.Client(api_key="AIzaSyAeMdaHga2nDhMK9wCBOVeCRbWJStmDaK0")

# Desired image dimensions
IMAGE_SIZE = (299, 299)

# Vocabulary size
VOCAB_SIZE = 10000
# Fixed length allowed for any sequence
SEQ_LENGTH = 25

# Project Directories
PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
DATASET_DIR_APP = ROOT / "datasets" / "flickr8k"
# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Per-layer units in the feed-forward network
FF_DIM = 512
BATCH_SIZE = 64
EPOCHS = 1
AUTOTUNE = tf.data.AUTOTUNE
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

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

vectorization = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH,
        standardize=custom_standardization,
    )

# Search all the image names in rlhf_rlmf.txt
rlhf_rlmf_file_names = []
def list_rlhf_rlmf_file_names():
  with open(str(ROOT) + "/rlhf_rlmf.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
            l = line.rstrip("\n")
            # Image name and captions are separated using a tab
            img_name, caption = l.split("\t")
            # Each image is repeated five times for the five different captions.
            # Each image name has a suffix `#(caption_number)`
            img_name = img_name.split("#")[0]
            if img_name not in rlhf_rlmf_file_names:
              rlhf_rlmf_file_names.append(img_name)
list_rlhf_rlmf_file_names()

def load_captions_data(filename):
    """Loads captions (text) data and maps them to corresponding images.

    Args:
        filename: Path to the text file containing caption data.

    Returns:
        caption_mapping: Dictionary mapping image names and the corresponding captions
        text_data: List containing all the available captions
    """

    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = {}
        text_data = []
        images_to_skip = set()

        for line in caption_data:
            line = line.rstrip("\n")
            # Image name and captions are separated using a tab
            img_name, caption = line.split("\t")

            # Each image is repeated five times for the five different captions.
            # Each image name has a suffix `#(caption_number)`
            img_name = img_name.split("#")[0]
            img_name = os.path.join(IMAGES_PATH, img_name.strip())

            # We will remove caption that are either too short to too long
            tokens = caption.strip().split()

            if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
                images_to_skip.add(img_name)
                continue

            if img_name.endswith("jpg") and img_name not in images_to_skip:
                # We will add a start and an end token to each caption
                caption = "<start> " + caption.strip() + " <end>"
                text_data.append(caption)

                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]

        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]

        return caption_mapping, text_data


def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """Split the captioning dataset into train and validation sets.

    Args:
        caption_data (dict): Dictionary containing the mapped caption data
        train_size (float): Fraction of all the full dataset to use as training data
        shuffle (bool): Whether to shuffle the dataset before splitting

    Returns:
        Traning and validation datasets as two separated dicts
    """

    # 1. Get the list of all image names
    all_images = list(caption_data.keys())

    # 2. Shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # 3. Split into training and validation sets
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    print(training_data)
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    # 4. Return the splits
    return training_data, validation_data

def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def process_input(img_path, captions):
    return decode_and_resize(img_path), vectorization(captions)

def make_dataset(images, captions):
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset

print("Extracting data")
command_wget_1 = 'wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'
command_wget_2 = 'wget -q https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip'
command_3 = 'unzip -qq ' + str(ROOT) + "/" + 'Flickr8k_Dataset.zip -d ' + str(ROOT)
command_4 = 'unzip -qq ' + str(ROOT) + "/" + 'Flickr8k_text.zip -d ' + str(ROOT)
#command_5 = 'rm Flickr8k_Dataset.zip Flickr8k_text.zip'
print("Download the files")
#os.system(command_wget_1)
#os.system(command_wget_2)
print("unzip dataset")
#os.system(command_3)
print("unzip text")
#os.system(command_4)
print("remove zip files")
#os.system(command_5)
print("Done extracting data")
# Pass the list of images and the list of corresponding captions
# Load the dataset
# For the training/validation set, only first caption is used

# Prepare new version of data
v = 1.0
def prepare_new_data_version():
  global v
  v += 0.1
  with open("Flickr8k.token_" + str(v) + ".txt", "w") as f:
    with open("Flickr8k.token.txt", "r") as f2:
      lines = f2.readlines()
      for line in lines:
            l = line.rstrip("\n")
            # Image name and captions are separated using a tab
            img_name, caption = l.split("\t")
            # Each image is repeated five times for the five different captions.
            # Each image name has a suffix `#(caption_number)`
            img_name = img_name.split("#")[0]
            if img_name not in rlhf_rlmf_file_names:
              f.write(line)
    f2.close()
    with open(str(ROOT) + "/rlhf_rlmf.txt", "r") as f1:
      rlines = f1.readlines()
      for rline in rlines:
        f.write(rline)
    f1.close()
  f.close()

# To be called during RLHF/RLMF workflow actions to run fine-tuning dynamically
prepare_new_data_version()          

captions_mapping, text_data = load_captions_data(str(ROOT) + "/Flickr8k_text/" + "Flickr8k.token_" + str(v) + ".txt")
vectorization.adapt(text_data)

# Split the dataset into training and validation sets
train_data, valid_data = train_val_split(captions_mapping)
print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))
list_of_v_images = list(valid_data.keys())
list_of_v_captions = list(valid_data.values())
train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
valid_dataset = make_dataset(list_of_v_images, list_of_v_captions)

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["caption"]} ]
        },
    ]
    return { "messages" : conversation }

if(v>1.0):
    '''
    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
        "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
        "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

        "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
        "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

        "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
        "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

        "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
        "unsloth/llava-1.5-7b-hf-bnb-4bit",
    ] # More models at https://huggingface.co/unsloth
    '''
    '''
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct",
        load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    )
    '''
    '''
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 16,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 16,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )
    '''
    instruction = "Caption this image in less than 25 words"

    converted_dataset = []
    for image_path, captions in train_data.items():
        for caption in captions:
            # Create a dictionary in the format expected by convert_to_conversation
            sample = {"image": image_path, "caption": caption}
            converted_dataset.append(convert_to_conversation(sample))

    '''
    FastVisionModel.for_training(model) # Enable for training!
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
        train_dataset = converted_dataset,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 30,
            # num_train_epochs = 1, # Set this instead of max_steps for full training runs
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",     # For Weights and Biases

            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_seq_length = 2048,
        ),
    )
    trainer_stats = trainer.train()
    '''
    '''
    model.push_to_hub("sindsub/llama3.2_11b_flickr8k_lora_model_"+str(v), token = "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") # Online saving
    tokenizer.push_to_hub("sindsub/llama3.2_11b_flickr8k_lora_model_"+str(v), token = "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") # Online saving
    '''
    model,tokenizer = FastVisionModel.from_pretrained(
    model_name = "sindsub/llama_flickr8k_lora_model", # Replace with your model name if different
    #tokenizer_name = "sindsub/lora_model", # Replace with your model name if different
    max_seq_length = 2048, # You can set this based on your training
    dtype = None, # Can be None, torch.float16, torch.bfloat16
    load_in_4bit = True, # Set to True if saved in 4bit, False for 16bit
    token = "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", # Use Hugging Face token if needed 
    )

    # Need to set the model to evaluation mode for inference
    FastVisionModel.for_inference(model) # Enable for inference!

    image = Image.open(train_data[0]["image"])
    instruction = "Caption this image in less than 25 words"

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                    use_cache = True, temperature = 1.5, min_p = 0.1)
    output = " ".join(_)
    print(output)
pass

# Else continue with EfficientNet model
# Data augmentation for image data
image_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.3),
    ]
)

vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1
valid_images = list(os.listdir(str(DATASET_DIR_APP)))

def get_cnn_model():
    global IMAGE_SIZE
    base_model = efficientnet.EfficientNetB0(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    # We freeze our feature extractor
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")

    def call(self, inputs, training, mask=None):
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training,
        )
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM,
            sequence_length=SEQ_LENGTH,
            vocab_size=VOCAB_SIZE,
        )
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax")

        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
            training=training,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [
                tf.expand_dims(batch_size, -1),
                tf.constant([1, 1], dtype=tf.int32),
            ],
            axis=0,
        )
        return tf.tile(mask, mult)


class ImageCaptioningModel(keras.Model):
    def __init__(
        self,
        cnn_model,
        encoder,
        decoder,
        num_captions_per_image=5,
        image_aug=None,
        **kwargs # Accept other arguments like 'trainable'
    ):
        super().__init__(**kwargs) # Pass them to the superclass
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(
            batch_seq_inp, encoder_out, training=training, mask=mask
        )
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, acc

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        if self.image_aug:
            batch_img = self.image_aug(batch_img)

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                loss, acc = self._compute_caption_loss_and_acc(
                    img_embed, batch_seq[:, i, :], training=True
                )

                # 3. Update loss and accuracy
                batch_loss += loss
                batch_acc += acc

            # 4. Get the list of all the trainable weights
            train_vars = (
                self.encoder.trainable_variables + self.decoder.trainable_variables
            )

            # 5. Get the gradients
            grads = tape.gradient(loss, train_vars)

            # 6. Update the trainable weights
            self.optimizer.apply_gradients(zip(grads, train_vars))

        # 7. Update the trackers
        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 8. Return the loss and accuracy values
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_tracker.result(),
        }

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        for i in range(self.num_captions_per_image):
            loss, acc = self._compute_caption_loss_and_acc(
                img_embed, batch_seq[:, i, :], training=False
            )

            # 3. Update batch loss and batch accuracy
            batch_loss += loss
            batch_acc += acc

        batch_acc /= float(self.num_captions_per_image)

        # 4. Update the trackers
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 5. Return the loss and accuracy values
        return {
            "loss": self.loss_tracker.result(),
            "acc": self.acc_tracker.result(),
        }

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]

    @classmethod
    def from_config(cls, config):
        # Deserialize the sub-models from their configurations
        cnn_model = keras.saving.deserialize_keras_object(config.pop("cnn_model"))
        encoder = keras.saving.deserialize_keras_object(config.pop("encoder"))
        decoder = keras.saving.deserialize_keras_object(config.pop("decoder"))
        # Deserialize the image_aug model if present
        image_aug_config = config.pop("image_aug", None)
        image_aug = keras.saving.deserialize_keras_object(image_aug_config) if image_aug_config else None

        # Create the ImageCaptioningModel instance with the deserialized sub-models
        return cls(
            cnn_model=cnn_model,
            encoder=encoder,
            decoder=decoder,
            image_aug=image_aug,
            **config # Pass any remaining config items to the constructor
        )

    def get_config(self):
        config = super().get_config()
        # Serialize the sub-models and add them to the config
        config.update({
            "cnn_model": keras.saving.serialize_keras_object(self.cnn_model),
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "decoder": keras.saving.serialize_keras_object(self.decoder),
            "num_captions_per_image": self.num_captions_per_image,
            "image_aug": keras.saving.serialize_keras_object(self.image_aug) if self.image_aug else None,
        })
        return config

cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model,
    encoder=encoder,
    decoder=decoder,
    image_aug=image_augmentation,
)

# Define the loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    reduction=None,
)

# EarlyStopping criteria
early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)


# Learning Rate Scheduler for the optimizer
class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate,
        )

    def get_config(self):
        return {
            "post_warmup_learning_rate": self.post_warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
        }

# Create a learning rate schedule
num_train_steps = len(train_dataset) * EPOCHS
num_warmup_steps = num_train_steps // 15
lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)
caption_model.compile(optimizer=keras.optimizers.Adam(lr_schedule), loss=cross_entropy)

# Fit the model
caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    callbacks=[early_stopping],
)
vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1
valid_images = list(valid_data.keys())

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
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
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

def read_image(filename=None):
    # Read image with opencv
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    # Extract width and height
    h, w = image.shape[:2]
    # # Read the image using OpenCV and convert it into the PIL format
    return Image.fromarray(image), w, h


def clean_results(results):
    """Clean the results for visualization."""
    # Use regex to find the JSON array within the string
    match = re.search(r'\[.*?\]', results, re.DOTALL)
    if match:
        return match.group(0)
    return ""

def inference(image, prompt, temp=0.5):
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",  # or "gemini-2.5-pro-exp-03-25"
        contents=[prompt, image],  # Provide both the text prompt and image as input
        config=types.GenerateContentConfig(
            temperature=temp,  # Controls creativity vs. determinism in output
        ),
    )
    return response.text  # Return the generated textual response

def yolo_result(image, caption_us):
    prompt = """
             Detect the 2d bounding box around:
             """
    # Fixed, plotting function depends on this.
    output_prompt = "Return just box_2d and labels, no additional text."
    gemini_caption_prompt = """caption?"""
    image, w, h = read_image(image, caption_us)
    if image is not None:
      print("Looking for " + caption_us)
      results = inference(image, prompt + caption_us + output_prompt)
      cln_results_str = clean_results(results)
      #print("Raw results:", results) # Added for debugging
      #print("Cleaned results string:", cln_results_str) # Added for debugging
      try:
        cln_results = json.loads(cln_results_str)  # Clean results, list convert
        if len(cln_results) == 0:
          print("Kidding")
          yolo_res = "Kidding"
          yolo_gemini_caption = inference(image, gemini_caption_prompt)
        else:
          print("Real")
          yolo_res = "Real"
      except json.JSONDecodeError as e:
        #print(f"Error decoding JSON: {e}")
        #print(f"Raw results after cleaning attempt: {cln_results_str}") # Added for debugging
        print("Exception, possibly exhausted tokens")
    return yolo_res, yolo_gemini_caption

def run_rlmf(image):
    if image is None:
        print("No image provided.Selecting random")
        image = np.random.choice(list_of_v_images)
    image = Image.fromarray(image).convert("RGB")
    caption_us = generate_caption(image)
    return image, caption_us, yolo_result(image, caption_us)

def done_with_rl_mf():
    PAT = "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"
    c0 = "git clone https://github.com/simanyas/ic_proj.git"
    c0_1 = "cp " + str(ROOT) + "/rlhf_rlmf.txt " + str(ROOT) + "/ic_proj/cpst_code_ic/ic_api/"
    c0_2 = "cd " + str(ROOT) + "/ic_proj/"
    c1 = "git add " + str(ROOT) + "/ic_proj/cpst_code_ic/ic_api/rlhf_rlmf.txt"
    c2 = "git commit -m \"adding rlhf_rlmf.txt\""
    c3 = "git push -u https://simanyas:" + PAT + "@github.com/simanyas/ic_proj.git main"
    os.system(c0)
    os.system(c0_1)
    os.system(c0_2)
    os.system(c1)
    os.system(c2)
    os.system(c3)
pass

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
