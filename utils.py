import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
import json
import re
import os
import string
import time

def update_tokenizer(tokenizer):
    tokenizer.add_special_tokens({"additional_special_tokens": ["<cls>"]})
    return tokenizer

def batch_iterator(iterable, size=1, return_index=False):
    l = len(iterable)
    for ndx in range(0, l, size):
        if return_index:
            yield (ndx, min(ndx + size, l))
        else:
            yield iterable[ndx:min(ndx + size, l)]

def load_model(model_name_or_path, model_class='causualLM', device='cpu'):

    from transformers import AutoTokenizer
    from models import FiDT5
    MODEL_CLASS = {"fid": FiDT5}[model_class]

    logger.info(f"Loading {model_name_or_path} ({model_class})")
    start_time = time.time()

    model = MODEL_CLASS.from_pretrained(
        model_name_or_path, 
        device_map='cuda' if torch.cuda.is_available() else 'cpu',
        torch_dtype=torch.bfloat16,
    )

    logger.info("Finish loading in %.2f sec." % (time.time() - start_time))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    return model, tokenizer

