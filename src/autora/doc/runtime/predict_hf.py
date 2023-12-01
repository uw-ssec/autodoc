import logging
from typing import Dict, List

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_path: str):
        # Load the model in 4bit quantization for faster inference on smaller GPUs
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map="auto"
        )
        logger.info("Model loaded")
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def predict(self, sys: str, instr: str, inputs: List[str]) -> List[str]:
        # Standard Llama2 template
        template = f"""
[INST]<<SYS>>
{sys}

{instr}

[INPUT]
[/INST]
"""
        logger.info(f"Generating {len(inputs)} predictions")
        prompts = [template.replace("[INPUT]", input) for input in inputs]
        # TODO: Make these parameters configurable
        sequences = self.pipeline(
            prompts,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=40,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=1000,
        )

        results = [sequence[0]["generated_text"] for sequence in sequences]
        logger.info(f"Generated {len(results)} results")
        return results

    def tokenize(self, input: List[str]) -> Dict[str, List[List[int]]]:
        tokens: Dict[str, List[List[int]]] = self.tokenizer(input)
        return tokens
