import logging
from typing import Dict, List

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from autora.doc.runtime.prompts import LLAMA2_INST_CLOSE, TEMP_LLAMA2

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_path: str):
        config = self.get_config()

        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **config,
        )
        logger.info("Model loaded")
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def predict(self, sys: str, instr: str, inputs: List[str]) -> List[str]:
        logger.info(f"Generating {len(inputs)} predictions")
        prompts = [TEMP_LLAMA2.format(sys=sys, instr=instr, input=input) for input in inputs]
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

        results = [Predictor.trim_prompt(sequence[0]["generated_text"]) for sequence in sequences]
        logger.info(f"Generated {len(results)} results")
        return results

    @staticmethod
    def trim_prompt(output: str) -> str:
        marker = output.find(LLAMA2_INST_CLOSE)
        if marker == -1:
            logger.warning(f"Could not find end of prompt marker '{LLAMA2_INST_CLOSE}' in '{output}'")
            return output
        return output[marker + len(LLAMA2_INST_CLOSE) :]

    def tokenize(self, input: List[str]) -> Dict[str, List[List[int]]]:
        tokens: Dict[str, List[List[int]]] = self.tokenizer(input)
        return tokens

    def get_config(self) -> Dict[str, str]:
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            # Load the model in 4bit quantization for faster inference on smaller GPUs
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
                "device_map": "auto",
            }
        else:
            return {}
