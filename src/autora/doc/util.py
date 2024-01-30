import json
from typing import Any, Dict, List, Tuple

from autora.doc.runtime.prompts import PromptBuilder


def load_file(json_file_path: str) -> List[Dict[str, Any]]:
    # Read and parse the JSON file
    with open(json_file_path, "r") as file:
        data: List[Dict[str, Any]] = json.load(file)
    return data


def get_prompts_from_file(prompts_file: str) -> List[str]:
    prompts_data = load_file(prompts_file)
    prompts_list = [PromptBuilder(p["SYS"], p["INSTR"]).build() for p in prompts_data]
    return prompts_list


def get_eval_result_from_prediction(
    prediction: Tuple[List[str], float, float], prompt: str
) -> Dict[str, Any]:
    eval_result = {
        "prediction": prediction[0],
        "bleu": prediction[1],
        "meteor": prediction[2],
        "prompt": prompt,
    }
    return eval_result
