import logging
from timeit import default_timer as timer
from typing import List

import jsonlines
import mlflow
import torch
import typer

from autora.doc.runtime.predict_hf import Predictor
from autora.doc.runtime.prompts import INSTR, SYS, InstructionPrompts, SystemPrompts

app = typer.Typer()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(module)s.%(funcName)s(): %(message)s",
)
logger = logging.getLogger(__name__)


@app.command()
def eval(data_file: str, model_path: str, sys_id: SystemPrompts, instruc_id: InstructionPrompts) -> List[str]:
    run = mlflow.active_run()

    sys_prompt = SYS[sys_id]
    instr_prompt = INSTR[instruc_id]
    if run is None:
        run = mlflow.start_run()
    with run:
        logger.info(f"Active run_id: {run.info.run_id}")
        logger.info(f"running predict with {data_file}")
        logger.info(f"model path: {model_path}")

        with jsonlines.open(data_file) as reader:
            items = [item for item in reader]
            inputs = [item["instruction"] for item in items]
            labels = [item["output"] for item in items]

        pred = Predictor(model_path)
        timer_start = timer()
        predictions = pred.predict(sys_prompt, instr_prompt, inputs)
        timer_end = timer()
        pred_time = timer_end - timer_start
        mlflow.log_metric("prediction_time/doc", pred_time / (len(inputs)))
        for i in range(len(inputs)):
            mlflow.log_text(labels[i], f"label_{i}.txt")
            mlflow.log_text(inputs[i], f"input_{i}.py")
            mlflow.log_text(predictions[i], f"prediction_{i}.txt")

        tokens = pred.tokenize(predictions)["input_ids"]
        total_tokens = sum([len(token) for token in tokens])
        mlflow.log_metric("total_tokens", total_tokens)
        mlflow.log_metric("tokens/sec", total_tokens / pred_time)
        return predictions


@app.command()
def generate(
    python_file: str,
    model_path: str = "meta-llama/llama-2-7b-chat-hf",
    output: str = "output.txt",
    sys_id: SystemPrompts = SystemPrompts.SYS_1,
    instruc_id: InstructionPrompts = InstructionPrompts.INSTR_SWEETP_1,
) -> None:
    with open(python_file, "r") as f:
        inputs = [f.read()]
    sys_prompt = SYS[sys_id]
    instr_prompt = INSTR[instruc_id]
    pred = Predictor(model_path)
    predictions = pred.predict(sys_prompt, instr_prompt, inputs)
    assert len(predictions) == 1, f"Expected only one output, got {len(predictions)}"
    logger.info(f"Writing output to {output}")
    with open(output, "w") as f:
        f.write(predictions[0])


@app.command()
def import_model(model_name: str) -> None:
    pass


if __name__ == "__main__":
    logger.info(f"Torch version: {torch.__version__} , Cuda available: {torch.cuda.is_available()}")

    mlflow.autolog()
    app()
