from pathlib import Path

from autora.doc.pipelines.main import predict
from autora.doc.runtime.prompts import InstructionPrompts, SystemPrompts

# dummy HF model for testing
TEST_HF_MODEL = "hf-internal-testing/tiny-random-FalconForCausalLM"


def test_predict() -> None:
    data = Path(__file__).parent.joinpath("../data/data.jsonl").resolve()
    outputs = predict(str(data), TEST_HF_MODEL, SystemPrompts.SYS_1, InstructionPrompts.INSTR_SWEETP_1)
    assert len(outputs) == 3, "Expected 3 outputs"
    for output in outputs:
        assert len(output) > 0, "Expected non-empty output"
