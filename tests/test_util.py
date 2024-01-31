from pathlib import Path

from autora.doc.util import get_eval_result_from_prediction, get_prompts_from_file, load_file


def test_load_file() -> None:
    prompts_file_path = Path(__file__).parent.joinpath("../data/autora/prompts/all_prompt.json").resolve()
    data = load_file(str(prompts_file_path))
    assert type(data) == list


def test_get_prompts_from_file() -> None:
    prompts_file_path = Path(__file__).parent.joinpath("../data/autora/prompts/all_prompt.json").resolve()
    prompts_list = get_prompts_from_file(str(prompts_file_path))

    assert len(prompts_list) == 3, "Expected 3 outputs"
    for prompt in prompts_list:
        assert type(prompt) == str


def test_get_eval_result_from_prediction() -> None:
    prediction = (["response1", "response2"], 0.8, 0.7)
    prompt = "prompt1"
    result = get_eval_result_from_prediction(prediction, prompt)
    expected_result = {
        "prediction": ["response1", "response2"],
        "bleu": 0.8,
        "meteor": 0.7,
        "prompt": "prompt1",
    }
    assert type(result) == dict  # Assert result is a dictionary
    assert result == expected_result  # Assert specific keys and values
