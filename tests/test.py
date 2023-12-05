from autora.doc.runtime.predict_hf import Predictor


def test_trim_prompt() -> None:
    """Verify the output of the `trim_prompt` function"""
    no_marker = "Generated text with no marker"
    output = Predictor.trim_prompt(no_marker)
    assert output == no_marker

    with_marker = """
The prompt is here
[/INST]
output
"""
    output = Predictor.trim_prompt(with_marker)
    assert output == "output\n"
