from typing import List, Optional


class EvalResult:
    def __init__(
        self,
        prediction: List[str],
        prompt: str,
        bleu_score: Optional[float] = None,
        meteor_score: Optional[float] = None,
    ):
        self.prediction = prediction
        self.prompt = prompt
        self.bleu_score = bleu_score
        self.meteor_score = meteor_score

    def __str__(self) -> str:
        return (
            f"prediction: {self.prediction}, prompt: {self.prompt},"
            f"bleu_score: {self.bleu_score}, meteor_score: {self.meteor_score} )"
        )
