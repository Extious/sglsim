from pathlib import Path

from sglsim.srt.utils.synthesis_prompt import SynthesisPromptBuilder, SynthesisWorkloadConfig


def _builder(tmp_path: Path, body: str) -> SynthesisPromptBuilder:
    prompt_path = tmp_path / "body.txt"
    prompt_path.write_text(body, encoding="utf-8")
    return SynthesisPromptBuilder(
        SynthesisWorkloadConfig(
            model_path="mock-model",
            prompt_path=prompt_path,
            seed=1,
            request_marker_template="request {rid}/",
        )
    )


def test_file_prompt_shared_body_differs_by_marker(tmp_path):
    builder = _builder(tmp_path, "shared benchmark body text")
    ids1 = builder.build_token_ids("1", 0)
    ids2 = builder.build_token_ids("2", 1)
    body = builder.body_token_ids()
    assert ids1[len(ids1) - len(body) :] == body
    assert ids2[len(ids2) - len(body) :] == body
    assert ids1 != ids2
    assert builder.build("1", 0).startswith("request 1/")
    assert builder.build("2", 1).startswith("request 2/")


def test_file_prompt_marker_prefix(tmp_path):
    builder = _builder(tmp_path, "xyz")
    ids = builder.build_token_ids("9", 0)
    marker = builder._encode("request 9/")
    assert ids[: len(marker)] == marker
