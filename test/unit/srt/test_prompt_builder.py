from sglsim.srt.utils.prompt_builder import PromptBuilder


def test_mock_prompt_builder_token_length():
    builder = PromptBuilder("mock-model", 16, seed=42)
    ids_a = builder.build_token_ids(0)
    ids_b = builder.build_token_ids(1)
    assert len(ids_a) == 16
    assert len(ids_b) == 16
    assert ids_a != ids_b
