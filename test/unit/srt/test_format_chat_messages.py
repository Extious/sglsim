from sglsim.srt.utils.tokenizer_runtime import format_chat_messages


def test_format_chat_messages_mock():
    text = format_chat_messages(
        "mock-model",
        [
            {"role": "user", "content": "sys\n\nq0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "q1"},
        ],
    )
    assert "<|user|>" in text
    assert "<|assistant|>" in text
    assert "q1" in text
    assert text.endswith("\n")
