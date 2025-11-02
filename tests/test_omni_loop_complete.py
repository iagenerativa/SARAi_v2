"""Coverage-focused tests for ``core.omni_loop``."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class DummyLFM2:
    """Callable stub that mimics llama-cpp style interface."""

    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or ["Draft response"]

    def __call__(self, prompt, **kwargs):
        if not self.responses:
            text = "Default response"
        else:
            text = self.responses.pop(0)
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        tokens = max(1, len(text.split()))
        return {
            "choices": [{"text": text}],
            "usage": {"completion_tokens": tokens}
        }


@pytest.fixture(autouse=True)
def patch_detect_and_pool(monkeypatch):
    """Provide deterministic skill detection and model pool mocks."""

    dummy_pool = MagicMock()
    draft_model = DummyLFM2(["Draft 1", "Draft 2", "Draft 3"])
    fallback_model = DummyLFM2(["Fallback response"])

    def pool_get(name):
        # First call returns draft, subsequent calls fallback
        if pool_get.call_count == 0:
            pool_get.call_count += 1
            return draft_model
        pool_get.call_count += 1
        return fallback_model

    pool_get.call_count = 0
    dummy_pool.get.side_effect = pool_get

    monkeypatch.setattr("core.model_pool.get_model_pool", lambda: dummy_pool)

    skill_config = {
        "skill_name": "draft",
        "system_prompt": "You are a fast drafting assistant.",
        "generation_params": {
            "max_tokens": 150,
            "temperature": 0.9,
            "top_p": 0.95,
            "stop": []
        },
        "full_prompt": "You are a fast drafting assistant.\n\nDraft 1"
    }

    monkeypatch.setattr("core.mcp.detect_and_apply_skill", lambda query, model_name="lfm2": skill_config)

    yield


def test_execute_loop_single_iteration_no_reflection():
    from core.omni_loop import OmniLoop

    loop = OmniLoop()
    result = loop.execute_loop(
        prompt="Summarize the document",
        enable_reflection=False,
        max_iterations=1
    )

    assert result["response"] == "Draft 1"
    assert len(result["iterations"]) == 1
    assert result["metadata"]["num_iterations"] == 1


def test_execute_loop_reflection_multiple_iterations():
    from core.omni_loop import OmniLoop

    loop = OmniLoop()
    result = loop.execute_loop(
        prompt="Explain quantum tunneling",
        enable_reflection=True,
        max_iterations=3
    )

    assert result["metadata"]["num_iterations"] >= 1
    assert result["metadata"]["num_iterations"] <= 3


def test_execute_loop_fallback_path(monkeypatch):
    from core.omni_loop import OmniLoop

    loop = OmniLoop()

    # Force _run_iteration to fail to trigger fallback
    def raise_iteration(*_, **__):
        raise RuntimeError("draft-error")

    monkeypatch.setattr(loop, "_run_iteration", raise_iteration)
    monkeypatch.setattr(loop, "_fallback_lfm2", lambda prompt: "Fallback response")

    result = loop.execute_loop(
        prompt="Handle failure gracefully",
        enable_reflection=False
    )

    assert result["fallback_used"] is True
    assert result["response"] == "Fallback response"


def test_run_iteration_uses_fallback_when_draft_fails(monkeypatch):
    from core.omni_loop import OmniLoop

    loop = OmniLoop()
    def raise_draft(*_, **__):
        raise RuntimeError("draft fail")

    monkeypatch.setattr(loop, "_call_draft_skill", raise_draft)

    monkeypatch.setattr(
        loop,
        "_call_local_lfm2",
        lambda prompt: {"text": "Local fallback", "tokens": 5, "tokens_per_second": 10.0}
    )

    iteration = loop._run_iteration(
        prompt="Improve answer",
        image_path=None,
        iteration=2,
        previous_response="Previous draft"
    )

    assert iteration.response == "Local fallback"
    assert iteration.source == "lfm2"
    assert iteration.corrected is True


def test_build_full_prompt_with_previous_response():
    from core.omni_loop import OmniLoop

    loop = OmniLoop()
    prompt = loop._build_full_prompt("Question", "Answer")

    assert "Previous attempt" in prompt
    assert "Question" in prompt


def test_build_reflection_prompt_handles_gpg_failure(monkeypatch):
    from core.omni_loop import OmniLoop

    # Force GPG signer to raise to cover fallback branch
    monkeypatch.setattr("core.gpg_signer.GPGSigner", lambda key_id: (_ for _ in ()).throw(RuntimeError("gpg missing")))

    loop = OmniLoop()
    prompt = loop._build_reflection_prompt("Question", "Draft", iteration=2)

    assert "Draft" in prompt
    assert "Question" in prompt


def test_calculate_confidence_scores():
    from core.omni_loop import OmniLoop

    loop = OmniLoop()
    confidence = loop._calculate_confidence(
        response="This is a concise explanation of tunneling effects in quantum mechanics.",
        prompt="Explain tunneling effects in quantum mechanics",
        iteration=3
    )

    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.2


def test_preprocess_image_disabled(tmp_path):
    from core.omni_loop import LoopConfig, OmniLoop

    config = LoopConfig(use_skill_image=False)
    loop = OmniLoop(config)
    sample = tmp_path / "img.png"
    sample.write_bytes(b"fake-data")

    processed = loop._preprocess_image(str(sample))
    assert processed == str(sample)


def test_build_result_metadata():
    from core.omni_loop import OmniLoop, LoopIteration

    loop = OmniLoop()
    iterations = [
        LoopIteration(1, "Draft", 0.4, False, 100.0, 10, 40.0, "draft"),
        LoopIteration(2, "Refined", 0.9, True, 120.0, 12, 48.0, "draft"),
    ]

    start = 0.0
    result = loop._build_result(iterations, start, fallback=False)

    assert result["auto_corrected"] is True
    assert result["metadata"]["total_tokens"] == 22
    assert result["metadata"]["num_iterations"] == 2


def test_loop_history_management():
    from core.omni_loop import OmniLoop

    loop = OmniLoop()
    loop.loop_history.append({"response": "a"})
    history = loop.get_loop_history()
    assert history[-1]["response"] == "a"

    loop.clear_history()
    assert loop.get_loop_history() == []


def test_loop_config_sanitizes_max_iterations(caplog):
    from core.omni_loop import LoopConfig, OmniLoop

    with caplog.at_level("WARNING"):
        loop = OmniLoop(LoopConfig(max_iterations=5))

    assert loop.config.max_iterations == 3
    assert "max_iterations" in caplog.text


def test_preprocess_image_skill_path(monkeypatch, tmp_path):
    from core.omni_loop import LoopConfig, OmniLoop

    skills_module = types.ModuleType("skills")
    skills_pb2 = types.ModuleType("skills_pb2")

    class FakeImageReq:
        def __init__(self, image_bytes, target_format, max_size):
            self.image_bytes = image_bytes
            self.target_format = target_format
            self.max_size = max_size

    skills_pb2.ImageReq = FakeImageReq
    skills_module.skills_pb2 = skills_pb2

    monkeypatch.setitem(sys.modules, "skills", skills_module)
    monkeypatch.setitem(sys.modules, "skills.skills_pb2", skills_pb2)

    class FakeClient:
        def Preprocess(self, request, timeout=5.0):
            return types.SimpleNamespace(image_bytes=b"processed", image_hash="123abc")

    class DummyPool:
        def get_skill_client(self, name):
            assert name == "image"
            return FakeClient()

    monkeypatch.setattr("core.model_pool.get_model_pool", lambda: DummyPool())

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"raw")

    loop = OmniLoop(LoopConfig(use_skill_image=True))
    result_path = loop._preprocess_image(str(image_path))

    assert result_path.endswith(".webp")
    assert Path(result_path).exists()


def test_build_reflection_prompt_signs_when_gpg_available(monkeypatch):
    from core.omni_loop import OmniLoop

    class FakeSigner:
        def __init__(self, key_id):
            self.key_id = key_id

        def sign_prompt(self, prompt):
            return prompt + "\n--SIGNED--"

    monkeypatch.setattr("core.gpg_signer.GPGSigner", FakeSigner)

    loop = OmniLoop()
    signed_prompt = loop._build_reflection_prompt("Task", "Draft", iteration=2)

    assert "--SIGNED--" in signed_prompt


def test_fallback_lfm2_handles_exception(monkeypatch):
    from core.omni_loop import OmniLoop

    loop = OmniLoop()

    def raise_error(prompt):
        raise RuntimeError("lfm2 failure")

    monkeypatch.setattr(loop, "_call_local_lfm2", raise_error)

    fallback_text = loop._fallback_lfm2("Explain fallback")
    assert "Lo siento" in fallback_text


def test_get_omni_loop_singleton():
    from core.omni_loop import OmniLoop, get_omni_loop

    instance = get_omni_loop()
    assert isinstance(instance, OmniLoop)
    assert get_omni_loop() is instance
