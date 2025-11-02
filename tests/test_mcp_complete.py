"""Comprehensive tests for ``core.mcp`` covering rules, cache and skills."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml


class DummyEmbedder:
    """Deterministic embedder for cache tests."""

    def __init__(self, values=None):
        self.values = values or np.linspace(-1.0, 1.0, 8)
        self.calls = 0

    def encode(self, _: str):
        self.calls += 1
        return np.array(self.values)


@pytest.fixture
def embedder():
    return DummyEmbedder()


@pytest.fixture
def mcp_config(tmp_path) -> Path:
    config = {
        "mcp": {
            "mode": "rules",
            "feedback_buffer_size": 10,
            "checkpoint_path": str(tmp_path / "checkpoint.pt"),
            "training": {
                "min_samples": 2,
                "learning_rate": 0.001,
                "batch_size": 4,
            },
        }
    }
    config_path = tmp_path / "models.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def test_mcp_cache_hit_and_miss(embedder):
    from core.mcp import MCPCache

    cache = MCPCache(embedder, ttl=60, quant_levels=16)
    cache.set("context", 0.6, 0.4)

    assert cache.get("context") == (0.6, 0.4)

    embedder.encode = lambda _: np.linspace(-1.0, 1.0, 8)[::-1]
    assert cache.get("new context") is None


def test_mcp_cache_ttl(monkeypatch, embedder):
    from core.mcp import MCPCache

    cache = MCPCache(embedder, ttl=1)

    base_time = 1_000.0
    monkeypatch.setattr("core.mcp.time.time", lambda: base_time)
    cache.set("ctx", 0.7, 0.3)

    monkeypatch.setattr("core.mcp.time.time", lambda: base_time + 0.5)
    assert cache.get("ctx") == (0.7, 0.3)

    monkeypatch.setattr("core.mcp.time.time", lambda: base_time + 2.0)
    assert cache.get("ctx") is None


def test_mcp_cache_clear_expired(monkeypatch, embedder):
    from core.mcp import MCPCache

    cache = MCPCache(embedder, ttl=1)
    base_time = 500.0

    monkeypatch.setattr("core.mcp.time.time", lambda: base_time)
    cache.set("ctx", 0.5, 0.5)

    monkeypatch.setattr("core.mcp.time.time", lambda: base_time + 5.0)
    cache.clear_expired()

    assert cache.get("ctx") is None


def test_mcp_rules_apply_technical_bias(mcp_config, embedder):
    from core.mcp import MCPRules

    rules = MCPRules(str(mcp_config), embedder=embedder)
    alpha, beta = rules.compute_weights(0.9, 0.1, context="bug report")

    assert alpha > beta
    assert pytest.approx(alpha + beta, rel=1e-6) == 1.0


def test_mcp_rules_cache_paths(monkeypatch, mcp_config, embedder):
    from core.mcp import MCPRules

    rules = MCPRules(str(mcp_config), embedder=embedder)

    # Force cache to return predefined value on second call
    first = rules.compute_weights(0.6, 0.5, context="explain x")
    second = rules.compute_weights(0.6, 0.5, context="explain x")

    assert first == second


def test_mcp_rules_adjust_with_feedback(mcp_config, embedder):
    from core.mcp import MCPRules

    rules = MCPRules(str(mcp_config), embedder=embedder)
    feedback_buffer = [
        {"alpha": 0.85, "beta": 0.15, "feedback": 0.0} for _ in range(5)
    ] + [
        {"alpha": 0.2, "beta": 0.8, "feedback": 0.0} for _ in range(5)
    ]

    alpha, beta = rules.compute_weights(0.9, 0.2, context="ctx", feedback_buffer=feedback_buffer)

    assert beta == pytest.approx(0.2, abs=1e-6)
    assert alpha == pytest.approx(0.8, abs=1e-6)


def test_mcp_initial_mode_rules(mcp_config):
    from core.mcp import MCP

    mcp = MCP(str(mcp_config))
    assert mcp.mode == "rules"

    weights = mcp.compute_weights({"hard": 0.8, "soft": 0.1}, context="tech question")
    assert len(weights) == 2


def test_mcp_switches_to_learned(monkeypatch, mcp_config):
    from core.mcp import MCP

    mcp = MCP(str(mcp_config))

    trained = {"called": 0}

    def fake_train(self):  # noqa: D401
        trained["called"] += 1

    monkeypatch.setattr(MCP, "_train_learned_mcp", fake_train, raising=False)

    mcp.add_feedback({"alpha": 0.8, "beta": 0.2, "feedback": 1})
    mcp.add_feedback({"alpha": 0.2, "beta": 0.8, "feedback": 1})

    assert trained["called"] == 1
    assert mcp.mode == "learned"


def test_mcp_compute_weights_in_learned_mode(monkeypatch, mcp_config):
    from core.mcp import MCP

    mcp = MCP(str(mcp_config))
    mcp.mode = "learned"
    mcp.learned_mcp = lambda features: (0.4, 0.6)

    alpha, beta = mcp.compute_weights({"hard": 0.2, "soft": 0.9})
    assert (alpha, beta) == (0.4, 0.6)


def test_route_to_skills_filters_threshold():
    from core.mcp import route_to_skills

    scores = {"hard": 0.9, "soft": 0.2, "programming": 0.85, "math": 0.1, "creative": 0.5}
    result = route_to_skills(scores, threshold=0.3, top_k=2)
    assert result == ["programming", "creative"]


def test_execute_skills_moe_success():
    from core.mcp import execute_skills_moe

    class DummySkill:
        def create_completion(self, prompt, **_):
            return {"choices": [{"text": f"Processed: {prompt}"}]}

    class DummyPool:
        def get_skill(self, skill_name):
            return DummySkill()

        def get(self, name):
            raise AssertionError(f"fallback not expected: {name}")

    scores = {"hard": 0.8, "programming": 0.9}
    responses = execute_skills_moe("write python", scores, DummyPool())

    assert "programming" in responses
    assert "write python" in responses["programming"]


def test_execute_skills_moe_fallback():
    from core.mcp import execute_skills_moe

    class DummyExpert:
        def create_completion(self, prompt, **_):
            return {"choices": [{"text": f"Expert: {prompt}"}]}

    class DummyPool:
        def get_skill(self, name):
            raise RuntimeError("no skills")

        def get(self, name):
            assert name == "expert_short"
            return DummyExpert()

    responses = execute_skills_moe("hello", {"hard": 0.2}, DummyPool())
    assert responses == {"expert_fallback": "Expert: hello"}


def test_execute_skills_moe_skill_error_triggers_fallback():
    from core.mcp import execute_skills_moe

    class FailingSkill:
        def create_completion(self, *_args, **_kwargs):
            raise RuntimeError("failure")

    class DummyExpert:
        def create_completion(self, prompt, **_):
            return {"choices": [{"text": f"Expert: {prompt}"}]}

    class DummyPool:
        def get_skill(self, name):
            return FailingSkill()

        def get(self, name):
            return DummyExpert()

    responses = execute_skills_moe("assist", {"hard": 0.7, "programming": 0.9}, DummyPool())
    assert responses["expert_fallback"].startswith("Expert: assist")


def test_reload_mcp_swaps_active(tmp_path, monkeypatch):
    from core import mcp as mcp_module

    mcp_module._mcp_active = None

    model_dir = tmp_path / "models" / "mcp"
    model_dir.mkdir(parents=True)
    (tmp_path / "state").mkdir(parents=True)

    (tmp_path / "state" / "mcp_reload_signal").write_text("signal")
    (model_dir / "mcp_active.pkl").write_text("data")

    mock_model = object()

    monkeypatch.setattr(mcp_module, "Path", lambda p: tmp_path / p)
    monkeypatch.setattr(mcp_module.torch, "load", lambda path: mock_model)

    assert mcp_module.reload_mcp() is True
    assert mcp_module._mcp_active is mock_model


def test_reload_mcp_no_signal(tmp_path, monkeypatch):
    from core import mcp as mcp_module

    monkeypatch.setattr(mcp_module, "Path", lambda p: tmp_path / p)
    assert mcp_module.reload_mcp() is False


def test_get_mcp_weights_uses_active(monkeypatch):
    from core import mcp as mcp_module

    fake_mcp = MagicMock()
    fake_mcp.compute_weights.return_value = (0.5, 0.5)

    monkeypatch.setattr(mcp_module, "_mcp_active", fake_mcp)
    monkeypatch.setattr(mcp_module, "reload_mcp", lambda: False)

    weights = mcp_module.get_mcp_weights({"hard": 0.6, "soft": 0.4}, context="ctx")
    assert weights == (0.5, 0.5)
    fake_mcp.compute_weights.assert_called_with(0.6, 0.4, "ctx")


def test_get_mcp_weights_initializes_when_none(monkeypatch):
    from core import mcp as mcp_module

    class StubMCP:
        def compute_weights(self, hard, soft, context=""):
            return hard, soft

    monkeypatch.setattr(mcp_module, "_mcp_active", None)
    monkeypatch.setattr(mcp_module, "reload_mcp", lambda: False)
    monkeypatch.setattr(mcp_module, "create_mcp", lambda: StubMCP())

    weights = mcp_module.get_mcp_weights({"hard": 0.4, "soft": 0.6}, context="ctx")
    assert weights == (0.4, 0.6)


def test_detect_and_apply_skill_programming():
    from core.mcp import detect_and_apply_skill

    config = detect_and_apply_skill("Escribe código Python para ordenar una lista", model_name="solar")
    assert config is not None
    assert config["skill_name"] == "programming"


def test_skill_listing_utilities():
    from core.mcp import get_skill_info, list_available_skills

    skills = list_available_skills()
    assert "programming" in skills

    info = get_skill_info("creative")
    assert info is not None
    assert info["name"] == "creative"


def test_detect_and_apply_skill_respects_model():
    from core.mcp import detect_and_apply_skill

    config = detect_and_apply_skill("necesito un draft inicial", model_name="lfm2")
    assert config is not None
    assert config["skill_name"] == "draft"
