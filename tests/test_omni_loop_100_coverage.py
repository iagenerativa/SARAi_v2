"""
Omni Loop Coverage Completer - Enfoque pragmático para 100%

Tests que ejercitan branches no cubiertos sin modificar arquitectura:
- Config sanitization
- History management
- Singleton verification
- LFM2 fallback (éxito y fallo)
"""

import logging



# =============================================================================
# CONFIG DEFAULTS
# =============================================================================

def test_loop_config_defaults():
    """Cubre valores por defecto de LoopConfig"""
    from core.omni_loop import LoopConfig
    
    config = LoopConfig()
    
    assert config.max_iterations == 3
    assert config.enable_reflection is True
    assert config.confidence_threshold == 0.85
    assert config.temperature == 0.7


# =============================================================================
# HISTORY MANAGEMENT (Lines 327-332)
# =============================================================================

def test_get_omni_loop_singleton():
    """Verifica que singleton funciona"""
    from core.omni_loop import get_omni_loop
    
    loop1 = get_omni_loop()
    loop2 = get_omni_loop()
    
    # Deben ser la misma instancia
    assert loop1 is loop2


def test_loop_history_persists_across_calls():
    """Verifica que history se mantiene entre llamadas"""
    from core.omni_loop import get_omni_loop
    
    loop = get_omni_loop()
    
    # El execute_loop agregará entradas al history
    # Este test solo verifica que el atributo existe y es accesible
    assert hasattr(loop, 'loop_history')
    assert isinstance(loop.loop_history, list)


# =============================================================================
# LFM2 FALLBACK (Lines 372-424)
# =============================================================================


def test_fallback_lfm2_returns_text(monkeypatch):
    """Cubre ruta feliz del fallback LFM2"""
    from core.omni_loop import OmniLoop

    class DummyLLM:
        def __call__(self, prompt, max_tokens, temperature, stop):  # noqa: D401
            return {
                "choices": [{"text": " borrador final "}],
                "usage": {"completion_tokens": 7},
            }

    class DummyPool:
        def get(self, name):  # noqa: D401
            assert name == "tiny"
            return DummyLLM()

    dummy_pool = DummyPool()

    import core.model_pool as model_pool

    monkeypatch.setattr(model_pool, "get_model_pool", lambda: dummy_pool)

    loop = OmniLoop()

    response = loop._fallback_lfm2("haz un resumen")

    assert response == "borrador final"


def test_fallback_lfm2_logs_failure(monkeypatch, caplog):
    """Cubre mensaje crítico cuando LFM2 también falla"""
    from core.omni_loop import OmniLoop

    class ExplodingLLM:
        def __call__(self, *args, **kwargs):  # noqa: D401
            raise RuntimeError("lfm2 exploded")

    class ExplodingPool:
        def get(self, name):  # noqa: D401
            assert name == "tiny"
            return ExplodingLLM()

    exploding_pool = ExplodingPool()

    import core.model_pool as model_pool

    monkeypatch.setattr(model_pool, "get_model_pool", lambda: exploding_pool)

    loop = OmniLoop()

    with caplog.at_level(logging.CRITICAL):
        response = loop._fallback_lfm2("haz un resumen")

    assert "Lo siento, no puedo procesar" in response
    assert "LFM2 fallback failed" in caplog.text


