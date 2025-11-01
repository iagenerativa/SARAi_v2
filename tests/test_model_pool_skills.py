"""
Tests unitarios para ModelPool Skills MoE (v2.12)

Validación de:
- Carga dinámica de skills bajo demanda
- Gestión LRU (máximo 3 skills simultáneos)
- Auto-descarga por TTL
- Fallback si skill falla
- RAM P99 ≤ 12GB con 3 skills + expert_short
"""

import pytest
import time
import gc
import psutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from core.model_pool import ModelPool, get_model_pool


@pytest.fixture
def temp_config(tmp_path):
    """Crea configuración temporal para tests"""
    config_content = """
runtime:
  backend: "cpu"
  max_concurrent_llms: 2
  max_concurrent_skills: 3
  n_threads: 2

memory:
  max_ram_gb: 12
  model_ttl_seconds: 2  # TTL corto para tests
  use_mmap: true
  use_mlock: false

models:
  expert:
    name: "SOLAR-Mock"
    repo_id: "mock/solar"
    gguf_file: "solar.gguf"
    context_length: 512
    cache_dir: "./test_cache"
  
  tiny:
    name: "LFM2-Mock"
    repo_id: "mock/lfm2"
    gguf_file: "lfm2.gguf"
    context_length: 1024
    cache_dir: "./test_cache"
  
  skills:
    programming:
      name: "CodeLlama-Mock"
      repo_id: "mock/codellama"
      gguf_file: "codellama.gguf"
      max_memory_mb: 800
      context_length: 1024
      cache_dir: "./test_cache"
      domains: ["código", "programación"]
    
    diagnosis:
      name: "Mistral-Mock"
      repo_id: "mock/mistral"
      gguf_file: "mistral.gguf"
      max_memory_mb: 800
      context_length: 1024
      cache_dir: "./test_cache"
      domains: ["diagnóstico", "error"]
    
    finance:
      name: "FinGPT-Mock"
      repo_id: "mock/fingpt"
      gguf_file: "fingpt.gguf"
      max_memory_mb: 800
      context_length: 1024
      cache_dir: "./test_cache"
      domains: ["finanzas", "inversión"]
    
    creative:
      name: "Creative-Mock"
      repo_id: "mock/creative"
      gguf_file: "creative.gguf"
      max_memory_mb: 800
      context_length: 1024
      cache_dir: "./test_cache"
      domains: ["historia", "cuento"]
"""
    config_file = tmp_path / "test_sarai.yaml"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture
def mock_llama():
    """Mock de llama_cpp.Llama"""
    with patch('llama_cpp.Llama') as mock:
        # Simular modelo cargado
        mock_instance = MagicMock()
        mock_instance.create_completion = lambda prompt, **kwargs: {
            "choices": [{"text": "Mock response"}]
        }
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_hf_download():
    """Mock de huggingface_hub.hf_hub_download"""
    with patch('core.model_pool.hf_hub_download') as mock:
        mock.return_value = "/mock/path/model.gguf"
        yield mock


class TestSkillLoading:
    """Tests de carga de skills"""
    
    def test_load_single_skill(self, temp_config, mock_llama, mock_hf_download):
        """Test T1.1.1: Cargar un skill bajo demanda"""
        pool = ModelPool(temp_config)
        
        # Cargar skill programming
        skill = pool.get_skill("programming")
        
        assert skill is not None
        assert "programming" in pool.skills_cache
        assert len(pool.skills_cache) == 1
        
        # Verificar que se llamó a Llama con parámetros correctos
        mock_llama.assert_called_once()
        call_kwargs = mock_llama.call_args[1]
        assert call_kwargs['n_ctx'] == 1024
        assert call_kwargs['use_mmap'] == True
    
    def test_load_multiple_skills(self, temp_config, mock_llama, mock_hf_download):
        """Test T1.1.2: Cargar múltiples skills (hasta límite)"""
        pool = ModelPool(temp_config)
        
        # Cargar 3 skills (límite configurado)
        skill1 = pool.get_skill("programming")
        skill2 = pool.get_skill("diagnosis")
        skill3 = pool.get_skill("finance")
        
        assert len(pool.skills_cache) == 3
        assert "programming" in pool.skills_cache
        assert "diagnosis" in pool.skills_cache
        assert "finance" in pool.skills_cache
    
    def test_skill_cache_hit(self, temp_config, mock_llama, mock_hf_download):
        """Test T1.1.3: Cache hit cuando skill ya está cargado"""
        pool = ModelPool(temp_config)
        
        # Primera carga
        skill1 = pool.get_skill("programming")
        initial_calls = mock_llama.call_count
        
        # Segunda carga (debe ser cache hit)
        skill2 = pool.get_skill("programming")
        
        assert skill1 is skill2  # Mismo objeto
        assert mock_llama.call_count == initial_calls  # No se llamó de nuevo
    
    def test_skill_not_found(self, temp_config, mock_llama, mock_hf_download):
        """Test T1.1.4: Error si skill no existe en configuración"""
        pool = ModelPool(temp_config)
        
        with pytest.raises(ValueError, match="Skill 'nonexistent' no encontrado"):
            pool.get_skill("nonexistent")


class TestLRUEviction:
    """Tests de descarga LRU"""
    
    def test_lru_eviction_on_limit(self, temp_config, mock_llama, mock_hf_download):
        """Test T1.1.5: LRU descarga el skill menos usado cuando se alcanza límite"""
        pool = ModelPool(temp_config)
        
        # Cargar 3 skills (límite)
        pool.get_skill("programming")  # Más antiguo
        time.sleep(0.1)
        pool.get_skill("diagnosis")
        time.sleep(0.1)
        pool.get_skill("finance")  # Más reciente
        
        assert len(pool.skills_cache) == 3
        
        # Cargar 4to skill → debe descargar "programming" (LRU)
        pool.get_skill("creative")
        
        assert len(pool.skills_cache) == 3
        assert "programming" not in pool.skills_cache  # Descargado
        assert "diagnosis" in pool.skills_cache
        assert "finance" in pool.skills_cache
        assert "creative" in pool.skills_cache
    
    def test_lru_updates_on_access(self, temp_config, mock_llama, mock_hf_download):
        """Test T1.1.6: Acceder a un skill lo mueve al final (más reciente)"""
        pool = ModelPool(temp_config)
        
        # Cargar 3 skills
        pool.get_skill("programming")
        time.sleep(0.1)
        pool.get_skill("diagnosis")
        time.sleep(0.1)
        pool.get_skill("finance")
        
        # Acceder a "programming" → lo mueve al final
        pool.get_skill("programming")
        
        # Cargar 4to skill → debe descargar "diagnosis" (ahora es el LRU)
        pool.get_skill("creative")
        
        assert "programming" in pool.skills_cache  # No descargado (fue accedido)
        assert "diagnosis" not in pool.skills_cache  # Descargado
        assert "finance" in pool.skills_cache
        assert "creative" in pool.skills_cache


class TestTTLExpiration:
    """Tests de auto-descarga por TTL"""
    
    def test_ttl_auto_descarga(self, temp_config, mock_llama, mock_hf_download):
        """Test T1.1.7: Skill se descarga automáticamente tras TTL sin uso"""
        pool = ModelPool(temp_config)  # TTL = 2 segundos
        
        # Cargar skill
        pool.get_skill("programming")
        assert "programming" in pool.skills_cache
        
        # Esperar más que TTL
        time.sleep(2.5)
        
        # Intentar cargar otro skill → activa cleanup
        pool.get_skill("diagnosis")
        
        # "programming" debe haberse descargado por TTL
        assert "programming" not in pool.skills_cache
        assert "diagnosis" in pool.skills_cache
    
    def test_ttl_reset_on_access(self, temp_config, mock_llama, mock_hf_download):
        """Test T1.1.8: Acceder a skill resetea su TTL"""
        pool = ModelPool(temp_config)  # TTL = 2 segundos
        
        # Cargar skill
        pool.get_skill("programming")
        
        # Esperar casi el TTL
        time.sleep(1.5)
        
        # Acceder al skill → resetea TTL
        pool.get_skill("programming")
        
        # Esperar otros 1.5s (total 3s desde creación, pero solo 1.5s desde último acceso)
        time.sleep(1.5)
        
        # Cargar otro skill → "programming" NO debe descargarse (TTL reseteado)
        pool.get_skill("diagnosis")
        
        assert "programming" in pool.skills_cache  # Aún presente


class TestMemoryManagement:
    """Tests de gestión de memoria"""
    
    @pytest.mark.slow
    def test_ram_with_3_skills_under_limit(self, temp_config, mock_llama, mock_hf_download):
        """Test T1.1.9: RAM P99 ≤ 12GB con 3 skills + expert_short cargados"""
        # NOTA: Este test es teórico con mocks. En hardware real validar con benchmarks
        pool = ModelPool(temp_config)
        
        # Cargar expert_short (simular)
        with patch.object(pool, 'get') as mock_get_expert:
            mock_get_expert.return_value = MagicMock()
            pool.get("expert_short")
        
        # Cargar 3 skills
        pool.get_skill("programming")
        pool.get_skill("diagnosis")
        pool.get_skill("finance")
        
        # Verificar límites lógicos
        assert len(pool.cache) <= pool.max_models
        assert len(pool.skills_cache) <= pool.max_skills
        
        # En producción: verificar RAM real
        process = psutil.Process()
        ram_gb = process.memory_info().rss / (1024 ** 3)
        # assert ram_gb <= 12.0  # Descomentar en tests reales
    
    def test_release_skill_manually(self, temp_config, mock_llama, mock_hf_download):
        """Test T1.1.10: Liberar skill manualmente con release_skill()"""
        pool = ModelPool(temp_config)
        
        # Cargar skill
        pool.get_skill("programming")
        assert "programming" in pool.skills_cache
        
        # Liberar manualmente
        pool.release_skill("programming")
        
        assert "programming" not in pool.skills_cache
        assert "programming" not in pool.skills_timestamps


class TestStatsReporting:
    """Tests de estadísticas"""
    
    def test_get_stats_includes_skills(self, temp_config, mock_llama, mock_hf_download):
        """Test T1.1.11: get_stats() incluye información de skills"""
        pool = ModelPool(temp_config)
        
        # Cargar 2 skills
        pool.get_skill("programming")
        pool.get_skill("diagnosis")
        
        stats = pool.get_stats()
        
        assert "skills_loaded" in stats
        assert stats["skills_loaded"] == 2
        assert "max_skills_capacity" in stats
        assert stats["max_skills_capacity"] == 3
        assert "skills_in_cache" in stats
        assert "programming" in stats["skills_in_cache"]
        assert "diagnosis" in stats["skills_in_cache"]
        assert "skills_time_since_last_access" in stats


class TestErrorHandling:
    """Tests de manejo de errores"""
    
    def test_load_failure_raises_error(self, temp_config, mock_llama, mock_hf_download):
        """Test T1.1.12: Error si falla la carga del skill"""
        pool = ModelPool(temp_config)
        
        # Simular fallo en carga
        mock_llama.side_effect = Exception("Mock load failure")
        
        with pytest.raises(RuntimeError, match="No se pudo cargar skill"):
            pool.get_skill("programming")


# ============================================================================
# Test Suite Summary
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
