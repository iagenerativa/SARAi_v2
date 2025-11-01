"""
Tests unitarios para MCP Skills MoE Runtime (v2.12 - T1.2)

Validación de:
- Routing de skills según scores TRM
- Integración con ModelPool.get_skill()
- Fallback si skill no disponible/falla
- Ejecución correcta de múltiples skills
- Límite de concurrencia respetado
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.mcp import route_to_skills, execute_skills_moe


class TestSkillRouting:
    """Tests de enrutamiento de skills"""
    
    def test_route_single_skill_above_threshold(self):
        """Test T1.2.1: Enrutar un solo skill sobre threshold"""
        scores = {
            'hard': 0.9,
            'soft': 0.2,
            'programming': 0.85,
            'diagnosis': 0.25,
            'finance': 0.1
        }
        
        skills = route_to_skills(scores, threshold=0.3, top_k=3)
        
        assert len(skills) == 1
        assert 'programming' in skills
        assert 'hard' not in skills  # hard/soft excluidos
        assert 'diagnosis' not in skills  # Bajo threshold
    
    def test_route_multiple_skills_top_k(self):
        """Test T1.2.2: Enrutar múltiples skills respetando top_k"""
        scores = {
            'hard': 0.9,
            'programming': 0.85,
            'diagnosis': 0.75,
            'finance': 0.65,
            'logic': 0.55,
            'creative': 0.45
        }
        
        # Top-3 skills sobre threshold=0.3
        skills = route_to_skills(scores, threshold=0.3, top_k=3)
        
        assert len(skills) == 3
        assert skills == ['programming', 'diagnosis', 'finance']  # Ordenados por score
    
    def test_route_no_skills_above_threshold(self):
        """Test T1.2.3: Sin skills sobre threshold retorna lista vacía"""
        scores = {
            'hard': 0.9,
            'soft': 0.8,
            'programming': 0.25,
            'diagnosis': 0.15
        }
        
        skills = route_to_skills(scores, threshold=0.3, top_k=3)
        
        assert len(skills) == 0
    
    def test_route_excludes_base_categories(self):
        """Test T1.2.4: Excluye hard/soft/web_query del routing"""
        scores = {
            'hard': 0.95,
            'soft': 0.85,
            'web_query': 0.75,
            'programming': 0.70
        }
        
        skills = route_to_skills(scores, threshold=0.3, top_k=3)
        
        assert len(skills) == 1
        assert skills == ['programming']
        assert 'hard' not in skills
        assert 'soft' not in skills
        assert 'web_query' not in skills


class TestSkillExecution:
    """Tests de ejecución de skills MoE"""
    
    @pytest.fixture
    def mock_model_pool(self):
        """Mock de ModelPool con skills"""
        pool = MagicMock()
        
        # Mock de skill model
        skill_model = MagicMock()
        skill_model.create_completion = lambda text, **kwargs: {
            "choices": [{"text": f"Response from skill"}]
        }
        
        pool.get_skill.return_value = skill_model
        
        # Mock de expert fallback
        expert_model = MagicMock()
        expert_model.create_completion = lambda text, **kwargs: {
            "choices": [{"text": "Expert fallback response"}]
        }
        
        pool.get.return_value = expert_model
        
        return pool
    
    def test_execute_single_skill(self, mock_model_pool):
        """Test T1.2.5: Ejecutar un solo skill correctamente"""
        scores = {
            'hard': 0.9,
            'programming': 0.85
        }
        
        responses = execute_skills_moe(
            input_text="Escribe una función Python",
            scores=scores,
            model_pool=mock_model_pool,
            threshold=0.3,
            top_k=3
        )
        
        assert 'programming' in responses
        assert isinstance(responses['programming'], str)
        assert len(responses['programming']) > 0
        
        # Verificar que se llamó get_skill
        mock_model_pool.get_skill.assert_called_once_with('programming')
    
    def test_execute_multiple_skills(self, mock_model_pool):
        """Test T1.2.6: Ejecutar múltiples skills"""
        scores = {
            'hard': 0.9,
            'programming': 0.85,
            'diagnosis': 0.75
        }
        
        responses = execute_skills_moe(
            input_text="Debug mi código Python",
            scores=scores,
            model_pool=mock_model_pool,
            threshold=0.3,
            top_k=3
        )
        
        assert len(responses) == 2
        assert 'programming' in responses
        assert 'diagnosis' in responses
        
        # Verificar que se llamó get_skill para ambos
        assert mock_model_pool.get_skill.call_count == 2
    
    def test_fallback_when_no_skills(self, mock_model_pool):
        """Test T1.2.7: Fallback a expert_short si no hay skills activos"""
        scores = {
            'hard': 0.9,
            'programming': 0.25  # Bajo threshold
        }
        
        responses = execute_skills_moe(
            input_text="Pregunta general",
            scores=scores,
            model_pool=mock_model_pool,
            threshold=0.3,
            top_k=3,
            enable_fallback=True
        )
        
        assert 'expert_fallback' in responses
        assert isinstance(responses['expert_fallback'], str)
        
        # Verificar que se llamó get() para expert
        mock_model_pool.get.assert_called_once_with('expert_short')
        
        # NO debe llamar get_skill
        mock_model_pool.get_skill.assert_not_called()
    
    def test_fallback_when_skill_fails(self, mock_model_pool):
        """Test T1.2.8: Fallback a expert si skill falla"""
        scores = {
            'hard': 0.9,
            'programming': 0.85
        }
        
        # Simular fallo en skill
        mock_model_pool.get_skill.side_effect = Exception("Skill load failed")
        
        responses = execute_skills_moe(
            input_text="Escribe código",
            scores=scores,
            model_pool=mock_model_pool,
            threshold=0.3,
            top_k=3,
            enable_fallback=True
        )
        
        # Debe haber fallback a expert
        assert 'expert_fallback' in responses
        
        # Verificar que intentó cargar skill
        mock_model_pool.get_skill.assert_called_once_with('programming')
        
        # Verificar que usó expert fallback
        mock_model_pool.get.assert_called_once_with('expert_short')
    
    def test_no_fallback_if_disabled(self, mock_model_pool):
        """Test T1.2.9: Sin fallback si enable_fallback=False"""
        scores = {
            'hard': 0.9,
            'programming': 0.25  # Bajo threshold
        }
        
        responses = execute_skills_moe(
            input_text="Pregunta",
            scores=scores,
            model_pool=mock_model_pool,
            threshold=0.3,
            top_k=3,
            enable_fallback=False
        )
        
        # Dict vacío si no hay skills y fallback deshabilitado
        assert responses == {}
        
        # NO debe llamar ni get_skill ni get
        mock_model_pool.get_skill.assert_not_called()
        mock_model_pool.get.assert_not_called()
    
    def test_partial_skill_failure(self, mock_model_pool):
        """Test T1.2.10: Fallo parcial (un skill falla, otro funciona)"""
        scores = {
            'hard': 0.9,
            'programming': 0.85,
            'diagnosis': 0.75
        }
        
        # Configurar mock: programming falla, diagnosis funciona
        def get_skill_side_effect(skill_name):
            if skill_name == 'programming':
                raise Exception("Programming skill failed")
            else:
                model = MagicMock()
                model.create_completion = lambda text, **kwargs: {
                    "choices": [{"text": f"Response from {skill_name}"}]
                }
                return model
        
        mock_model_pool.get_skill.side_effect = get_skill_side_effect
        
        responses = execute_skills_moe(
            input_text="Debug código",
            scores=scores,
            model_pool=mock_model_pool,
            threshold=0.3,
            top_k=3,
            enable_fallback=True
        )
        
        # Solo debe tener respuesta de diagnosis
        assert 'diagnosis' in responses
        assert 'programming' not in responses
        
        # NO debe activar fallback porque diagnosis funcionó
        assert 'expert_fallback' not in responses


class TestConcurrencyLimits:
    """Tests de límites de concurrencia"""
    
    def test_respects_max_skills_limit(self, mock_model_pool=None):
        """Test T1.2.11: Respeta límite top_k (máximo skills simultáneos)"""
        scores = {
            'programming': 0.90,
            'diagnosis': 0.85,
            'finance': 0.80,
            'logic': 0.75,
            'creative': 0.70,
            'reasoning': 0.65
        }
        
        # Solo top-3 aunque 6 están sobre threshold
        skills = route_to_skills(scores, threshold=0.3, top_k=3)
        
        assert len(skills) <= 3
        assert skills == ['programming', 'diagnosis', 'finance']


class TestIntegrationWithModelPool:
    """Tests de integración con ModelPool real"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_load_skill_via_moe(self):
        """
        Test T1.2.12: Integración end-to-end con ModelPool real
        
        Requiere:
        - config/sarai.yaml válido con skills configurados
        - Modelos GGUF descargados en cache (o conexión HuggingFace)
        - Al menos 8GB RAM disponible
        
        Ejecutar con: pytest -v -m integration
        """
        from core.model_pool import ModelPool
        from core.mcp import execute_skills_moe
        import os
        
        # Skip si no existe config
        config_path = "config/sarai.yaml"
        if not os.path.exists(config_path):
            pytest.skip("Config sarai.yaml no encontrado")
        
        # Skip si estamos en CI sin modelos
        if os.getenv("CI") == "true" and not os.getenv("SKIP_MODEL_DOWNLOAD"):
            pytest.skip("CI sin modelos GGUF - usar mocks")
        
        try:
            # Cargar ModelPool real
            pool = ModelPool(config_path)
            
            # Simular scores TRM-Router que activan skill programming
            scores = {
                'hard': 0.9,
                'soft': 0.2,
                'programming': 0.85,
                'diagnosis': 0.25
            }
            
            # Ejecutar MoE con skill real
            responses = execute_skills_moe(
                input_text="Escribe una función Python para ordenar una lista",
                scores=scores,
                model_pool=pool,
                threshold=0.3,
                top_k=1  # Solo 1 skill para test rápido
            )
            
            # Validaciones
            assert 'programming' in responses, "Debe activar skill programming"
            assert isinstance(responses['programming'], str), "Debe retornar string"
            assert len(responses['programming']) > 10, "Respuesta muy corta (posible error)"
            
            # Verificar que el skill está en cache del pool
            stats = pool.get_stats()
            assert 'programming' in stats['skills_in_cache'], "Skill debe estar en cache"
            
            # Cleanup
            pool.release_skill('programming')
            
            print("✅ Test de integración MoE exitoso")
        
        except ImportError as e:
            pytest.skip(f"Dependencias faltantes: {e}")
        
        except Exception as e:
            pytest.fail(f"Integración MoE falló: {e}")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_skill_lru_with_real_models(self):
        """
        Test T1.2.13: Validar LRU con modelos GGUF reales
        
        Carga 4 skills para forzar eviction del primero
        """
        from core.model_pool import ModelPool
        import os
        
        config_path = "config/sarai.yaml"
        if not os.path.exists(config_path):
            pytest.skip("Config sarai.yaml no encontrado")
        
        if os.getenv("CI") == "true":
            pytest.skip("CI - test requiere múltiples modelos GGUF")
        
        try:
            pool = ModelPool(config_path)
            
            # Cargar 3 skills (límite configurado)
            skill1 = pool.get_skill("programming")
            assert "programming" in pool.skills_cache
            
            skill2 = pool.get_skill("diagnosis")
            assert "diagnosis" in pool.skills_cache
            
            skill3 = pool.get_skill("finance")
            assert "finance" in pool.skills_cache
            assert len(pool.skills_cache) == 3
            
            # Cargar 4to skill → debe descargar "programming" (LRU)
            skill4 = pool.get_skill("logic")
            
            # Verificar LRU funcionó
            assert "programming" not in pool.skills_cache, "Programming debió ser descargado (LRU)"
            assert "logic" in pool.skills_cache
            assert len(pool.skills_cache) == 3, "Debe mantener máximo 3 skills"
            
            # Cleanup
            for skill in ["diagnosis", "finance", "logic"]:
                if skill in pool.skills_cache:
                    pool.release_skill(skill)
            
            print("✅ Test LRU con modelos reales exitoso")
        
        except Exception as e:
            pytest.fail(f"Test LRU falló: {e}")


# ============================================================================
# Test Suite Summary
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
