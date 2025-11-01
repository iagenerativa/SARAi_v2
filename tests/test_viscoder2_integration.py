"""
Test para VisCoder2-7B Integration
===================================
Valida que el modelo VisCoder2 esté correctamente integrado en el Unified Wrapper.
"""

import pytest
from core.unified_model_wrapper import get_model, list_available_models
from core.skill_configs import PROGRAMMING_SKILL


class TestVisCoder2Integration:
    """Suite de tests para VisCoder2-7B"""
    
    def test_viscoder2_in_available_models(self):
        """Verifica que viscoder2 esté en la lista de modelos disponibles"""
        models = list_available_models()
        assert "viscoder2" in models, "viscoder2 debe estar en models.yaml"
    
    def test_viscoder2_wrapper_creation(self):
        """Verifica que se pueda crear wrapper de VisCoder2"""
        try:
            viscoder = get_model("viscoder2")
            assert viscoder is not None
            assert viscoder.name == "viscoder2"
        except FileNotFoundError:
            pytest.skip("Modelo VisCoder2 no descargado aún (ejecutar scripts/download_viscoder2.py)")
    
    def test_programming_skill_uses_viscoder2(self):
        """Verifica que el skill de programming use VisCoder2"""
        assert PROGRAMMING_SKILL.preferred_model == "viscoder2"
        assert PROGRAMMING_SKILL.name == "programming"
        assert PROGRAMMING_SKILL.temperature == 0.3  # Baja para código preciso
    
    def test_viscoder2_config_structure(self):
        """Verifica la estructura del config de VisCoder2 en YAML"""
        import yaml
        
        with open("config/models.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        assert "viscoder2" in config
        
        viscoder_config = config["viscoder2"]
        
        # Campos obligatorios
        assert viscoder_config["backend"] == "gguf"
        assert viscoder_config["type"] == "text"
        assert "model_path" in viscoder_config
        assert "repo_id" in viscoder_config
        assert "gguf_file" in viscoder_config
        
        # Especialización
        assert viscoder_config["specialty"] == "code_generation"
        assert "use_cases" in viscoder_config
        
        # Parámetros optimizados para código
        assert viscoder_config["temperature"] == 0.3  # Baja para precisión
        assert viscoder_config["n_ctx"] == 4096  # Contexto largo
        assert viscoder_config["priority"] == 8  # Alta prioridad
    
    @pytest.mark.skipif(
        True,  # Skip por defecto (requiere modelo descargado)
        reason="Requiere modelo VisCoder2 descargado"
    )
    def test_viscoder2_code_generation(self):
        """Test de generación de código con VisCoder2"""
        viscoder = get_model("viscoder2")
        
        prompt = """Write a Python function that calculates the factorial of a number.
Include:
- Type hints
- Docstring
- Error handling for negative numbers
- Recursive implementation"""
        
        response = viscoder.invoke(prompt)
        
        # Validaciones básicas
        assert isinstance(response, str)
        assert len(response) > 50
        assert "def " in response  # Debe contener definición de función
        assert ":" in response  # Sintaxis Python
    
    def test_viscoder2_system_prompt(self):
        """Verifica que el system prompt de VisCoder2 esté configurado"""
        import yaml
        
        with open("config/models.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        viscoder_config = config["viscoder2"]
        
        assert "system_prompt" in viscoder_config
        system_prompt = viscoder_config["system_prompt"]
        
        # Validar contenido del system prompt
        assert "VisCoder2" in system_prompt
        assert "programming" in system_prompt.lower()
        assert "code" in system_prompt.lower()
    
    def test_viscoder2_memory_config(self):
        """Verifica configuración de memoria para VisCoder2"""
        import yaml
        
        with open("config/models.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        viscoder_config = config["viscoder2"]
        
        # Load on demand (solo cuando se necesita)
        assert viscoder_config["load_on_demand"] is True
        
        # Límite de memoria razonable (~4.5GB)
        assert viscoder_config["max_memory_mb"] == 4500
        
        # Prioridad alta (especialista)
        assert viscoder_config["priority"] == 8


class TestProgrammingSkillVisCoder2:
    """Tests específicos del skill de programming con VisCoder2"""
    
    def test_skill_routing_to_viscoder2(self):
        """Verifica que queries de código rutean a VisCoder2"""
        from core.mcp import detect_and_apply_skill
        
        # Queries típicas de programación
        code_queries = [
            "Write a Python function to sort a list",
            "Debug this JavaScript code",
            "Implementa un algoritmo de búsqueda binaria",
            "Crea una clase para gestionar usuarios"
        ]
        
        for query in code_queries:
            skill = detect_and_apply_skill(query, agent_type="auto")
            
            if skill:  # Si detecta skill
                assert skill.get("name") == "programming"
                # El skill programming usa viscoder2
                assert PROGRAMMING_SKILL.preferred_model == "viscoder2"
    
    def test_programming_skill_parameters(self):
        """Verifica parámetros optimizados para código"""
        skill = PROGRAMMING_SKILL
        
        # Temperature baja para código preciso
        assert skill.temperature == 0.3
        
        # Max tokens alto para funciones/clases completas
        assert skill.max_tokens == 3072
        
        # Top_p conservador
        assert skill.top_p == 0.85
    
    def test_programming_keywords_coverage(self):
        """Verifica que keywords cubran casos comunes"""
        keywords = PROGRAMMING_SKILL.keywords
        
        essential_keywords = [
            "code", "código",
            "función", "clase",
            "debug", "test",
            "implementa", "algoritmo"
        ]
        
        for kw in essential_keywords:
            assert kw in keywords, f"Keyword '{kw}' debe estar presente"


# ========================================
# TESTS DE COMPARACIÓN VISCODER2 vs SOLAR
# ========================================

class TestVisCoder2vSOLAR:
    """Compara VisCoder2 (especialista) vs SOLAR (general)"""
    
    def test_context_length_comparison(self):
        """VisCoder2 debe tener más contexto que SOLAR short"""
        import yaml
        
        with open("config/models.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        viscoder_ctx = config["viscoder2"]["n_ctx"]
        solar_short_ctx = config["solar_short"]["n_ctx"]
        
        # VisCoder2 tiene contexto largo (4096) vs SOLAR short (512)
        assert viscoder_ctx > solar_short_ctx
        assert viscoder_ctx == 4096
    
    def test_temperature_comparison(self):
        """VisCoder2 debe tener temperatura más baja (código preciso)"""
        import yaml
        
        with open("config/models.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        viscoder_temp = config["viscoder2"]["temperature"]
        solar_temp = config["solar_short"]["temperature"]
        
        # VisCoder2: 0.3 (preciso), SOLAR: 0.7 (balanced)
        assert viscoder_temp < solar_temp
        assert viscoder_temp == 0.3
    
    def test_specialty_field_present(self):
        """VisCoder2 debe tener campo specialty (especialista)"""
        import yaml
        
        with open("config/models.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        assert "specialty" in config["viscoder2"]
        assert config["viscoder2"]["specialty"] == "code_generation"
        
        # SOLAR no tiene specialty (modelo general)
        assert "specialty" not in config.get("solar_short", {})


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
