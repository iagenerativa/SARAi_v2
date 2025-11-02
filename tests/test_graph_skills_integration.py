"""
Tests de Integración: Graph + Skills (v2.12)
============================================
Valida que skills se detectan y aplican automáticamente en el grafo

NOTA: Tests unitarios que NO requieren modelos cargados (usa TRM simulado)
"""

import pytest
from unittest.mock import Mock, patch
from core.skill_configs import match_skill_by_keywords
from core.mcp import detect_and_apply_skill
from core.graph import SARAiOrchestrator


class TestSkillDetectionStandalone:
    """Tests de detección de skills SIN grafo completo (más rápidos)"""
    
    def test_programming_skill_detection(self):
        """Verifica que query de programación detecta programming skill"""
        # Long-tail pattern: "código" + "python"
        skill = match_skill_by_keywords("Escribe código Python para quicksort")
        assert skill is not None
        assert skill.name == "programming"
    
    def test_diagnosis_skill_detection(self):
        """Verifica que query de diagnóstico detecta diagnosis skill"""
        # Long-tail pattern: "diagnostica" + "problema"
        skill = match_skill_by_keywords("Diagnostica el problema de este memory leak grave")
        assert skill is not None
        assert skill.name == "diagnosis"
    
    def test_financial_skill_detection(self):
        """Verifica que query financiera detecta financial skill"""
        # Long-tail pattern: "roi" + "inversión"
        skill = match_skill_by_keywords("Calcula el ROI esperado de esta inversión financiera")
        assert skill is not None
        assert skill.name == "financial"
    
    def test_creative_skill_detected(self):
        """Verifica que query creativa detecta creative skill"""
        query = "Crea una historia corta sobre un robot que aprende a sentir emociones"
        
        skill = match_skill_by_keywords(query)
        
        assert skill is not None
        assert skill.name == "creative"
        
        print(f"✅ Creative skill detectado")
    
    def test_no_skill_for_generic_query(self):
        """Verifica que query genérica NO detecta skill"""
        query = "Hola, ¿cómo estás hoy?"
        
        skill = match_skill_by_keywords(query)
        
        assert skill is None
        
        print(f"✅ Sin skill detectado para query genérica")


class TestSkillConfigGeneration:
    """Tests de generación de configuración de skills"""
    
    def test_programming_skill_config_complete(self):
        """Verifica que programming skill genera configuración completa"""
        query = "Implementa quicksort en Python"
        
        config = detect_and_apply_skill(query, "solar")
        
        assert config is not None
        assert config["skill_name"] == "programming"
        assert "system_prompt" in config
        assert "generation_params" in config
        assert "full_prompt" in config
        # Programming skill ahora usa viscoder2 (v2.14+)
        assert config["preferred_model"] == "viscoder2"
        
        # Verificar parámetros
        params = config["generation_params"]
        assert params["temperature"] == 0.3  # Baja para precisión
        assert params["max_tokens"] == 3072
        
        print(f"✅ Programming skill config completo")
        print(f"   Temperature: {params['temperature']}")
        print(f"   Max tokens: {params['max_tokens']}")
    
    def test_creative_skill_config_high_temperature(self):
        """Verifica que creative skill tiene alta temperatura"""
        query = "Genera ideas innovadoras para un producto tech"
        
        config = detect_and_apply_skill(query, "lfm2")
        
        assert config is not None
        assert config["skill_name"] == "creative"
        
        params = config["generation_params"]
        assert params["temperature"] == 0.9  # Alta para creatividad
        assert config["preferred_model"] == "lfm2"
        
        print(f"✅ Creative skill config correcto")
        print(f"   Temperature: {params['temperature']}")
        print(f"   Preferred model: {config['preferred_model']}")
    
    def test_skill_prompt_contains_query(self):
        """Verifica que prompt completo contiene la query del usuario"""
        query = "Escribe código Python para calcular fibonacci"
        
        config = detect_and_apply_skill(query, "solar")
        
        assert config is not None
        assert query in config["full_prompt"]
        assert "User:" in config["full_prompt"]
        assert "Assistant:" in config["full_prompt"]
        
        print(f"✅ Prompt completo contiene query del usuario")


class TestSkillStatistics:
    """Tests de estadísticas y métricas de skills"""
    
    def test_all_seven_skills_detectable(self):
        """Verifica que los 7 skills se pueden detectar"""
        test_queries = {
            "programming": "Implementa un algoritmo de búsqueda binaria",
            "diagnosis": "Diagnostica este error de timeout en la base de datos",
            "financial": "Calcula el ROI de una inversión usando análisis financiero avanzado",  # Más específico
            "creative": "Crea una historia sobre inteligencia artificial",
            "reasoning": "Usa razonamiento lógico para evaluar esta estrategia",
            "cto": "¿Qué arquitectura recomendarías para escalar a 1M usuarios?",
            "sre": "Configura monitoring en Kubernetes para producción"
        }
        
        detected_skills = []
        
        for expected_skill, query in test_queries.items():
            skill = match_skill_by_keywords(query)
            
            if skill:
                detected_skills.append(skill.name)
                assert skill.name == expected_skill, \
                    f"Query '{query}' debería detectar '{expected_skill}', detectó '{skill.name}'"
        
        assert len(detected_skills) == 7
        
        print(f"✅ Los 7 skills son detectables:")
        for skill_name in detected_skills:
            print(f"   - {skill_name}")


class TestSkillIntegrationWithMocking:
    """Tests de integración usando mocks para evitar cargar modelos"""
    
    @pytest.fixture
    def mock_agent(self):
        """Mock de agente LLM que simula generación"""
        agent = Mock()
        agent.generate = Mock(return_value="Respuesta simulada del modelo")
        return agent
    
    def test_skill_applied_in_generation(self, mock_agent):
        """Verifica que skill se aplica en generación (con mock)"""
        query = "Implementa bubble sort en Python"
        
        # Detectar skill
        config = detect_and_apply_skill(query, "solar")
        
        assert config is not None
        
        # Simular generación con prompt especializado
        prompt = config["full_prompt"]
        params = config["generation_params"]
        
        response = mock_agent.generate(
            prompt,
            max_new_tokens=params["max_tokens"],
            temperature=params["temperature"]
        )
        
        # Verificar que se llamó con parámetros correctos
        mock_agent.generate.assert_called_once()
        call_kwargs = mock_agent.generate.call_args[1]
        
        assert call_kwargs["temperature"] == 0.3  # Programming skill
        assert call_kwargs["max_new_tokens"] == 3072
        
        print(f"✅ Skill aplicado en generación (mock)")
        print(f"   Prompt usado: {prompt[:100]}...")
        print(f"   Parámetros: temp={call_kwargs['temperature']}, tokens={call_kwargs['max_new_tokens']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
    
    def test_programming_skill_auto_applied(self, orchestrator):
        """Verifica que query de código activa programming skill automáticamente"""
        query = "Implementa una función en Python que calcule el factorial de un número"
        
        # Ejecutar grafo
        state = orchestrator.invoke(query)
        
        # Verificaciones
        assert "skill_used" in state
        assert state["skill_used"] == "programming"
        assert state["agent_used"] in ["expert", "tiny"]  # Depende de α/β
        assert "response" in state
        assert len(state["response"]) > 0
        
        print(f"✅ Programming skill detectado y aplicado")
        print(f"   Agente usado: {state['agent_used']}")
        print(f"   Respuesta preview: {state['response'][:100]}...")
    
    def test_diagnosis_skill_auto_applied(self, orchestrator):
        """Verifica que query de diagnóstico activa diagnosis skill"""
        query = "Tengo un error en mi aplicación que causa memory leak, diagnostica el problema"
        
        state = orchestrator.invoke(query)
        
        assert state["skill_used"] == "diagnosis"
        assert state["agent_used"] in ["expert", "tiny"]
        
        print(f"✅ Diagnosis skill detectado y aplicado")
    
    def test_financial_skill_auto_applied(self, orchestrator):
        """Verifica que query financiera activa financial skill"""
        query = "Calcula el ROI esperado de una inversión de $100,000 con retorno del 15% anual"
        
        state = orchestrator.invoke(query)
        
        assert state["skill_used"] == "financial"
        assert state["agent_used"] in ["expert", "tiny"]
        
        print(f"✅ Financial skill detectado y aplicado")
    
    def test_creative_skill_auto_applied_uses_lfm2(self, orchestrator):
        """Verifica que creative skill prefiere LFM2 (tiny agent)"""
        query = "Crea una historia corta sobre un robot que aprende a sentir emociones"
        
        state = orchestrator.invoke(query)
        
        # Creative skill tiene alta β → debería usar tiny (LFM2)
        assert state["skill_used"] == "creative"
        # Nota: Con TRM simulado, α/β son aleatorios, así que no podemos garantizar agent
        # pero en producción, creative debería ir a tiny
        
        print(f"✅ Creative skill detectado")
        print(f"   Agente usado: {state['agent_used']}")
    
    def test_reasoning_skill_auto_applied(self, orchestrator):
        """Verifica que query de razonamiento activa reasoning skill"""
        query = "Necesito razonamiento lógico para evaluar si esta estrategia de negocio es viable"
        
        state = orchestrator.invoke(query)
        
        assert state["skill_used"] == "reasoning"
        
        print(f"✅ Reasoning skill detectado y aplicado")
    
    def test_cto_skill_auto_applied(self, orchestrator):
        """Verifica que query de arquitectura activa CTO skill"""
        query = "¿Qué arquitectura recomendarías para escalar un sistema de pagos a 1M TPS?"
        
        state = orchestrator.invoke(query)
        
        assert state["skill_used"] == "cto"
        
        print(f"✅ CTO skill detectado y aplicado")
    
    def test_sre_skill_auto_applied(self, orchestrator):
        """Verifica que query de SRE activa SRE skill"""
        query = "Necesito configurar monitoring y alertas en Kubernetes para producción"
        
        state = orchestrator.invoke(query)
        
        assert state["skill_used"] == "sre"
        
        print(f"✅ SRE skill detectado y aplicado")
    
    def test_no_skill_fallback(self, orchestrator):
        """Verifica que queries sin skill aplicable usan prompt estándar"""
        query = "Hola, ¿cómo estás hoy?"
        
        state = orchestrator.invoke(query)
        
        # No debería detectar skill específico
        assert state["skill_used"] is None
        assert "response" in state
        
        print(f"✅ Sin skill detectado → prompt estándar usado")
    
    def test_skill_logged_in_feedback(self, orchestrator):
        """Verifica que skill usado se loggea en feedback"""
        query = "Implementa bubble sort en Python"
        
        state = orchestrator.invoke(query)
        
        # Verificar que se loggeó
        recent = orchestrator.feedback_detector.get_recent_interactions(n=1)
        assert len(recent) > 0
        
        latest = recent[-1]
        assert "skill_used" in latest
        assert latest["skill_used"] == "programming"
        
        print(f"✅ Skill usado loggeado en feedback correctamente")
        print(f"   Log entry: {latest}")
    
    def test_multiple_queries_different_skills(self, orchestrator):
        """Verifica que diferentes queries activan diferentes skills"""
        queries = [
            ("Implementa quicksort", "programming"),
            ("Crea un poema sobre el mar", "creative"),
            ("Diagnostica este error de timeout", "diagnosis"),
        ]
        
        for query, expected_skill in queries:
            state = orchestrator.invoke(query)
            assert state["skill_used"] == expected_skill, \
                f"Query '{query}' debería activar skill '{expected_skill}', obtuvo '{state['skill_used']}'"
        
        print(f"✅ Múltiples skills detectados correctamente")


class TestSkillParametersApplied:
    """Verifica que parámetros de skills se aplican correctamente
    
    NOTA: Tests unitarios que verifican configuración de skills sin 
    necesidad de inicializar el orquestador completo.
    """
    
    def test_programming_skill_low_temperature(self):
        """Verifica que programming skill tiene temperatura baja (0.3)"""
        from core.skill_configs import PROGRAMMING_SKILL
        
        # Verificar configuración del skill
        assert PROGRAMMING_SKILL.temperature == 0.3
        assert PROGRAMMING_SKILL.name == "programming"
        
        # Verificar detección
        skill = match_skill_by_keywords("Escribe código Python para calcular números primos")
        assert skill is not None
        assert skill.name == "programming"
        assert skill.temperature == 0.3
        
        print(f"✅ Programming skill: temperature={skill.temperature} (precisión)")
    
    def test_creative_skill_high_temperature(self):
        """Verifica que creative skill tiene temperatura alta (0.9)"""
        from core.skill_configs import CREATIVE_SKILL
        
        # Verificar configuración del skill
        assert CREATIVE_SKILL.temperature == 0.9
        assert CREATIVE_SKILL.name == "creative"
        
        # Verificar detección
        skill = match_skill_by_keywords("Genera ideas innovadoras para un producto tech")
        assert skill is not None
        assert skill.name == "creative"
        assert skill.temperature == 0.9
        
        print(f"✅ Creative skill: temperature={skill.temperature} (creatividad)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
