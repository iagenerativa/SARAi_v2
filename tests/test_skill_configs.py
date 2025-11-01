"""
Tests para Skill Configs - Sistema de prompting especializado
===========================================================
Valida que los skills NO cargan modelos, solo configuran prompts
"""

import pytest
from core.skill_configs import (
    SkillConfig,
    PROGRAMMING_SKILL,
    DIAGNOSIS_SKILL,
    FINANCIAL_SKILL,
    CREATIVE_SKILL,
    REASONING_SKILL,
    CTO_SKILL,
    SRE_SKILL,
    ALL_SKILLS,
    get_skill,
    list_skills,
    match_skill_by_keywords,
    get_skills_info
)
from core.mcp import detect_and_apply_skill, list_available_skills, get_skill_info


class TestSkillConfig:
    """Tests para la clase SkillConfig"""
    
    def test_skill_config_creation(self):
        """Verifica que se puede crear una configuración de skill"""
        skill = SkillConfig(
            name="test_skill",
            description="Test skill",
            system_prompt="You are a test assistant",
            keywords=["test", "example"],
            temperature=0.5,
            max_tokens=1024,
            preferred_model="solar"
        )
        
        assert skill.name == "test_skill"
        assert skill.description == "Test skill"
        assert "test" in skill.keywords
        assert skill.temperature == 0.5
        assert skill.max_tokens == 1024
        assert skill.preferred_model == "solar"
    
    def test_build_prompt(self):
        """Verifica construcción de prompt completo"""
        skill = SkillConfig(
            name="test",
            description="Test",
            system_prompt="System: Test",
            keywords=["test"]
        )
        
        full_prompt = skill.build_prompt("Hello world")
        
        assert "System: Test" in full_prompt
        assert "Hello world" in full_prompt
        assert "User:" in full_prompt
        assert "Assistant:" in full_prompt
    
    def test_get_generation_params(self):
        """Verifica obtención de parámetros de generación"""
        skill = SkillConfig(
            name="test",
            description="Test",
            system_prompt="Test",
            keywords=["test"],
            temperature=0.8,
            max_tokens=2048,
            top_p=0.95
        )
        
        params = skill.get_generation_params()
        
        assert params["temperature"] == 0.8
        assert params["max_tokens"] == 2048
        assert params["top_p"] == 0.95
        assert "stop" in params
    
    def test_to_dict(self):
        """Verifica serialización a diccionario"""
        skill = SkillConfig(
            name="test",
            description="Test skill",
            system_prompt="Test prompt",
            keywords=["test"],
            temperature=0.7
        )
        
        skill_dict = skill.to_dict()
        
        assert skill_dict["name"] == "test"
        assert skill_dict["description"] == "Test skill"
        assert skill_dict["system_prompt"] == "Test prompt"
        assert skill_dict["temperature"] == 0.7


class TestPredefinedSkills:
    """Tests para skills predefinidos"""
    
    def test_programming_skill_exists(self):
        """Verifica que el skill de programación existe"""
        assert PROGRAMMING_SKILL.name == "programming"
        assert "code" in PROGRAMMING_SKILL.keywords
        assert PROGRAMMING_SKILL.preferred_model == "solar"
        assert PROGRAMMING_SKILL.temperature == 0.3  # Baja para precisión
    
    def test_diagnosis_skill_exists(self):
        """Verifica que el skill de diagnóstico existe"""
        assert DIAGNOSIS_SKILL.name == "diagnosis"
        assert "problema" in DIAGNOSIS_SKILL.keywords
        assert DIAGNOSIS_SKILL.preferred_model == "solar"
    
    def test_financial_skill_exists(self):
        """Verifica que el skill financiero existe"""
        assert FINANCIAL_SKILL.name == "financial"
        assert "financiero" in FINANCIAL_SKILL.keywords
        assert FINANCIAL_SKILL.preferred_model == "solar"
    
    def test_creative_skill_exists(self):
        """Verifica que el skill creativo existe"""
        assert CREATIVE_SKILL.name == "creative"
        assert "crea" in CREATIVE_SKILL.keywords
        assert CREATIVE_SKILL.preferred_model == "lfm2"  # LFM2 mejor para soft
        assert CREATIVE_SKILL.temperature == 0.9  # Alta para creatividad
    
    def test_reasoning_skill_exists(self):
        """Verifica que el skill de razonamiento existe"""
        assert REASONING_SKILL.name == "reasoning"
        assert "razonamiento" in REASONING_SKILL.keywords
        assert REASONING_SKILL.preferred_model == "solar"
    
    def test_cto_skill_exists(self):
        """Verifica que el skill CTO existe"""
        assert CTO_SKILL.name == "cto"
        assert "arquitectura" in CTO_SKILL.keywords
        assert CTO_SKILL.preferred_model == "solar"
    
    def test_sre_skill_exists(self):
        """Verifica que el skill SRE existe"""
        assert SRE_SKILL.name == "sre"
        assert "kubernetes" in SRE_SKILL.keywords
        assert SRE_SKILL.preferred_model == "solar"
    
    def test_all_skills_registry(self):
        """Verifica que todos los skills están registrados"""
        assert len(ALL_SKILLS) == 7
        assert "programming" in ALL_SKILLS
        assert "diagnosis" in ALL_SKILLS
        assert "financial" in ALL_SKILLS
        assert "creative" in ALL_SKILLS
        assert "reasoning" in ALL_SKILLS
        assert "cto" in ALL_SKILLS
        assert "sre" in ALL_SKILLS


class TestSkillUtilityFunctions:
    """Tests para funciones de utilidad del sistema de skills"""
    
    def test_get_skill_by_name(self):
        """Verifica obtención de skill por nombre"""
        skill = get_skill("programming")
        assert skill is not None
        assert skill.name == "programming"
        
        skill = get_skill("PROGRAMMING")  # Case insensitive
        assert skill is not None
        
        skill = get_skill("nonexistent")
        assert skill is None
    
    def test_list_skills(self):
        """Verifica listado de todos los skills"""
        skills = list_skills()
        assert len(skills) == 7
        assert "programming" in skills
        assert "creative" in skills
    
    def test_match_skill_by_keywords_programming(self):
        """Verifica detección de skill por keywords - Programming"""
        # Long-tail: "código" + "python"
        skill = match_skill_by_keywords("Escribe código Python que calcule fibonacci")
        assert skill is not None
        assert skill.name == "programming"
    
    def test_match_skill_by_keywords_diagnosis(self):
        """Verifica detección de skill por keywords - Diagnosis"""
        # Long-tail: "diagnostica" + "problema"
        skill = match_skill_by_keywords("Diagnostica el problema de este error grave")
        assert skill is not None
        assert skill.name == "diagnosis"
    
    def test_match_skill_by_keywords_financial(self):
        """Verifica detección de skill por keywords - Financial"""
        # Long-tail: "roi" + "inversión"
        skill = match_skill_by_keywords("Calcula el ROI de esta inversión financiera")
        assert skill is not None
        assert skill.name == "financial"
    
    def test_match_skill_by_keywords_creative(self):
        """Verifica detección de skill por keywords - Creative"""
        # Long-tail: "crea" + "historia"
        skill = match_skill_by_keywords("Crea una historia corta sobre un robot")
        assert skill is not None
        assert skill.name == "creative"
    
    def test_match_skill_by_keywords_reasoning(self):
        """Verifica detección de skill por keywords - Reasoning"""
        # Long-tail: "razonamiento" + "lógico"
        skill = match_skill_by_keywords("Necesito razonamiento lógico para esta estrategia compleja")
        assert skill is not None
        assert skill.name == "reasoning"
    
    def test_match_skill_by_keywords_cto(self):
        """Verifica detección de skill por keywords - CTO"""
        # Long-tail: "arquitectura" + "sistema"
        skill = match_skill_by_keywords("¿Qué arquitectura de sistema recomendarías para escalar?")
        assert skill is not None
        assert skill.name == "cto"
    
    def test_match_skill_by_keywords_sre(self):
        """Verifica detección de skill por keywords - SRE"""
        # Long-tail: "kubernetes" + "cluster"
        skill = match_skill_by_keywords("Necesito configurar monitoring en un cluster Kubernetes")
        assert skill is not None
        assert skill.name == "sre"
    
    def test_match_skill_no_match(self):
        """Verifica que retorna None si no hay match"""
        skill = match_skill_by_keywords("Hola, ¿cómo estás?")
        assert skill is None
    
    def test_get_skills_info(self):
        """Verifica obtención de info resumida de todos los skills"""
        info = get_skills_info()
        
        assert len(info) == 7
        assert "programming" in info
        assert "description" in info["programming"]
        assert "keywords" in info["programming"]
        assert "temperature" in info["programming"]


class TestMCPSkillIntegration:
    """Tests para integración de skills con MCP"""
    
    def test_detect_and_apply_skill_programming(self):
        """Verifica detección y aplicación de skill desde MCP"""
        result = detect_and_apply_skill("Implementa un algoritmo de búsqueda binaria", "solar")
        
        assert result is not None
        assert result["skill_name"] == "programming"
        assert "system_prompt" in result
        assert "generation_params" in result
        assert "full_prompt" in result
        assert result["preferred_model"] == "solar"
    
    def test_detect_and_apply_skill_creative(self):
        """Verifica detección de skill creativo"""
        result = detect_and_apply_skill("Escribe un poema sobre el océano", "solar")
        
        assert result is not None
        assert result["skill_name"] == "creative"
        assert result["preferred_model"] == "lfm2"  # Debe recomendar LFM2
    
    def test_detect_and_apply_skill_no_match(self):
        """Verifica que retorna None si no hay skill aplicable"""
        result = detect_and_apply_skill("Hola", "solar")
        assert result is None
    
    def test_list_available_skills_from_mcp(self):
        """Verifica listado de skills desde MCP"""
        skills = list_available_skills()
        assert len(skills) == 7
    
    def test_get_skill_info_from_mcp(self):
        """Verifica obtención de info de skill desde MCP"""
        info = get_skill_info("programming")
        
        assert info is not None
        assert info["name"] == "programming"
        assert "system_prompt" in info
        assert "keywords" in info
    
    def test_get_skill_info_nonexistent(self):
        """Verifica que retorna None para skill inexistente"""
        info = get_skill_info("nonexistent")
        assert info is None


class TestSkillPromptGeneration:
    """Tests para generación de prompts completos"""
    
    def test_programming_prompt_generation(self):
        """Verifica generación de prompt para programming skill"""
        query = "Implementa quicksort en Python"
        result = detect_and_apply_skill(query, "solar")
        
        assert result is not None
        full_prompt = result["full_prompt"]
        
        assert "programming" in full_prompt.lower()
        assert query in full_prompt
        assert "User:" in full_prompt
        assert "Assistant:" in full_prompt
    
    def test_diagnosis_prompt_generation(self):
        """Verifica generación de prompt para diagnosis skill"""
        query = "Mi aplicación tiene un memory leak grave, diagnostica el origen del problema"
        result = detect_and_apply_skill(query, "solar")
        
        assert result is not None
        full_prompt = result["full_prompt"]
        
        assert "diagnostic" in full_prompt.lower()
        assert query in full_prompt
    
    def test_creative_prompt_generation(self):
        """Verifica generación de prompt para creative skill"""
        query = "Genera ideas para un producto innovador"
        result = detect_and_apply_skill(query, "lfm2")
        
        assert result is not None
        full_prompt = result["full_prompt"]
        
        assert "creative" in full_prompt.lower()
        assert query in full_prompt


class TestSkillParameters:
    """Tests para validación de parámetros de skills"""
    
    def test_programming_low_temperature(self):
        """Verifica que programming tiene baja temperature para precisión"""
        assert PROGRAMMING_SKILL.temperature == 0.3
    
    def test_creative_high_temperature(self):
        """Verifica que creative tiene alta temperature para creatividad"""
        assert CREATIVE_SKILL.temperature == 0.9
    
    def test_all_skills_have_valid_temperature(self):
        """Verifica que todos los skills tienen temperature válida"""
        for skill_name, skill in ALL_SKILLS.items():
            assert 0.0 <= skill.temperature <= 1.0, f"{skill_name} tiene temperature inválida"
    
    def test_all_skills_have_valid_max_tokens(self):
        """Verifica que todos los skills tienen max_tokens válido"""
        for skill_name, skill in ALL_SKILLS.items():
            assert skill.max_tokens > 0, f"{skill_name} tiene max_tokens inválido"
            assert skill.max_tokens <= 4096, f"{skill_name} excede max_tokens permitido"
    
    def test_all_skills_have_keywords(self):
        """Verifica que todos los skills tienen keywords"""
        for skill_name, skill in ALL_SKILLS.items():
            assert len(skill.keywords) > 0, f"{skill_name} no tiene keywords"
    
    def test_all_skills_have_system_prompt(self):
        """Verifica que todos los skills tienen system prompt"""
        for skill_name, skill in ALL_SKILLS.items():
            assert len(skill.system_prompt) > 0, f"{skill_name} no tiene system_prompt"
            assert len(skill.system_prompt) > 50, f"{skill_name} tiene system_prompt demasiado corto"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
