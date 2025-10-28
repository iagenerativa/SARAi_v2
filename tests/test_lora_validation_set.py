"""
Unit Tests para LoRA Validation Dataset
========================================

Valida que data/lora_validation_set.jsonl tiene el formato correcto
y contiene queries de calidad.
"""

import pytest
import json
from pathlib import Path


class TestLoRAValidationSet:
    """Tests para dataset de validación de LoRA"""
    
    @pytest.fixture
    def validation_set_path(self):
        """Path al validation set"""
        return Path("data/lora_validation_set.jsonl")
    
    def test_file_exists(self, validation_set_path):
        """Test: Archivo existe"""
        assert validation_set_path.exists(), "Validation set not found"
    
    def test_json_format(self, validation_set_path):
        """Test: Todas las líneas son JSON válido"""
        with open(validation_set_path) as f:
            for i, line in enumerate(f, start=1):
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON at line {i}: {e}")
    
    def test_required_fields(self, validation_set_path):
        """Test: Todas las entradas tienen campos requeridos"""
        required_fields = ["input", "expected_keywords", "category", "difficulty"]
        
        with open(validation_set_path) as f:
            for i, line in enumerate(f, start=1):
                entry = json.loads(line)
                
                for field in required_fields:
                    assert field in entry, f"Line {i}: Missing field '{field}'"
    
    def test_expected_keywords_is_list(self, validation_set_path):
        """Test: expected_keywords es una lista no vacía"""
        with open(validation_set_path) as f:
            for i, line in enumerate(f, start=1):
                entry = json.loads(line)
                
                keywords = entry["expected_keywords"]
                assert isinstance(keywords, list), f"Line {i}: expected_keywords must be list"
                assert len(keywords) > 0, f"Line {i}: expected_keywords cannot be empty"
    
    def test_minimum_queries(self, validation_set_path):
        """Test: Al menos 50 queries en el dataset"""
        with open(validation_set_path) as f:
            count = sum(1 for _ in f)
        
        assert count >= 50, f"Need at least 50 queries, found {count}"
    
    def test_category_distribution(self, validation_set_path):
        """Test: Diversidad de categorías"""
        categories = set()
        
        with open(validation_set_path) as f:
            for line in f:
                entry = json.loads(line)
                categories.add(entry["category"])
        
        # Al menos 5 categorías diferentes
        assert len(categories) >= 5, f"Need at least 5 categories, found {len(categories)}"
    
    def test_difficulty_levels(self, validation_set_path):
        """Test: Mezcla de dificultades"""
        difficulties = set()
        
        with open(validation_set_path) as f:
            for line in f:
                entry = json.loads(line)
                difficulties.add(entry["difficulty"])
        
        # Debe tener basic, intermediate, advanced
        expected_levels = {"basic", "intermediate", "advanced"}
        assert expected_levels.issubset(difficulties), \
            f"Missing difficulty levels: {expected_levels - difficulties}"
    
    def test_keyword_quality(self, validation_set_path):
        """Test: Keywords son relevantes (no muy genéricos)"""
        # Palabras muy genéricas que no deberían estar solas
        too_generic = {"es", "un", "una", "el", "la", "de", "en", "a"}
        
        with open(validation_set_path) as f:
            for i, line in enumerate(f, start=1):
                entry = json.loads(line)
                
                for keyword in entry["expected_keywords"]:
                    # Excepciones: números, fórmulas químicas (con dígitos), siglas
                    is_number = keyword.isdigit()
                    is_formula = any(char.isdigit() for char in keyword)  # CO2, H2O, O3
                    is_acronym = keyword.isupper() and len(keyword) <= 5  # TCP, HTTP, URSS
                    
                    # Keywords deben tener al menos 3 caracteres (excepto casos especiales)
                    if not (is_number or is_formula or is_acronym):
                        assert len(keyword) >= 3, \
                            f"Line {i}: Keyword '{keyword}' too short"
                    
                    # No deben ser solo palabras genéricas
                    assert keyword.lower() not in too_generic, \
                        f"Line {i}: Keyword '{keyword}' too generic"
    
    def test_input_diversity(self, validation_set_path):
        """Test: No hay inputs duplicados"""
        inputs_seen = set()
        
        with open(validation_set_path) as f:
            for i, line in enumerate(f, start=1):
                entry = json.loads(line)
                input_text = entry["input"]
                
                assert input_text not in inputs_seen, \
                    f"Line {i}: Duplicate input found: {input_text}"
                
                inputs_seen.add(input_text)
    
    def test_sample_query_structure(self, validation_set_path):
        """Test: Estructura de una query de ejemplo"""
        with open(validation_set_path) as f:
            first_line = f.readline()
            entry = json.loads(first_line)
        
        # Verificar estructura completa
        assert isinstance(entry["input"], str)
        assert isinstance(entry["expected_keywords"], list)
        assert isinstance(entry["category"], str)
        assert isinstance(entry["difficulty"], str)
        
        # Verificar que input es una pregunta o comando
        assert len(entry["input"]) > 10, "Input too short"
        
        # Verificar que hay al menos 3 keywords
        assert len(entry["expected_keywords"]) >= 3, "Need at least 3 keywords"


class TestLoRAValidationIntegration:
    """Integration tests con LoRA trainer"""
    
    @pytest.mark.integration
    def test_validation_set_loadable_by_lora_trainer(self):
        """Test: El validation set puede ser cargado por LoRANightlyTrainer"""
        from pathlib import Path
        import json
        
        validation_set_path = Path("data/lora_validation_set.jsonl")
        
        # Simular carga por LoRANightlyTrainer
        validation_queries = []
        with open(validation_set_path) as f:
            for line in f:
                validation_queries.append(json.loads(line))
        
        assert len(validation_queries) >= 50
        
        # Sample 10 queries (como hace _validate_lora)
        import random
        sample = random.sample(validation_queries, min(10, len(validation_queries)))
        
        assert len(sample) == 10
        
        # Cada query debe tener la estructura esperada
        for query in sample:
            assert "input" in query
            assert "expected_keywords" in query
