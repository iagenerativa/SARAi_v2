"""
Tests básicos para TRM-Classifier
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.trm_classifier import TRMClassifierSimulated


class TestTRMClassifier:
    """Tests para el clasificador TRM"""
    
    def setup_method(self):
        """Setup antes de cada test"""
        self.classifier = TRMClassifierSimulated()
    
    def test_hard_intent_detection(self):
        """Test: Detecta correctamente intención técnica"""
        prompts_hard = [
            "Configura SSH en Linux",
            "Error 404 en servidor Apache",
            "¿Cómo funciona el algoritmo de Dijkstra?",
            "Necesito depurar código Python"
        ]
        
        for prompt in prompts_hard:
            result = self.classifier.invoke(prompt)
            assert result["hard"] > 0.5, f"Hard score bajo para: {prompt}"
            print(f"✓ {prompt[:30]}... → hard={result['hard']:.2f}")
    
    def test_soft_intent_detection(self):
        """Test: Detecta correctamente intención emocional"""
        prompts_soft = [
            "Estoy muy triste hoy",
            "Me siento frustrado con mi trabajo",
            "Necesito motivación para seguir",
            "Gracias por tu ayuda, me siento mejor"
        ]
        
        for prompt in prompts_soft:
            result = self.classifier.invoke(prompt)
            assert result["soft"] > 0.5, f"Soft score bajo para: {prompt}"
            print(f"✓ {prompt[:30]}... → soft={result['soft']:.2f}")
    
    def test_hybrid_intent(self):
        """Test: Detecta correctamente intención híbrida"""
        prompts_hybrid = [
            "Explícame Python como a un niño",
            "Ayúdame a entender git, estoy perdido"
        ]
        
        for prompt in prompts_hybrid:
            result = self.classifier.invoke(prompt)
            # En híbrido, ambos deberían ser > 0.3
            assert result["hard"] > 0.3, f"Hard muy bajo en híbrido: {prompt}"
            assert result["soft"] > 0.3, f"Soft muy bajo en híbrido: {prompt}"
            print(f"✓ {prompt[:30]}... → hard={result['hard']:.2f}, soft={result['soft']:.2f}")
    
    def test_scores_in_range(self):
        """Test: Los scores están en rango [0, 1]"""
        prompts = [
            "Test 1",
            "¿Cómo estás?",
            "Error crítico",
            "Gracias"
        ]
        
        for prompt in prompts:
            result = self.classifier.invoke(prompt)
            assert 0.0 <= result["hard"] <= 1.0, f"Hard fuera de rango: {result['hard']}"
            assert 0.0 <= result["soft"] <= 1.0, f"Soft fuera de rango: {result['soft']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
