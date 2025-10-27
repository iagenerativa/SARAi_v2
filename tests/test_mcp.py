"""
Tests para el Meta Control Plane
"""

import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.mcp import MCPRules


class TestMCP:
    """Tests para el MCP"""
    
    def setup_method(self):
        """Setup antes de cada test"""
        self.mcp = MCPRules()
    
    def test_weights_sum_to_one(self):
        """Test: α + β = 1.0"""
        test_cases = [
            (0.9, 0.1),  # Hard puro
            (0.1, 0.9),  # Soft puro
            (0.5, 0.5),  # Híbrido
            (0.7, 0.4),  # Mixto
        ]
        
        for hard, soft in test_cases:
            alpha, beta = self.mcp.compute_weights(hard, soft)
            total = alpha + beta
            assert abs(total - 1.0) < 0.01, f"α + β = {total} (esperado 1.0)"
            print(f"✓ hard={hard}, soft={soft} → α={alpha:.2f}, β={beta:.2f} (sum={total:.2f})")
    
    def test_hard_task_routing(self):
        """Test: Tareas técnicas usan mostly hard-agent"""
        hard, soft = 0.95, 0.1
        alpha, beta = self.mcp.compute_weights(hard, soft)
        
        assert alpha > 0.9, f"Alpha debería ser >0.9 para tarea técnica, got {alpha}"
        assert beta < 0.1, f"Beta debería ser <0.1 para tarea técnica, got {beta}"
        print(f"✓ Tarea técnica → α={alpha:.2f} (>0.9)")
    
    def test_soft_task_routing(self):
        """Test: Tareas emocionales usan mostly soft-agent"""
        hard, soft = 0.1, 0.95
        alpha, beta = self.mcp.compute_weights(hard, soft)
        
        assert beta > 0.7, f"Beta debería ser >0.7 para tarea emocional, got {beta}"
        assert alpha < 0.3, f"Alpha debería ser <0.3 para tarea emocional, got {alpha}"
        print(f"✓ Tarea emocional → β={beta:.2f} (>0.7)")
    
    def test_hybrid_task_routing(self):
        """Test: Tareas híbridas usan ambos agentes"""
        hard, soft = 0.6, 0.6
        alpha, beta = self.mcp.compute_weights(hard, soft)
        
        # Ambos deberían estar balanceados
        assert 0.4 <= alpha <= 0.7, f"Alpha desbalanceado en híbrido: {alpha}"
        assert 0.3 <= beta <= 0.6, f"Beta desbalanceado en híbrido: {beta}"
        print(f"✓ Tarea híbrida → α={alpha:.2f}, β={beta:.2f} (balanceado)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
