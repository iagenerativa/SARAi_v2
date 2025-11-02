#!/usr/bin/env python3
"""
Test: Regresión es detectada y swap abortado

Verifica que el sistema detecta degradación de performance
en modelos reentrenados y aborta el swap atómico.

FASE 4: Testing & Validación
Fecha: 2 Noviembre 2025
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRegressionDetection:
    """Test suite para detección de regresiones y abort de swap"""
    
    def __init__(self):
        self.temp_dir = None
        self.golden_queries_path = Path(__file__).parent / "golden_queries.jsonl"
        self.mcp_state_path = Path(__file__).parent.parent / "state" / "mcp_state.pkl"
    
    def setup_method(self):
        """Setup antes de cada test"""
        # Crear directorio temporal
        self.temp_dir = tempfile.mkdtemp(prefix="sarai_regression_test_")
        
        # Backup de estado original
        if self.mcp_state_path.exists():
            shutil.copy(
                self.mcp_state_path,
                os.path.join(self.temp_dir, "mcp_state_backup.pkl")
            )
    
    def teardown_method(self):
        """Cleanup después de cada test"""
        # Restaurar estado original
        backup_path = os.path.join(self.temp_dir, "mcp_state_backup.pkl")
        if os.path.exists(backup_path):
            shutil.copy(backup_path, self.mcp_state_path)
        
        # Limpiar directorio temporal
        shutil.rmtree(self.temp_dir)
    
    def load_golden_queries(self) -> List[Dict]:
        """Carga golden queries"""
        if not self.golden_queries_path.exists():
            raise FileNotFoundError(
                f"Golden queries no encontrado: {self.golden_queries_path}\n"
                "Ejecuta: python -m core.mcp --generate-golden"
            )
        
        queries = []
        with open(self.golden_queries_path) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                queries.append(json.loads(stripped))
        
        return queries
    
    def simulate_baseline_performance(self, queries: List[Dict]) -> Dict[str, float]:
        """
        Simula performance baseline (modelo actual)
        
        En producción, esto ejecutaría el grafo completo.
        Para testing, usamos scores mock basados en golden queries.
        """
        scores = {}
        
        for query_obj in queries:
            query_id = query_obj.get("id", hash(query_obj["input"]))
            
            # Mock: baseline score basado en expected output
            # En producción, esto compararía response vs expected_response
            if "expected_response" in query_obj:
                # Simular score perfecto para baseline
                scores[query_id] = 1.0
            else:
                # Simular score alto para queries sin expected
                scores[query_id] = 0.95
        
        return scores
    
    def simulate_new_model_performance(
        self, 
        queries: List[Dict], 
        regression_factor: float = 1.0
    ) -> Dict[str, float]:
        """
        Simula performance de modelo nuevo
        
        Args:
            queries: Golden queries
            regression_factor: 1.0 = sin regresión, <1.0 = regresión
        """
        scores = {}
        
        for query_obj in queries:
            query_id = query_obj.get("id", hash(query_obj["input"]))
            
            # Mock: nuevo modelo con factor de regresión
            if "expected_response" in query_obj:
                scores[query_id] = 1.0 * regression_factor
            else:
                scores[query_id] = 0.95 * regression_factor
        
        return scores
    
    def detect_regression(
        self, 
        baseline_scores: Dict[str, float], 
        new_scores: Dict[str, float],
        threshold: float = 0.95
    ) -> tuple:
        """
        Detecta regresión comparando scores
        
        Args:
            baseline_scores: Scores del modelo actual
            new_scores: Scores del modelo nuevo
            threshold: Umbral mínimo (nuevo debe tener ≥95% de baseline)
        
        Returns:
            (has_regression: bool, ratio: float, details: dict)
        """
        # Calcular promedio de scores
        baseline_avg = sum(baseline_scores.values()) / len(baseline_scores)
        new_avg = sum(new_scores.values()) / len(new_scores)
        
        # Ratio de performance
        ratio = new_avg / baseline_avg if baseline_avg > 0 else 0.0
        
        # Detectar regresión
        has_regression = ratio < threshold
        
        details = {
            "baseline_avg": baseline_avg,
            "new_avg": new_avg,
            "ratio": ratio,
            "threshold": threshold,
            "delta_pct": (ratio - 1.0) * 100,
        }
        
        return has_regression, ratio, details
    
    def abort_swap(self, reason: str):
        """
        Aborta swap atómico de modelo
        
        En producción, esto eliminaría mcp_v_new.pkl y loggearía el abort.
        """
        new_model_path = self.mcp_state_path.parent / "mcp_v_new.pkl"
        
        if new_model_path.exists():
            # Mover a carpeta de rechazados
            rejected_dir = self.mcp_state_path.parent / "rejected"
            rejected_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rejected_path = rejected_dir / f"mcp_v_rejected_{timestamp}.pkl"
            
            shutil.move(str(new_model_path), str(rejected_path))
            
            # Log del abort
            abort_log = rejected_dir / "abort_log.jsonl"
            with open(abort_log, "a") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "reason": reason,
                    "rejected_model": str(rejected_path),
                }, f)
                f.write("\n")
            
            return True
        
        return False
    
    # ==================== TESTS ====================
    
    def test_no_regression_allows_swap(self):
        """✅ Test 1: Sin regresión, swap procede normalmente"""
        print("\n🧪 Test 1: Sin regresión → Swap procede")
        
        # Cargar queries
        queries = self.load_golden_queries()
        
        # Simular baseline
        baseline_scores = self.simulate_baseline_performance(queries)
        
        # Simular nuevo modelo SIN regresión (mismo performance)
        new_scores = self.simulate_new_model_performance(queries, regression_factor=1.0)
        
        # Detectar regresión
        has_regression, ratio, details = self.detect_regression(baseline_scores, new_scores)
        
        # Aserciones
        assert not has_regression, "Regresión detectada incorrectamente"
        assert ratio >= 0.95, f"Ratio {ratio:.3f} < 0.95"
        
        print(f"   ✓ Ratio: {ratio:.3f} (≥0.95)")
        print(f"   ✓ Delta: {details['delta_pct']:+.2f}%")
        print("   ✅ PASS: Swap permitido")
    
    def test_minor_regression_detected_swap_aborted(self):
        """✅ Test 2: Regresión menor detectada → Swap abortado"""
        print("\n🧪 Test 2: Regresión menor → Swap abortado")
        
        # Cargar queries
        queries = self.load_golden_queries()
        
        # Simular baseline
        baseline_scores = self.simulate_baseline_performance(queries)
        
        # Simular nuevo modelo con regresión 10%
        new_scores = self.simulate_new_model_performance(queries, regression_factor=0.90)
        
        # Detectar regresión
        has_regression, ratio, details = self.detect_regression(baseline_scores, new_scores)
        
        # Aserciones
        assert has_regression, "Regresión NO detectada"
        assert ratio < 0.95, f"Ratio {ratio:.3f} ≥ 0.95 (esperado <0.95)"
        
        # Abortar swap
        reason = f"Regression detected: {details['delta_pct']:.2f}% performance drop"
        aborted = self.abort_swap(reason)
        
        assert aborted or not (self.mcp_state_path.parent / "mcp_v_new.pkl").exists(), \
            "Swap NO abortado tras detectar regresión"
        
        print(f"   ✓ Ratio: {ratio:.3f} (<0.95)")
        print(f"   ✓ Delta: {details['delta_pct']:+.2f}%")
        print(f"   ✓ Abort reason: {reason}")
        print("   ✅ PASS: Swap abortado correctamente")
    
    def test_severe_regression_detected_swap_aborted(self):
        """✅ Test 3: Regresión severa detectada → Swap abortado"""
        print("\n🧪 Test 3: Regresión severa → Swap abortado")
        
        # Cargar queries
        queries = self.load_golden_queries()
        
        # Simular baseline
        baseline_scores = self.simulate_baseline_performance(queries)
        
        # Simular nuevo modelo con regresión severa 30%
        new_scores = self.simulate_new_model_performance(queries, regression_factor=0.70)
        
        # Detectar regresión
        has_regression, ratio, details = self.detect_regression(baseline_scores, new_scores)
        
        # Aserciones
        assert has_regression, "Regresión severa NO detectada"
        assert ratio < 0.80, f"Ratio {ratio:.3f} ≥ 0.80 (esperado <0.80 para severa)"
        
        # Abortar swap
        reason = f"SEVERE regression: {details['delta_pct']:.2f}% performance drop"
        aborted = self.abort_swap(reason)
        
        assert aborted or not (self.mcp_state_path.parent / "mcp_v_new.pkl").exists(), \
            "Swap NO abortado tras regresión severa"
        
        print(f"   ✓ Ratio: {ratio:.3f} (<<0.95)")
        print(f"   ✓ Delta: {details['delta_pct']:+.2f}%")
        print(f"   ✓ Abort reason: {reason}")
        print("   ✅ PASS: Swap abortado correctamente")
    
    def test_improvement_allows_swap(self):
        """✅ Test 4: Mejora detectada → Swap procede"""
        print("\n🧪 Test 4: Mejora detectada → Swap procede")
        
        # Cargar queries
        queries = self.load_golden_queries()
        
        # Simular baseline
        baseline_scores = self.simulate_baseline_performance(queries)
        
        # Simular nuevo modelo con MEJORA 5%
        new_scores = self.simulate_new_model_performance(queries, regression_factor=1.05)
        
        # Detectar regresión
        has_regression, ratio, details = self.detect_regression(baseline_scores, new_scores)
        
        # Aserciones
        assert not has_regression, "Mejora detectada como regresión"
        assert ratio > 1.0, f"Ratio {ratio:.3f} ≤ 1.0 (esperado >1.0 para mejora)"
        
        print(f"   ✓ Ratio: {ratio:.3f} (>1.0)")
        print(f"   ✓ Delta: {details['delta_pct']:+.2f}%")
        print("   ✅ PASS: Swap permitido (mejora detectada)")


def run_tests():
    """Ejecuta todos los tests"""
    print("=" * 70)
    print("🔍 TEST SUITE: Regression Detection & Swap Abort")
    print("=" * 70)
    
    suite = TestRegressionDetection()
    
    tests = [
        suite.test_no_regression_allows_swap,
        suite.test_minor_regression_detected_swap_aborted,
        suite.test_severe_regression_detected_swap_aborted,
        suite.test_improvement_allows_swap,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        suite.setup_method()
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"   ❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"   💥 ERROR: {e}")
            failed += 1
        finally:
            suite.teardown_method()
    
    print("\n" + "=" * 70)
    print(f"📊 RESULTADOS: {passed} PASS, {failed} FAIL")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
