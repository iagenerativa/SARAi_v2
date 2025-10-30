"""
Test para v2.16 Risk #5: Timeout Dinámico
Valida que _calculate_timeout() funciona correctamente según n_ctx
"""

import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_pool import _calculate_timeout


def test_timeout_table():
    """Valida tabla de referencia de timeouts"""
    
    # Casos de prueba: (n_ctx, timeout_esperado)
    test_cases = [
        (512, 15),   # 10 + (512/1024)*10 = 15
        (1024, 20),  # 10 + (1024/1024)*10 = 20
        (2048, 30),  # 10 + (2048/1024)*10 = 30
        (4096, 50),  # 10 + (4096/1024)*10 = 50
        (8192, 60),  # 10 + (8192/1024)*10 = 90 → min(90, 60) = 60
        (16384, 60), # 10 + (16384/1024)*10 = 170 → min(170, 60) = 60
    ]
    
    print("🧪 Testing _calculate_timeout() con tabla de referencia:")
    print(f"{'n_ctx':<10} {'Esperado':<15} {'Obtenido':<15} {'Estado'}")
    print("-" * 55)
    
    all_passed = True
    
    for n_ctx, expected_timeout in test_cases:
        actual_timeout = _calculate_timeout(n_ctx)
        passed = actual_timeout == expected_timeout
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"{n_ctx:<10} {expected_timeout:<15} {actual_timeout:<15} {status}")
        
        if not passed:
            all_passed = False
    
    print("-" * 55)
    
    if all_passed:
        print("✅ TODOS LOS TESTS PASARON")
        return True
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        return False


def test_timeout_edge_cases():
    """Valida casos extremos"""
    
    print("\n🧪 Testing casos extremos:")
    print(f"{'Caso':<30} {'n_ctx':<10} {'Timeout':<10} {'Estado'}")
    print("-" * 55)
    
    edge_cases = [
        ("Contexto mínimo", 256, 12),   # 10 + (256/1024)*10 ≈ 12
        ("Contexto típico small", 512, 15),
        ("Contexto típico medium", 2048, 30),
        ("Contexto típico large", 4096, 50),
        ("Límite superior 8K", 8192, 60),
        ("Límite superior 16K", 16384, 60),
        ("Extremo 32K", 32768, 60),  # Debe respetar max 60s
    ]
    
    all_passed = True
    
    for case_name, n_ctx, expected_max in edge_cases:
        actual_timeout = _calculate_timeout(n_ctx)
        # Para casos extremos, validar que no exceda max
        passed = actual_timeout <= 60 and (actual_timeout == expected_max if n_ctx <= 8192 else True)
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"{case_name:<30} {n_ctx:<10} {actual_timeout:<10} {status}")
        
        if not passed:
            all_passed = False
    
    print("-" * 55)
    
    if all_passed:
        print("✅ TODOS LOS EDGE CASES PASARON")
        return True
    else:
        print("❌ ALGUNOS EDGE CASES FALLARON")
        return False


def test_timeout_monotonicity():
    """Valida que timeout crece monótonamente con n_ctx (hasta el límite)"""
    
    print("\n🧪 Testing monotonía de timeout:")
    
    contexts = [256, 512, 1024, 2048, 4096, 8192, 16384]
    timeouts = [_calculate_timeout(ctx) for ctx in contexts]
    
    print(f"{'n_ctx':<10} {'Timeout':<10}")
    print("-" * 25)
    
    for ctx, timeout in zip(contexts, timeouts):
        print(f"{ctx:<10} {timeout:<10}")
    
    print("-" * 25)
    
    # Validar que crece hasta el límite, luego se estabiliza
    is_monotonic = True
    for i in range(len(timeouts) - 1):
        if timeouts[i] > timeouts[i+1]:
            print(f"❌ No monotónico: {timeouts[i]} > {timeouts[i+1]}")
            is_monotonic = False
    
    if is_monotonic:
        print("✅ Timeout es monotónico (crece o se mantiene)")
        return True
    else:
        print("❌ Timeout NO es monotónico")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("v2.16 Risk #5: Test de Timeout Dinámico")
    print("=" * 60)
    
    # Ejecutar todos los tests
    test1_passed = test_timeout_table()
    test2_passed = test_timeout_edge_cases()
    test3_passed = test_timeout_monotonicity()
    
    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    
    if test1_passed and test2_passed and test3_passed:
        print("✅ TODOS LOS TESTS PASARON - Risk #5 RESUELTO")
        sys.exit(0)
    else:
        print("❌ ALGUNOS TESTS FALLARON - REVISAR IMPLEMENTACIÓN")
        sys.exit(1)
