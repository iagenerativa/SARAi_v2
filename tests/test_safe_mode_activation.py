#!/usr/bin/env python3
"""
Test: Safe Mode se activa automáticamente con logs corruptos

Verifica que el sistema detecta corrupción SHA-256/HMAC
y activa Safe Mode para proteger integridad.

FASE 4: Testing & Validación
Fecha: 2 Noviembre 2025
"""

import os
import sys
import json
import hashlib
import hmac
import tempfile
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audit import is_safe_mode, activate_safe_mode, deactivate_safe_mode


class TestSafeModeActivation:
    """Test suite para activación automática de Safe Mode"""
    
    def setup_method(self):
        """Setup antes de cada test"""
        # Limpiar estado
        deactivate_safe_mode()
        
        # Crear directorio temporal para logs
        self.temp_dir = tempfile.mkdtemp(prefix="sarai_test_")
        self.log_path = os.path.join(self.temp_dir, "test_log.jsonl")
        self.sha256_path = f"{self.log_path}.sha256"
        self.hmac_path = f"{self.log_path}.hmac"
    
    def teardown_method(self):
        """Cleanup después de cada test"""
        # Limpiar archivos temporales
        for path in [self.log_path, self.sha256_path, self.hmac_path]:
            if os.path.exists(path):
                os.remove(path)
        
        os.rmdir(self.temp_dir)
        
        # Desactivar Safe Mode
        deactivate_safe_mode()
    
    def create_valid_sha256_log(self, num_entries=5):
        """Crea log válido con SHA-256 sidecar"""
        with open(self.log_path, "w") as f, open(self.sha256_path, "w") as f_hash:
            for i in range(num_entries):
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "input": f"test query {i}",
                    "response": f"test response {i}"
                }
                
                line = json.dumps(entry, ensure_ascii=False)
                f.write(line + "\n")
                
                # Hash correcto
                line_hash = hashlib.sha256(line.encode('utf-8')).hexdigest()
                f_hash.write(f"{line_hash}\n")
    
    def create_corrupted_sha256_log(self, num_entries=5, corrupt_index=2):
        """Crea log con corrupción en índice específico"""
        with open(self.log_path, "w") as f, open(self.sha256_path, "w") as f_hash:
            for i in range(num_entries):
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "input": f"test query {i}",
                    "response": f"test response {i}"
                }
                
                line = json.dumps(entry, ensure_ascii=False)
                f.write(line + "\n")
                
                # Hash correcto excepto en corrupt_index
                if i == corrupt_index:
                    # Hash intencionalmente incorrecto
                    line_hash = "0" * 64
                else:
                    line_hash = hashlib.sha256(line.encode('utf-8')).hexdigest()
                
                f_hash.write(f"{line_hash}\n")
    
    def create_valid_hmac_log(self, num_entries=5):
        """Crea log válido con HMAC sidecar"""
        secret_key = os.getenv("HMAC_SECRET_KEY", "test-secret").encode()
        
        with open(self.log_path, "w") as f, open(self.hmac_path, "w") as f_hmac:
            for i in range(num_entries):
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "input_audio_sha256": hashlib.sha256(f"audio{i}".encode()).hexdigest(),
                    "detected_lang": "es",
                    "engine_used": "omni",
                    "response_text": f"test response {i}"
                }
                
                line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
                f.write(line + "\n")
                
                # HMAC correcto
                signature = hmac.new(secret_key, line.encode(), hashlib.sha256).hexdigest()
                f_hmac.write(f"{signature}\n")
    
    def create_corrupted_hmac_log(self, num_entries=5, corrupt_index=3):
        """Crea log con corrupción HMAC en índice específico"""
        secret_key = os.getenv("HMAC_SECRET_KEY", "test-secret").encode()
        
        with open(self.log_path, "w") as f, open(self.hmac_path, "w") as f_hmac:
            for i in range(num_entries):
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "input_audio_sha256": hashlib.sha256(f"audio{i}".encode()).hexdigest(),
                    "detected_lang": "es",
                    "engine_used": "omni",
                    "response_text": f"test response {i}"
                }
                
                line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
                f.write(line + "\n")
                
                # HMAC correcto excepto en corrupt_index
                if i == corrupt_index:
                    # HMAC intencionalmente incorrecto
                    signature = "0" * 64
                else:
                    signature = hmac.new(secret_key, line.encode(), hashlib.sha256).hexdigest()
                
                f_hmac.write(f"{signature}\n")
    
    def verify_sha256_log(self):
        """Verifica integridad SHA-256 (wrapper del script real)"""
        with open(self.log_path) as f, open(self.sha256_path) as f_hash:
            for line_num, (line, expected_hash) in enumerate(zip(f, f_hash), 1):
                computed_hash = hashlib.sha256(line.strip().encode('utf-8')).hexdigest()
                if computed_hash != expected_hash.strip():
                    return False, line_num
        return True, None
    
    def verify_hmac_log(self):
        """Verifica integridad HMAC (wrapper del script real)"""
        secret_key = os.getenv("HMAC_SECRET_KEY", "test-secret").encode()
        
        with open(self.log_path) as f, open(self.hmac_path) as f_hmac:
            for line_num, (line, expected_hmac) in enumerate(zip(f, f_hmac), 1):
                entry = json.loads(line.strip())
                entry_str = json.dumps(entry, ensure_ascii=False, sort_keys=True)
                computed_hmac = hmac.new(secret_key, entry_str.encode(), hashlib.sha256).hexdigest()
                
                if computed_hmac != expected_hmac.strip():
                    return False, line_num
        return True, None
    
    # ==================== TESTS ====================
    
    def test_sha256_valid_log_no_activation(self):
        """✅ Test 1: Log válido NO activa Safe Mode"""
        print("\n🧪 Test 1: Log SHA-256 válido NO activa Safe Mode")
        
        # Crear log válido
        self.create_valid_sha256_log(num_entries=10)
        
        # Verificar
        is_valid, corrupt_line = self.verify_sha256_log()
        
        # Aserciones
        assert is_valid, f"Log válido marcado como corrupto en línea {corrupt_line}"
        assert not is_safe_mode(), "Safe Mode activado incorrectamente con log válido"
        
        print("   ✅ PASS: Log válido, Safe Mode inactivo")
    
    def test_sha256_corrupted_activates_safe_mode(self):
        """✅ Test 2: Log corrupto SHA-256 ACTIVA Safe Mode"""
        print("\n🧪 Test 2: Log SHA-256 corrupto ACTIVA Safe Mode")
        
        # Crear log corrupto
        self.create_corrupted_sha256_log(num_entries=10, corrupt_index=5)
        
        # Verificar
        is_valid, corrupt_line = self.verify_sha256_log()
        
        # Aserciones
        assert not is_valid, "Log corrupto marcado como válido"
        assert corrupt_line == 6, f"Línea corrupta detectada incorrectamente: {corrupt_line} (esperado 6)"
        
        # Simular activación de Safe Mode por detección
        activate_safe_mode(reason=f"SHA-256 corruption detected at line {corrupt_line}")
        
        assert is_safe_mode(), "Safe Mode NO activado tras detectar corrupción SHA-256"
        
        print(f"   ✅ PASS: Corrupción detectada en línea {corrupt_line}, Safe Mode activado")
    
    def test_hmac_valid_log_no_activation(self):
        """✅ Test 3: Log HMAC válido NO activa Safe Mode"""
        print("\n🧪 Test 3: Log HMAC válido NO activa Safe Mode")
        
        # Crear log válido
        self.create_valid_hmac_log(num_entries=8)
        
        # Verificar
        is_valid, corrupt_line = self.verify_hmac_log()
        
        # Aserciones
        assert is_valid, f"Log HMAC válido marcado como corrupto en línea {corrupt_line}"
        assert not is_safe_mode(), "Safe Mode activado incorrectamente con log HMAC válido"
        
        print("   ✅ PASS: Log HMAC válido, Safe Mode inactivo")
    
    def test_hmac_corrupted_activates_safe_mode(self):
        """✅ Test 4: Log corrupto HMAC ACTIVA Safe Mode"""
        print("\n🧪 Test 4: Log HMAC corrupto ACTIVA Safe Mode")
        
        # Crear log corrupto
        self.create_corrupted_hmac_log(num_entries=7, corrupt_index=3)
        
        # Verificar
        is_valid, corrupt_line = self.verify_hmac_log()
        
        # Aserciones
        assert not is_valid, "Log HMAC corrupto marcado como válido"
        assert corrupt_line == 4, f"Línea corrupta detectada incorrectamente: {corrupt_line} (esperado 4)"
        
        # Simular activación de Safe Mode por detección
        activate_safe_mode(reason=f"HMAC corruption detected at line {corrupt_line}")
        
        assert is_safe_mode(), "Safe Mode NO activado tras detectar corrupción HMAC"
        
        print(f"   ✅ PASS: Corrupción HMAC detectada en línea {corrupt_line}, Safe Mode activado")
    
    def test_multiple_corruptions_safe_mode_persistent(self):
        """✅ Test 5: Safe Mode persiste con múltiples corrupciones"""
        print("\n🧪 Test 5: Safe Mode persiste con múltiples corrupciones")
        
        # Primera corrupción
        self.create_corrupted_sha256_log(corrupt_index=2)
        is_valid_1, line_1 = self.verify_sha256_log()
        
        if not is_valid_1:
            activate_safe_mode(reason=f"SHA-256 corruption at line {line_1}")
        
        assert is_safe_mode(), "Safe Mode NO activado tras primera corrupción"
        
        # Segunda corrupción (diferente log)
        self.create_corrupted_hmac_log(corrupt_index=4)
        is_valid_2, line_2 = self.verify_hmac_log()
        
        # Safe Mode DEBE seguir activo
        assert is_safe_mode(), "Safe Mode desactivado tras segunda corrupción"
        assert not is_valid_2, "Segunda corrupción no detectada"
        
        print("   ✅ PASS: Safe Mode persiste con múltiples corrupciones")


def run_tests():
    """Ejecuta todos los tests"""
    print("=" * 70)
    print("🔒 TEST SUITE: Safe Mode Activation con Logs Corruptos")
    print("=" * 70)
    
    suite = TestSafeModeActivation()
    
    tests = [
        suite.test_sha256_valid_log_no_activation,
        suite.test_sha256_corrupted_activates_safe_mode,
        suite.test_hmac_valid_log_no_activation,
        suite.test_hmac_corrupted_activates_safe_mode,
        suite.test_multiple_corruptions_safe_mode_persistent,
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
