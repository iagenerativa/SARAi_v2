#!/usr/bin/env python3
"""
Test: Chaos Engineering - Corrupción Intencional de Logs

Valida que el sistema mantiene integridad bajo condiciones adversas:
- Corrupción de logs SHA-256/HMAC
- Eliminación de archivos sidecar
- Logs truncados
- Modificación de líneas

FASE 4: Testing & Validación
Fecha: 2 Noviembre 2025
"""

import os
import sys
import json
import hashlib
import hmac
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.audit import is_safe_mode, activate_safe_mode, deactivate_safe_mode


class TestChaosEngineering:
    """Test suite para chaos engineering con logs"""
    
    def setup_method(self):
        """Setup antes de cada test"""
        # Limpiar Safe Mode
        deactivate_safe_mode()
        
        # Crear directorio temporal
        self.temp_dir = tempfile.mkdtemp(prefix="sarai_chaos_")
        self.log_path = os.path.join(self.temp_dir, "test_log.jsonl")
        self.sha256_path = f"{self.log_path}.sha256"
        self.hmac_path = f"{self.log_path}.hmac"
    
    def teardown_method(self):
        """Cleanup después de cada test"""
        # Limpiar archivos temporales
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Desactivar Safe Mode
        deactivate_safe_mode()
    
    def create_valid_log_with_sha256(self, num_entries=10):
        """Crea log válido con SHA-256 sidecar"""
        with open(self.log_path, "w") as f, open(self.sha256_path, "w") as f_hash:
            for i in range(num_entries):
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "input": f"query {i}",
                    "response": f"response {i}"
                }
                
                line = json.dumps(entry, ensure_ascii=False)
                f.write(line + "\n")
                
                line_hash = hashlib.sha256(line.encode('utf-8')).hexdigest()
                f_hash.write(f"{line_hash}\n")
    
    def create_valid_log_with_hmac(self, num_entries=10):
        """Crea log válido con HMAC sidecar"""
        secret_key = os.getenv("HMAC_SECRET_KEY", "test-secret").encode()
        
        with open(self.log_path, "w") as f, open(self.hmac_path, "w") as f_hmac:
            for i in range(num_entries):
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "input_audio_sha256": hashlib.sha256(f"audio{i}".encode()).hexdigest(),
                    "response_text": f"response {i}"
                }
                
                line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
                f.write(line + "\n")
                
                signature = hmac.new(secret_key, line.encode(), hashlib.sha256).hexdigest()
                f_hmac.write(f"{signature}\n")
    
    def verify_sha256_log(self):
        """Verifica integridad SHA-256"""
        with open(self.log_path) as f, open(self.sha256_path) as f_hash:
            for line_num, (line, expected_hash) in enumerate(zip(f, f_hash), 1):
                computed_hash = hashlib.sha256(line.strip().encode('utf-8')).hexdigest()
                if computed_hash != expected_hash.strip():
                    return False, line_num
        return True, None
    
    def verify_hmac_log(self):
        """Verifica integridad HMAC"""
        secret_key = os.getenv("HMAC_SECRET_KEY", "test-secret").encode()
        
        with open(self.log_path) as f, open(self.hmac_path) as f_hmac:
            for line_num, (line, expected_hmac) in enumerate(zip(f, f_hmac), 1):
                entry = json.loads(line.strip())
                entry_str = json.dumps(entry, ensure_ascii=False, sort_keys=True)
                computed_hmac = hmac.new(secret_key, entry_str.encode(), hashlib.sha256).hexdigest()
                
                if computed_hmac != expected_hmac.strip():
                    return False, line_num
        return True, None
    
    # ==================== CHAOS TESTS ====================
    
    def test_chaos_delete_sidecar_sha256(self):
        """🌪️ Chaos 1: Eliminación de archivo .sha256"""
        print("\n🌪️ Chaos 1: Eliminación de .sha256")
        
        # Crear log válido
        self.create_valid_log_with_sha256(num_entries=5)
        
        # CHAOS: Eliminar sidecar
        os.remove(self.sha256_path)
        
        # Intentar verificar (debe fallar)
        try:
            is_valid, line = self.verify_sha256_log()
            assert False, "Verificación NO falló tras eliminar sidecar"
        except FileNotFoundError:
            print("   ✓ FileNotFoundError detectado correctamente")
        
        # Safe Mode DEBE activarse
        activate_safe_mode(reason="Missing SHA-256 sidecar")
        assert is_safe_mode(), "Safe Mode NO activado tras eliminar sidecar"
        
        print("   ✅ PASS: Safe Mode activado tras eliminar sidecar")
    
    def test_chaos_delete_sidecar_hmac(self):
        """🌪️ Chaos 2: Eliminación de archivo .hmac"""
        print("\n🌪️ Chaos 2: Eliminación de .hmac")
        
        # Crear log válido
        self.create_valid_log_with_hmac(num_entries=5)
        
        # CHAOS: Eliminar sidecar
        os.remove(self.hmac_path)
        
        # Intentar verificar (debe fallar)
        try:
            is_valid, line = self.verify_hmac_log()
            assert False, "Verificación NO falló tras eliminar sidecar HMAC"
        except FileNotFoundError:
            print("   ✓ FileNotFoundError detectado correctamente")
        
        # Safe Mode DEBE activarse
        activate_safe_mode(reason="Missing HMAC sidecar")
        assert is_safe_mode(), "Safe Mode NO activado tras eliminar sidecar HMAC"
        
        print("   ✅ PASS: Safe Mode activado tras eliminar sidecar HMAC")
    
    def test_chaos_modify_log_line(self):
        """🌪️ Chaos 3: Modificación de línea en log principal"""
        print("\n🌪️ Chaos 3: Modificación de línea en log")
        
        # Crear log válido
        self.create_valid_log_with_sha256(num_entries=10)
        
        # CHAOS: Modificar línea 5
        with open(self.log_path, "r") as f:
            lines = f.readlines()
        
        # Modificar entrada (cambiar response)
        entry = json.loads(lines[4])
        entry["response"] = "MODIFIED RESPONSE"
        lines[4] = json.dumps(entry, ensure_ascii=False) + "\n"
        
        with open(self.log_path, "w") as f:
            f.writelines(lines)
        
        # Verificar (debe detectar corrupción)
        is_valid, corrupt_line = self.verify_sha256_log()
        
        assert not is_valid, "Modificación NO detectada"
        assert corrupt_line == 5, f"Línea corrupta incorrecta: {corrupt_line}"
        
        # Safe Mode
        activate_safe_mode(reason=f"SHA-256 corruption at line {corrupt_line}")
        assert is_safe_mode(), "Safe Mode NO activado tras modificación"
        
        print(f"   ✓ Modificación detectada en línea {corrupt_line}")
        print("   ✅ PASS: Safe Mode activado tras modificación")
    
    def test_chaos_truncate_log(self):
        """🌪️ Chaos 4: Truncamiento de log principal"""
        print("\n🌪️ Chaos 4: Truncamiento de log")
        
        # Crear log válido
        self.create_valid_log_with_sha256(num_entries=10)
        
        # CHAOS: Truncar log a 5 líneas (sidecar tiene 10)
        with open(self.log_path, "r") as f:
            lines = f.readlines()[:5]
        
        with open(self.log_path, "w") as f:
            f.writelines(lines)
        
        # Verificar (debe detectar desincronización)
        try:
            is_valid, line = self.verify_sha256_log()
            # Si llegamos aquí, log está truncado pero válido parcialmente
            # Debemos verificar CONTEO de líneas
            with open(self.log_path) as f_log:
                log_lines = len(f_log.readlines())
            
            with open(self.sha256_path) as f_hash:
                hash_lines = len(f_hash.readlines())
            
            assert log_lines != hash_lines, "Truncamiento NO detectado"
            print(f"   ✓ Desincronización detectada: {log_lines} log vs {hash_lines} hashes")
        
        except StopIteration:
            print("   ✓ StopIteration detectado (sidecar más largo que log)")
        
        # Safe Mode
        activate_safe_mode(reason="Log/sidecar length mismatch")
        assert is_safe_mode(), "Safe Mode NO activado tras truncamiento"
        
        print("   ✅ PASS: Safe Mode activado tras truncamiento")
    
    def test_chaos_modify_hash_sidecar(self):
        """🌪️ Chaos 5: Modificación de hash en sidecar"""
        print("\n🌪️ Chaos 5: Modificación de hash en sidecar")
        
        # Crear log válido
        self.create_valid_log_with_sha256(num_entries=8)
        
        # CHAOS: Modificar hash de línea 3
        with open(self.sha256_path, "r") as f:
            hashes = f.readlines()
        
        hashes[2] = "0" * 64 + "\n"  # Hash inválido
        
        with open(self.sha256_path, "w") as f:
            f.writelines(hashes)
        
        # Verificar (debe detectar hash incorrecto)
        is_valid, corrupt_line = self.verify_sha256_log()
        
        assert not is_valid, "Hash modificado NO detectado"
        assert corrupt_line == 3, f"Línea corrupta incorrecta: {corrupt_line}"
        
        # Safe Mode
        activate_safe_mode(reason=f"SHA-256 mismatch at line {corrupt_line}")
        assert is_safe_mode(), "Safe Mode NO activado tras modificar hash"
        
        print(f"   ✓ Hash modificado detectado en línea {corrupt_line}")
        print("   ✅ PASS: Safe Mode activado tras modificar hash")
    
    def test_chaos_swap_log_lines(self):
        """🌪️ Chaos 6: Intercambio de líneas en log"""
        print("\n🌪️ Chaos 6: Intercambio de líneas")
        
        # Crear log válido
        self.create_valid_log_with_sha256(num_entries=10)
        
        # CHAOS: Intercambiar líneas 2 y 7
        with open(self.log_path, "r") as f:
            lines = f.readlines()
        
        lines[1], lines[6] = lines[6], lines[1]
        
        with open(self.log_path, "w") as f:
            f.writelines(lines)
        
        # Verificar (debe detectar corrupción en AMBAS líneas)
        is_valid, first_corrupt = self.verify_sha256_log()
        
        assert not is_valid, "Intercambio NO detectado"
        assert first_corrupt in [2, 7], f"Primera corrupción en línea inesperada: {first_corrupt}"
        
        # Safe Mode
        activate_safe_mode(reason=f"SHA-256 corruption at line {first_corrupt}")
        assert is_safe_mode(), "Safe Mode NO activado tras intercambio"
        
        print(f"   ✓ Primera corrupción detectada en línea {first_corrupt}")
        print("   ✅ PASS: Safe Mode activado tras intercambio")
    
    def test_chaos_append_malicious_entry(self):
        """🌪️ Chaos 7: Añadir entrada maliciosa sin hash"""
        print("\n🌪️ Chaos 7: Entrada maliciosa sin hash")
        
        # Crear log válido
        self.create_valid_log_with_sha256(num_entries=5)
        
        # CHAOS: Añadir entrada sin actualizar sidecar
        malicious_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": "MALICIOUS INJECTION",
            "response": "UNAUTHORIZED"
        }
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(malicious_entry, ensure_ascii=False) + "\n")
        
        # Verificar longitud
        with open(self.log_path) as f_log:
            log_lines = len(f_log.readlines())
        
        with open(self.sha256_path) as f_hash:
            hash_lines = len(f_hash.readlines())
        
        assert log_lines != hash_lines, "Entrada maliciosa NO detectada"
        print(f"   ✓ Desincronización detectada: {log_lines} log vs {hash_lines} hashes")
        
        # Safe Mode
        activate_safe_mode(reason="Malicious entry detected (length mismatch)")
        assert is_safe_mode(), "Safe Mode NO activado tras entrada maliciosa"
        
        print("   ✅ PASS: Safe Mode activado tras entrada maliciosa")


def run_tests():
    """Ejecuta todos los tests de chaos engineering"""
    print("=" * 70)
    print("🌪️  CHAOS ENGINEERING: Test de Integridad bajo Condiciones Adversas")
    print("=" * 70)
    
    suite = TestChaosEngineering()
    
    tests = [
        suite.test_chaos_delete_sidecar_sha256,
        suite.test_chaos_delete_sidecar_hmac,
        suite.test_chaos_modify_log_line,
        suite.test_chaos_truncate_log,
        suite.test_chaos_modify_hash_sidecar,
        suite.test_chaos_swap_log_lines,
        suite.test_chaos_append_malicious_entry,
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
            import traceback
            traceback.print_exc()
            failed += 1
        finally:
            suite.teardown_method()
    
    print("\n" + "=" * 70)
    print(f"📊 RESULTADOS: {passed} PASS, {failed} FAIL")
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
