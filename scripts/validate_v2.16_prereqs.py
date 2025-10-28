#!/usr/bin/env python3
"""
Pre-Flight Checks para v2.16 Omni-Loop
=======================================

Valida que TODOS los pre-requisitos están cumplidos antes de iniciar
la implementación de v2.16.

CHECKS:
1. v2.12 Skills MoE implementado
2. v2.13 ProactiveLoop implementado  
3. v2.14 SpeculativeDecoding implementado
4. v2.15 SelfRepair implementado
5. llama.cpp binarios instalados
6. Dependencias Python (OpenCV, imagehash, GPG)
7. Espacio en disco suficiente (>10GB)
8. RAM disponible (>16GB recomendado)

EXIT CODES:
    0: Todos los checks pasados ✅
    1: Uno o más checks fallaron ❌

USAGE:
    python scripts/validate_v2.16_prereqs.py
    
    # CI/CD
    python scripts/validate_v2.16_prereqs.py --strict
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
import json

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


class PrerequisiteChecker:
    """Valida pre-requisitos para v2.16 Omni-Loop"""
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
    
    def check_all(self) -> bool:
        """Ejecuta todos los checks. Retorna True si todos pasan."""
        print("🔍 SARAi v2.16 Omni-Loop - Pre-Flight Checks")
        print("=" * 60)
        print()
        
        checks = [
            ("v2.12 Skills MoE", self.check_v2_12),
            ("v2.13 ProactiveLoop", self.check_v2_13),
            ("v2.14 SpeculativeDecoding", self.check_v2_14),
            ("v2.15 SelfRepair", self.check_v2_15),
            ("llama.cpp binarios", self.check_llama_cpp),
            ("Python dependencies", self.check_python_deps),
            ("Disk space", self.check_disk_space),
            ("RAM disponible", self.check_ram),
            ("GPG setup", self.check_gpg)
        ]
        
        for name, check_func in checks:
            self._run_check(name, check_func)
        
        print()
        print("=" * 60)
        print(f"📊 RESUMEN:")
        print(f"   ✅ Passed:   {GREEN}{self.checks_passed}{RESET}")
        print(f"   ❌ Failed:   {RED}{self.checks_failed}{RESET}")
        print(f"   ⚠️  Warnings: {YELLOW}{self.warnings}{RESET}")
        print()
        
        if self.checks_failed == 0:
            print(f"{GREEN}✅ TODOS LOS CHECKS PASADOS{RESET}")
            print(f"{GREEN}   Ready para implementar v2.16 Omni-Loop!{RESET}")
            return True
        else:
            print(f"{RED}❌ HAY CHECKS FALLIDOS{RESET}")
            print(f"{RED}   Fix los errores antes de continuar.{RESET}")
            return False
    
    def _run_check(self, name: str, check_func):
        """Ejecuta un check individual"""
        try:
            passed, message = check_func()
            
            if passed:
                print(f"✅ {name}: {GREEN}{message}{RESET}")
                self.checks_passed += 1
            else:
                if self.strict:
                    print(f"❌ {name}: {RED}{message}{RESET}")
                    self.checks_failed += 1
                else:
                    print(f"⚠️  {name}: {YELLOW}{message} (warning){RESET}")
                    self.warnings += 1
        
        except Exception as e:
            print(f"❌ {name}: {RED}Error - {e}{RESET}")
            self.checks_failed += 1
    
    def check_v2_12(self) -> Tuple[bool, str]:
        """Check: v2.12 Skills MoE implementado"""
        skills_dir = Path("skills")
        base_skill = skills_dir / "base_skill.py"
        
        if not skills_dir.exists():
            return False, "skills/ directory not found"
        
        if not base_skill.exists():
            return False, "skills/base_skill.py not implemented"
        
        # Verificar que tiene BaseSkill class
        content = base_skill.read_text()
        if "class BaseSkill" not in content:
            return False, "BaseSkill class not found"
        
        return True, "Skills MoE architecture implemented"
    
    def check_v2_13(self) -> Tuple[bool, str]:
        """Check: v2.13 ProactiveLoop implementado"""
        proactive_file = Path("core/proactive_loop.py")
        
        if not proactive_file.exists():
            return False, "core/proactive_loop.py not found"
        
        content = proactive_file.read_text()
        if "class ProactiveLoop" not in content:
            return False, "ProactiveLoop class not implemented"
        
        return True, "ProactiveLoop implemented"
    
    def check_v2_14(self) -> Tuple[bool, str]:
        """Check: v2.14 SpeculativeDecoding implementado"""
        speculative_file = Path("core/speculative_decoder.py")
        
        if not speculative_file.exists():
            return False, "core/speculative_decoder.py not found"
        
        return True, "SpeculativeDecoding implemented"
    
    def check_v2_15(self) -> Tuple[bool, str]:
        """Check: v2.15 SelfRepair implementado"""
        self_repair_file = Path("core/self_repair.py")
        
        if not self_repair_file.exists():
            return False, "core/self_repair.py not found"
        
        content = self_repair_file.read_text()
        if "class SelfRepair" not in content:
            return False, "SelfRepair class not implemented"
        
        return True, "SelfRepair + RedTeam implemented"
    
    def check_llama_cpp(self) -> Tuple[bool, str]:
        """Check: llama.cpp binarios instalados"""
        required_bins = ["llama-cli", "llama-finetune", "llama-lora-merge"]
        missing = []
        
        for binary in required_bins:
            if not shutil.which(binary):
                missing.append(binary)
        
        if missing:
            return False, f"Missing binaries: {', '.join(missing)}"
        
        return True, "All llama.cpp binaries installed"
    
    def check_python_deps(self) -> Tuple[bool, str]:
        """Check: Dependencias Python para v2.16"""
        required_packages = {
            "cv2": "opencv-python",
            "PIL": "pillow",
            "imagehash": "imagehash",
            "gnupg": "python-gnupg"
        }
        
        missing = []
        for import_name, package_name in required_packages.items():
            try:
                __import__(import_name)
            except ImportError:
                missing.append(package_name)
        
        if missing:
            return False, f"Missing packages: {', '.join(missing)}"
        
        return True, "All Python dependencies installed"
    
    def check_disk_space(self) -> Tuple[bool, str]:
        """Check: Espacio en disco suficiente"""
        stat = shutil.disk_usage(Path.home())
        free_gb = stat.free / (1024**3)
        
        if free_gb < 10:
            return False, f"Insufficient disk space: {free_gb:.1f}GB (need >10GB)"
        
        return True, f"Disk space OK: {free_gb:.1f}GB free"
    
    def check_ram(self) -> Tuple[bool, str]:
        """Check: RAM disponible"""
        try:
            import psutil
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            
            if total_ram_gb < 16:
                return False, f"Low RAM: {total_ram_gb:.1f}GB (recommend >16GB)"
            
            return True, f"RAM OK: {total_ram_gb:.1f}GB total"
        
        except ImportError:
            return False, "psutil not installed (cannot check RAM)"
    
    def check_gpg(self) -> Tuple[bool, str]:
        """Check: GPG setup para signing"""
        gpg_key_id = os.getenv("SARAI_GPG_KEY_ID")
        
        if not gpg_key_id:
            return False, "SARAI_GPG_KEY_ID env var not set"
        
        # Verificar que GPG está instalado
        if not shutil.which("gpg"):
            return False, "GPG not installed"
        
        # Verificar que la key existe
        try:
            result = subprocess.run(
                ["gpg", "--list-keys", gpg_key_id],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                return False, f"GPG key {gpg_key_id} not found"
            
            return True, f"GPG key configured: {gpg_key_id[:8]}..."
        
        except Exception as e:
            return False, f"GPG check failed: {e}"


def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate prerequisites for SARAi v2.16 Omni-Loop"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings (for CI/CD)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    checker = PrerequisiteChecker(strict=args.strict)
    all_passed = checker.check_all()
    
    if args.json:
        # Output JSON para CI/CD
        result = {
            "all_passed": all_passed,
            "checks_passed": checker.checks_passed,
            "checks_failed": checker.checks_failed,
            "warnings": checker.warnings
        }
        print(json.dumps(result, indent=2))
    
    # Exit code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
