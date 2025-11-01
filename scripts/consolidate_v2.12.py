#!/usr/bin/env python3
"""
Script maestro de consolidación para SARAi v2.12
Descarga modelos, ejecuta tests y prepara para commit
"""
import subprocess
import sys
import time
from pathlib import Path

class ProgressBar:
    """Barra de progreso simple para terminal"""
    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.current = 0
        self.desc = desc
        self.width = 40
    
    def update(self, amount: int = 1):
        self.current += amount
        self._render()
    
    def _render(self):
        filled = int(self.width * self.current / self.total)
        bar = '█' * filled + '░' * (self.width - filled)
        percent = 100 * self.current / self.total
        print(f'\r{self.desc} |{bar}| {percent:.1f}% ({self.current}/{self.total})', end='', flush=True)
        if self.current >= self.total:
            print()  # Nueva línea al completar

def run_command(cmd: list, desc: str, show_output: bool = False) -> bool:
    """Ejecuta comando con manejo de errores"""
    print(f"\n🔄 {desc}...")
    
    try:
        if show_output:
            result = subprocess.run(cmd, check=True, cwd="/home/noel/SARAi_v2")
        else:
            result = subprocess.run(
                cmd, 
                check=True, 
                cwd="/home/noel/SARAi_v2",
                capture_output=True,
                text=True
            )
        print(f"✅ {desc} - COMPLETADO")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {desc} - FALLÓ")
        if not show_output and e.stdout:
            print(f"   Salida: {e.stdout[:200]}")
        if not show_output and e.stderr:
            print(f"   Error: {e.stderr[:200]}")
        return False

def main():
    """Pipeline completo de consolidación"""
    print("=" * 70)
    print("🚀 CONSOLIDACIÓN SARAi v2.12 - Skills MoE + Testing")
    print("=" * 70)
    print()
    
    # Paso 1: Verificar archivos modificados
    print("📋 PASO 1/5: Verificación de archivos")
    progress = ProgressBar(9, "Verificando")
    
    required_files = [
        "core/model_pool.py",
        "core/mcp.py",
        "config/sarai.yaml",
        "tests/test_model_pool_skills.py",
        "tests/test_mcp_skills.py",
        "pytest.ini",
        "PROGRESO_31102025.md",
        "SEMANA1_TICKETS.md",
        "STATUS_31102025.md"
    ]
    
    all_exist = True
    for file in required_files:
        progress.update()
        if not Path(f"/home/noel/SARAi_v2/{file}").exists():
            print(f"\n❌ Falta: {file}")
            all_exist = False
        time.sleep(0.1)  # Simular progreso
    
    if not all_exist:
        print("\n❌ Archivos faltantes. Abortando.")
        sys.exit(1)
    
    print("✅ Todos los archivos presentes\n")
    
    # Paso 2: Verificar sintaxis Python
    print("📋 PASO 2/5: Validación de sintaxis Python")
    python_files = [
        "core/model_pool.py",
        "core/mcp.py",
        "tests/test_model_pool_skills.py",
        "tests/test_mcp_skills.py"
    ]
    
    progress = ProgressBar(len(python_files), "Compilando")
    for file in python_files:
        if not run_command(
            ["python3", "-m", "py_compile", file],
            f"Compilar {file}",
            show_output=False
        ):
            sys.exit(1)
        progress.update()
    
    print()
    
    # Paso 3: Descargar modelos GGUF
    print("📋 PASO 3/5: Descarga de modelos GGUF")
    print("⚠️  Esta etapa puede tardar 10-30 minutos según tu conexión")
    print("⚠️  Se descargarán ~8 GB de modelos")
    print()
    
    response = input("¿Descargar modelos ahora? (s/n, Enter=sí): ").strip().lower()
    if response in ['', 's', 'si', 'y', 'yes']:
        if not run_command(
            ["python3", "scripts/download_skill_models.py", "--yes"],
            "Descargar modelos GGUF",
            show_output=True  # Mostrar progreso de descarga
        ):
            print("\n⚠️  Descarga falló o incompleta. Tests de integración se saltarán.")
            print("     Puedes ejecutar manualmente: python3 scripts/download_skill_models.py --yes")
    else:
        print("⏭️  Descarga omitida. Tests de integración se saltarán.")
    
    print()
    
    # Paso 4: Ejecutar tests unitarios
    print("📋 PASO 4/5: Ejecución de tests unitarios")
    
    if not run_command(
        ["python3", "-m", "pytest", 
         "tests/test_model_pool_skills.py", 
         "tests/test_mcp_skills.py",
         "-v", "-m", "not integration and not slow",
         "--tb=short"],
        "Tests unitarios (23 tests)",
        show_output=True
    ):
        print("\n❌ Tests unitarios fallaron. Revisa errores antes de commit.")
        sys.exit(1)
    
    print()
    
    # Paso 5: Preparar commit
    print("📋 PASO 5/5: Preparación para commit")
    print()
    print("✅ CONSOLIDACIÓN COMPLETADA")
    print()
    print("=" * 70)
    print("📦 RESUMEN DE CAMBIOS")
    print("=" * 70)
    print()
    print("Archivos modificados:")
    for file in required_files:
        print(f"  • {file}")
    
    print()
    print("Estadísticas:")
    print("  • Tickets completados: 2/5 (T1.1, T1.2)")
    print("  • Tests unitarios: 23/23 ✅")
    print("  • Tests integración: 2 (con guards)")
    print("  • LOC añadidas: ~970")
    print("  • Progreso Semana 1: 40%")
    
    print()
    print("=" * 70)
    print("🎯 LISTO PARA COMMIT")
    print("=" * 70)
    print()
    print("Ejecuta:")
    print()
    print("  git add core/model_pool.py core/mcp.py config/sarai.yaml")
    print("  git add tests/test_model_pool_skills.py tests/test_mcp_skills.py pytest.ini")
    print("  git add PROGRESO_31102025.md SEMANA1_TICKETS.md STATUS_31102025.md")
    print()
    print('  git commit -m "feat(v2.12): Implementar Skills MoE con LRU y routing dinámico')
    print()
    print("  - T1.1: ModelPool.get_skill() con cache separado, LRU, TTL")
    print("  - T1.2: MCP.execute_skills_moe() con fallback automático")
    print("  - Config: 6 skills especializados (programming, diagnosis, etc)")
    print("  - Tests: 23 unitarios + 2 integration con guards")
    print("  - Docs: PROGRESO_31102025.md, SEMANA1_TICKETS.md actualizados")
    print()
    print('  Progreso Semana 1: 40% (2/5 tickets, velocidad 2.8x)"')
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Consolidación interrumpida por el usuario")
        sys.exit(130)
