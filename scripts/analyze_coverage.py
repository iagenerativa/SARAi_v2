#!/usr/bin/env python3
"""
Coverage Analyzer para SARAi

FASE 5: Optimización - Análisis de Cobertura de Tests
Fecha: 2 Noviembre 2025

Uso:
    python scripts/analyze_coverage.py --run-tests --html
    python scripts/analyze_coverage.py --report --threshold 80
    python scripts/analyze_coverage.py --diff baseline.json
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class CoverageAnalyzer:
    """Analizador de cobertura de tests"""
    
    def __init__(self, output_dir: str = "reports/coverage"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.project_root = Path(__file__).parent.parent
    
    def run_tests_with_coverage(self, test_path: str = "tests/", html: bool = False):
        """
        Ejecutar tests con pytest-cov
        
        Args:
            test_path: Path de tests a ejecutar
            html: Generar reporte HTML
        """
        print("🧪 Ejecutando tests con coverage...")
        
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_path,
            "--cov=core",
            "--cov=agents",
            "--cov=sarai",
            "--cov-report=term",
            f"--cov-report=json:{self.output_dir}/coverage_{self.timestamp}.json",
        ]
        
        if html:
            html_dir = self.output_dir / f"html_{self.timestamp}"
            cmd.append(f"--cov-report=html:{html_dir}")
            print(f"📊 Reporte HTML: {html_dir}/index.html")
        
        # Ejecutar pytest
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode == 0:
            print("✅ Tests ejecutados con éxito")
        else:
            print(f"⚠️  Tests fallaron (exit code: {result.returncode})")
        
        return result.returncode
    
    def load_coverage_report(self, report_path: str) -> Dict:
        """Cargar reporte de coverage JSON"""
        with open(report_path) as f:
            return json.load(f)
    
    def analyze_coverage(self, report_path: str = None) -> Dict:
        """
        Analizar coverage report
        
        Returns:
            {
                "total_coverage": 85.4,
                "files": {...},
                "missing_lines": {...},
                "summary": {...}
            }
        """
        if report_path is None:
            # Buscar último reporte
            reports = sorted(self.output_dir.glob("coverage_*.json"))
            if not reports:
                raise FileNotFoundError("No coverage reports encontrados. Ejecuta --run-tests primero.")
            report_path = reports[-1]
        
        print(f"📖 Analizando coverage: {report_path}")
        
        data = self.load_coverage_report(report_path)
        
        total_coverage = data.get("totals", {}).get("percent_covered", 0.0)
        
        # Analizar por archivo
        files_coverage = {}
        missing_lines_summary = {}
        
        for filepath, file_data in data.get("files", {}).items():
            summary = file_data.get("summary", {})
            missing_lines = file_data.get("missing_lines", [])
            
            files_coverage[filepath] = {
                "percent_covered": summary.get("percent_covered", 0.0),
                "num_statements": summary.get("num_statements", 0),
                "missing_lines_count": len(missing_lines),
            }
            
            if missing_lines:
                missing_lines_summary[filepath] = missing_lines
        
        # Ordenar archivos por coverage (menor a mayor)
        sorted_files = sorted(
            files_coverage.items(),
            key=lambda x: x[1]["percent_covered"]
        )
        
        return {
            "total_coverage": total_coverage,
            "files": dict(sorted_files),
            "missing_lines": missing_lines_summary,
            "timestamp": self.timestamp,
            "report_path": str(report_path),
        }
    
    def generate_report(self, analysis: Dict, threshold: float = 80.0):
        """
        Generar reporte legible de coverage
        
        Args:
            analysis: Resultado de analyze_coverage()
            threshold: Umbral mínimo de coverage
        """
        total = analysis["total_coverage"]
        
        print("\n" + "=" * 80)
        print("📊 COVERAGE REPORT")
        print("=" * 80)
        print(f"Total Coverage: {total:.2f}%")
        print(f"Threshold:      {threshold:.2f}%")
        
        if total >= threshold:
            print(f"Status:         ✅ PASS")
        else:
            print(f"Status:         ❌ FAIL (bajo threshold)")
        
        print("=" * 80)
        print()
        
        # Top 10 archivos con MENOR coverage
        print("🔴 TOP 10 ARCHIVOS CON MENOR COVERAGE")
        print("-" * 80)
        
        files = analysis["files"]
        for i, (filepath, data) in enumerate(list(files.items())[:10], 1):
            percent = data["percent_covered"]
            missing = data["missing_lines_count"]
            
            status = "✅" if percent >= threshold else "❌"
            
            # Truncar path para legibilidad
            short_path = filepath.replace(str(self.project_root), "")
            if len(short_path) > 60:
                short_path = "..." + short_path[-57:]
            
            print(f"{status} {i:2d}. {short_path:60s} {percent:6.2f}% ({missing} líneas faltantes)")
        
        print()
        
        # Guardar reporte en archivo
        report_file = self.output_dir / f"coverage_report_{self.timestamp}.txt"
        with open(report_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("COVERAGE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Coverage: {total:.2f}%\n")
            f.write(f"Threshold: {threshold:.2f}%\n")
            f.write(f"Status: {'PASS' if total >= threshold else 'FAIL'}\n\n")
            
            f.write("FILES WITH LOW COVERAGE (<80%):\n")
            f.write("-" * 80 + "\n")
            
            for filepath, data in files.items():
                if data["percent_covered"] < 80.0:
                    f.write(f"{filepath}:\n")
                    f.write(f"  Coverage: {data['percent_covered']:.2f}%\n")
                    f.write(f"  Missing lines: {data['missing_lines_count']}\n")
                    
                    # Listar líneas faltantes
                    if filepath in analysis["missing_lines"]:
                        missing = analysis["missing_lines"][filepath]
                        f.write(f"  Lines: {missing[:20]}")  # Primeras 20
                        if len(missing) > 20:
                            f.write(f"... (+{len(missing) - 20} more)")
                        f.write("\n")
                    f.write("\n")
        
        print(f"📄 Reporte guardado: {report_file}")
        
        return total >= threshold
    
    def compare_coverage(self, baseline_path: str, current_path: str = None):
        """
        Comparar coverage con baseline
        
        Args:
            baseline_path: Path al baseline JSON
            current_path: Path al current JSON (o usar último)
        """
        if current_path is None:
            reports = sorted(self.output_dir.glob("coverage_*.json"))
            if not reports:
                raise FileNotFoundError("No current coverage encontrado")
            current_path = reports[-1]
        
        print(f"📊 Comparando coverage:")
        print(f"  Baseline: {baseline_path}")
        print(f"  Current:  {current_path}")
        print()
        
        baseline = self.load_coverage_report(baseline_path)
        current = self.load_coverage_report(current_path)
        
        baseline_total = baseline.get("totals", {}).get("percent_covered", 0.0)
        current_total = current.get("totals", {}).get("percent_covered", 0.0)
        
        delta = current_total - baseline_total
        
        print("=" * 80)
        print("📈 COVERAGE COMPARISON")
        print("=" * 80)
        print(f"Baseline: {baseline_total:.2f}%")
        print(f"Current:  {current_total:.2f}%")
        print(f"Delta:    {delta:+.2f}%")
        
        if delta >= 0:
            print(f"Status:   ✅ IMPROVED")
        else:
            print(f"Status:   ⚠️  REGRESSED")
        
        print("=" * 80)
        
        return delta


def main():
    parser = argparse.ArgumentParser(
        description="Coverage Analyzer para SARAi"
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Ejecutar tests con coverage"
    )
    parser.add_argument(
        "--test-path",
        default="tests/",
        help="Path de tests a ejecutar"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generar reporte HTML"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generar reporte de coverage"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="Threshold mínimo de coverage (%)"
    )
    parser.add_argument(
        "--diff",
        metavar="BASELINE",
        help="Comparar con baseline JSON"
    )
    parser.add_argument(
        "--output",
        default="reports/coverage",
        help="Directorio de output"
    )
    
    args = parser.parse_args()
    
    analyzer = CoverageAnalyzer(output_dir=args.output)
    
    print("=" * 80)
    print("📊 SARAi Coverage Analyzer")
    print("=" * 80)
    print()
    
    exit_code = 0
    
    # 1. Ejecutar tests
    if args.run_tests:
        exit_code = analyzer.run_tests_with_coverage(
            test_path=args.test_path,
            html=args.html
        )
        print()
    
    # 2. Generar reporte
    if args.report or args.run_tests:
        analysis = analyzer.analyze_coverage()
        passed = analyzer.generate_report(analysis, threshold=args.threshold)
        
        if not passed:
            exit_code = 1
    
    # 3. Comparar con baseline
    if args.diff:
        delta = analyzer.compare_coverage(baseline_path=args.diff)
        
        if delta < 0:
            print("\n⚠️  WARNING: Coverage ha disminuido vs baseline")
            exit_code = 1
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
