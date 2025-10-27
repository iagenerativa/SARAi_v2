#!/usr/bin/env python3
"""
Detecta flags de CPU y BLAS para compilar llama-cpp-python
Salida: cadena lista para CMAKE_ARGS

Uso:
    CMAKE_ARGS=$(python scripts/cpu_flags.py) pip install llama-cpp-python
"""
import platform
import sys


def detect_cpu_flags():
    """Detecta arquitectura y retorna flags de compilación para llama.cpp"""
    flags = []
    
    # 1. Arquitectura base
    arch = platform.machine().lower()
    
    if arch in ("x86_64", "amd64"):
        # Intel/AMD: AVX2 + F16C para cuantización rápida
        flags.extend([
            "-DLLAMA_AVX=ON",
            "-DLLAMA_AVX2=ON",
            "-DLLAMA_F16C=ON"
        ])
    
    elif arch in ("arm64", "aarch64"):
        # ARM (M1/M2/M3, Graviton): NEON automático
        flags.append("-DLLAMA_NATIVE=ON")
    
    else:
        # Otras arquitecturas: best-effort con optimizaciones nativas
        flags.append("-DLLAMA_NATIVE=ON")
    
    # 2. BLAS detection vía numpy
    try:
        import numpy as np
        from io import StringIO
        
        # Captura la salida de show_config
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        np.show_config()
        sys.stdout = old_stdout
        cfg = buffer.getvalue().lower()
        
        if "openblas" in cfg:
            flags.extend([
                "-DLLAMA_BLAS=ON",
                "-DLLAMA_BLAS_VENDOR=OpenBLAS"
            ])
        elif "mkl" in cfg:
            flags.extend([
                "-DLLAMA_BLAS=ON",
                "-DLLAMA_BLAS_VENDOR=Intel10_64lp"
            ])
    except Exception:
        # No BLAS disponible, usar solo CPU
        pass
    
    return " ".join(flags)


def main():
    """Punto de entrada: imprime flags sin newline para captura en Makefile"""
    print(detect_cpu_flags(), end="")


if __name__ == "__main__":
    main()
