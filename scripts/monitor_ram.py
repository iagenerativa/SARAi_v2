#!/usr/bin/env python3
"""
Monitor de RAM en tiempo real para SARAi v2.2
Visualiza uso de memoria y modelos cargados en el ModelPool
"""

import os
import sys
import time
import psutil
from datetime import datetime


def get_ram_usage():
    """
    Obtiene uso de RAM del sistema
    
    Returns:
        Tuple (usado_gb, total_gb, porcentaje)
    """
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    total_gb = mem.total / (1024**3)
    percent = mem.percent
    return used_gb, total_gb, percent


def get_process_memory():
    """
    Obtiene memoria del proceso actual de Python
    
    Returns:
        Float con GB usados por el proceso
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024**3)


def print_bar(value, max_value, width=40, label=""):
    """
    Imprime barra de progreso en consola
    
    Args:
        value: Valor actual
        max_value: Valor máximo
        width: Ancho de la barra en caracteres
        label: Etiqueta descriptiva
    """
    filled = int(width * value / max_value)
    bar = "█" * filled + "░" * (width - filled)
    percent = (value / max_value) * 100
    
    # Colorear según nivel de uso
    if percent < 60:
        color = "\033[92m"  # Verde
    elif percent < 80:
        color = "\033[93m"  # Amarillo
    else:
        color = "\033[91m"  # Rojo
    
    reset = "\033[0m"
    
    print(f"{label:20s} {color}{bar}{reset} {value:.2f}/{max_value:.2f} GB ({percent:.1f}%)")


def monitor_loop(interval=2):
    """
    Loop de monitoreo continuo
    
    Args:
        interval: Segundos entre actualizaciones
    """
    try:
        while True:
            # Limpiar pantalla
            os.system('clear' if os.name != 'nt' else 'cls')
            
            # Header
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("="*70)
            print(f"{'SARAi v2.2 - Monitor de RAM':^70}")
            print(f"{timestamp:^70}")
            print("="*70)
            print()
            
            # Uso de RAM del sistema
            used_gb, total_gb, percent = get_ram_usage()
            print_bar(used_gb, total_gb, label="Sistema")
            
            # Uso del proceso Python
            process_gb = get_process_memory()
            print_bar(process_gb, total_gb, label="Proceso Python")
            
            # Advertencia si supera límite
            if used_gb > 12:
                print("\n⚠️  ADVERTENCIA: Uso de RAM supera 12GB (límite SARAi)")
                print("   Considera cerrar aplicaciones o reducir max_concurrent_llms")
            
            # Información adicional
            print("\n" + "-"*70)
            print(f"Disponible: {(total_gb - used_gb):.2f} GB")
            print(f"Swap usado: {psutil.swap_memory().percent:.1f}%")
            
            # Intentar obtener stats del ModelPool (si está corriendo)
            try:
                # Aquí podrías hacer una conexión IPC al proceso de SARAi
                # para obtener stats en tiempo real del ModelPool
                print("\n[ModelPool Stats: Conectar IPC para ver modelos cargados]")
            except:
                pass
            
            print("\n" + "-"*70)
            print(f"Actualizando cada {interval}s... (Ctrl+C para salir)")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitor detenido")
        sys.exit(0)


def main():
    """
    Punto de entrada del script
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor de RAM para SARAi v2.2")
    parser.add_argument(
        "--interval",
        type=int,
        default=2,
        help="Segundos entre actualizaciones (default: 2)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Iniciando monitor de RAM...")
    print("   (Presiona Ctrl+C para salir)\n")
    time.sleep(1)
    
    monitor_loop(args.interval)


if __name__ == "__main__":
    main()
