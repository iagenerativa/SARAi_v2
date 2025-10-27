"""
SARAi v2 - Sistema de AGI Local Híbrido
Punto de entrada principal
"""

import sys
import argparse
from pathlib import Path

from core.graph import create_orchestrator


def main():
    """Bucle interactivo principal"""
    parser = argparse.ArgumentParser(description="SARAi v2 - AGI Local")
    parser.add_argument("--use-real-trm", action="store_true",
                       help="Usar TRM real (requiere entrenamiento previo)")
    parser.add_argument("--stats", action="store_true",
                       help="Mostrar estadísticas de rendimiento")
    parser.add_argument("--days", type=int, default=7,
                       help="Días para estadísticas (default: 7)")
    
    args = parser.parse_args()
    
    # Mostrar estadísticas si se solicita
    if args.stats:
        from core.feedback import get_feedback_detector
        detector = get_feedback_detector()
        stats = detector.compute_statistics(args.days)
        
        print("\n📊 Estadísticas de SARAi")
        print("=" * 50)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        print("=" * 50)
        return
    
    # Inicializar orquestador
    print("\n" + "="*60)
    print("🧠 SARAi v2 - Sistema de AGI Local Híbrido")
    print("="*60)
    print("Hardware: CPU-only, 16GB RAM")
    print("Modelos: SOLAR-10.7B (expert) + LFM2-1.2B (tiny)")
    print("Embeddings: EmbeddingGemma-300M")
    print("="*60 + "\n")
    
    use_simulated = not args.use_real_trm
    if use_simulated:
        print("⚠️  Modo: TRM-Classifier simulado (basado en keywords)")
        print("💡 Entrena el TRM real para mejor clasificación\n")
    
    orchestrator = create_orchestrator(use_simulated_trm=use_simulated)
    
    print("\n💬 SARAi listo. Escribe 'salir' para terminar.")
    print("📝 Comandos especiales:")
    print("  - 'stats': Ver estadísticas de rendimiento")
    print("  - 'clear': Limpiar pantalla")
    print("  - 'help': Mostrar ayuda")
    print("\n" + "-"*60 + "\n")
    
    # Bucle interactivo
    while True:
        try:
            user_input = input("Tú: ").strip()
            
            if not user_input:
                continue
            
            # Comandos especiales
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("\n👋 Hasta pronto. SARAi apagándose...")
                break
            
            elif user_input.lower() == "stats":
                stats = orchestrator.get_statistics(days=7)
                print("\n📊 Estadísticas (últimos 7 días):")
                for key, value in stats.items():
                    if isinstance(value, dict):
                        print(f"\n{key}:")
                        for k, v in value.items():
                            print(f"  {k}: {v}")
                    else:
                        print(f"{key}: {value}")
                print()
                continue
            
            elif user_input.lower() == "clear":
                import os
                os.system('clear' if sys.platform != 'win32' else 'cls')
                continue
            
            elif user_input.lower() == "help":
                print("\n📖 Ayuda de SARAi:")
                print("  - Haz preguntas técnicas o emocionales")
                print("  - El sistema detecta automáticamente la intención")
                print("  - Se adapta según tu feedback implícito")
                print("  - Usa 'stats' para ver rendimiento")
                print()
                continue
            
            # Procesar input
            print("\n🤔 Procesando...\n")
            response = orchestrator.invoke(user_input)
            
            print(f"SARAi: {response}\n")
            print("-"*60 + "\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Hasta pronto. SARAi apagándose...")
            break
        
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            print("Intenta de nuevo o escribe 'salir' para terminar.\n")


if __name__ == "__main__":
    main()
