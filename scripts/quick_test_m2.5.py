#!/usr/bin/env python3
"""
Quick Test M2.5 - Verifica si todo está listo para consolidación
Sin levantar servicios Docker, solo verifica componentes locales.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def check_hf_auth():
    """Verifica autenticación HuggingFace"""
    print("📋 1. Verificar HuggingFace Auth")
    print("-" * 40)
    
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"✅ Autenticado como: {user['name']}")
        return True
    except Exception as e:
        print(f"❌ No autenticado: {e}")
        print("\nPara autenticarte:")
        print("  1. Ejecuta: huggingface-cli login")
        print("  2. Token: https://huggingface.co/settings/tokens")
        print("  3. Acepta: https://huggingface.co/google/embeddinggemma-300m-qat-q4_0-unquantized")
        return False

def check_trm_checkpoint():
    """Verifica que TRM esté entrenado"""
    print("\n📋 2. Verificar TRM Checkpoint")
    print("-" * 40)
    
    checkpoint_path = "models/trm_classifier/checkpoint.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        print("\nPara entrenar el TRM:")
        print("  python3 scripts/train_trm_v2.py")
        return False
    
    print(f"✅ Checkpoint encontrado: {checkpoint_path}")
    
    # Verificar que se puede cargar
    try:
        import torch
        checkpoint = torch.load(checkpoint_path)
        
        print(f"   Época: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Loss: {checkpoint.get('loss', 'N/A'):.4f}")
        
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.2%}")
        
        return True
    except Exception as e:
        print(f"❌ Error cargando checkpoint: {e}")
        return False

def check_embedding_model():
    """Verifica que EmbeddingGemma se puede cargar"""
    print("\n📋 3. Verificar EmbeddingGemma")
    print("-" * 40)
    
    try:
        from core.embeddings import get_embedding_model
        
        print("Cargando modelo (puede tardar ~30s la primera vez)...")
        embedder = get_embedding_model()
        
        # Test de encoding
        test_text = "Test encoding"
        embedding = embedder.encode(test_text)
        
        print(f"✅ EmbeddingGemma cargado")
        print(f"   Dimensión: {embedding.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error cargando EmbeddingGemma: {e}")
        
        if "gated repo" in str(e).lower():
            print("\n⚠️  El modelo requiere aceptar términos:")
            print("   https://huggingface.co/google/embeddinggemma-300m-qat-q4_0-unquantized")
        
        return False

def check_trm_with_embeddings():
    """Verifica TRM con embeddings reales"""
    print("\n📋 4. Verificar TRM + EmbeddingGemma")
    print("-" * 40)
    
    try:
        import torch
        from core.embeddings import get_embedding_model
        from core.trm_classifier import TRMClassifierDual
        
        # Cargar modelos
        embedder = get_embedding_model()
        
        trm = TRMClassifierDual()
        checkpoint = torch.load("models/trm_classifier/checkpoint.pth")
        trm.load_state_dict(checkpoint['model_state_dict'])
        trm.eval()
        
        # Test queries
        test_cases = [
            ("¿Quién ganó el Oscar 2025?", "web"),
            ("¿Cómo está el clima en Tokio?", "web"),
            ("¿Cómo configurar SSH?", "hard"),
            ("Me siento frustrado", "soft")
        ]
        
        print("\nTest de clasificación:")
        all_correct = True
        
        for query, expected_type in test_cases:
            emb = embedder.encode(query)
            emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                scores = trm.forward(emb_tensor)
            
            print(f"\n  '{query}'")
            print(f"    hard={scores['hard']:.3f}, soft={scores['soft']:.3f}, web={scores['web_query']:.3f}")
            
            # Validar
            if expected_type == "web":
                if scores['web_query'] > 0.7:
                    print(f"    ✅ Web query detectada correctamente")
                else:
                    print(f"    ❌ Debería ser web_query > 0.7")
                    all_correct = False
            
            elif expected_type == "hard":
                if scores['hard'] > scores['soft'] and scores['web_query'] < 0.7:
                    print(f"    ✅ Hard query correcta")
                else:
                    print(f"    ❌ Debería ser hard dominante")
                    all_correct = False
            
            elif expected_type == "soft":
                if scores['soft'] > scores['hard'] and scores['web_query'] < 0.7:
                    print(f"    ✅ Soft query correcta")
                else:
                    print(f"    ❌ Debería ser soft dominante")
                    all_correct = False
        
        if all_correct:
            print("\n✅ TRM + Embeddings: FUNCIONA")
            return True
        else:
            print("\n⚠️  Algunos tests fallaron")
            return False
            
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ejecuta todos los checks"""
    print("🎯 M2.5 QUICK TEST - Verificación Local")
    print("=" * 50)
    print()
    
    checks = [
        ("HuggingFace Auth", check_hf_auth),
        ("TRM Checkpoint", check_trm_checkpoint),
        ("EmbeddingGemma", check_embedding_model),
        ("TRM + Embeddings", check_trm_with_embeddings)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrumpido por el usuario")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error inesperado en {name}: {e}")
            results.append((name, False))
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN")
    print("=" * 50)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    all_passed = all(result for _, result in results)
    
    print()
    if all_passed:
        print("🎉 ¡TODOS LOS CHECKS PASARON!")
        print("\nSiguiente paso:")
        print("  bash scripts/consolidate_m2.5.sh")
        return 0
    else:
        print("⚠️  Algunos checks fallaron.")
        print("\nResuelve los problemas antes de continuar.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
