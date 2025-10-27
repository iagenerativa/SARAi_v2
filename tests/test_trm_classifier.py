#!/usr/bin/env python3
"""
tests/test_trm_classifier.py - Test completo de TRM-Classifier v2.11

Copyright (c) 2025 Noel
Licencia: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
No se permite uso comercial sin permiso del autor.

---

Test exhaustivo de las 3 cabezas del TRM-Classifier entrenado:
- hard: Detección de queries técnicas
- soft: Detección de queries emocionales
- web_query: Detección de queries web (RAG)

Usa embeddings TF-IDF similares al entrenamiento para validación realista.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict

from core.trm_classifier import TRMClassifierDual


# ============================================================================
# TEST DATASET (60 queries balanceadas)
# ============================================================================

TEST_QUERIES = {
    "hard": [
        # Queries técnicas puras (hard > 0.7, soft < 0.3, web_query < 0.3)
        "¿Cómo configurar SSH en Ubuntu 22.04?",
        "Error al importar numpy en Python 3.11",
        "Diferencia entre async/await y Promises en JavaScript",
        "¿Cómo optimizar una query SQL con JOIN?",
        "Bug en mi código: segmentation fault en C++",
        "Instalar Docker en Raspberry Pi 4",
        "¿Qué es una función recursiva en Python?",
        "Configurar firewall con iptables",
        "¿Cómo usar git rebase interactivo?",
        "Diferencia entre class y interface en Java",
        "¿Cómo funciona el garbage collector en Python?",
        "Error 500 en mi API REST con Express",
        "¿Qué es un algoritmo de ordenamiento quicksort?",
        "Configurar NGINX como reverse proxy",
        "¿Cómo crear un virtual environment en Python?",
        "Diferencia entre TCP y UDP",
        "¿Qué es una variable de entorno en Linux?",
        "¿Cómo funciona el protocolo HTTPS?",
        "Error de compilación en GCC con flags -O3",
        "¿Qué es un índice en bases de datos?",
    ],
    
    "soft": [
        # Queries emocionales puras (soft > 0.7, hard < 0.3, web_query < 0.3)
        "Me siento frustrado con este bug, llevo horas",
        "Gracias por tu ayuda, eres muy claro explicando",
        "Estoy perdido, no entiendo nada de este código",
        "Explícame Python como si fuera un niño de 10 años",
        "Estoy cansado de intentar y fallar",
        "¿Puedes ser más paciente conmigo?",
        "No me juzgues, soy principiante en esto",
        "Agradezco mucho tu tiempo",
        "Me siento motivado a seguir aprendiendo",
        "Esto es muy difícil para mí",
        "¿Puedes explicármelo de forma más simple?",
        "Me preocupa no estar entendiendo bien",
        "Necesito que me guíes paso a paso",
        "Siento que no soy bueno para programar",
        "Me emociona aprender cosas nuevas",
        "¿Puedes ayudarme sin hacerme sentir tonto?",
        "Estoy confundido con tanta información",
        "Agradezco tu paciencia conmigo",
        "Me siento inseguro con mi código",
        "¿Puedes motivarme un poco?",
    ],
    
    "web_query": [
        # Queries web puras (web_query > 0.7, hard < 0.3, soft < 0.3)
        "¿Quién ganó el Oscar 2025?",
        "¿Cuándo fue la final de la Copa del Mundo 2026?",
        "¿Cómo está el clima en Tokio hoy?",
        "Precio actual de Bitcoin",
        "Últimas noticias de tecnología",
        "¿Quién es el presidente actual de Francia?",
        "Resultados del partido Real Madrid vs Barcelona hoy",
        "¿Qué pasó en las elecciones de Argentina 2024?",
        "Stock price de Tesla ahora",
        "¿Cuándo sale la nueva película de Marvel?",
        "Noticias de SpaceX esta semana",
        "¿Dónde está ubicada la Torre Eiffel?",
        "¿Cuál es el precio del dólar hoy?",
        "Últimos terremotos en Chile",
        "¿Quién ganó el Nobel de Física 2025?",
        "Clima en Nueva York ahora",
        "¿Cuándo es el próximo eclipse solar?",
        "Resultados de la fórmula 1 hoy",
        "¿Qué edad tiene Elon Musk?",
        "Noticias de inteligencia artificial esta semana",
    ]
}


# ============================================================================
# EMBEDDING GENERATOR (TF-IDF como en entrenamiento)
# ============================================================================

class EmbeddingGenerator:
    """Genera embeddings TF-IDF de 2048-D para testing"""
    
    def __init__(self):
        self.vectorizer = None
        self.fitted = False
    
    def fit(self, texts: List[str]):
        """Ajusta TF-IDF en corpus de test"""
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()
        self.fitted = True
        
        print(f"   TF-IDF fitted: {tfidf_matrix.shape[0]} docs, {tfidf_matrix.shape[1]} features")
    
    def transform(self, text: str) -> torch.Tensor:
        """Transforma un texto a embedding 2048-D"""
        if not self.fitted:
            raise ValueError("EmbeddingGenerator no ha sido fitted")
        
        # TF-IDF
        tfidf_vec = self.vectorizer.transform([text]).toarray()[0]
        
        # Padding a 2048-D
        if len(tfidf_vec) < 2048:
            padding = np.zeros(2048 - len(tfidf_vec))
            embedding = np.hstack([tfidf_vec, padding])
        else:
            # Truncar (si hay más de 2048)
            embedding = tfidf_vec[:2048]
        
        # Normalizar
        norm = np.linalg.norm(embedding)
        embedding = embedding / (norm + 1e-8)
        
        return torch.tensor(embedding, dtype=torch.float32)


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_category(
    classifier: TRMClassifierDual,
    embedder: EmbeddingGenerator,
    category: str,
    queries: List[str],
    expected_primary: str,
    threshold: float = 0.7
) -> Dict:
    """Test una categoría completa"""
    
    print(f"\n🔍 Categoría: {category.upper()}")
    print(f"   Objetivo: {expected_primary} > {threshold}")
    print("-" * 80)
    
    passed = 0
    total = len(queries)
    
    scores_list = []
    
    for query in queries:
        # Generar embedding
        embedding = embedder.transform(query)
        
        # Clasificar
        with torch.no_grad():
            scores = classifier.invoke(embedding)
        
        scores_list.append(scores)
        
        # Validar
        primary_score = scores[expected_primary]
        is_correct = primary_score > threshold
        
        if is_correct:
            passed += 1
            status = "✅"
        else:
            status = "❌"
        
        # Mostrar resultado
        print(f"{status} {query[:65]:<65}")
        print(f"   hard={scores['hard']:.3f}, soft={scores['soft']:.3f}, web_query={scores['web_query']:.3f}")
    
    # Estadísticas
    accuracy = (passed / total * 100) if total > 0 else 0
    
    # Promedios de scores
    avg_hard = np.mean([s['hard'] for s in scores_list])
    avg_soft = np.mean([s['soft'] for s in scores_list])
    avg_web = np.mean([s['web_query'] for s in scores_list])
    
    print(f"\n📊 Resumen {category}:")
    print(f"   Accuracy: {passed}/{total} ({accuracy:.1f}%)")
    print(f"   Avg scores: hard={avg_hard:.3f}, soft={avg_soft:.3f}, web_query={avg_web:.3f}")
    
    return {
        "category": category,
        "passed": passed,
        "total": total,
        "accuracy": accuracy,
        "avg_hard": avg_hard,
        "avg_soft": avg_soft,
        "avg_web_query": avg_web,
        "expected_primary": expected_primary
    }


def main():
    """Test completo de TRM-Classifier v2.11"""
    
    print("=" * 80)
    print("🧪 TEST EXHAUSTIVO TRM-CLASSIFIER v2.11 - CHECKPOINT ENTRENADO")
    print("=" * 80)
    print("Objetivo: Validar las 3 cabezas con el modelo entrenado real")
    print("Método: Embeddings TF-IDF del checkpoint (consistencia garantizada)")
    print("=" * 80)
    
    # Cargar modelo entrenado Y vectorizador
    print("\n📥 Cargando TRM-Classifier desde checkpoint...")
    try:
        checkpoint_path = "./models/trm_classifier/checkpoint.pth"
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
        
        # Verificar que tiene vectorizador (v2.11)
        if 'vectorizer' not in checkpoint:
            print("⚠️  Checkpoint antiguo sin vectorizador")
            print("   Reentrenando con: python3 scripts/train_trm_v2.py --samples 5000")
            return 1
        
        # Cargar modelo
        classifier = TRMClassifierDual()
        classifier.load_state_dict(checkpoint['state_dict'])
        classifier.eval()
        
        print("✅ Checkpoint cargado exitosamente")
        print(f"   Época: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return 1
    
    # Preparar embeddings CON EL MISMO VECTORIZADOR del checkpoint
    print("\n🔧 Preparando generador de embeddings...")
    embedder = EmbeddingGenerator()
    
    # CRÍTICO: Usar vectorizador del checkpoint (no crear uno nuevo)
    embedder.vectorizer = checkpoint['vectorizer']
    embedder.fitted = True
    
    print("✅ Embeddings preparados con vectorizador del checkpoint")
    
    # Ejecutar tests por categoría
    print("\n" + "=" * 80)
    print("📊 EJECUTANDO TESTS POR CATEGORÍA")
    print("=" * 80)
    
    results = []
    
    # Test 1: Hard
    result_hard = test_category(
        classifier, embedder, "hard", TEST_QUERIES["hard"],
        expected_primary="hard", threshold=0.7
    )
    results.append(result_hard)
    
    # Test 2: Soft
    result_soft = test_category(
        classifier, embedder, "soft", TEST_QUERIES["soft"],
        expected_primary="soft", threshold=0.7
    )
    results.append(result_soft)
    
    # Test 3: Web Query
    result_web = test_category(
        classifier, embedder, "web_query", TEST_QUERIES["web_query"],
        expected_primary="web_query", threshold=0.7
    )
    results.append(result_web)
    
    # Resumen final
    print("\n" + "=" * 80)
    print("📈 RESUMEN FINAL - PREPARACIÓN PARA M2.5")
    print("=" * 80)
    
    total_passed = sum(r["passed"] for r in results)
    total_queries = sum(r["total"] for r in results)
    overall_accuracy = (total_passed / total_queries * 100) if total_queries > 0 else 0
    
    print(f"\n{'Cabeza':<15} {'Accuracy':<12} {'Promedio Score':<20} {'Estado':<10}")
    print("-" * 80)
    
    for result in results:
        cat = result["category"]
        acc = result["accuracy"]
        expected = result["expected_primary"]
        
        if expected == "hard":
            avg_score = result["avg_hard"]
        elif expected == "soft":
            avg_score = result["avg_soft"]
        else:
            avg_score = result["avg_web_query"]
        
        status = "✅ PASS" if acc >= 80 else "⚠️ REVIEW" if acc >= 60 else "❌ FAIL"
        
        print(f"{cat:<15} {acc:>5.1f}% ({result['passed']}/{result['total']}){avg_score:>10.3f}          {status:<10}")
    
    print("-" * 80)
    print(f"{'TOTAL':<15} {overall_accuracy:>5.1f}% ({total_passed}/{total_queries})")
    print("=" * 80)
    
    # Criterios de aprobación
    print("\n🎯 CRITERIOS DE APROBACIÓN:")
    print("-" * 80)
    
    hard_pass = result_hard["accuracy"] >= 80
    soft_pass = result_soft["accuracy"] >= 80
    web_pass = result_web["accuracy"] >= 80
    overall_pass = overall_accuracy >= 80
    
    print(f"{'Hard detection (≥80%)':<40} {'✅ PASS' if hard_pass else '❌ FAIL'}")
    print(f"{'Soft detection (≥80%)':<40} {'✅ PASS' if soft_pass else '❌ FAIL'}")
    print(f"{'Web_query detection (≥80%)':<40} {'✅ PASS' if web_pass else '❌ FAIL'}")
    print(f"{'Overall accuracy (≥80%)':<40} {'✅ PASS' if overall_pass else '❌ FAIL'}")
    
    # Conclusión
    print("\n" + "=" * 80)
    
    if hard_pass and soft_pass and web_pass:
        print("✅ SISTEMA LISTO PARA M2.5 (Integrar RAG en Graph)")
        print("=" * 80)
        print("\nLas 3 cabezas del TRM-Classifier funcionan correctamente:")
        print(f"  • Hard: {result_hard['accuracy']:.1f}% accuracy (objetivo: ≥80%)")
        print(f"  • Soft: {result_soft['accuracy']:.1f}% accuracy (objetivo: ≥80%)")
        print(f"  • Web_query: {result_web['accuracy']:.1f}% accuracy (objetivo: ≥80%)")
        print("\n🚀 Próximo paso: Implementar routing RAG en core/graph.py")
        return 0
    else:
        print("⚠️ SISTEMA REQUIERE AJUSTES ANTES DE M2.5")
        print("=" * 80)
        print("\nProblemas detectados:")
        
        if not hard_pass:
            print(f"  ❌ Hard: {result_hard['accuracy']:.1f}% < 80%")
            print("     → Reentrenar con más ejemplos técnicos")
        
        if not soft_pass:
            print(f"  ❌ Soft: {result_soft['accuracy']:.1f}% < 80%")
            print("     → Reentrenar con más ejemplos emocionales")
        
        if not web_pass:
            print(f"  ❌ Web_query: {result_web['accuracy']:.1f}% < 80%")
            print("     → Reentrenar con más ejemplos web")
        
        print("\n💡 Recomendación: Ejecutar python3 scripts/train_trm_v2.py con dataset ampliado")
        return 1


if __name__ == "__main__":
    exit(main())
