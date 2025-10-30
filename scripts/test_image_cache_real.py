"""
Test de Cache LRU+TTL con Dataset Real de Imágenes
v2.16 Risk #6: Validación con imágenes reales + análisis multimodal

Uso:
    python scripts/test_image_cache_real.py --image-dir /path/to/images
    python scripts/test_image_cache_real.py --image-dir /path/to/images --analyze
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.image_preprocessor import ImagePreprocessor


def scan_image_directory(image_dir: Path) -> List[Path]:
    """
    Escanea directorio y retorna lista de imágenes
    
    Args:
        image_dir: Directorio con imágenes
    
    Returns:
        Lista de paths a archivos de imagen
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
    
    images = []
    for ext in valid_extensions:
        images.extend(image_dir.glob(f"*{ext}"))
        images.extend(image_dir.glob(f"*{ext.upper()}"))
    
    return sorted(images)


def test_cache_with_real_images(image_paths: List[Path], analyze: bool = False):
    """
    Test de cache con imágenes reales
    
    Args:
        image_paths: Lista de paths a imágenes
        analyze: Si True, intenta analizar contenido con Omni (futuro)
    """
    print("=" * 70)
    print("🖼️  TEST DE CACHE CON IMÁGENES REALES")
    print("=" * 70)
    
    # Inicializar preprocessor
    preprocessor = ImagePreprocessor(
        cache_dir="state/image_cache_test",
        metadata_file="state/image_cache_metadata_test.json",
        ttl_days=7,
        max_cache_mb=200
    )
    
    print(f"\n📁 Imágenes encontradas: {len(image_paths)}")
    print("-" * 70)
    
    # Fase 1: Cargar todas las imágenes (primera pasada)
    print("\n🔄 FASE 1: Primera carga (todas MISS esperadas)")
    print("-" * 70)
    
    load_times = []
    
    for i, img_path in enumerate(image_paths, 1):
        # Leer bytes de la imagen
        img_bytes = img_path.read_bytes()
        size_mb = len(img_bytes) / (1024 * 1024)
        
        print(f"\n[{i}/{len(image_paths)}] {img_path.name}")
        print(f"  Tamaño: {size_mb:.2f} MB")
        
        # Procesar y cachear
        start = time.time()
        cached_path = preprocessor.preprocess(img_bytes, image_id=img_path.stem)
        elapsed = time.time() - start
        
        load_times.append(elapsed)
        print(f"  Tiempo: {elapsed*1000:.1f} ms")
        
        # Análisis opcional (requiere Omni-7B cargado)
        if analyze and i == 1:  # Solo primera imagen por ahora
            print(f"  🔍 Análisis multimodal: PENDIENTE (Omni-7B no integrado aún)")
            # TODO: Cuando agents/omni_native.py esté listo:
            # from agents.omni_native import analyze_image
            # description = analyze_image(cached_path)
            # print(f"  📝 Descripción: {description}")
    
    # Estadísticas de primera carga
    print("\n" + "=" * 70)
    print("📊 ESTADÍSTICAS PRIMERA CARGA")
    print("=" * 70)
    
    stats = preprocessor.get_stats()
    print(f"Cache size: {stats['cache_size_mb']:.2f} MB / {stats['max_cache_mb']} MB ({stats['usage_percent']:.1f}%)")
    print(f"Total entries: {stats['total_entries']}")
    print(f"Tiempo promedio: {sum(load_times)/len(load_times)*1000:.1f} ms")
    print(f"Tiempo total: {sum(load_times):.2f} s")
    
    # Fase 2: Re-acceso (todas HIT esperadas)
    print("\n" + "=" * 70)
    print("🔄 FASE 2: Re-acceso (todas HIT esperadas)")
    print("=" * 70)
    
    hit_times = []
    cache_hits = 0
    
    for i, img_path in enumerate(image_paths, 1):
        img_bytes = img_path.read_bytes()
        
        print(f"\n[{i}/{len(image_paths)}] {img_path.name}")
        
        start = time.time()
        cached_path = preprocessor.preprocess(img_bytes, image_id=img_path.stem)
        elapsed = time.time() - start
        
        hit_times.append(elapsed)
        
        # Verificar si fue HIT (tiempo debería ser ~0)
        is_hit = elapsed < 0.01  # <10ms = HIT
        if is_hit:
            cache_hits += 1
        
        status = "✅ HIT" if is_hit else "❌ MISS"
        print(f"  Tiempo: {elapsed*1000:.1f} ms - {status}")
    
    # Estadísticas de cache hits
    print("\n" + "=" * 70)
    print("📊 ESTADÍSTICAS CACHE HITS")
    print("=" * 70)
    
    hit_rate = (cache_hits / len(image_paths)) * 100
    print(f"Hit rate: {hit_rate:.1f}% ({cache_hits}/{len(image_paths)})")
    print(f"Tiempo promedio HIT: {sum(hit_times)/len(hit_times)*1000:.1f} ms")
    
    speedup = (sum(load_times) / sum(hit_times)) if sum(hit_times) > 0 else 0
    print(f"Speedup: {speedup:.1f}x más rápido")
    
    # Fase 3: Cleanup test (simular TTL)
    print("\n" + "=" * 70)
    print("🧹 FASE 3: Test de Cleanup LRU+TTL")
    print("=" * 70)
    
    print("\nSimulando entradas expiradas (modificando timestamps)...")
    
    # Hacer que la mitad de las imágenes sean "viejas"
    now = time.time()
    ttl_seconds = 7 * 86400
    old_count = 0
    
    for i, img_hash in enumerate(list(preprocessor.lru_cache.keys())):
        if i % 2 == 0:  # Cada segunda imagen
            old_time = now - ttl_seconds - 3600  # 7 días + 1 hora
            file_path, _, size = preprocessor.lru_cache[img_hash]
            preprocessor.lru_cache[img_hash] = (file_path, old_time, size)
            preprocessor.metadata["entries"][img_hash]["last_access"] = old_time
            old_count += 1
    
    print(f"Marcadas como expiradas: {old_count} imágenes")
    
    # Ejecutar cleanup
    print("\nEjecutando cleanup híbrido...")
    freed_mb = preprocessor.cleanup_lru_ttl_hybrid()
    
    stats_after = preprocessor.get_stats()
    print(f"\nResultados:")
    print(f"  Espacio liberado: {freed_mb:.2f} MB")
    print(f"  Entradas restantes: {stats_after['total_entries']}")
    print(f"  Cache size: {stats_after['cache_size_mb']:.2f} MB")
    
    # Validación final
    print("\n" + "=" * 70)
    print("✅ VALIDACIÓN FINAL")
    print("=" * 70)
    
    validations = []
    
    # 1. Hit rate debe ser 100% en segunda pasada
    val1 = hit_rate == 100.0
    validations.append(("Hit rate 100%", val1))
    
    # 2. Cleanup debe haber liberado espacio
    val2 = freed_mb > 0
    validations.append(("Cleanup liberó espacio", val2))
    
    # 3. Cache debe estar bajo límite
    val3 = stats_after['cache_size_mb'] <= 200
    validations.append(("Cache bajo límite", val3))
    
    # 4. Speedup debe ser significativo (>10x)
    val4 = speedup > 10
    validations.append(("Speedup >10x", val4))
    
    for check, passed in validations:
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
    
    all_passed = all(v[1] for v in validations)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 TODOS LOS TESTS PASARON - Risk #6 VALIDADO CON DATOS REALES")
        return 0
    else:
        print("⚠️  ALGUNOS TESTS FALLARON")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Test de cache de imágenes con dataset real"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=".",
        help="Directorio con imágenes de test"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Intentar análisis multimodal con Omni-7B (requiere implementación)"
    )
    
    args = parser.parse_args()
    
    image_dir = Path(args.image_dir)
    
    if not image_dir.exists():
        print(f"❌ Directorio no existe: {image_dir}")
        return 1
    
    # Escanear imágenes
    image_paths = scan_image_directory(image_dir)
    
    if not image_paths:
        print(f"❌ No se encontraron imágenes en: {image_dir}")
        print("   Formatos soportados: .jpg, .jpeg, .png, .bmp, .webp, .gif")
        return 1
    
    # Ejecutar test
    return test_cache_with_real_images(image_paths, analyze=args.analyze)


if __name__ == "__main__":
    sys.exit(main())
