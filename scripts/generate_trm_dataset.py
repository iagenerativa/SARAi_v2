#!/usr/bin/env python3
"""
scripts/generate_trm_dataset.py - Generador de Dataset Sintético para TRM-Classifier

Copyright (c) 2025 Noel
Licencia: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
No se permite uso comercial sin permiso del autor.

---

Genera ejemplos sintéticos de queries con labels (hard, soft, web_query)
para entrenar el TRM-Classifier v2.11 con la nueva cabeza web_query.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple


# ============================================================================
# TEMPLATES DE QUERIES POR CATEGORÍA
# ============================================================================

HARD_QUERIES = [
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
]

SOFT_QUERIES = [
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
]

WEB_QUERIES = [
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

# Templates para generar variaciones
HARD_TEMPLATES = [
    "¿Cómo {acción} {tecnología} en {contexto}?",
    "Error al {acción} {tecnología}",
    "Diferencia entre {concepto1} y {concepto2}",
    "Bug en {contexto}: {error}",
    "¿Qué es {concepto} en {tecnología}?",
    "Configurar {tecnología} con {herramienta}",
]

SOFT_TEMPLATES = [
    "Me siento {emoción} con {contexto}",
    "{agradecimiento} por tu {cualidad}",
    "Estoy {estado_emocional}, {razón}",
    "¿Puedes {solicitud_emocional}?",
    "{emoción} mucho tu {cualidad}",
]

WEB_TEMPLATES = [
    "¿Quién {verbo} {evento} {año}?",
    "¿Cuándo {verbo_temporal} {evento}?",
    "¿Cómo está {estado} en {lugar} {tiempo}?",
    "Precio {adjetivo} de {commodity}",
    "Últimas noticias de {tema}",
    "¿Dónde está {lugar}?",
    "Resultados del {evento} {tiempo}",
]

# Vocabulario para templates
VOCABULARIO = {
    "acción": ["instalar", "configurar", "optimizar", "depurar", "compilar"],
    "tecnología": ["Docker", "Kubernetes", "PostgreSQL", "Redis", "MongoDB"],
    "contexto": ["Ubuntu 22.04", "Windows 11", "macOS", "Raspberry Pi"],
    "concepto1": ["mutex", "semaphore", "array", "lista", "tupla"],
    "concepto2": ["lock", "monitor", "vector", "conjunto", "diccionario"],
    "error": ["segmentation fault", "memory leak", "null pointer", "timeout"],
    "herramienta": ["systemd", "supervisor", "pm2", "gunicorn"],
    "emoción": ["frustrado", "perdido", "confundido", "motivado", "emocionado"],
    "agradecimiento": ["Gracias", "Agradezco", "Aprecio"],
    "cualidad": ["paciencia", "ayuda", "claridad", "tiempo", "dedicación"],
    "estado_emocional": ["cansado", "inseguro", "motivado", "perdido"],
    "razón": ["llevo horas", "no entiendo", "es muy difícil", "necesito ayuda"],
    "solicitud_emocional": ["ser más paciente", "explicarlo más simple", "guiarme paso a paso"],
    "verbo": ["ganó", "fundó", "descubrió", "inventó", "creó"],
    "evento": ["el Oscar", "la Copa del Mundo", "el Nobel", "las elecciones"],
    "año": ["2024", "2025", "2026"],
    "verbo_temporal": ["fue", "es", "será"],
    "estado": ["el clima", "el tiempo", "la temperatura"],
    "lugar": ["Tokio", "París", "Nueva York", "Londres", "Sydney"],
    "tiempo": ["hoy", "ahora", "esta semana", "ayer"],
    "adjetivo": ["actual", "de hoy", "en tiempo real"],
    "commodity": ["Bitcoin", "Ethereum", "oro", "petróleo", "dólar"],
    "tema": ["tecnología", "deportes", "política", "ciencia", "economía"],
}


# ============================================================================
# GENERACIÓN DE EJEMPLOS
# ============================================================================

def generar_variacion(template: str, vocab: Dict) -> str:
    """Genera una variación de un template usando vocabulario"""
    resultado = template
    for placeholder, opciones in vocab.items():
        if f"{{{placeholder}}}" in resultado:
            resultado = resultado.replace(f"{{{placeholder}}}", random.choice(opciones))
    return resultado


def generar_ejemplos_hard(n: int) -> List[Tuple[str, Dict[str, float]]]:
    """Genera n ejemplos de queries técnicas (hard)"""
    ejemplos = []
    
    # Añadir ejemplos base
    for query in HARD_QUERIES[:min(n // 2, len(HARD_QUERIES))]:
        ejemplos.append((query, {"hard": 0.9, "soft": 0.1, "web_query": 0.05}))
    
    # Generar variaciones con templates
    for _ in range(n - len(ejemplos)):
        template = random.choice(HARD_TEMPLATES)
        query = generar_variacion(template, VOCABULARIO)
        ejemplos.append((query, {"hard": 0.85, "soft": 0.15, "web_query": 0.05}))
    
    return ejemplos


def generar_ejemplos_soft(n: int) -> List[Tuple[str, Dict[str, float]]]:
    """Genera n ejemplos de queries emocionales (soft)"""
    ejemplos = []
    
    # Añadir ejemplos base
    for query in SOFT_QUERIES[:min(n // 2, len(SOFT_QUERIES))]:
        ejemplos.append((query, {"hard": 0.1, "soft": 0.9, "web_query": 0.05}))
    
    # Generar variaciones con templates
    for _ in range(n - len(ejemplos)):
        template = random.choice(SOFT_TEMPLATES)
        query = generar_variacion(template, VOCABULARIO)
        ejemplos.append((query, {"hard": 0.15, "soft": 0.85, "web_query": 0.05}))
    
    return ejemplos


def generar_ejemplos_web(n: int) -> List[Tuple[str, Dict[str, float]]]:
    """Genera n ejemplos de queries de búsqueda web (web_query)"""
    ejemplos = []
    
    # Añadir ejemplos base
    for query in WEB_QUERIES[:min(n // 2, len(WEB_QUERIES))]:
        ejemplos.append((query, {"hard": 0.05, "soft": 0.05, "web_query": 0.95}))
    
    # Generar variaciones con templates
    for _ in range(n - len(ejemplos)):
        template = random.choice(WEB_TEMPLATES)
        query = generar_variacion(template, VOCABULARIO)
        ejemplos.append((query, {"hard": 0.1, "soft": 0.1, "web_query": 0.9}))
    
    return ejemplos


def generar_ejemplos_hibridos(n: int) -> List[Tuple[str, Dict[str, float]]]:
    """Genera n ejemplos híbridos (combinaciones)"""
    ejemplos = []
    
    hibridos = [
        ("¿Cómo está el precio de Bitcoin ahora? Necesito ayuda urgente", 
         {"hard": 0.3, "soft": 0.4, "web_query": 0.7}),
        ("Explícame de forma simple quién ganó el Nobel 2025",
         {"hard": 0.2, "soft": 0.6, "web_query": 0.8}),
        ("Estoy frustrado, no entiendo el error de mi código con Docker",
         {"hard": 0.7, "soft": 0.6, "web_query": 0.1}),
        ("¿Puedes ayudarme a entender las últimas noticias de IA?",
         {"hard": 0.2, "soft": 0.5, "web_query": 0.7}),
        ("Me siento perdido configurando SSH, ¿hay alguna guía reciente?",
         {"hard": 0.6, "soft": 0.5, "web_query": 0.4}),
    ]
    
    for _ in range(n):
        ejemplos.append(random.choice(hibridos))
    
    return ejemplos


def aplicar_data_augmentation(ejemplos: List[Tuple[str, Dict[str, float]]], factor: int = 2) -> List[Tuple[str, Dict[str, float]]]:
    """
    Aplica data augmentation a los ejemplos existentes
    
    Técnicas:
    - Parafraseo simple (cambiar orden de palabras)
    - Variaciones con sinónimos
    - Cambios de mayúsculas/minúsculas en palabras clave
    
    Args:
        ejemplos: Lista de (query, labels)
        factor: Cuántos ejemplos generar por cada original (default: 2)
    
    Returns:
        Lista ampliada con variaciones
    """
    import random
    
    # Sinónimos comunes
    sinonimos = {
        "configurar": ["configurar", "setear", "ajustar", "establecer"],
        "error": ["error", "fallo", "problema", "bug"],
        "instalar": ["instalar", "montar", "deployar", "poner"],
        "explicar": ["explicar", "aclarar", "detallar", "describir"],
        "ayuda": ["ayuda", "asistencia", "apoyo", "soporte"],
        "gracias": ["gracias", "agradezco", "muchas gracias", "te agradezco"],
        "ganó": ["ganó", "obtuvo", "consiguió", "logró"],
        "clima": ["clima", "tiempo", "temperatura", "condiciones"],
        "precio": ["precio", "valor", "costo", "cotización"],
        "noticias": ["noticias", "novedades", "info", "información"],
    }
    
    ejemplos_aumentados = list(ejemplos)  # Copiar originales
    
    for query, labels in ejemplos[:len(ejemplos) // factor]:  # Aumentar subset
        # Variación 1: Cambiar sinónimos
        query_aug = query
        for palabra, variantes in sinonimos.items():
            if palabra in query.lower():
                query_aug = query.replace(palabra, random.choice(variantes))
                break
        
        if query_aug != query:
            ejemplos_aumentados.append((query_aug, labels))
    
    return ejemplos_aumentados


def generar_dataset(total: int = 5000, output_path: str = "data/trm_training.jsonl", augmentation: bool = True):
    """
    Genera dataset balanceado para entrenamiento (v2.11 robusto)
    
    Distribución:
    - 35% hard (queries técnicas)
    - 25% soft (queries emocionales)
    - 30% web_query (búsquedas web)
    - 10% híbridos (combinaciones)
    
    Args:
        total: Número total de ejemplos (default: 5000 para mejor generalización)
        output_path: Ruta de salida
        augmentation: Si aplicar data augmentation (default: True)
    """
    print(f"🔧 Generando {total} ejemplos de entrenamiento...")
    
    n_hard = int(total * 0.35)
    n_soft = int(total * 0.25)
    n_web = int(total * 0.30)
    n_hibrido = total - n_hard - n_soft - n_web
    
    ejemplos = []
    ejemplos.extend(generar_ejemplos_hard(n_hard))
    ejemplos.extend(generar_ejemplos_soft(n_soft))
    ejemplos.extend(generar_ejemplos_web(n_web))
    ejemplos.extend(generar_ejemplos_hibridos(n_hibrido))
    
    # Data augmentation (v2.11)
    if augmentation:
        print("🔄 Aplicando data augmentation...")
        ejemplos_antes = len(ejemplos)
        ejemplos = aplicar_data_augmentation(ejemplos, factor=2)
        print(f"   Ejemplos aumentados: {ejemplos_antes} → {len(ejemplos)}")
        
        # Ajustar al total deseado
        if len(ejemplos) > total:
            ejemplos = ejemplos[:total]
        elif len(ejemplos) < total:
            # Completar con ejemplos adicionales si es necesario
            faltantes = total - len(ejemplos)
            ejemplos.extend(generar_ejemplos_hibridos(faltantes))
    
    # Shuffle
    random.shuffle(ejemplos)
    
    # Guardar
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for query, labels in ejemplos:
            entry = {
                "text": query,
                "hard": labels["hard"],
                "soft": labels["soft"],
                "web_query": labels["web_query"]
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"✅ Dataset guardado en {output_path}")
    print(f"   • {n_hard} ejemplos hard ({n_hard/total*100:.1f}%)")
    print(f"   • {n_soft} ejemplos soft ({n_soft/total*100:.1f}%)")
    print(f"   • {n_web} ejemplos web_query ({n_web/total*100:.1f}%)")
    print(f"   • {n_hibrido} ejemplos híbridos ({n_hibrido/total*100:.1f}%)")
    print(f"   Total: {len(ejemplos)} ejemplos")
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generar dataset sintético para TRM-Classifier v2.11")
    parser.add_argument("--samples", type=int, default=5000, 
                        help="Número total de ejemplos a generar (default: 5000)")
    parser.add_argument("--output", type=str, default="data/trm_training.jsonl",
                        help="Ruta de salida del dataset")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Deshabilitar data augmentation")
    
    args = parser.parse_args()
    
    generar_dataset(
        total=args.samples, 
        output_path=args.output,
        augmentation=not args.no_augmentation
    )
