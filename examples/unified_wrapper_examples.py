"""
SARAi v2.14 - Ejemplos de Uso del Unified Model Wrapper
========================================================

Ejemplos prácticos de cómo usar el Unified Model Wrapper en diferentes escenarios.
"""

# ============================================================================
# EJEMPLO 1: Uso Básico - Cargar y usar un modelo
# ============================================================================

from core.unified_model_wrapper import get_model

# Cargar modelo SOLAR (Ollama backend)
solar = get_model("solar_short")

# Invocar con texto simple
response = solar.invoke("¿Qué es la inteligencia artificial?")
print(f"Respuesta: {response}")

# Invocar con configuración personalizada
response_custom = solar.invoke(
    "Explica Python en 50 palabras",
    config={
        "temperature": 0.9,
        "max_tokens": 150
    }
)
print(f"Respuesta personalizada: {response_custom}")


# ============================================================================
# EJEMPLO 2: Embeddings para Clasificación Semántica
# ============================================================================

import numpy as np
from core.unified_model_wrapper import get_model

# Cargar modelo de embeddings
embeddings = get_model("embeddings")

# Generar embedding de consulta
query = "¿Cómo instalar Python en Ubuntu?"
query_embedding = embeddings.invoke(query)

print(f"Dimensión del vector: {query_embedding.shape}")  # (768,)

# Categorías predefinidas
categories = [
    "pregunta técnica de programación",
    "saludo casual",
    "despedida",
    "consulta sobre clima"
]

# Generar embeddings de categorías (batch)
category_embeddings = embeddings.batch_encode(categories)
print(f"Embeddings de categorías: {category_embeddings.shape}")  # (4, 768)

# Calcular similitud coseno
similarities = np.dot(category_embeddings, query_embedding)
best_match_idx = np.argmax(similarities)
best_category = categories[best_match_idx]

print(f"Mejor categoría: {best_category}")
print(f"Similitud: {similarities[best_match_idx]:.3f}")


# ============================================================================
# EJEMPLO 3: Pipeline LangChain LCEL
# ============================================================================

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from core.unified_model_wrapper import get_model

# Cargar modelo
lfm2 = get_model("lfm2")

# Crear template de prompt
prompt_template = ChatPromptTemplate.from_template(
    """Eres un asistente {estilo}.

Usuario: {pregunta}

Responde de forma {estilo}:"""
)

# Crear pipeline LCEL
chain = prompt_template | lfm2 | StrOutputParser()

# Invocar con variables
response = chain.invoke({
    "estilo": "empático y cercano",
    "pregunta": "Estoy frustrado con este error de Python"
})

print(f"Respuesta empática: {response}")


# ============================================================================
# EJEMPLO 4: Fallback Chain (Resiliencia)
# ============================================================================

from langchain_core.runnables import RunnableBranch
from core.unified_model_wrapper import get_model

# Cargar modelos para fallback
solar = get_model("solar_short")
lfm2 = get_model("lfm2")

# Crear cadena con fallback automático
chain_with_fallback = solar.with_fallbacks([lfm2])

# Si SOLAR falla (ej. Ollama down), automáticamente usa LFM2
try:
    response = chain_with_fallback.invoke("Hola, ¿cómo estás?")
    print(f"Respuesta (con fallback): {response}")
except Exception as e:
    print(f"Error incluso con fallback: {e}")


# ============================================================================
# EJEMPLO 5: Multimodal - Procesar Imagen
# ============================================================================

from core.unified_model_wrapper import get_model

# Cargar modelo multimodal
qwen_vl = get_model("qwen3_vl")

# Procesar imagen desde archivo
response_image = qwen_vl.invoke({
    "text": "¿Qué aparece en esta imagen? Describe detalladamente.",
    "image": "diagrams/arquitectura_sarai.png"
})

print(f"Descripción de imagen: {response_image}")

# Procesar imagen desde base64
import base64

with open("assets/logo.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

response_b64 = qwen_vl.invoke({
    "text": "Identifica los colores principales",
    "image": f"data:image/png;base64,{image_data}"
})

print(f"Colores identificados: {response_b64}")


# ============================================================================
# EJEMPLO 6: Async Invocation (Alto Rendimiento)
# ============================================================================

import asyncio
from core.unified_model_wrapper import get_model

async def process_queries_async():
    """Procesar múltiples queries en paralelo"""
    solar = get_model("solar_short")
    
    queries = [
        "¿Qué es Python?",
        "¿Qué es JavaScript?",
        "¿Qué es Rust?"
    ]
    
    # Invocar en paralelo
    tasks = [solar.ainvoke(q) for q in queries]
    responses = await asyncio.gather(*tasks)
    
    for query, response in zip(queries, responses):
        print(f"Q: {query}")
        print(f"A: {response}\n")

# Ejecutar
asyncio.run(process_queries_async())


# ============================================================================
# EJEMPLO 7: Streaming (Respuestas en Tiempo Real)
# ============================================================================

from core.unified_model_wrapper import get_model

solar = get_model("solar_short")

print("Respuesta en streaming:")
for chunk in solar.stream("Cuenta del 1 al 20"):
    print(chunk, end="", flush=True)
print("\n")


# ============================================================================
# EJEMPLO 8: Batch Processing (Eficiencia)
# ============================================================================

from core.unified_model_wrapper import get_model

solar = get_model("solar_short")

# Procesar múltiples inputs en batch
queries = [
    "¿Qué es IA?",
    "¿Qué es ML?",
    "¿Qué es DL?"
]

responses = solar.batch(queries, config={"temperature": 0.7})

for q, r in zip(queries, responses):
    print(f"Q: {q}")
    print(f"A: {r}\n")


# ============================================================================
# EJEMPLO 9: Gestión Manual de Memoria
# ============================================================================

from core.unified_model_wrapper import ModelRegistry

# Obtener registro
registry = ModelRegistry()

# Ver modelos cargados
loaded_models = registry.get_loaded_models()
print(f"Modelos en memoria: {loaded_models}")

# Descargar modelo manualmente
registry.unload_model("solar_short")
print("SOLAR descargado manualmente")

# Cargar de nuevo cuando se necesite
solar = get_model("solar_short")  # Auto-carga
print("SOLAR recargado")

# Descargar TODOS los modelos
registry.unload_all()
print("Todos los modelos descargados")


# ============================================================================
# EJEMPLO 10: TRM + Embeddings + MCP (Pipeline Completo)
# ============================================================================

from core.unified_model_wrapper import get_model
from core.trm_classifier import TRMClassifierDual
from core.mcp import MCP

def classify_and_respond(user_input: str) -> str:
    """Pipeline completo: TRM → Embeddings → MCP → Agent"""
    
    # 1. Generar embeddings
    embeddings = get_model("embeddings")
    input_embedding = embeddings.invoke(user_input)
    
    # 2. Clasificar con TRM
    trm = TRMClassifierDual()  # Cargado desde checkpoint
    scores = trm.forward(input_embedding)
    
    print(f"Scores TRM: hard={scores['hard']:.3f}, soft={scores['soft']:.3f}")
    
    # 3. Calcular pesos con MCP
    mcp = MCP()
    alpha, beta = mcp.compute_weights(scores, user_input)
    
    print(f"Pesos MCP: α={alpha:.3f}, β={beta:.3f}")
    
    # 4. Seleccionar agente
    if alpha > 0.7:
        model = get_model("solar_short")
        agent_type = "SOLAR (Técnico)"
    else:
        model = get_model("lfm2")
        agent_type = "LFM2 (Soft)"
    
    # 5. Generar respuesta
    response = model.invoke(user_input)
    
    print(f"Agente usado: {agent_type}")
    
    return response

# Probar pipeline completo
query_tecnico = "¿Cómo configurar SSH en Ubuntu 22.04?"
response1 = classify_and_respond(query_tecnico)
print(f"Respuesta técnica: {response1}\n")

query_emocional = "Estoy frustrado, no entiendo este error"
response2 = classify_and_respond(query_emocional)
print(f"Respuesta empática: {response2}\n")


# ============================================================================
# EJEMPLO 11: Validar Config y Listar Modelos
# ============================================================================

from core.unified_model_wrapper import list_available_models, get_model

# Listar todos los modelos disponibles
available = list_available_models()
print(f"Modelos disponibles: {available}")

# Validar que un modelo existe antes de usarlo
if "solar_short" in available:
    solar = get_model("solar_short")
    print("SOLAR cargado correctamente")
else:
    print("Error: SOLAR no disponible en config")


# ============================================================================
# EJEMPLO 12: Cambiar Backend sin Cambiar Código
# ============================================================================

"""
Escenario: Migrar de Ollama a GGUF local

ANTES (config/models.yaml):
solar_short:
  backend: "ollama"
  api_url: "http://192.168.0.251:11434"
  model_name: "solar-10.7b-q4_k_m"

DESPUÉS (config/models.yaml):
solar_short:
  backend: "gguf"
  model_path: "models/cache/solar/solar-10.7b.Q4_K_M.gguf"
  n_ctx: 512

EL CÓDIGO NO CAMBIA:
"""

from core.unified_model_wrapper import get_model

# Mismo código funciona con CUALQUIER backend
solar = get_model("solar_short")
response = solar.invoke("Hola")
print(response)

# Solo cambias YAML, no Python ✅


# ============================================================================
# EJEMPLO 13: Agregar Modelo Cloud (OpenAI)
# ============================================================================

"""
1. Agregar a config/models.yaml:

gpt4:
  name: "GPT-4 Turbo"
  backend: "openai_api"
  api_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"  # Lee desde env var
  model_name: "gpt-4-turbo-preview"
  max_tokens: 2048
  temperature: 0.7

2. Configurar env var:
export OPENAI_API_KEY="sk-..."

3. Usar:
"""

from core.unified_model_wrapper import get_model

gpt4 = get_model("gpt4")
response = gpt4.invoke("Explica teoría cuántica en 100 palabras")
print(f"GPT-4: {response}")


# ============================================================================
# EJEMPLO 14: Error Handling Robusto
# ============================================================================

from core.unified_model_wrapper import get_model

def safe_invoke(model_name: str, prompt: str) -> str:
    """Invocación robusta con manejo de errores"""
    try:
        model = get_model(model_name)
    except ValueError as e:
        return f"Error: Modelo '{model_name}' no encontrado. {e}"
    
    try:
        response = model.invoke(prompt)
        return response
    except ConnectionError:
        return "Error: No se pudo conectar al modelo (Ollama down?)"
    except Exception as e:
        return f"Error inesperado: {e}"

# Probar con modelo inexistente
result1 = safe_invoke("modelo_fake", "Hola")
print(result1)

# Probar con modelo real
result2 = safe_invoke("lfm2", "Hola")
print(result2)


# ============================================================================
# EJEMPLO 15: Benchmark de Latencia
# ============================================================================

import time
from core.unified_model_wrapper import get_model

def benchmark_model(model_name: str, prompt: str, iterations: int = 5):
    """Medir latencia promedio de un modelo"""
    model = get_model(model_name)
    
    latencies = []
    for i in range(iterations):
        start = time.time()
        response = model.invoke(prompt)
        end = time.time()
        
        latency = end - start
        latencies.append(latency)
        print(f"Iteración {i+1}: {latency:.2f}s")
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"\nLatencia promedio ({model_name}): {avg_latency:.2f}s")
    
    return avg_latency

# Comparar SOLAR vs LFM2
print("=== Benchmark SOLAR ===")
solar_latency = benchmark_model("solar_short", "Hola", iterations=3)

print("\n=== Benchmark LFM2 ===")
lfm2_latency = benchmark_model("lfm2", "Hola", iterations=3)

print(f"\nDiferencia: {abs(solar_latency - lfm2_latency):.2f}s")


if __name__ == "__main__":
    print("SARAi v2.14 - Unified Model Wrapper Examples")
    print("=" * 60)
    print("Ejecuta cada ejemplo individualmente copiando el código")
    print("o descomenta las secciones que quieras probar.")
