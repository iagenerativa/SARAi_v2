# Configuración de Servidor Ollama para Desarrollo

## Resumen

SARAi v2.16 ahora soporta **servidor Ollama remoto** para desarrollo, permitiendo:

- ✅ Servidor de desarrollo en red local (<OLLAMA_HOST>:11434)
- ✅ SOLAR-10.7B disponible sin descargar GGUF localmente
- ✅ Configuración centralizada en `.env`
- ✅ Fallback automático a GGUF nativo si servidor no disponible

## Arquitectura

```
┌─────────────────────────────────────────┐
│  Desarrollo (Laptop i5 6ª Gen)          │
│  ┌────────────────────────────────┐    │
│  │ SARAi v2.16                     │    │
│  │ ┌────────────────────────┐     │    │
│  │ │ agents/solar_ollama.py │─────┼────┼──► HTTP
│  │ └────────────────────────┘     │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
                    │
                    │ HTTP REST
                    │ (OLLAMA_BASE_URL)
                    ▼
┌─────────────────────────────────────────┐
│  Servidor Ollama (<OLLAMA_HOST>:11434)  │
│  ┌────────────────────────────────┐    │
│  │ SOLAR-10.7B                     │    │
│  │ fblgit/UNA-SOLAR-10.7B-         │    │
│  │ Instruct-v1.0:Q5_K_M            │    │
│  │                                 │    │
│  │ Tamaño: 7.5 GB                  │    │
│  │ Cuantización: Q5_K_M            │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

## Configuración Rápida

### 1. Editar `.env`

El archivo `.env` ya está pre-configurado para desarrollo:

```bash
# Servidor Ollama (Desarrollo)
OLLAMA_BASE_URL=http://<OLLAMA_HOST>:11434
SOLAR_MODEL_NAME=fblgit/UNA-SOLAR-10.7B-Instruct-v1.0:Q5_K_M
OLLAMA_TIMEOUT=120
OLLAMA_RETRIES=3
```

**Para producción**, cambia a:

```bash
OLLAMA_BASE_URL=http://localhost:11434
SOLAR_MODEL_NAME=solar:10.7b
```

### 2. Verificar Conexión

```bash
# Opción A: Script de prueba
python scripts/test_ollama_connection.py

# Opción B: cURL manual
curl http://<OLLAMA_HOST>:11434/api/tags
```

**Salida esperada**:

```
✅ Cliente Ollama inicializado:
   Servidor: http://<OLLAMA_HOST>:11434
   Modelo: fblgit/UNA-SOLAR-10.7B-Instruct-v1.0:Q5_K_M
   Timeout: 120s
   Reintentos: 3
✅ Modelo fblgit/UNA-SOLAR-10.7B-Instruct-v1.0:Q5_K_M disponible en servidor
```

### 3. Usar en tu Código

```python
from agents.solar_ollama import SolarOllama

# Lee configuración de .env automáticamente
client = SolarOllama()

# Generación simple
response = client.generate("¿Qué es la IA?", max_tokens=200)
print(response)

# Estilo Upstage (formato oficial)
response = client.generate_upstage_style(
    prompt="Explica Python en una frase",
    system_prompt="Eres un tutor conciso"
)

# Chat multi-turno
messages = [
    {"role": "system", "content": "Eres un asistente técnico"},
    {"role": "user", "content": "Hola"},
    {"role": "assistant", "content": "Hola, ¿en qué puedo ayudarte?"},
    {"role": "user", "content": "¿Qué es Docker?"}
]
response = client.chat(messages)
```

## Características

### Reintentos Automáticos

El cliente reintenta automáticamente en caso de:

- **Timeout**: Backoff exponencial (2^n segundos)
- **Error de red**: 3 intentos por defecto
- **Servidor caído**: Excepción clara con sugerencias

```python
# Configurar reintentos personalizados
client = SolarOllama(
    timeout=60,      # 60 segundos
    retries=5        # 5 intentos
)
```

### Streaming

```python
# Generación con streaming (token por token)
for token in client.generate("Cuenta hasta 10", stream=True):
    print(token, end="", flush=True)
```

### Formato Upstage Oficial

El método `generate_upstage_style()` usa el formato oficial de SOLAR:

```
### System:
{system_prompt}

### User:
{user_prompt}

### Assistant:
{respuesta}
```

Esto garantiza **máxima calidad** de respuestas según entrenamiento de Upstage.

## Troubleshooting

### Error: "No se pudo conectar al servidor Ollama"

**Causas**:
1. Servidor Ollama no está corriendo
2. IP incorrecta en `.env`
3. Firewall bloqueando puerto 11434

**Soluciones**:

```bash
# 1. Verificar que el servidor esté corriendo
ssh user@<OLLAMA_HOST> 'systemctl status ollama'

# 2. Verificar acceso HTTP
curl http://<OLLAMA_HOST>:11434/api/tags

# 3. Verificar firewall
ssh user@<OLLAMA_HOST> 'sudo ufw status'

# 4. Probar con localhost (si Ollama está local)
OLLAMA_BASE_URL=http://localhost:11434 python scripts/test_ollama_connection.py
```

### Error: "Modelo X NO encontrado en servidor"

**Solución**:

```bash
# Conectar al servidor y descargar modelo
ssh user@<OLLAMA_HOST>
ollama pull fblgit/UNA-SOLAR-10.7B-Instruct-v1.0:Q5_K_M

# O usar modelo oficial
ollama pull solar:10.7b
```

### Timeout constante

**Solución**: Aumentar timeout en `.env`

```bash
OLLAMA_TIMEOUT=300  # 5 minutos
```

## Comparativa: GGUF Nativo vs Ollama HTTP

| Aspecto | GGUF Nativo (`solar_native.py`) | Ollama HTTP (`solar_ollama.py`) |
|---------|----------------------------------|----------------------------------|
| **Velocidad** | 2.11-2.20 tok/s | 2.0-2.5 tok/s (según red) |
| **RAM Local** | 11.8 GB | ~200 MB (solo cliente) |
| **Setup** | Descarga GGUF 6.1 GB | Solo configurar `.env` |
| **Latencia** | ~30ms (local) | ~50-100ms (red local) |
| **Red** | No requiere | Requiere (LAN/WAN) |
| **Escalabilidad** | 1 instancia | N instancias compartidas |
| **Ideal para** | Producción local | Desarrollo compartido |

## Producción: Estrategia Híbrida

**Recomendación v2.16**:

```python
# core/model_pool.py (pseudo-código)
def get_solar():
    """Estrategia híbrida: Ollama primary, GGUF fallback"""
    
    # Intento 1: Ollama HTTP (más recursos, servidor dedicado)
    try:
        from agents.solar_ollama import SolarOllama
        return SolarOllama()
    except ConnectionError:
        # Intento 2: GGUF nativo (fallback local)
        from agents.solar_native import SolarNative
        return SolarNative()
```

**Beneficios**:
- ✅ **Alta disponibilidad**: Ollama cae → GGUF sigue funcionando
- ✅ **Flexibilidad**: Desarrollo usa Ollama, producción usa GGUF
- ✅ **Debugging**: Mismo código, diferente backend

## Siguiente Paso: Validación

```bash
# Ejecutar test completo
python scripts/test_ollama_connection.py

# Si pasa: listo para integrar en SARAi
# Si falla: revisar troubleshooting arriba
```

## Referencias

- **Ollama API**: https://github.com/ollama/ollama/blob/main/docs/api.md
- **SOLAR Oficial**: https://huggingface.co/Upstage/SOLAR-10.7B-v1.0
- **UNA-SOLAR Q5_K_M**: https://huggingface.co/fblgit/UNA-SOLAR-10.7B-Instruct-v1.0
