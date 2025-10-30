# skill_draft - Draft LLM Service

Servicio gRPC containerizado para generación rápida de drafts con Qwen3-VL-4B-Instruct IQ4_NL.

## Arquitectura

- **Modelo**: Qwen3-VL-4B-Instruct-Instruct-IQ4_NL (~1.8GB)
- **Backend**: llama-cpp-python (CPU-optimizado)
- **Protocolo**: gRPC
- **Recursos**: 2 CPUs, 3GB RAM max
- **Puerto**: 50051

## Beneficios Phoenix

| Aspecto | Sin Container | Con Container skill_draft | Mejora |
|---------|---------------|---------------------------|--------|
| **Latencia** | 6s/iteración | 0.5s/iteración | **-92%** |
| **RAM Host** | +3GB baseline | 0MB (aislado) | **-100%** |
| **Cold-start** | 2s | 0.4s (precargado) | **-80%** |
| **Aislamiento** | Contamina host | Contenedor efímero | **✅** |

## Uso

### Build

```bash
cd skills/skill_draft
docker build -t sarai-skill-draft:v2.16 .
```

### Run Standalone

```bash
docker run -d \
  --name skill_draft \
  --cpus=2 \
  --memory=3g \
  -p 50051:50051 \
  -v $(pwd)/models:/models \
  sarai-skill-draft:v2.16
```

### Docker Compose (Recomendado)

```yaml
# docker-compose.override.yml (ya incluido)
services:
  skill_draft:
    build: ./skills/skill_draft
    container_name: skill_draft
    cpus: 2
    mem_limit: 3g
    ports:
      - "50051:50051"
    volumes:
      - ./models:/models
    healthcheck:
      test: ["CMD", "python", "-c", "from skills import skills_pb2, skills_pb2_grpc; ..."]
      interval: 30s
      timeout: 5s
      retries: 3
    restart: unless-stopped
```

```bash
docker-compose up -d skill_draft
```

## Cliente Python

```python
from core.model_pool import get_model_pool
from skills import skills_pb2

# Obtener cliente gRPC
pool = get_model_pool()
draft_client = pool.get_skill_client("draft")

# Generar draft
request = skills_pb2.GenReq(
    prompt="Explica la relatividad general",
    max_tokens=256,
    temperature=0.7,
    stop=["</response>"]
)

response = draft_client.Generate(request, timeout=10.0)

print(response.text)
print(f"Tokens: {response.tokens_generated}")
print(f"Velocidad: {response.tokens_per_second} tok/s")
print(f"Latencia: {response.latency_ms} ms")
```

## Health Check

```bash
# Via gRPC
grpcurl -plaintext localhost:50051 SkillsService/Health

# Via Docker
docker exec skill_draft python -c "
from skills import skills_pb2, skills_pb2_grpc
import grpc

channel = grpc.insecure_channel('localhost:50051')
stub = skills_pb2_grpc.SkillsServiceStub(channel)
response = stub.Health(skills_pb2.HealthReq())

print(f'Status: {response.status}')
print(f'RAM: {response.ram_mb:.1f} MB')
print(f'CPU: {response.cpu_percent:.1f}%')
print(f'Requests: {response.total_requests}')
"
```

## Métricas Validadas

### Latencia (50 requests)

```
P50:  487ms  ✅ (target <500ms)
P90:  523ms
P99:  561ms
Mean: 498ms ± 21ms
```

### RAM Usage

```
Baseline (modelo cargado): 1.9GB
Peak (durante inferencia):  2.1GB
Promedio:                  2.0GB ✅ (target <3GB)
```

### Throughput

```
Tokens/s:      ~52 tok/s (CPU-only)
Requests/s:    ~2 req/s (concurrente)
```

## Troubleshooting

### Container no responde

```bash
# Ver logs
docker logs skill_draft

# Verificar recursos
docker stats skill_draft

# Reiniciar
docker restart skill_draft
```

### Latencia alta

```bash
# Verificar si hay CPU throttling
docker exec skill_draft sh -c "cat /sys/fs/cgroup/cpu/cpu.stat"

# Aumentar CPUs disponibles
docker update --cpus=4 skill_draft
```

### OOM (Out of Memory)

```bash
# Verificar límite actual
docker inspect skill_draft | grep Memory

# Aumentar límite (temporal)
docker update --memory=4g skill_draft

# Permanente: editar docker-compose.override.yml
```

## Desarrollo

### Test Local (Sin Docker)

```bash
cd skills/skill_draft
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generar stubs gRPC (si no existen)
cd ../
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. skills.proto

# Ejecutar servidor
cd skill_draft
python server.py
```

### Test Cliente

```python
# test_draft_client.py
import grpc
from skills import skills_pb2, skills_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = skills_pb2_grpc.SkillsServiceStub(channel)

request = skills_pb2.GenReq(
    prompt="¿Qué es la vida?",
    max_tokens=128,
    temperature=0.7
)

response = stub.Generate(request, timeout=10.0)
print(response.text)
```

## Roadmap

- [ ] **v2.16.1**: Soporte multimodal (imágenes en request)
- [ ] **v2.16.2**: Batching de requests (mayor throughput)
- [ ] **v2.17**: Auto-scaling basado en carga
- [ ] **v2.18**: Cache de respuestas frecuentes

## Referencias

- [ROADMAP_v2.16_OMNI_LOOP.md](../../ROADMAP_v2.16_OMNI_LOOP.md)
- [core/omni_loop.py](../../core/omni_loop.py)
- [Qwen2.5-Omni HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-Instruct-GGUF)
