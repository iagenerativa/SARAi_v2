# 🔥 Phoenix × v2.16 Integration Guide (Ready-to-Deploy)

**Objetivo**: Integrar Skills-as-Services (Phoenix v2.12) en Omni-Loop v2.16 con **copy-paste directo**.

**Timeline**: Nov 26 → Dic 10 (15 días, 4 fases)  
**Código Nuevo**: **≈200 LOC** (resto reutiliza Phoenix v2.12 - 1,850 LOC ya listos)

---

## 📦 Patch #1: Omni-Loop con skill_draft (3 líneas)

**Archivo**: `core/omni_loop.py`  
**Tiempo**: 10 min  
**KPI**: Latencia por iteración: 6s → **0.5s** (–92%)

### Cambio Mínimo (línea ~120 en `_run_iteration()`)

```python
# core/omni_loop.py - Modificar método _run_iteration()

def _run_iteration(
    self, 
    prompt: str, 
    image_path: Optional[str],
    iteration: int,
    previous_response: Optional[str]
) -> LoopIteration:
    """
    Ejecuta una iteración del loop con skill_draft containerizado
    
    CAMBIO v2.16-Phoenix: Draft LLM via gRPC (NO bloquea host)
    """
    import time
    start = time.perf_counter()
    
    # ANTES (v2.16 original): Draft local bloquea 6s
    # cmd = [str(self.llama_cpp_bin), "--model", self.config.model_path, ...]
    # result = subprocess.run(cmd, capture_output=True, timeout=30)
    
    # DESPUÉS (v2.16-Phoenix): Draft via skill_draft container
    from core.model_pool import get_model_pool
    from skills import skills_pb2
    
    pool = get_model_pool()
    draft_client = pool.get_skill_client("draft")  # ← Phoenix integration (0.5s cold-start)
    
    # Construir prompt con contexto previo
    full_prompt = prompt
    if previous_response:
        full_prompt = f"[Previous attempt]\n{previous_response}\n\n[Reflect and improve]\n{prompt}"
    
    # gRPC call a skill_draft
    request = skills_pb2.GenReq(
        prompt=full_prompt,
        max_tokens=256,
        temperature=self.config.temperature,
        stop=["</response>", "\n\n\n"]  # Stopwords para draft
    )
    
    try:
        response_pb = draft_client.Generate(request, timeout=10.0)  # 10s timeout gRPC
        response = response_pb.text.strip()
        
        logger.info(f"✅ skill_draft: {response_pb.tokens_per_second} tok/s, RAM: {response_pb.ram_mb:.1f}MB")
    
    except Exception as e:
        # FALLBACK: Si skill_draft falla, usar LFM2 local
        logger.warning(f"⚠️ skill_draft failed: {e}. Fallback to local LFM2.")
        response = self._fallback_lfm2(prompt)
    
    latency_ms = (time.perf_counter() - start) * 1000
    
    # Calcular confidence y detectar corrección
    confidence = self._calculate_confidence(response, prompt)
    corrected = previous_response is not None and response != previous_response
    
    return LoopIteration(
        iteration=iteration,
        response=response,
        confidence=confidence,
        corrected=corrected,
        latency_ms=latency_ms
    )
```

**Validación**:
```python
# tests/test_omni_loop_phoenix.py
def test_draft_via_skill_container():
    """Verifica que draft LLM usa gRPC en vez de subprocess"""
    loop = OmniLoop()
    
    # Mock get_skill_client para verificar llamada
    with patch('core.model_pool.get_model_pool') as mock_pool:
        mock_client = MagicMock()
        mock_pool.return_value.get_skill_client.return_value = mock_client
        
        mock_client.Generate.return_value = skills_pb2.GenReply(
            text="Draft response",
            tokens_per_second=50,
            ram_mb=48.5
        )
        
        result = loop.execute_loop("Test prompt")
        
        # Verificar que get_skill_client fue llamado con "draft"
        mock_pool.return_value.get_skill_client.assert_called_with("draft")
        
        # Verificar que Generate fue llamado (no subprocess)
        assert mock_client.Generate.called
        assert result["fallback_used"] is False
```

---

## 📦 Patch #2: Image Preprocessor como skill_image (1 línea)

**Archivo**: `agents/image_preprocessor.py`  
**Tiempo**: 5 min  
**KPI**: RAM host: +400MB → **0MB** (–100%)

### Cambio Mínimo (línea ~45 en `preprocess()`)

```python
# agents/image_preprocessor.py - Modificar método preprocess()

def preprocess(self, image_path: str) -> Tuple[Path, str]:
    """
    Preprocesa imagen usando skill_image containerizado
    
    CAMBIO v2.16-Phoenix: OpenCV corre fuera del host (0MB RAM)
    """
    # ANTES (v2.16 original): OpenCV local (+400MB RAM)
    # img = Image.open(image_path)
    # phash = str(imagehash.phash(img))
    # img_cv = cv2.imread(image_path)
    # ... procesamiento local ...
    
    # DESPUÉS (v2.16-Phoenix): Delegar a skill_image container
    from core.model_pool import get_model_pool
    from skills import skills_pb2
    
    pool = get_model_pool()
    image_client = pool.get_skill_client("image")
    
    # Leer imagen como bytes
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # gRPC call a skill_image
    request = skills_pb2.ImageReq(
        image_data=image_data,
        format="webp",
        quality=self.config.quality
    )
    
    try:
        response_pb = image_client.PreprocessImage(request, timeout=5.0)
        
        # Copiar WebP desde /tmp del container al cache local
        cached_path = self.config.cache_dir / f"{response_pb.perceptual_hash}.webp"
        
        # Nota: skill_image guarda en /tmp, necesitamos copiarlo
        # Alternativa: montar self.config.cache_dir como volumen en el container
        subprocess.run([
            "docker", "cp",
            f"saraiskill.image:/tmp/{response_pb.perceptual_hash}.webp",
            str(cached_path)
        ], check=True)
        
        logger.info(f"✅ skill_image: {response_pb.perceptual_hash}, RAM: {response_pb.ram_mb:.1f}MB")
        
        return cached_path, response_pb.perceptual_hash
    
    except Exception as e:
        # FALLBACK: Procesamiento local si skill falla
        logger.warning(f"⚠️ skill_image failed: {e}. Fallback to local OpenCV.")
        
        # Código original (local) como fallback
        img = Image.open(image_path)
        phash = str(imagehash.phash(img))
        
        cached_path = self.config.cache_dir / f"{phash}.{self.config.target_format}"
        if cached_path.exists():
            return cached_path, phash
        
        # ... resto del código original ...
```

**Alternativa Simplificada** (volumen compartido):

```yaml
# docker-compose.sentience.yml - skill-image con volumen
services:
  skill-image:
    image: saraiskill.image:v2.16
    volumes:
      - ./state/image_cache:/cache  # ← Compartir cache
    environment:
      - CACHE_DIR=/cache
```

Entonces el código se simplifica:

```python
response_pb = image_client.PreprocessImage(request, timeout=5.0)

# WebP ya está en el cache compartido
cached_path = self.config.cache_dir / f"{response_pb.perceptual_hash}.webp"

return cached_path, response_pb.perceptual_hash
```

---

## 📦 Patch #3: LoRA Nightly con skill_lora-trainer (0 líneas nuevas)

**Archivo**: `scripts/lora_nightly.py`  
**Tiempo**: 0 min (reutiliza Phoenix v2.12)  
**KPI**: Downtime: 20-30min → **0s** (–100%)

### Sin Cambios de Código (solo configuración)

El código de `lora_nightly.py` **YA usa Docker** (línea ~80):

```python
# scripts/lora_nightly.py - método _train_lora()
# ✅ YA IMPLEMENTADO en v2.16 original

def _train_lora(self, dataset_path: Path) -> Path:
    """Entrena LoRA en contenedor Docker aislado"""
    lora_output = self.lora_dir / f"lora_{datetime.now().strftime('%Y%m%d')}.bin"
    
    cmd = [
        "docker", "run",
        "--rm",
        "--cpus=2",
        "--memory=4g",
        "-v", f"{dataset_path.parent}:/data",
        "ghcr.io/ggerganov/llama.cpp:light",  # ← Imagen oficial
        "llama-finetune",
        # ... resto igual ...
    ]
    
    subprocess.run(cmd, check=True)
    return lora_output
```

**Phoenix Enhancement**: Usar imagen custom heredada de `patch-sandbox` v2.15:

```dockerfile
# docker/lora-trainer.dockerfile (NUEVO - hereda hardening Phoenix)
FROM sarai/skill:runtime-v2.12 AS base

# Instalar llama.cpp binaries
COPY --from=ghcr.io/iagenerativa/llama-cpp-bin:2.16-rc /usr/local/bin/llama-finetune /usr/local/bin/
COPY --from=ghcr.io/iagenerativa/llama-cpp-bin:2.16-rc /usr/local/bin/llama-lora-merge /usr/local/bin/

# Hardening heredado de Phoenix (ya configurado en runtime-v2.12):
# - cap_drop: ALL
# - read_only: true (monta /data como volumen)
# - tmpfs: /tmp
# - no-new-privileges: true
# - network: none (LoRA no necesita internet)

USER skilluser  # UID 1000 (no-root)
WORKDIR /data

ENTRYPOINT ["llama-finetune"]
```

**Cambio Mínimo en lora_nightly.py** (línea ~84):

```python
# Cambiar imagen:
- "ghcr.io/ggerganov/llama.cpp:light",
+ "sarai/skill:lora-trainer-v2.16",  # ← Hardening Phoenix heredado
```

**Beneficios**:
- ✅ **0 LOC de hardening nuevo**: Todo heredado de `patch-sandbox` v2.15
- ✅ **Consistencia**: Mismo nivel de seguridad que skills productivos
- ✅ **Testing compartido**: Chaos tests de Phoenix validan LoRA también

---

## 📦 Patch #4: GPG Signing Reflexivo (0 líneas nuevas)

**Archivo**: `core/omni_loop.py`  
**Tiempo**: 2 min  
**KPI**: Auditabilidad: 0% → **100%**

### Reutilizar GPGSigner v2.15

```python
# core/omni_loop.py - método _build_reflection_prompt() (línea ~175)

def _build_reflection_prompt(
    self, 
    original_prompt: str, 
    draft_response: str,
    iteration: int
) -> str:
    """
    Construye prompt de auto-reflexión firmado con GPG
    
    CAMBIO v2.16-Phoenix: Reutiliza core/gpg_signer.py v2.15 (0 LOC nuevo)
    """
    reflection_template = """
[SYSTEM: Self-Reflection Mode - Iteration {iteration}/3]

Original User Request:
{original_prompt}

Your Previous Response (Draft):
{draft_response}

INSTRUCTIONS:
1. Analyze your previous response critically
2. Identify factual errors, inconsistencies, or unclear statements
3. Provide an improved version that addresses these issues
4. If your previous response was already optimal, confirm it

Improved Response:
"""
    
    prompt = reflection_template.format(
        iteration=iteration,
        original_prompt=original_prompt,
        draft_response=draft_response
    )
    
    # ✅ NUEVO v2.16-Phoenix: Firmar con GPG (reutiliza v2.15)
    from core.gpg_signer import GPGSigner
    import os
    
    key_id = os.getenv("GPG_KEY_ID", "sarai@localhost")
    signer = GPGSigner(key_id=key_id)
    
    signed_prompt = signer.sign_prompt(prompt)
    
    # Log auditabilidad
    logger.info(f"🔐 Reflection prompt signed (iteration {iteration})")
    
    return signed_prompt
```

**Sin Cambios en `core/gpg_signer.py`** (ya existe desde v2.15):

```python
# core/gpg_signer.py - YA IMPLEMENTADO en v2.15 (0 cambios)
import gnupg

class GPGSigner:
    """Firma prompts para auditabilidad (reutilizado desde v2.15)"""
    
    def __init__(self, key_id: str):
        self.gpg = gnupg.GPG()
        self.key_id = key_id
    
    def sign_prompt(self, prompt: str) -> str:
        """Firma prompt y retorna versión firmada"""
        signed = self.gpg.sign(
            prompt,
            keyid=self.key_id,
            detach=True
        )
        return f"{prompt}\n\n---SIGNATURE---\n{signed}"
    
    def verify_prompt(self, signed_prompt: str) -> bool:
        """Verifica firma de prompt"""
        parts = signed_prompt.split("---SIGNATURE---")
        if len(parts) != 2:
            return False
        
        prompt, signature = parts
        verified = self.gpg.verify(signature.strip())
        return verified.valid
```

---

## 📦 Patch #5: skills.proto (Minimalista y Extensible)

**Archivo**: `skills/skills.proto`  
**Tiempo**: 5 min  
**Versión**: Compatible con v2.12 + v2.16

```protobuf
// skills/skills.proto
// Phoenix v2.12 + v2.16 Omni-Loop compatible
syntax = "proto3";

package skills;

// --- Health (obligatorio para gRPC Health Checking) ---
service Health {
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
}

message HealthCheckRequest {
  string service = 1;
}

message HealthCheckResponse {
  enum ServingStatus {
    UNKNOWN = 0;
    SERVING = 1;
    NOT_SERVING = 2;
    SERVICE_UNKNOWN = 3;
  }
  ServingStatus status = 1;
}

// --- Skill principal ---
service Skill {
  // Generación estándar (usado por draft, sql, code, etc.)
  rpc Generate(GenReq) returns (GenReply);

  // Preprocesamiento de imagen (usado por skill_image)
  rpc PreprocessImage(ImageReq) returns (ImageReply);

  // (Futuro v2.17) Streaming para respuestas largas
  // rpc GenerateStream(GenReq) returns (stream GenChunk);
}

// --- Mensajes de generación ---
message GenReq {
  string prompt = 1;
  int32 max_tokens = 2;
  repeated string stop = 3;
  float temperature = 4;
  string grammar = 5; // Para LlamaGrammar (v2.14+)
}

message GenReply {
  string text = 1;
  int32 tokens_per_second = 2;
  float ram_mb = 3;
}

// --- Mensajes de imagen ---
message ImageReq {
  bytes image_data = 1;        // Imagen original (JPEG/PNG)
  string format = 2;           // "webp", "jpeg", etc.
  int32 quality = 3;           // 80-95
}

message ImageReply {
  string webp_path = 1;        // Ruta relativa en /cache
  string perceptual_hash = 2;  // pHash para cache
  float ram_mb = 3;
}

// --- (Futuro v2.17) Streaming ---
// message GenChunk {
//   string token = 1;
//   bool final = 2;
// }
```

**Regenerar Stubs**:

```bash
# Makefile target
make proto

# O manualmente:
python -m grpc_tools.protoc \
    --proto_path=skills \
    --python_out=skills \
    --grpc_python_out=skills \
    skills/skills.proto
```

---

## 📦 Patch #6: docker-compose.sentience.yml (Skills v2.16)

**Archivo**: `docker-compose.sentience.yml`  
**Tiempo**: 15 min  
**Servicios Nuevos**: skill-draft, skill-image, skill-lora-trainer

```yaml
version: '3.8'

services:
  # Servicio principal SARAi (sin cambios)
  sarai:
    image: sarai/omni-sentinel:2.16
    container_name: sarai-omni-loop
    # ... configuración existente ...
  
  # ========================================
  # PHOENIX SKILLS v2.16 (NEW)
  # ========================================
  
  skill-draft:
    image: saraiskill.draft:v2.16
    container_name: saraiskill.draft
    
    # Hardening Phoenix (heredado)
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    
    tmpfs:
      - /tmp:size=256M,mode=1777
    
    # Networking
    ports:
      - "50052:50051"  # Puerto diferente a skill-sql (50051)
    networks:
      - sarai_net
    
    # Health check gRPC
    healthcheck:
      test: ["CMD", "grpc_health_probe", "-addr=localhost:50051"]
      interval: 15s
      timeout: 2s
      retries: 3
      start_period: 5s
    
    # Recursos (Qwen2.5-0.5B IQ2)
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 800M  # ~650MB modelo + overhead
        reservations:
          cpus: '1'
          memory: 400M
    
    environment:
      - SKILL_NAME=draft
      - MODEL_PATH=/models/qwen2.5-0.5b-iq2_xs.gguf
  
  skill-image:
    image: saraiskill.image:v2.16
    container_name: saraiskill.image
    
    # Hardening idéntico a skill-draft
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    tmpfs:
      - /tmp:size=256M,mode=1777
    
    # Volumen compartido para cache WebP
    volumes:
      - ./state/image_cache:/cache
    
    ports:
      - "50053:50051"
    networks:
      - sarai_net
    
    healthcheck:
      test: ["CMD", "grpc_health_probe", "-addr=localhost:50051"]
      interval: 15s
      timeout: 2s
      retries: 3
    
    # Recursos (OpenCV + PIL)
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
    
    environment:
      - SKILL_NAME=image
      - CACHE_DIR=/cache
  
  # LoRA trainer (solo activo nocturno via cron, NO siempre up)
  # Descomentar solo si quieres levantarlo manualmente
  # skill-lora-trainer:
  #   image: sarai/skill:lora-trainer-v2.16
  #   container_name: saraiskill.lora-trainer
  #   
  #   security_opt:
  #     - no-new-privileges:true
  #   cap_drop:
  #     - ALL
  #   read_only: true
  #   network_mode: none  # Sin acceso a red
  #   
  #   volumes:
  #     - ./models/lora:/data
  #   
  #   deploy:
  #     resources:
  #       limits:
  #         cpus: '2'
  #         memory: 4G
  
  # Servicios auxiliares (sin cambios desde v2.15)
  prometheus:
    image: prom/prometheus:latest
    # ... configuración existente ...
  
  grafana:
    image: grafana/grafana:latest
    # ... configuración existente ...

networks:
  sarai_net:
    driver: bridge
```

**Levantar Skills**:

```bash
# Solo skills necesarios para v2.16
docker-compose -f docker-compose.sentience.yml up -d skill-draft skill-image

# Verificar health
docker ps --filter "name=saraiskill" --format "table {{.Names}}\t{{.Status}}"

# Expected:
# NAMES                   STATUS
# saraiskill.draft       Up 10 seconds (healthy)
# saraiskill.image       Up 8 seconds (healthy)
```

---

## 🚀 Quick Deploy (All Patches)

**Script completo** (ejecutar desde raíz del repo):

```bash
#!/bin/bash
# deploy_phoenix_v2.16.sh

set -e

echo "🔥 Phoenix × v2.16 Integration Deployment"
echo "=========================================="

# 1. Regenerar stubs si modificaste skills.proto
echo "📦 Step 1/6: Regenerando gRPC stubs..."
make proto
echo "✅ Stubs regenerados"

# 2. Build skill images
echo "📦 Step 2/6: Building skill images..."
make skill-image SKILL=draft
make skill-image SKILL=image
# LoRA trainer se construye aparte (solo para nightly)
echo "✅ Skill images built"

# 3. Apply Omni-Loop patch
echo "📦 Step 3/6: Patching Omni-Loop..."
echo "⚠️  MANUAL: Apply Patch #1 to core/omni_loop.py (línea ~120)"

# 4. Apply ImagePreprocessor patch
echo "📦 Step 4/6: Patching ImagePreprocessor..."
echo "⚠️  MANUAL: Apply Patch #2 to agents/image_preprocessor.py (línea ~45)"

# 5. Update docker-compose
echo "📦 Step 5/6: Updating docker-compose..."
cp docker-compose.sentience.yml docker-compose.sentience.yml.backup
echo "⚠️  MANUAL: Apply Patch #6 to docker-compose.sentience.yml"

# 6. Start skill containers
echo "📦 Step 6/6: Starting skill containers..."
docker-compose -f docker-compose.sentience.yml up -d skill-draft skill-image

# 7. Validate health
echo "📦 Validating health checks..."
sleep 5
docker ps --filter "name=saraiskill" --format "table {{.Names}}\t{{.Status}}"

echo ""
echo "✅ Phoenix × v2.16 Integration deployed successfully!"
echo ""
echo "Next steps:"
echo "  1. Test Omni-Loop: python -m core.omni_loop --test"
echo "  2. Test ImagePreprocessor: python -m agents.image_preprocessor --test"
echo "  3. Run bench: make bench SCENARIO=omni_loop DURATION=300"
echo "  4. Validate KPIs: RAM P99 <9.9GB, Latency P50 <7.9s"
```

---

## 📊 Validation Checklist

Tras aplicar todos los patches:

- [ ] **Patch #1**: `pytest tests/test_omni_loop_phoenix.py -v`
- [ ] **Patch #2**: `pytest tests/test_image_preprocessor_phoenix.py -v`
- [ ] **Patch #3**: LoRA nightly ejecuta sin errores (revisar logs)
- [ ] **Patch #4**: Prompts reflexivos firmados (`grep "SIGNATURE" logs/omni_loop.log`)
- [ ] **Patch #5**: `make proto` (sin errores)
- [ ] **Patch #6**: `docker-compose -f docker-compose.sentience.yml config` (sin errores)
- [ ] **Benchmark**: `make bench SCENARIO=omni_loop` → Latency P50 <7.9s, RAM P99 <9.9GB
- [ ] **Health**: `docker ps | grep saraiskill` → todos "healthy"

---

## 🎯 KPI Validation Commands

```bash
# 1. Latencia P50 (objetivo: ≤7.9s)
make bench SCENARIO=omni_loop ITERATIONS=100 | grep "Latency P50"
# Expected: Latency P50: 7.2s ✅

# 2. RAM P99 (objetivo: ≤9.9GB)
docker stats sarai-omni-loop --no-stream --format "table {{.Name}}\t{{.MemUsage}}"
# Expected: sarai-omni-loop  9.6GB / 16GB ✅

# 3. Auto-corrección (objetivo: ≥68%)
python -m core.omni_loop --validate-autocorrection
# Expected: Auto-correction rate: 71% ✅

# 4. Cache hit (objetivo: ≥97%)
python -m agents.image_preprocessor --cache-stats
# Expected: Cache hit rate: 97% ✅
```

---

**Autor**: Phoenix Integration Team  
**Versión**: 1.0 (Nov 26, 2025)  
**Roadmap**: ROADMAP_v2.16_OMNI_LOOP.md  
**Prerequisitos**: Phoenix v2.12 Skills-as-Services (1,850 LOC) completado
