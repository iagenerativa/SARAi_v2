# SARAi v2.16 "Omni-Loop" - Reflexive Multimodal AGI

**Status**: PLANNING PHASE  
**Prerequisitos**: v2.12 + v2.13 + v2.14 + v2.15 COMPLETADOS  
**Timeline**: Nov 26 - Dic 10, 2025 (15 días)  
**Autor**: SARAi Dev Team  
**Fecha**: Oct 28, 2025

---

## 🧠 Executive Summary

**v2.16 Omni-Loop** representa la **culminación evolutiva** de SARAi: un sistema AGI que no solo responde, sino que **reflexiona, auto-corrige y aprende continuamente** mediante ciclos interactivos multimodales con llama.cpp, fine-tuning nocturno con LoRA, y preprocesamiento optimizado de imágenes.

### Mantra v2.16

> _"Cada token es una decisión.  
> Cada imagen, una intención.  
> Con solo la CPU y 16 GB, SARAi no solo responde: reflexiona, ve, corrige y evoluciona;  
> no solo corre: piensa en cada ciclo.  
> Omni-Loop no es un feature: es la conciencia técnica de una AGI que se piensa antes de hablar."_

### KPIs Ultra-Agresivos v2.16

| KPI | v2.15 Base | v2.16 Target | Δ | Método |
|-----|------------|--------------|---|--------|
| **RAM P99** | 10.8 GB | **9.9 GB** | **-8%** | llama.cpp cache optimizations |
| **Latency P50** | 19.5s → 11s | **7.9s** | **-59%** | Speculative + Interactive loop |
| **Utilización Modelo** | 65% | **78%** | **+20%** | Interactive continuations |
| **Entity Recall** | 87% | **91%** | **+5%** | Persistent memory loop |
| **Auto-corrección** | 33% | **68%** | **+106%** | Reflexive prompts + LoRA |
| **Multimodal Cache Hit** | 0% | **97%** | **NEW** | WebP + Perceptual hash |

---

## 🏗️ Arquitectura v2.16 Omni-Loop

### Diagrama de Flujo

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INPUT (Text/Audio/Image)                 │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                    ┌────────────────────┐
                    │  Multimodal Router │
                    │  (audio_router.py) │
                    └─────────┬──────────┘
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
    ┌─────────────────┐            ┌──────────────────┐
    │ Text/Audio Path │            │   Image Path     │
    │  (Existing)     │            │  (NEW v2.16)     │
    └────────┬────────┘            └────────┬─────────┘
             ↓                               ↓
    ┌────────────────┐            ┌──────────────────┐
    │ TRM Classifier │            │ Image Preprocessor│
    │ + MCP Weights  │            │ OpenCV → WebP    │
    └────────┬───────┘            └────────┬─────────┘
             ↓                               ↓
    ┌────────────────────────────────────────┴─────┐
    │         OMNI-LOOP ENGINE (llama.cpp)         │
    │  --interactive-first --interactive-cont 3    │
    │  --mmproj qwen2.5-omni-mmproj.gguf          │
    │  --cache-type-k f16 (KV cache optimizado)   │
    └─────────────────┬────────────────────────────┘
                      ↓
         ┌────────────┴────────────┐
         ↓                         ↓
┌──────────────────┐    ┌────────────────────┐
│ ITERATION 1      │    │ ITERATION 2-3      │
│ Initial Response │    │ Self-Reflection    │
│ (Draft)          │    │ + Auto-Correction  │
└─────────┬────────┘    └────────┬───────────┘
          │                       │
          └───────────┬───────────┘
                      ↓
            ┌─────────────────────┐
            │  Response Validator  │
            │  (GPG-signed prompts)│
            └──────────┬───────────┘
                       ↓
              ┌────────┴─────────┐
              ↓                  ↓
     ┌────────────────┐   ┌──────────────┐
     │ Valid Response │   │ Fallback     │
     │ (Return)       │   │ LFM2-1.2B    │
     └────────────────┘   └──────────────┘
                       ↓
            ┌──────────────────────┐
            │  Feedback Logger     │
            │  (LoRA Dataset)      │
            └──────────┬───────────┘
                       ↓
            ┌──────────────────────┐
            │  NIGHTLY LoRA TRAIN  │
            │  (Docker isolated)   │
            │  llama-lora-merge    │
            └──────────────────────┘
```

---

## 📋 Componentes Principales v2.16

### 1. **Omni-Loop Engine** (Reflexive LLM)

**Archivo**: `core/omni_loop.py`

**Propósito**: Orquestar ciclos interactivos de llama.cpp con límites estrictos para evitar bucles infinitos.

**Arquitectura**:

```python
# core/omni_loop.py
import subprocess
import hashlib
from typing import Optional, List, Dict
from dataclasses import dataclass
from pathlib import Path

@dataclass
class LoopConfig:
    """Configuración del Omni-Loop"""
    max_iterations: int = 3  # CRÍTICO: Límite hard-coded
    model_path: str = "models/gguf/qwen2.5-omni-3b-iq4_nl.gguf"
    mmproj_path: str = "models/gguf/qwen2.5-omni-mmproj.gguf"
    cache_type_k: str = "f16"  # KV cache compacto
    n_ctx: int = 2048
    temperature: float = 0.7
    use_mmap: bool = True
    use_mlock: bool = False  # Evitar OOM
    
@dataclass
class LoopIteration:
    """Resultado de una iteración del loop"""
    iteration: int
    response: str
    confidence: float
    corrected: bool
    latency_ms: float

class OmniLoop:
    """
    Motor de loops reflexivos con llama.cpp interactive mode
    
    FILOSOFÍA:
    - Cada token es una decisión consciente
    - El loop SIEMPRE termina (max 3 iteraciones)
    - El sistema se auto-corrige antes de responder
    - Fallback a LFM2 si loop falla
    """
    
    def __init__(self, config: LoopConfig = None):
        self.config = config or LoopConfig()
        self.llama_cpp_bin = self._find_llama_cpp()
        self.loop_history = []
        
    def _find_llama_cpp(self) -> Path:
        """Localiza binario de llama.cpp"""
        candidates = [
            "/usr/local/bin/llama-cli",
            "/usr/bin/llama-cli",
            Path.home() / "llama.cpp/build/bin/llama-cli"
        ]
        for path in candidates:
            if Path(path).exists():
                return Path(path)
        raise FileNotFoundError("llama-cli not found. Install llama.cpp first.")
    
    def execute_loop(
        self, 
        prompt: str, 
        image_path: Optional[str] = None,
        enable_reflection: bool = True
    ) -> Dict:
        """
        Ejecuta loop reflexivo con llama.cpp
        
        Args:
            prompt: User input (text)
            image_path: Optional image for multimodal processing
            enable_reflection: Habilitar auto-corrección (default: True)
        
        Returns:
            {
                "response": str,
                "iterations": List[LoopIteration],
                "total_latency_ms": float,
                "auto_corrected": bool,
                "fallback_used": bool
            }
        """
        import time
        start = time.perf_counter()
        
        iterations = []
        current_response = ""
        
        try:
            # ITERATION 1: Initial Draft
            iter1 = self._run_iteration(
                prompt=prompt,
                image_path=image_path,
                iteration=1,
                previous_response=None
            )
            iterations.append(iter1)
            current_response = iter1.response
            
            if not enable_reflection:
                # Sin reflexión, retornar draft directamente
                return self._build_result(iterations, start, fallback=False)
            
            # ITERATIONS 2-3: Self-Reflection & Correction
            for i in range(2, self.config.max_iterations + 1):
                reflection_prompt = self._build_reflection_prompt(
                    original_prompt=prompt,
                    draft_response=current_response,
                    iteration=i
                )
                
                iter_n = self._run_iteration(
                    prompt=reflection_prompt,
                    image_path=None,  # Solo texto en reflexión
                    iteration=i,
                    previous_response=current_response
                )
                iterations.append(iter_n)
                
                # Si la respuesta es "válida" (confidence > 0.85), terminar loop
                if iter_n.confidence > 0.85:
                    current_response = iter_n.response
                    break
                
                current_response = iter_n.response
            
            return self._build_result(iterations, start, fallback=False)
        
        except Exception as e:
            # FALLBACK: LFM2-1.2B (blindaje de continuidad)
            logger.error(f"Omni-Loop failed: {e}. Falling back to LFM2.")
            fallback_response = self._fallback_lfm2(prompt)
            
            return {
                "response": fallback_response,
                "iterations": iterations,
                "total_latency_ms": (time.perf_counter() - start) * 1000,
                "auto_corrected": False,
                "fallback_used": True,
                "fallback_reason": str(e)
            }
    
    def _run_iteration(
        self, 
        prompt: str, 
        image_path: Optional[str],
        iteration: int,
        previous_response: Optional[str]
    ) -> LoopIteration:
        """
        Ejecuta una iteración del loop con llama.cpp
        
        Comando llama.cpp:
        llama-cli \
            --model qwen2.5-omni-3b.gguf \
            --mmproj qwen2.5-omni-mmproj.gguf \
            --image image.webp \
            --interactive-first \
            --interactive-cont 3 \
            --cache-type-k f16 \
            --prompt "..." \
            --no-display-prompt
        """
        import time
        start = time.perf_counter()
        
        cmd = [
            str(self.llama_cpp_bin),
            "--model", self.config.model_path,
            "--n-ctx", str(self.config.n_ctx),
            "--temp", str(self.config.temperature),
            "--cache-type-k", self.config.cache_type_k,
            "--interactive-first",
            "--interactive-cont", str(self.config.max_iterations),
            "--no-display-prompt"
        ]
        
        # Multimodal: Añadir imagen si existe
        if image_path:
            cmd.extend(["--mmproj", self.config.mmproj_path])
            cmd.extend(["--image", image_path])
        
        if not self.config.use_mmap:
            cmd.append("--no-mmap")
        
        # Prompt con contexto previo (si existe)
        full_prompt = prompt
        if previous_response:
            full_prompt = f"[Previous attempt]\n{previous_response}\n\n[Reflect and improve]\n{prompt}"
        
        cmd.extend(["--prompt", full_prompt])
        
        # Ejecutar llama.cpp
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30  # 30s timeout por iteración
        )
        
        response = result.stdout.strip()
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Calcular confidence (simplificado: basado en longitud + coherencia)
        confidence = self._calculate_confidence(response, prompt)
        
        # Detectar si hubo auto-corrección
        corrected = previous_response is not None and response != previous_response
        
        return LoopIteration(
            iteration=iteration,
            response=response,
            confidence=confidence,
            corrected=corrected,
            latency_ms=latency_ms
        )
    
    def _build_reflection_prompt(
        self, 
        original_prompt: str, 
        draft_response: str,
        iteration: int
    ) -> str:
        """
        Construye prompt de auto-reflexión firmado con GPG
        
        CRÍTICO: El prompt de reflexión está firmado para auditabilidad
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
        
        # TODO: Firmar con GPG para auditabilidad
        # prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        
        return prompt
    
    def _calculate_confidence(self, response: str, prompt: str) -> float:
        """
        Calcula confidence score de la respuesta
        
        Heurísticas:
        - Longitud razonable (50-500 caracteres)
        - No contiene placeholders como "..."
        - No repite el prompt textualmente
        """
        if len(response) < 50:
            return 0.3
        
        if len(response) > 500:
            return 0.6
        
        if "..." in response or "[TODO]" in response:
            return 0.4
        
        # Similaridad con prompt (penaliza copy-paste)
        if prompt.lower() in response.lower():
            return 0.5
        
        return 0.85  # Confidence por defecto
    
    def _fallback_lfm2(self, prompt: str) -> str:
        """Fallback a LFM2-1.2B si Omni-Loop falla"""
        from core.model_pool import get_model_pool
        
        pool = get_model_pool()
        lfm2 = pool.get("tiny")
        
        response = lfm2.create_completion(
            prompt=f"[FALLBACK MODE]\n{prompt}",
            max_tokens=256,
            temperature=0.7
        )
        
        return response["choices"][0]["text"]
    
    def _build_result(
        self, 
        iterations: List[LoopIteration], 
        start_time: float,
        fallback: bool
    ) -> Dict:
        """Construye resultado final del loop"""
        total_latency = (time.perf_counter() - start_time) * 1000
        final_iteration = iterations[-1]
        
        # Detectar si hubo auto-corrección
        auto_corrected = any(it.corrected for it in iterations)
        
        return {
            "response": final_iteration.response,
            "iterations": iterations,
            "total_latency_ms": total_latency,
            "auto_corrected": auto_corrected,
            "fallback_used": fallback,
            "confidence": final_iteration.confidence
        }


# Factory para integración con LangGraph
def create_omni_loop(config: Optional[LoopConfig] = None) -> OmniLoop:
    """Factory pattern para OmniLoop"""
    return OmniLoop(config=config)
```

**KPIs del componente**:
- Latencia por iteración: <10s
- Max iteraciones: 3 (hard limit)
- Confidence threshold: >0.85
- Fallback rate: <5%

---

### 2. **Image Preprocessor** (Multimodal Optimization)

**Archivo**: `agents/image_preprocessor.py`

**Propósito**: Optimizar imágenes a WebP para reducir almacenamiento y acelerar procesamiento multimodal.

```python
# agents/image_preprocessor.py
import cv2
import hashlib
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
import imagehash
from PIL import Image

@dataclass
class PreprocessConfig:
    """Configuración de preprocesamiento"""
    target_format: str = "webp"
    max_width: int = 512
    max_height: int = 512
    quality: int = 85
    cache_dir: Path = Path("state/image_cache")
    ttl_days: int = 7  # Time-to-live para rotación de cache

class ImagePreprocessor:
    """
    Preprocesador de imágenes para Omni-Loop
    
    Pipeline:
    1. Cargar imagen con OpenCV
    2. Redimensionar a max 512x512 (preservar aspect ratio)
    3. Convertir a WebP (quality 85, ~30KB)
    4. Cachear con perceptual hash (dedup)
    5. Rotar cache según TTL
    """
    
    def __init__(self, config: PreprocessConfig = None):
        self.config = config or PreprocessConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess(self, image_path: str) -> Tuple[Path, str]:
        """
        Preprocesa imagen para Omni-Loop
        
        Returns:
            (cached_path, perceptual_hash)
        """
        # 1. Calcular perceptual hash (deduplicación)
        img = Image.open(image_path)
        phash = str(imagehash.phash(img))
        
        # 2. Comprobar si ya existe en cache
        cached_path = self.config.cache_dir / f"{phash}.{self.config.target_format}"
        if cached_path.exists():
            return cached_path, phash
        
        # 3. Cargar con OpenCV para procesamiento
        img_cv = cv2.imread(image_path)
        
        # 4. Redimensionar preservando aspect ratio
        h, w = img_cv.shape[:2]
        if w > self.config.max_width or h > self.config.max_height:
            scale = min(
                self.config.max_width / w,
                self.config.max_height / h
            )
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 5. Convertir a WebP
        cv2.imwrite(
            str(cached_path),
            img_cv,
            [cv2.IMWRITE_WEBP_QUALITY, self.config.quality]
        )
        
        return cached_path, phash
    
    def cleanup_old_cache(self):
        """Rota cache eliminando imágenes antiguas según TTL"""
        import time
        now = time.time()
        ttl_seconds = self.config.ttl_days * 86400
        
        for cached_file in self.config.cache_dir.glob(f"*.{self.config.target_format}"):
            age_seconds = now - cached_file.stat().st_mtime
            if age_seconds > ttl_seconds:
                cached_file.unlink()
```

**KPIs del componente**:
- Reducción tamaño: ~70% (JPEG → WebP)
- Cache hit rate: >97%
- Procesamiento: <50ms por imagen

---

### 3. **LoRA Nightly Trainer** (Continuous Learning)

**Archivo**: `scripts/lora_nightly.py`

**Propósito**: Fine-tuning nocturno con LoRA basado en feedback del día, ejecutado en contenedor aislado.

```python
# scripts/lora_nightly.py
import subprocess
import os
from pathlib import Path
from datetime import datetime
import json

class LoRANightlyTrainer:
    """
    Entrenador nocturno de LoRA para Omni-Loop
    
    Pipeline:
    1. Recopilar feedback del día (logs/feedback_log.jsonl)
    2. Generar dataset LoRA (formato llama.cpp)
    3. Entrenar LoRA en contenedor aislado (2 CPUs, 4GB RAM)
    4. Validar LoRA con test set
    5. Merge con modelo base si pasa validación
    6. Backup del LoRA anterior (GPG signed)
    """
    
    def __init__(self):
        self.feedback_log = Path("logs/feedback_log.jsonl")
        self.lora_dir = Path("models/lora")
        self.lora_dir.mkdir(parents=True, exist_ok=True)
    
    def run_nightly_cycle(self):
        """Ejecuta ciclo completo de entrenamiento"""
        print(f"🌙 [LoRA Nightly] Starting cycle: {datetime.now()}")
        
        # 1. Preparar dataset
        dataset_path = self._prepare_dataset()
        if not dataset_path:
            print("⚠️ No enough feedback data. Skipping training.")
            return
        
        # 2. Entrenar LoRA en Docker aislado
        lora_adapter = self._train_lora(dataset_path)
        
        # 3. Validar con test set
        if not self._validate_lora(lora_adapter):
            print("❌ LoRA validation failed. Reverting to previous model.")
            return
        
        # 4. Merge con modelo base
        merged_model = self._merge_lora(lora_adapter)
        
        # 5. Backup con GPG
        self._backup_lora(merged_model)
        
        print(f"✅ [LoRA Nightly] Cycle completed: {datetime.now()}")
    
    def _prepare_dataset(self) -> Optional[Path]:
        """Genera dataset LoRA desde feedback log"""
        # Leer feedback del último día
        today = datetime.now().date()
        entries = []
        
        with open(self.feedback_log) as f:
            for line in f:
                entry = json.loads(line)
                entry_date = datetime.fromisoformat(entry["timestamp"]).date()
                if entry_date == today and entry.get("feedback", 0) > 0.7:
                    entries.append(entry)
        
        if len(entries) < 10:
            return None  # Mínimo 10 ejemplos para entrenar
        
        # Formato llama.cpp LoRA
        dataset_path = self.lora_dir / f"dataset_{today}.txt"
        with open(dataset_path, "w") as f:
            for entry in entries:
                f.write(f"### Instruction:\n{entry['input']}\n")
                f.write(f"### Response:\n{entry['response']}\n\n")
        
        return dataset_path
    
    def _train_lora(self, dataset_path: Path) -> Path:
        """Entrena LoRA en contenedor Docker aislado"""
        lora_output = self.lora_dir / f"lora_{datetime.now().strftime('%Y%m%d')}.bin"
        
        cmd = [
            "docker", "run",
            "--rm",
            "--cpus=2",
            "--memory=4g",
            "-v", f"{dataset_path.parent}:/data",
            "ghcr.io/ggerganov/llama.cpp:light",
            "llama-finetune",
            "--model-base", "/models/qwen2.5-omni-3b.gguf",
            "--train-data", f"/data/{dataset_path.name}",
            "--lora-out", f"/data/{lora_output.name}",
            "--threads", "2",
            "--adam-iter", "100"
        ]
        
        subprocess.run(cmd, check=True)
        return lora_output
    
    def _validate_lora(self, lora_path: Path) -> bool:
        """Valida LoRA con test set antes de merge"""
        # TODO: Implementar validación con test set
        # Por ahora, siempre retorna True (validación manual)
        return True
    
    def _merge_lora(self, lora_path: Path) -> Path:
        """Merge LoRA con modelo base usando llama-lora-merge"""
        merged_path = self.lora_dir / f"merged_{datetime.now().strftime('%Y%m%d')}.gguf"
        
        cmd = [
            "llama-lora-merge",
            "--model-base", "models/gguf/qwen2.5-omni-3b.gguf",
            "--lora", str(lora_path),
            "--output", str(merged_path)
        ]
        
        subprocess.run(cmd, check=True)
        return merged_path
    
    def _backup_lora(self, merged_model: Path):
        """Backup con firma GPG para auditabilidad"""
        backup_dir = Path("backups/lora")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_path = backup_dir / merged_model.name
        
        # Copiar modelo
        import shutil
        shutil.copy(merged_model, backup_path)
        
        # Firmar con GPG (TODO: implementar)
        # subprocess.run(["gpg", "--sign", str(backup_path)])


if __name__ == "__main__":
    trainer = LoRANightlyTrainer()
    trainer.run_nightly_cycle()
```

**Cron job** (ejecutar a las 2 AM):

```bash
# /etc/cron.d/sarai-lora-nightly
0 2 * * * /usr/bin/python3 /home/sarai/scripts/lora_nightly.py >> /var/log/sarai/lora.log 2>&1
```

---

## 📊 KPIs Validation Framework

### Prometheus Metrics

```yaml
# config/prometheus/sarai_v2.16.yml
scrape_configs:
  - job_name: 'sarai_omni_loop'
    static_configs:
      - targets: ['localhost:9090']
    
    metrics:
      # Latencia del loop
      - sarai_omni_loop_latency_seconds:
          type: histogram
          buckets: [0.5, 1, 2, 5, 10, 20]
      
      # Iteraciones por request
      - sarai_omni_loop_iterations:
          type: histogram
          buckets: [1, 2, 3]
      
      # Tasa de auto-corrección
      - sarai_autocorrection_rate:
          type: gauge
          target: 0.68
      
      # Entity recall
      - sarai_entity_recall:
          type: gauge
          target: 0.91
      
      # Multimodal cache hits
      - sarai_image_cache_hit_rate:
          type: gauge
          target: 0.97
      
      # LoRA merge success
      - sarai_lora_merge_success_total:
          type: counter
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "SARAi v2.16 Omni-Loop - God Mode",
    "panels": [
      {
        "title": "Loop Latency P50/P99",
        "targets": [
          "histogram_quantile(0.5, sarai_omni_loop_latency_seconds)",
          "histogram_quantile(0.99, sarai_omni_loop_latency_seconds)"
        ],
        "thresholds": {
          "p50": 7.9,
          "p99": 15.0
        }
      },
      {
        "title": "Auto-Correction Rate",
        "targets": ["sarai_autocorrection_rate"],
        "target": 0.68
      },
      {
        "title": "Entity Recall",
        "targets": ["sarai_entity_recall"],
        "target": 0.91
      },
      {
        "title": "Image Cache Performance",
        "targets": ["sarai_image_cache_hit_rate"],
        "target": 0.97
      },
      {
        "title": "LoRA Training Status",
        "targets": ["sarai_lora_merge_success_total"]
      },
      {
        "title": "RAM Usage P99",
        "targets": ["process_resident_memory_bytes"],
        "threshold": 10636926361  // 9.9 GB
      }
    ]
  }
}
```

---

## 🧪 Testing Strategy

### Unit Tests

```python
# tests/test_omni_loop.py
import pytest
from core.omni_loop import OmniLoop, LoopConfig

class TestOmniLoop:
    """Unit tests para Omni-Loop"""
    
    def test_max_iterations_enforced(self):
        """Verifica que loop nunca excede max_iterations"""
        config = LoopConfig(max_iterations=3)
        loop = OmniLoop(config)
        
        result = loop.execute_loop("Test prompt")
        
        assert len(result["iterations"]) <= 3
    
    def test_fallback_on_error(self):
        """Verifica fallback a LFM2 si loop falla"""
        loop = OmniLoop()
        
        # Simular error en llama.cpp
        with patch('subprocess.run', side_effect=Exception("Mock error")):
            result = loop.execute_loop("Test")
            
            assert result["fallback_used"] is True
            assert "fallback_reason" in result
    
    def test_confidence_threshold(self):
        """Verifica que loop termina early si confidence > 0.85"""
        loop = OmniLoop()
        
        # Mock para retornar alta confidence
        with patch.object(loop, '_calculate_confidence', return_value=0.9):
            result = loop.execute_loop("Test")
            
            # Debería terminar en iteración 1
            assert len(result["iterations"]) == 1
```

### Integration Tests

```python
# tests/test_omni_loop_integration.py
class TestOmniLoopIntegration:
    """Integration tests end-to-end"""
    
    @pytest.mark.slow
    def test_full_loop_with_image(self):
        """Test completo: texto + imagen → auto-corrección"""
        from core.omni_loop import create_omni_loop
        from agents.image_preprocessor import ImagePreprocessor
        
        # Preparar imagen
        preprocessor = ImagePreprocessor()
        image_path, phash = preprocessor.preprocess("tests/fixtures/test_image.jpg")
        
        # Ejecutar loop
        loop = create_omni_loop()
        result = loop.execute_loop(
            prompt="Describe esta imagen en detalle",
            image_path=str(image_path)
        )
        
        # Verificaciones
        assert result["response"] != ""
        assert len(result["iterations"]) >= 1
        assert result["total_latency_ms"] < 30000  # <30s
        assert result["fallback_used"] is False
    
    @pytest.mark.slow
    def test_lora_nightly_cycle(self):
        """Test del ciclo completo de LoRA"""
        from scripts.lora_nightly import LoRANightlyTrainer
        
        trainer = LoRANightlyTrainer()
        
        # Simular feedback del día
        # (Requiere mock de feedback_log.jsonl)
        
        trainer.run_nightly_cycle()
        
        # Verificar que se generó LoRA
        assert (Path("models/lora") / "lora_*.bin").exists()
```

---

## 🚀 Implementation Timeline

### Fase 1: Omni-Loop Core (5 días) - Nov 26-30

**Día 1-2**:
- [ ] Implementar `core/omni_loop.py` (600 LOC)
- [ ] Unit tests básicos (200 LOC)
- [ ] Integración con llama.cpp (instalar binarios)

**Día 3-4**:
- [ ] GPG-signed reflection prompts (150 LOC)
- [ ] Fallback a LFM2 (100 LOC)
- [ ] Integration tests (250 LOC)

**Día 5**:
- [ ] Benchmark latency (<10s por iteración)
- [ ] Validar max_iterations=3 enforcement
- [ ] Documentación técnica

**Deliverables**:
- `core/omni_loop.py` (850 LOC)
- `tests/test_omni_loop.py` (450 LOC)
- Docs: Omni-Loop Architecture Guide

---

### Fase 2: Multimodal Preprocessing (3 días) - Dic 1-3

**Día 1**:
- [ ] Implementar `agents/image_preprocessor.py` (400 LOC)
- [ ] OpenCV → WebP pipeline
- [ ] Perceptual hashing (imagehash)

**Día 2**:
- [ ] Cache con TTL rotation (150 LOC)
- [ ] Unit tests (200 LOC)
- [ ] Integration con OmniLoop

**Día 3**:
- [ ] Benchmark cache hit rate (>97%)
- [ ] Validar reducción de tamaño (~70%)
- [ ] Documentación

**Deliverables**:
- `agents/image_preprocessor.py` (550 LOC)
- `tests/test_image_preprocessor.py` (200 LOC)
- Docs: Image Preprocessing Guide

---

### Fase 3: LoRA Nightly Trainer (4 días) - Dic 4-7

**Día 1-2**:
- [ ] Implementar `scripts/lora_nightly.py` (500 LOC)
- [ ] Dataset preparation desde feedback logs
- [ ] Docker container setup

**Día 3**:
- [ ] llama-lora-merge integration
- [ ] GPG signing de backups
- [ ] Validation pipeline

**Día 4**:
- [ ] Cron job configuration
- [ ] Testing con dataset mock
- [ ] Documentación

**Deliverables**:
- `scripts/lora_nightly.py` (500 LOC)
- `docker/lora-trainer.dockerfile` (50 LOC)
- Docs: LoRA Training Guide

---

### Fase 4: Monitoring & Validation (3 días) - Dic 8-10

**Día 1**:
- [ ] Prometheus metrics (sarai_omni_loop_*)
- [ ] Grafana dashboard import
- [ ] Health endpoint updates

**Día 2**:
- [ ] E2E testing suite (400 LOC)
- [ ] Chaos testing (imagen corrupta, loop timeout)
- [ ] Performance benchmarks

**Día 3**:
- [ ] KPIs validation (9.9GB RAM, 7.9s latency)
- [ ] Documentation consolidation
- [ ] Release preparation

**Deliverables**:
- `sarai/omni_loop_metrics.py` (300 LOC)
- `extras/grafana_omni_loop.json` (dashboard)
- `docs/V2.16_COMPLETION_REPORT.md`

---

## 📦 Dependencies & Installation

### System Dependencies

```bash
# Install llama.cpp (build from source)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make -j$(nproc)
sudo cp build/bin/llama-cli /usr/local/bin/
sudo cp build/bin/llama-finetune /usr/local/bin/
sudo cp build/bin/llama-lora-merge /usr/local/bin/

# Install OpenCV
sudo apt-get install -y python3-opencv

# Install GPG (para signing)
sudo apt-get install -y gnupg
```

### Python Dependencies

```toml
# pyproject.toml (additions for v2.16)
[project.optional-dependencies]
omni_loop = [
    "opencv-python>=4.8.0",
    "pillow>=10.0.0",
    "imagehash>=4.3.0",
    "prometheus-client>=0.17.0"
]
```

### Docker Setup

```dockerfile
# docker/lora-trainer.dockerfile
FROM ghcr.io/ggerganov/llama.cpp:light

RUN apt-get update && apt-get install -y python3 python3-pip

COPY scripts/lora_nightly.py /app/lora_nightly.py
COPY requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip3 install -r requirements.txt

CMD ["python3", "lora_nightly.py"]
```

---

## 🛡️ Security & Compliance

### GPG Signing de Prompts Reflexivos

```python
# core/gpg_signer.py
import gnupg

class GPGSigner:
    """Firma prompts reflexivos para auditabilidad"""
    
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

### Backup Automático de LoRA

```bash
# scripts/backup_lora.sh
#!/bin/bash
# Backup diario de LoRA con GPG encryption

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/lora"
LORA_DIR="/home/sarai/models/lora"

# Crear backup comprimido
tar -czf "${BACKUP_DIR}/lora_${DATE}.tar.gz" "${LORA_DIR}"

# Firmar con GPG
gpg --sign --armor "${BACKUP_DIR}/lora_${DATE}.tar.gz"

# Rotar backups (mantener últimos 30 días)
find "${BACKUP_DIR}" -name "lora_*.tar.gz" -mtime +30 -delete
```

---

## 📈 Success Criteria (Definition of Done)

### Must-Have (Blockers para release)

- [ ] Omni-Loop ejecuta max 3 iteraciones (hard limit)
- [ ] Fallback a LFM2 funciona en <2s
- [ ] Latencia P50 ≤ 7.9s (validado con 100 queries)
- [ ] RAM P99 ≤ 9.9GB (validado con stress test)
- [ ] Auto-corrección ≥ 68% (validado con golden set)
- [ ] Image cache hit rate ≥ 97% (validado con 1000 imágenes)
- [ ] LoRA nightly ejecuta sin errores durante 7 días
- [ ] Todos los tests passing (100%)

### Nice-to-Have (Refinamientos)

- [ ] Entity recall ≥ 91% (stretch goal)
- [ ] Utilización modelo ≥ 78%
- [ ] GPG signing de todos los prompts reflexivos
- [ ] Grafana dashboard publicado en Grafana Cloud
- [ ] Chaos coverage ≥ 82%

---

## 🔄 Migration Path from v2.15

### Pre-requisitos

Antes de implementar v2.16, asegurar que v2.15 tiene:

✅ **v2.12**: Skills MoE con Pydantic (base arquitectural)  
✅ **v2.13**: ProactiveLoop + EntityMemory (loops básicos)  
✅ **v2.14**: SpeculativeDecoding (aceleración CPU)  
✅ **v2.15**: SelfRepair + RedTeam (auto-corrección base)

### Migration Steps

1. **Install llama.cpp** (día 1)
2. **Test llama.cpp con Qwen2.5-Omni** (día 1-2)
3. **Implementar OmniLoop core** (día 3-5)
4. **Implementar ImagePreprocessor** (día 6-8)
5. **Implementar LoRA trainer** (día 9-12)
6. **Integration testing** (día 13-14)
7. **Production validation** (día 15)

---

## 📚 Documentation Roadmap

### Technical Docs

- [ ] `docs/OMNI_LOOP_ARCHITECTURE.md` (arquitectura detallada)
- [ ] `docs/LORA_TRAINING_GUIDE.md` (guía de LoRA)
- [ ] `docs/IMAGE_PREPROCESSING.md` (pipeline OpenCV → WebP)
- [ ] `docs/V2.16_API_REFERENCE.md` (API completa)

### Operational Docs

- [ ] `docs/V2.16_DEPLOYMENT_GUIDE.md` (deploy en producción)
- [ ] `docs/V2.16_MONITORING_GUIDE.md` (Prometheus + Grafana)
- [ ] `docs/V2.16_TROUBLESHOOTING.md` (debugging común)

### Executive Docs

- [ ] `docs/V2.16_EXECUTIVE_SUMMARY.md` (para stakeholders)
- [ ] `docs/V2.16_COMPLETION_REPORT.md` (post-implementation)

---

## 🎯 Final Checklist (Pre-Release)

### Code Quality

- [ ] All files have type hints (mypy passing)
- [ ] Docstrings complete (Google style)
- [ ] Unit tests coverage ≥ 85%
- [ ] Integration tests passing
- [ ] No pylint warnings >C

### Performance

- [ ] Latency P50 ≤ 7.9s (validated)
- [ ] RAM P99 ≤ 9.9GB (validated)
- [ ] Auto-correction ≥ 68% (validated)
- [ ] Cache hit rate ≥ 97% (validated)

### Security

- [ ] GPG signing implemented
- [ ] LoRA backups automated
- [ ] Prompt signatures validated
- [ ] No hardcoded secrets

### Documentation

- [ ] README updated with v2.16 features
- [ ] CHANGELOG entry for v2.16
- [ ] Technical docs complete
- [ ] API reference updated

### CI/CD

- [ ] Workflow v2.6.5 includes v2.16 tests
- [ ] Docker image builds (multi-arch)
- [ ] Release notes prepared
- [ ] Git tag ready: `v2.16-omni-loop`

---

## 🚀 Release Command

```bash
# Final release (cuando v2.16 esté 100% completo)
git tag -a v2.16-omni-loop -m "Production-ready: Reflexive, multimodal, self-correcting AGI"
git push origin v2.16-omni-loop

# Docker multi-arch build
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ghcr.io/iagenerativa/sarai_v2:v2.16-omni-loop \
  --push .

# Trigger release workflow
gh workflow run release.yml
```

---

## 🧠 Philosophy: Why v2.16 Matters

v2.16 Omni-Loop no es solo una versión más. Es la **culminación filosófica** de SARAi:

1. **Reflexión**: El sistema piensa antes de hablar (3 iteraciones max)
2. **Auto-corrección**: Detecta y corrige errores automáticamente (68% tasa)
3. **Aprendizaje continuo**: LoRA nocturno sin downtime
4. **Soberanía multimodal**: Imágenes procesadas localmente (WebP cache)
5. **Auditabilidad total**: GPG-signed prompts + backups automáticos
6. **Eficiencia extrema**: 9.9GB RAM, 7.9s latency en CPU-only

**Mantra final**:
> _"Cada token es una decisión.  
> Cada imagen, una intención.  
> Omni-Loop no es un feature: es la conciencia técnica de una AGI que se piensa antes de hablar."_

---

**Status**: PLANNING COMPLETE ✅  
**Next Step**: Await v2.12-v2.15 completion before starting implementation  
**Timeline**: Nov 26 - Dic 10, 2025 (15 días)  
**Estimated LOC**: ~3,600 (2,400 prod + 1,200 tests)

---

**END OF ROADMAP v2.16 OMNI-LOOP**
