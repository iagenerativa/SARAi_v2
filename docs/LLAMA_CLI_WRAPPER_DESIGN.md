# Diseño: Wrapper Python para llama-cpp-bin

**Versión**: v2.14 Pre-FASE 3  
**Fecha**: 2025-01-XX  
**Propósito**: Reemplazar `llama-cpp-python` con llamadas al binario `llama-cli` del contenedor Docker personalizado

---

## 🎯 Objetivo

Crear un wrapper Python que:
1. ✅ **Llame a `llama-cli`** del contenedor Docker `llama-cpp-bin`
2. ✅ **Mantenga compatibilidad** con la interfaz de `llama_cpp.Llama`
3. ✅ **Siga patrones LangChain** (StateGraph para orquestación)
4. ✅ **Evite código spaghetti** (separación clara de responsabilidades)
5. ✅ **Soporte n_ctx dinámico** (context-aware como en model_pool actual)
6. ✅ **Timeout adaptativo** basado en n_ctx (v2.16 Risk #5)

---

## 📐 Arquitectura

### Componente 1: `core/llama_cli_wrapper.py`

**Clase principal**: `LlamaCLIWrapper`

```python
"""
Wrapper Python para llama-cli (llama.cpp binario)

Reemplaza llama_cpp.Llama manteniendo compatibilidad de interfaz.
Usa el contenedor Docker llama-cpp-bin personalizado.

Filosofía:
- Llamadas via subprocess al binario llama-cli
- Timeout dinámico según n_ctx (Risk #5)
- Fallback a llama-cpp-python si Docker no disponible
- JSON para comunicación (entrada/salida estándar)
"""

from typing import Dict, Any, Optional, List, Union
import subprocess
import json
import os
from pathlib import Path


class LlamaCLIWrapper:
    """
    Wrapper compatible con llama_cpp.Llama
    
    Uso:
        llm = LlamaCLIWrapper(
            model_path="/models/gguf/solar.gguf",
            n_ctx=2048,
            n_threads=6
        )
        
        response = llm(
            prompt="¿Qué es Python?",
            max_tokens=150,
            temperature=0.7
        )
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        use_mmap: bool = True,
        use_mlock: bool = False,
        verbose: bool = False,
        docker_image: str = "ghcr.io/iagenerativa/llama-cpp-bin:2.16-rc"
    ):
        """
        Args:
            model_path: Ruta al archivo GGUF
            n_ctx: Tamaño del contexto
            n_threads: Threads para inferencia
            use_mmap: Usar memory mapping
            use_mlock: Lock de memoria (no recomendado >12GB RAM)
            verbose: Logging detallado
            docker_image: Imagen Docker con llama-cli
        """
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.verbose = verbose
        self.docker_image = docker_image
        
        # Timeout dinámico según n_ctx (v2.16 Risk #5)
        self.timeout = self._calculate_timeout(n_ctx)
        
        # Verificar disponibilidad de Docker
        self.docker_available = self._check_docker()
        
        if not self.docker_available and verbose:
            print(f"⚠️  Docker no disponible. Fallback a llama-cpp-python")
    
    def _calculate_timeout(self, n_ctx: int) -> int:
        """
        Timeout dinámico basado en contexto
        
        Fórmula: timeout = 10s + (n_ctx / 1024) * 10s
        Max: 60s
        """
        base_timeout = 10
        scaling_factor = 10
        timeout = base_timeout + (n_ctx / 1024) * scaling_factor
        return min(int(timeout), 60)
    
    def _check_docker(self) -> bool:
        """Verifica si Docker está disponible"""
        try:
            result = subprocess.run(
                ["docker", "ps"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def __call__(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Genera respuesta (interfaz compatible con llama_cpp.Llama)
        
        Returns:
            {
                "choices": [{"text": "respuesta generada"}]
            }
        """
        if self.docker_available:
            return self._call_docker(prompt, max_tokens, temperature, top_p, stop)
        else:
            return self._call_fallback(prompt, max_tokens, temperature, top_p, stop)
    
    def _call_docker(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Llamada al binario llama-cli via Docker
        
        Comando:
            docker run --rm \
                -v /path/to/models:/models:ro \
                ghcr.io/iagenerativa/llama-cpp-bin:2.16-rc \
                llama-cli \
                    --model /models/solar.gguf \
                    --ctx-size 2048 \
                    --threads 6 \
                    --temp 0.7 \
                    --top-p 0.95 \
                    --n-predict 150 \
                    --prompt "texto"
        """
        # Construir comando
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{self.model_path.parent}:/models:ro",
            self.docker_image,
            "llama-cli",
            "--model", f"/models/{self.model_path.name}",
            "--ctx-size", str(self.n_ctx),
            "--threads", str(self.n_threads),
            "--temp", str(temperature),
            "--top-p", str(top_p),
            "--n-predict", str(max_tokens),
            "--prompt", prompt
        ]
        
        # Agregar flags opcionales
        if not self.use_mmap:
            cmd.append("--no-mmap")
        
        if self.use_mlock:
            cmd.append("--mlock")
        
        if stop:
            for stop_seq in stop:
                cmd.extend(["--reverse-prompt", stop_seq])
        
        # Ejecutar con timeout
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"llama-cli error: {result.stderr}")
            
            # Parsear salida
            # llama-cli genera texto directo, no JSON
            # Necesitamos extraer solo la respuesta
            response_text = self._extract_response(result.stdout, prompt)
            
            return {
                "choices": [{"text": response_text}]
            }
        
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"llama-cli timeout después de {self.timeout}s")
    
    def _extract_response(self, stdout: str, prompt: str) -> str:
        """
        Extrae respuesta del output de llama-cli
        
        llama-cli devuelve:
            [metadata de carga]
            [prompt completo]
            [respuesta generada]
            [estadísticas]
        
        Necesitamos solo la respuesta.
        """
        # Estrategia: Buscar el prompt en el output y tomar todo después
        if prompt in stdout:
            parts = stdout.split(prompt, 1)
            if len(parts) > 1:
                response = parts[1].strip()
                
                # Eliminar estadísticas finales (líneas que empiezan con "llama_")
                lines = response.split('\n')
                clean_lines = [l for l in lines if not l.strip().startswith('llama_')]
                
                return '\n'.join(clean_lines).strip()
        
        # Fallback: devolver todo el stdout si no encontramos el prompt
        return stdout.strip()
    
    def _call_fallback(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Fallback a llama-cpp-python si Docker no disponible
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "Docker no disponible y llama-cpp-python no instalado. "
                "Instala uno de los dos."
            )
        
        llm = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            use_mmap=self.use_mmap,
            use_mlock=self.use_mlock,
            verbose=self.verbose
        )
        
        return llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )
    
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.8,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat completion (formato OpenAI-compatible)
        
        Args:
            messages: Lista de mensajes [{"role": "user", "content": "..."}]
        
        Returns:
            {
                "choices": [{
                    "message": {"role": "assistant", "content": "respuesta"}
                }]
            }
        """
        # Convertir mensajes a prompt plano
        prompt = self._messages_to_prompt(messages)
        
        # Llamar a generación normal
        response = self(prompt, max_tokens, temperature, **kwargs)
        
        # Convertir a formato chat
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response["choices"][0]["text"]
                }
            }]
        }
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convierte mensajes de chat a prompt plano
        
        Ejemplo:
            [
                {"role": "system", "content": "Eres un asistente."},
                {"role": "user", "content": "Hola"}
            ]
            
            → "Eres un asistente.\n\nUser: Hola\nAssistant:"
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(content)
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Agregar prompt final para asistente
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
```

---

## 🔧 Integración con `model_pool.py`

**Modificación**: `_load_gguf_cpu()` en `core/model_pool.py`

```python
def _load_gguf_cpu(
    self,
    model_cfg: Dict[str, Any],
    context_length: Optional[int] = None,
    prefetch: bool = False
) -> Any:
    """
    Carga GGUF con llama-cli wrapper (v2.14+)
    
    Cambios v2.14:
    - Usa LlamaCLIWrapper en lugar de llama_cpp.Llama
    - Fallback automático si Docker no disponible
    """
    from core.llama_cli_wrapper import LlamaCLIWrapper
    
    # Descargar archivo GGUF
    gguf_file = model_cfg.get('gguf_file')
    model_path = hf_hub_download(
        repo_id=model_cfg['repo_id'],
        filename=gguf_file,
        cache_dir=model_cfg.get('cache_dir', './models/cache')
    )
    
    # Determinar n_ctx
    n_ctx = context_length if context_length is not None else model_cfg.get('context_length', 2048)
    
    # Determinar n_threads
    runtime_cfg = self.config.get('runtime', {})
    if prefetch:
        n_threads = 1
    else:
        n_threads = runtime_cfg.get('n_threads', max(1, os.cpu_count() - 2))
    
    # Configuración de memoria
    memory_cfg = self.config.get('memory', {})
    use_mmap = memory_cfg.get('use_mmap', True)
    use_mlock = memory_cfg.get('use_mlock', False)
    
    print(f"[ModelPool] Cargando GGUF via llama-cli wrapper: n_ctx={n_ctx}")
    
    # ✅ NUEVO v2.14: Usar wrapper personalizado
    return LlamaCLIWrapper(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        use_mmap=use_mmap,
        use_mlock=use_mlock,
        verbose=False
    )
```

---

## 🧪 Tests de Validación

**Archivo**: `tests/test_llama_cli_wrapper.py`

```python
"""
Tests para LlamaCLIWrapper

Validación:
1. Interfaz compatible con llama_cpp.Llama
2. Timeout dinámico funciona
3. Fallback a llama-cpp-python si Docker no disponible
4. Chat completion formato OpenAI
"""

import pytest
from core.llama_cli_wrapper import LlamaCLIWrapper


def test_wrapper_initialization():
    """Test: Inicialización del wrapper"""
    wrapper = LlamaCLIWrapper(
        model_path="models/gguf/test.gguf",
        n_ctx=512,
        n_threads=2
    )
    
    assert wrapper.n_ctx == 512
    assert wrapper.timeout == 15  # 10 + (512/1024)*10 = 15s


def test_timeout_calculation():
    """Test: Cálculo de timeout dinámico"""
    wrapper_small = LlamaCLIWrapper(model_path="test.gguf", n_ctx=512)
    assert wrapper_small.timeout == 15
    
    wrapper_medium = LlamaCLIWrapper(model_path="test.gguf", n_ctx=2048)
    assert wrapper_medium.timeout == 30
    
    wrapper_large = LlamaCLIWrapper(model_path="test.gguf", n_ctx=8192)
    assert wrapper_large.timeout == 60  # Max cap


@pytest.mark.skipif(not docker_available(), reason="Docker no disponible")
def test_docker_call():
    """Test: Llamada via Docker funciona"""
    wrapper = LlamaCLIWrapper(
        model_path="models/gguf/lfm2.gguf",
        n_ctx=512
    )
    
    response = wrapper(
        prompt="¿Qué es Python?",
        max_tokens=50,
        temperature=0.7
    )
    
    assert "choices" in response
    assert len(response["choices"]) > 0
    assert "text" in response["choices"][0]


def test_fallback_to_python():
    """Test: Fallback a llama-cpp-python si Docker no disponible"""
    wrapper = LlamaCLIWrapper(
        model_path="models/gguf/lfm2.gguf",
        n_ctx=512,
        docker_image="invalid-image-name"  # Forzar fallback
    )
    
    # Debe funcionar con fallback
    response = wrapper(
        prompt="Test",
        max_tokens=10
    )
    
    assert "choices" in response
```

---

## 📊 Beneficios vs. llama-cpp-python directo

| Aspecto | llama-cpp-python directo | LlamaCLIWrapper |
|---------|-------------------------|-----------------|
| **Binarios** | Python wheels genéricos | Compilados custom (AVX2/AVX512/NEON) |
| **Tamaño** | ~100-200 MB | ~18 MB (UPX compressed) |
| **Control** | Dependencias pip | Contenedor versionado |
| **Auditoría** | Difícil (black box) | SHA-256 checksums + GPG |
| **Multi-arch** | Wheels limitados | amd64 + arm64 nativos |
| **Portabilidad** | Requiere compilación | Docker portable |
| **Fallback** | N/A | Automático a Python si Docker falla |

---

## 🚀 Implementación FASE 3

### Paso 1: Crear wrapper (2h)
- `core/llama_cli_wrapper.py` con clase completa
- Tests en `tests/test_llama_cli_wrapper.py`
- Validar timeout dinámico

### Paso 2: Integrar en model_pool (1h)
- Modificar `_load_gguf_cpu()` para usar wrapper
- Preservar fallback a llama-cpp-python
- Tests de regresión

### Paso 3: Validar end-to-end (1h)
- Ejecutar SARAi con wrapper
- Medir latencia (debe ser ≤ latencia actual)
- Validar RAM (no debe aumentar)

### Paso 4: Documentar (30min)
- Actualizar copilot-instructions.md
- Agregar sección "llama-cpp-bin Wrapper Usage"
- KPIs de latencia/RAM

**Total estimado**: 4.5h (solo wrapper, antes del sandbox completo)

---

## ✅ Checklist de Implementación

- [ ] Crear `core/llama_cli_wrapper.py`
- [ ] Implementar `LlamaCLIWrapper.__call__()`
- [ ] Implementar `LlamaCLIWrapper.create_chat_completion()`
- [ ] Agregar timeout dinámico `_calculate_timeout()`
- [ ] Implementar fallback a llama-cpp-python
- [ ] Crear tests en `tests/test_llama_cli_wrapper.py`
- [ ] Modificar `model_pool.py` para usar wrapper
- [ ] Ejecutar tests de regresión
- [ ] Validar latencia end-to-end (≤ baseline)
- [ ] Validar RAM (≤ 12 GB P99)
- [ ] Documentar en copilot-instructions.md

---

**Mantra v2.14**:  
_"El wrapper es transparente. El código no debe saber si llama a Docker o Python.  
La abstracción perfecta es invisible."_
