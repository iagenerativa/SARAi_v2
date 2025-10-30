# SARAi v2.18 - Arquitectura TRUE Full-Duplex

## 🚨 Problema Fundamental v2.17

### Threading NO es Paralelismo Real

```python
# ❌ FALSO paralelismo (v2.17)
import threading

input_thread = threading.Thread(target=capture_audio)   # Thread 1
output_thread = threading.Thread(target=play_audio)     # Thread 2

# PROBLEMA: Python GIL (Global Interpreter Lock)
# - Solo 1 thread ejecuta código Python a la vez
# - Los threads se alternan cada 5ms
# - NO hay ejecución simultánea REAL
# - threading.Event() BLOQUEA ambos threads
```

**Resultado**: Sistema con "turnos invisibles"
- Usuario habla → STT espera → LLM procesa → TTS espera → Audio reproduce
- **TODO EN SERIE**, aunque parezca paralelo

---

## ✅ Solución v2.18: Multiprocessing

### Arquitectura de 3 Procesos Independientes

```
┌─────────────────────────────────────────────────────────────┐
│                    PROCESO 1: AudioEngine                   │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │  Micrófono   │ ──────> │ Input Buffer │ ──┐             │
│  │  (PortAudio) │         │   (Queue)    │   │             │
│  └──────────────┘         └──────────────┘   │             │
│                                               │             │
│                                               ▼             │
│  ┌──────────────┐         ┌──────────────┐  CALLBACK      │
│  │   Altavoz    │ <────── │Output Buffer │  (C thread,    │
│  │  (PortAudio) │         │   (Queue)    │   NO GIL)      │
│  └──────────────┘         └──────────────┘                 │
└─────────────────────────────────────────────────────────────┘
                │                           ▲
                │ Audio chunks              │ Audio chunks
                │ (numpy arrays)            │ (numpy arrays)
                ▼                           │
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│     PROCESO 2: STTProcessor     │  │    PROCESO 3: LLMProcessor      │
│                                 │  │                                 │
│  ┌──────────────┐               │  │               ┌──────────────┐  │
│  │ Vosk Engine  │               │  │               │  LFM2 (GGUF) │  │
│  │  (Streaming) │               │  │               │  n_threads=4 │  │
│  └──────┬───────┘               │  │               └──────┬───────┘  │
│         │                       │  │                      │          │
│         ▼                       │  │                      ▼          │
│  Audio → Text                   │  │              Text → Response    │
│                                 │  │                      │          │
│  ┌──────────────┐               │  │               ┌──────▼───────┐  │
│  │ Text Queue   │ ─────────────────────────────>   │ MeloTTS      │  │
│  │ (mp.Queue)   │               │  │               │ (preload)    │  │
│  └──────────────┘               │  │               └──────┬───────┘  │
│                                 │  │                      │          │
└─────────────────────────────────┘  └──────────────────────┼──────────┘
                                                            │
                                                            ▼
                                                 Audio chunks → AudioEngine
```

---

## Comparación Threading vs Multiprocessing

| Aspecto | Threading (v2.17) | Multiprocessing (v2.18) |
|---------|-------------------|-------------------------|
| **Paralelismo Real** | ❌ NO (GIL) | ✅ SÍ (procesos separados) |
| **Interferencia** | ✅ Threads comparten GIL | ❌ Procesos independientes |
| **Bloqueos** | ✅ `threading.Event()` bloquea | ❌ Solo shared memory queues |
| **CPU Cores** | ❌ 1 core (turnos) | ✅ 3 cores simultáneos |
| **Latencia STT** | 🟡 Espera a que TTS termine | ✅ Continuo, sin esperas |
| **Latencia TTS** | 🟡 Espera a que STT termine | ✅ Continuo, sin esperas |
| **Interrupciones** | 🟡 Artificial (checks cada 100ms) | ✅ Nativa (audio duplex) |

---

## Flujo Temporal: Threading vs Multiprocessing

### v2.17 (Threading - SERIAL)

```
Timeline (threading con GIL):

0ms    Usuario empieza a hablar
       ├─ InputThread ACTIVO (captura audio)
       │  └─ OutputThread BLOQUEADO (esperando)
       │
1500ms Usuario termina de hablar (VAD detecta silencio)
       ├─ InputThread procesa STT (Vosk)
       │  └─ OutputThread BLOQUEADO
       │
2000ms STT completo: "¿Cuál es el clima?"
       ├─ InputThread envía a queue
       │  └─ OutputThread BLOQUEADO
       │
2100ms OutputThread ACTIVA (recibe mensaje)
       ├─ InputThread BLOQUEADO (esperando)
       │  └─ LLM genera respuesta
       │
3500ms LLM termina: "El clima es soleado"
       ├─ InputThread BLOQUEADO
       │  └─ TTS sintetiza (MeloTTS)
       │
4200ms TTS completo (audio listo)
       ├─ InputThread BLOQUEADO
       │  └─ Audio reproduce
       │
6000ms Audio termina
       └─ InputThread REACTIVA (vuelve a escuchar)

TOTAL: 6000ms (usuario escucha respuesta)
PROBLEMA: Si usuario habla durante 2100-6000ms, SE IGNORA
```

### v2.18 (Multiprocessing - PARALELO)

```
Timeline (multiprocessing sin GIL):

0ms    Usuario empieza a hablar
       ├─ AudioEngine captura (Proceso 1)
       ├─ STTProcessor procesa (Proceso 2) ← SIMULTÁNEO
       └─ LLMProcessor IDLE (Proceso 3)
       
       TODOS los procesos corren en paralelo
       
1500ms Usuario termina de hablar (VAD detecta silencio)
       ├─ AudioEngine sigue capturando (NUNCA para)
       ├─ STTProcessor emite texto: "¿Cuál es el clima?"
       └─ LLMProcessor recibe texto, empieza LLM
       
2100ms LLM genera respuesta (mientras STT sigue escuchando)
       ├─ AudioEngine sigue capturando (CONTINUO)
       ├─ STTProcessor SIGUE procesando (sin bloqueo)
       └─ LLMProcessor sintetiza TTS
       
2800ms TTS completo, audio enviado a AudioEngine
       ├─ AudioEngine reproduce MIENTRAS captura
       │  └─ Input y Output simultáneos (duplex real)
       ├─ STTProcessor procesa nuevos inputs (sin interrumpir)
       └─ LLMProcessor IDLE
       
4600ms Audio termina
       └─ Usuario puede haber hablado DURANTE el audio
          ├─ STT ya procesó nuevas palabras
          └─ LLM puede responder INMEDIATAMENTE

TOTAL: 2800ms (primera síntesis completa)
BENEFICIO: Usuario puede interrumpir EN CUALQUIER MOMENTO
```

---

## Ventajas Técnicas del Multiprocessing

### 1. **Sin GIL (Global Interpreter Lock)**

```python
# Threading (v2.17)
# GIL permite solo 1 thread ejecutando Python a la vez
import threading
lock = threading.Lock()  # Locks explícitos
event = threading.Event()  # Bloqueos implícitos

# Multiprocessing (v2.18)
# Cada proceso tiene su propio intérprete Python
import multiprocessing as mp
# NO hay GIL compartido
# Procesos realmente paralelos
```

### 2. **Comunicación Ultra-Rápida (Shared Memory)**

```python
# Threading (v2.17)
queue_thread = queue.Queue()  # Memoria compartida (pero GIL)

# Multiprocessing (v2.18)
queue_proc = mp.Queue()  # IPC optimizado
# Bajo el capó: pipes, shared memory segments
# Sin serialización Python (numpy arrays directos)
```

### 3. **Audio Full-Duplex NATIVO (PortAudio)**

```python
# v2.17: Dos streams separados (input + output)
input_stream = sd.InputStream(callback=input_callback)
output_stream = sd.OutputStream(callback=output_callback)
# Problema: NO sincronizados, latencias desalineadas

# v2.18: UN stream duplex
stream = sd.Stream(callback=audio_callback)
# Input y output en el MISMO callback
# Sincronización perfecta por hardware
# Sin overhead de Python
```

---

## Métricas de Rendimiento Esperadas

| Métrica | v2.17 (Threading) | v2.18 (Multiprocessing) | Mejora |
|---------|-------------------|-------------------------|--------|
| **Latencia STT** | 500ms (bloqueado por TTS) | 200ms (continuo) | 60% ↓ |
| **Latencia TTS** | 700ms (espera STT) | 600ms (paralelo) | 14% ↓ |
| **Interrupciones** | 100ms (artificial) | <10ms (nativo) | 90% ↓ |
| **CPU Usage** | 1 core (100%) | 3 cores (30% c/u) | Escalable |
| **Turno invisible** | SÍ (GIL) | NO (procesos) | Eliminado |
| **Backpressure** | Bloqueos | Queue full → drop | Resiliente |

---

## Patrones de Código Críticos

### 1. Audio Callback (C thread, sin GIL)

```python
def _audio_callback(self, indata, outdata, frames, time_info, status):
    """
    Este callback corre en un thread C de PortAudio.
    NO está sujeto al GIL de Python.
    Ejecución: <1ms por llamada (cada 100ms).
    """
    # CAPTURAR (no bloqueante)
    try:
        self.input_buffer.put_nowait(indata[:, 0].copy())
    except queue.Full:
        pass  # Descartar si saturado
    
    # REPRODUCIR (no bloqueante)
    try:
        outdata[:] = self.output_buffer.get_nowait().reshape(-1, 1)
    except queue.Empty:
        outdata.fill(0)  # Silencio si no hay audio
```

### 2. Proceso STT (independiente)

```python
def run(self):
    """Loop INFINITO sin bloqueos externos"""
    while self.running.value:
        try:
            audio = self.input_buffer.get(timeout=0.1)
            
            # Procesar con Vosk (NO bloqueado por TTS)
            text = session.feed_audio(audio)
            
            if text:
                self.text_queue.put({"text": text, "timestamp": time.time()})
        
        except queue.Empty:
            continue  # No hay audio, loop continúa
```

### 3. Proceso LLM (independiente)

```python
def run(self):
    """Loop INFINITO sin bloqueos externos"""
    while self.running.value:
        try:
            user_input = self.text_queue.get(timeout=0.1)
            
            # LLM + TTS (NO bloqueado por STT)
            response = llm(user_input["text"])
            audio = tts.synthesize(response)
            
            # Enviar audio en chunks al buffer
            for chunk in split_audio(audio, chunk_size=1600):
                self.audio_output_buffer.put(chunk)
        
        except queue.Empty:
            continue  # No hay input, loop continúa
```

---

## Testing de Verdadero Paralelismo

### Test 1: Latencia STT Durante TTS

```python
# Hablar MIENTRAS SARAi responde
# Threading (v2.17): STT se ignora o se retrasa
# Multiprocessing (v2.18): STT procesa inmediatamente

# Comando de test:
python -m tests.test_true_fullduplex --test stt_during_tts
```

### Test 2: CPU Core Utilization

```bash
# Monitorear uso de CPU por proceso
htop  # Ver que 3 procesos Python usan cores diferentes

# Threading: 1 proceso al 100%
# Multiprocessing: 3 procesos al 30-40% cada uno
```

### Test 3: Interrupciones Naturales

```python
# Usuario interrumpe a SARAi a mitad de respuesta
# Threading (v2.17): Espera a chunk de 100ms
# Multiprocessing (v2.18): <10ms (callback C)

# Comando de test:
python -m tests.test_true_fullduplex --test interruption_latency
```

---

## Migración desde v2.17

### Cambios en `orchestrator.py`

```python
# v2.17 (ELIMINAR)
from threading import Thread, Event, Queue

# v2.18 (REEMPLAZAR)
from multiprocessing import Process, Queue, Event, Value
from core.layer1_io.true_fullduplex import TrueFullDuplexOrchestrator

# Uso:
orchestrator = TrueFullDuplexOrchestrator()
orchestrator.run_forever()
```

### Cambios en Configuración

```yaml
# config/sarai.yaml

full_duplex:
  # v2.17 (threading)
  # use_threading: true  # OBSOLETO
  
  # v2.18 (multiprocessing)
  use_multiprocessing: true
  
  audio:
    sample_rate: 16000
    blocksize: 1600  # 100ms chunks
    duplex: true  # Stream bidireccional
  
  processes:
    stt:
      name: "STT-Process"
      priority: "high"  # Nice -10
    
    llm:
      name: "LLM-Process"
      priority: "normal"  # Nice 0
```

---

## Limitaciones y Trade-offs

### Ventajas del Multiprocessing

✅ **Paralelismo real**: Sin GIL, ejecución simultánea  
✅ **Escalabilidad**: Usa múltiples CPU cores  
✅ **Aislamiento**: Crash en un proceso no afecta a otros  
✅ **Latencia nativa**: Audio duplex sin overhead Python  

### Desventajas del Multiprocessing

⚠️ **Overhead de startup**: Procesos tardan ~200-300ms en iniciar  
⚠️ **Memoria duplicada**: Cada proceso carga sus modelos  
⚠️ **IPC overhead**: Queues usan serialización (aunque numpy optimizado)  
⚠️ **Debugging complejo**: Logs de 3 procesos simultáneos  

### Trade-off Aceptado

**Memoria**: ~+500MB por proceso duplicado de Python  
**Beneficio**: Latencia -60%, interrupciones <10ms, 0 turnos invisibles  

**Conclusión**: La memoria extra es un precio bajo por verdadero paralelismo.

---

## Roadmap de Optimizaciones Futuras

### v2.19: Shared Memory para Modelos

```python
# Compartir modelos GGUF entre procesos (mmap)
from multiprocessing import shared_memory

# LFM2 cargado UNA vez, compartido por procesos
shm = shared_memory.SharedMemory(create=True, size=1_200_000_000)
```

### v2.20: Process Pool con Warm-up

```python
# Pre-fork procesos al inicio (evitar startup overhead)
from multiprocessing import Pool

pool = Pool(processes=3, initializer=warmup_models)
```

### v2.21: NUMA Awareness

```bash
# Asignar procesos a CPU cores específicos
taskset -c 0-1 python -m stt_process
taskset -c 2-3 python -m llm_process
```

---

## Conclusión

**v2.17 (Threading)**: Falso paralelismo, turnos invisibles, GIL bloqueante  
**v2.18 (Multiprocessing)**: Paralelismo real, 3 cores, audio duplex nativo

**Resultado**: Sistema que REALMENTE trabaja como el cerebro humano:
- Input y Output simultáneos
- Sin esperas artificiales
- Sin turnos invisibles
- Interrupciones naturales <10ms

**Filosofía v2.18**: 
_"El cerebro no espera turnos para escuchar y hablar.  
SARAi tampoco."_
