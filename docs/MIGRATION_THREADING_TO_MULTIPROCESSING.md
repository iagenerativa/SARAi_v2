# Migración Threading → Multiprocessing: Guía Rápida

## 🔥 Cambios Críticos

### 1. Imports

```python
# ❌ ANTES (v2.17)
from threading import Thread, Event, Queue
import queue

# ✅ AHORA (v2.18)
from multiprocessing import Process, Event, Queue, Value
import multiprocessing as mp
```

### 2. Creación de Procesos vs Threads

```python
# ❌ ANTES
worker = Thread(target=my_function, args=(param,))
worker.start()

# ✅ AHORA
worker = Process(target=my_function, args=(param,))
worker.start()
```

### 3. Flags Compartidos

```python
# ❌ ANTES (threading.Event)
flag = Event()
flag.set()  # Activa
flag.is_set()  # Consulta

# ✅ AHORA (mp.Value)
flag = mp.Value('b', False)  # 'b' = boolean
flag.value = True  # Activa
flag.value  # Consulta
```

### 4. Queues

```python
# ❌ ANTES (queue.Queue - solo threads)
q = queue.Queue(maxsize=10)

# ✅ AHORA (mp.Queue - entre procesos)
q = mp.Queue(maxsize=10)

# API idéntica:
q.put(item)
q.get()
q.put_nowait(item)  # No bloquea
q.get_nowait()
```

### 5. Detener Procesos

```python
# ❌ ANTES (threads terminan con el programa)
worker.join()

# ✅ AHORA (procesos necesitan terminación explícita)
worker.join(timeout=2.0)
if worker.is_alive():
    worker.terminate()  # Forzar si no responde
```

---

## 🎯 Patrón de Migración: `orchestrator.py`

### ANTES (v2.17 - Threading)

```python
# core/layer1_io/orchestrator.py (v2.17)
from threading import Thread, Event
import queue

class FullDuplexOrchestrator:
    def __init__(self):
        # Flag compartido (threading)
        self.user_speaking = Event()
        
        # Queue compartida (threading)
        self.decision_queue = queue.Queue()
        
        # Threads
        self.input_thread = None
        self.output_thread = None
    
    def start(self):
        # Crear threads
        self.input_thread = Thread(
            target=self._input_loop,
            daemon=True
        )
        self.output_thread = Thread(
            target=self._output_loop,
            daemon=True
        )
        
        # Iniciar
        self.input_thread.start()
        self.output_thread.start()
    
    def stop(self):
        # Threads daemon terminan automáticamente
        pass
```

### DESPUÉS (v2.18 - Multiprocessing)

```python
# core/layer1_io/true_fullduplex.py (v2.18)
from multiprocessing import Process, Queue, Value
import multiprocessing as mp

class TrueFullDuplexOrchestrator:
    def __init__(self):
        # Flag compartido (multiprocessing)
        self.running = mp.Value('b', False)
        
        # Queues compartidas (multiprocessing)
        self.text_queue = mp.Queue(maxsize=10)
        
        # Motor de audio con sus propias queues
        self.audio_engine = FullDuplexAudioEngine()
        
        # Procesos
        self.stt_process = None
        self.llm_process = None
    
    def start(self):
        # Activar flag
        self.running.value = True
        
        # Iniciar motor de audio
        self.audio_engine.start()
        
        # Crear procesos
        self.stt_process = Process(
            target=STTProcessor(
                self.audio_engine.input_buffer,
                self.text_queue,
                self.running
            ).run,
            name="STT-Process"
        )
        
        self.llm_process = Process(
            target=LLMProcessor(
                self.text_queue,
                self.audio_engine.output_buffer,
                self.running
            ).run,
            name="LLM-Process"
        )
        
        # Iniciar procesos
        self.stt_process.start()
        self.llm_process.start()
    
    def stop(self):
        # Señal de parada
        self.running.value = False
        
        # Esperar procesos (con timeout)
        if self.stt_process:
            self.stt_process.join(timeout=2.0)
            if self.stt_process.is_alive():
                self.stt_process.terminate()
        
        if self.llm_process:
            self.llm_process.join(timeout=2.0)
            if self.llm_process.is_alive():
                self.llm_process.terminate()
        
        # Detener audio
        self.audio_engine.stop()
```

---

## 🚨 Gotchas y Errores Comunes

### 1. **Serialización de Objetos**

```python
# ❌ ERROR: Modelos no son serializables
tts_engine = MeloTTSEngine()
p = Process(target=worker, args=(tts_engine,))  # CRASH

# ✅ SOLUCIÓN: Cargar modelos DENTRO del proceso
def worker():
    tts_engine = MeloTTSEngine()  # Cargar aquí
    # ...

p = Process(target=worker)
```

### 2. **Estado Global NO Compartido**

```python
# ❌ ERROR: Variables globales NO se comparten
global_counter = 0

def worker():
    global global_counter
    global_counter += 1  # NO se refleja en el proceso padre

# ✅ SOLUCIÓN: Usar mp.Value o mp.Array
shared_counter = mp.Value('i', 0)  # 'i' = int

def worker(counter):
    counter.value += 1  # SÍ se comparte
```

### 3. **Locks en Multiprocessing**

```python
# ❌ ANTES (threading.Lock)
from threading import Lock
lock = Lock()

# ✅ AHORA (mp.Lock)
from multiprocessing import Lock
lock = mp.Lock()

# Uso idéntico:
with lock:
    # Sección crítica
    pass
```

### 4. **Print Debugging**

```python
# ⚠️ PROBLEMA: Print de múltiples procesos se mezcla
print("Proceso 1")
print("Proceso 2")
# Output: "PProrocceessoo  12"

# ✅ SOLUCIÓN: Prefijos únicos
def worker(name):
    print(f"[{name}] Mensaje")
    # Output: "[STT] Mensaje"
```

### 5. **Daemon Processes**

```python
# ❌ ERROR: Daemon processes mueren al cerrar el padre
p = Process(target=worker, daemon=True)
# Si el padre termina, worker muere ABRUPTAMENTE

# ✅ SOLUCIÓN: NO usar daemon, hacer shutdown explícito
p = Process(target=worker, daemon=False)
# Llamar p.terminate() o usar mp.Value('b', False)
```

---

## 🔧 Debugging de Multiprocessing

### Ver Procesos Activos

```bash
# Ver todos los procesos Python
ps aux | grep python

# Ver jerarquía de procesos
pstree -p $(pidof python)
```

### Logs Estructurados

```python
# Cada proceso loggea con su PID
import os
import logging

def setup_logger(process_name):
    logger = logging.getLogger(process_name)
    handler = logging.FileHandler(f"logs/{process_name}_{os.getpid()}.log")
    logger.addHandler(handler)
    return logger

# En STT Process
logger = setup_logger("STT")
logger.info("Procesando audio...")

# En LLM Process
logger = setup_logger("LLM")
logger.info("Generando respuesta...")
```

### Profiling de Procesos

```python
# Ver uso de CPU por proceso
import psutil

for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
    if 'python' in proc.info['name'].lower():
        print(f"PID {proc.info['pid']}: {proc.info['cpu_percent']}%")
```

---

## 📊 Comparación de Rendimiento

### Latencia de Comunicación

```python
# Benchmark: put() → get() latency
import time

# Threading (v2.17)
q = queue.Queue()
start = time.perf_counter()
q.put("test")
q.get()
threading_latency = time.perf_counter() - start
# ~10 microsegundos

# Multiprocessing (v2.18)
q = mp.Queue()
start = time.perf_counter()
q.put("test")
q.get()
mp_latency = time.perf_counter() - start
# ~50 microsegundos (5x más lento, pero aceptable)
```

### CPU Utilization

```python
# Threading: 1 core al 100%
# Multiprocessing: 3 cores al 33% cada uno
# → MISMO throughput total, MEJOR latencia individual
```

---

## ✅ Checklist de Migración

- [ ] Reemplazar `threading` imports por `multiprocessing`
- [ ] Cambiar `Thread` → `Process`
- [ ] Cambiar `threading.Event()` → `mp.Value('b', False)`
- [ ] Cambiar `queue.Queue()` → `mp.Queue()`
- [ ] Mover carga de modelos DENTRO de los procesos
- [ ] Añadir `terminate()` en shutdown
- [ ] Añadir timeout a `join()`
- [ ] Probar serialización de parámetros
- [ ] Añadir prefijos a prints para debugging
- [ ] Verificar que no hay variables globales compartidas
- [ ] Testear con `test_true_fullduplex.py`

---

## 🚀 Cómo Ejecutar

```bash
# Test completo
python tests/test_true_fullduplex.py

# Test específico
python tests/test_true_fullduplex.py --test stt_during_tts

# Ejecutar orquestador directamente
python -m core.layer1_io.true_fullduplex
```

---

## 📚 Referencias

- [Python Multiprocessing Docs](https://docs.python.org/3/library/multiprocessing.html)
- [GIL Explanation](https://realpython.com/python-gil/)
- [PortAudio Duplex](http://portaudio.com/docs/v19-doxydocs/writing_a_callback.html)
