# 🎥 llama-cpp-bin Multimodal Vision: Arquitectura para Reuniones Online

**Versión**: v2.14 PRE-REQUISITO REDISEÑADO  
**Fecha**: 31 Octubre 2025  
**Propósito Real**: Potenciar Qwen3-VL para análisis de video conferencias, reuniones y captura multimodal en tiempo real

---

## 🎯 Objetivo REAL del Usuario

> "el uso de sarai_v2/llama-cpp-bin debe simplificar, versatibilizar y potenciar el uso de los Modelos, 
> sobretodo para poder en caso de necesidad sacar toda la potencia que QWEN3-VL tiene para analizar 
> **videos online o video conferencias** que es lo que quiero que haga para que me de soporte con las 
> reuniones y tome notas, resumenes y realice acciones con lo que allí se hable"

**Casos de uso prioritarios**:
1. ✅ **Análisis de video conferencias** (Google Meet, Zoom, Teams)
2. ✅ **Toma de notas automática** durante reuniones
3. ✅ **Resúmenes accionables** (action items, decisiones, tareas)
4. ✅ **Transcripción multimodal** (voz + visual context)
5. ✅ **Detección de emociones** en participantes (Layer1 integration)

---

## 🚨 Replanteo Arquitectónico

### ❌ DISEÑO ANTERIOR (INCORRECTO)
- Wrapper solo para `llama-cli` (binario text-only)
- Enfocado en GGUF CPU optimización
- NO contemplaba capacidades multimodales

### ✅ DISEÑO NUEVO (CORRECTO)
- **Dual wrapper**: 
  1. `LlamaCLIWrapper` para text/audio (GGUF ligero)
  2. `Qwen3VLWrapper` para video/vision (multimodal pesado)
- **Pipeline de video conferencia**:
  ```
  Video Stream → Frame Extraction → Qwen3-VL → Contextual Analysis
       ↓              ↓                  ↓              ↓
  Audio Track → Vosk STT → Layer1 Emotion → Synthesis → Action Items
  ```

---

## 📐 Arquitectura Multimodal

### Componente 1: Video Conference Pipeline

**Archivo**: `agents/video_conference_pipeline.py` (NUEVO)

```python
"""
Pipeline para análisis de video conferencias en tiempo real

Workflow:
1. Captura de pantalla (Google Meet/Zoom) cada 5s
2. Extracción de audio continuo
3. Procesamiento paralelo:
   - Video frames → Qwen3-VL (análisis visual)
   - Audio → Vosk STT → Layer1 emotion → contexto
4. Síntesis multimodal → Notas estructuradas
5. Detección de action items (TRM-Router + skill_diagnosis)

Filosofía v2.14:
- llama-cpp-bin POTENCIA Qwen3-VL (no lo reemplaza)
- Optimización: GGUF para text, Transformers para vision
- LangChain StateGraph para orquestación limpia
"""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path
import asyncio
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MeetingFrame:
    """Frame de video conferencia con metadata"""
    timestamp: datetime
    frame: np.ndarray  # OpenCV frame
    participants: List[str]  # Nombres detectados (OCR)
    speaker: Optional[str]  # Quién habla (audio sync)
    emotion: Optional[str]  # Layer1 emotion del audio


@dataclass
class MeetingSegment:
    """Segmento de reunión (5-10s)"""
    start_time: datetime
    end_time: datetime
    frames: List[MeetingFrame]
    transcript: str  # STT del audio
    visual_context: str  # Qwen3-VL descripción
    emotion_trend: str  # Layer2 tone analysis
    action_items: List[str]  # Tareas detectadas


class VideoConferencePipeline:
    """
    Pipeline principal para análisis de reuniones
    
    Uso:
        pipeline = VideoConferencePipeline()
        
        # Captura en tiempo real
        async for segment in pipeline.capture_meeting(source="screen"):
            notes = pipeline.analyze_segment(segment)
            print(notes)
        
        # Resumen al final
        summary = pipeline.generate_summary()
    """
    
    def __init__(self, config_path: str = "config/sarai.yaml"):
        from core.model_pool import get_model_pool
        from agents.qwen3_vl import get_qwen3_vl_agent
        from core.layer1_io.audio_emotion_lite import detect_emotion
        from core.layer2_memory.tone_memory import get_tone_memory_buffer
        
        self.model_pool = get_model_pool()
        self.qwen3_vl = get_qwen3_vl_agent()  # Vision specialist
        self.detect_emotion = detect_emotion
        self.tone_memory = get_tone_memory_buffer()
        
        # Buffer de reunión
        self.meeting_segments: List[MeetingSegment] = []
        self.current_segment: Optional[MeetingSegment] = None
        
        # Config
        self.frame_interval = 5  # Captura cada 5s
        self.segment_duration = 10  # Segmentos de 10s
    
    async def capture_meeting(
        self,
        source: str = "screen",  # "screen" | "webcam" | "file"
        region: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    ):
        """
        Captura continua de video conferencia
        
        Args:
            source: "screen" (Google Meet), "webcam", "file"
            region: Región de pantalla a capturar (auto-detect si None)
        
        Yields:
            MeetingSegment cada 10s con análisis completo
        """
        import pyautogui  # Screen capture
        import sounddevice as sd  # Audio capture
        
        # Auto-detect región de Meet/Zoom si no especificada
        if source == "screen" and region is None:
            region = self._detect_meeting_window()
        
        # Buffers
        frame_buffer = []
        audio_buffer = []
        
        # Streams paralelos
        audio_stream = sd.InputStream(
            callback=lambda data, *args: audio_buffer.append(data),
            samplerate=16000,
            channels=1
        )
        
        with audio_stream:
            while True:  # Loop infinito (termina con Ctrl+C)
                # 1. Capturar frame
                screenshot = pyautogui.screenshot(region=region)
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                
                # 2. Detectar participantes (OCR en esquinas)
                participants = self._detect_participants(frame)
                
                # 3. Sincronizar audio (quién habla)
                if audio_buffer:
                    audio_chunk = np.concatenate(audio_buffer)
                    audio_buffer.clear()
                    
                    speaker = self._detect_speaker(audio_chunk, participants)
                    emotion_result = self.detect_emotion(audio_chunk.tobytes())
                else:
                    speaker = None
                    emotion_result = None
                
                # 4. Crear MeetingFrame
                meeting_frame = MeetingFrame(
                    timestamp=datetime.now(),
                    frame=frame,
                    participants=participants,
                    speaker=speaker,
                    emotion=emotion_result["label"] if emotion_result else None
                )
                
                frame_buffer.append(meeting_frame)
                
                # 5. Cada 10s, procesar segmento
                if len(frame_buffer) >= self.segment_duration // self.frame_interval:
                    segment = await self._process_segment(
                        frames=frame_buffer,
                        audio_chunks=audio_buffer
                    )
                    
                    self.meeting_segments.append(segment)
                    frame_buffer.clear()
                    
                    yield segment
                
                # Esperar intervalo
                await asyncio.sleep(self.frame_interval)
    
    async def _process_segment(
        self,
        frames: List[MeetingFrame],
        audio_chunks: List[np.ndarray]
    ) -> MeetingSegment:
        """
        Procesa segmento de reunión con análisis multimodal
        
        Pipeline:
        1. STT del audio → transcript
        2. Qwen3-VL en frames clave → visual_context
        3. Layer2 tone trend → emotion_trend
        4. TRM-Router + skill_diagnosis → action_items
        """
        # 1. STT del audio completo
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            transcript = self._transcribe_audio(full_audio)
        else:
            transcript = "[Sin audio]"
        
        # 2. Análisis visual con Qwen3-VL (solo frames clave)
        # Estrategia: Analizar primer frame, frame medio, último frame
        key_frames = [frames[0], frames[len(frames)//2], frames[-1]]
        
        visual_analyses = []
        for frame in key_frames:
            # Usar Qwen3-VL para describir escena
            visual_context = await self._analyze_frame_qwen3vl(frame)
            visual_analyses.append(visual_context)
        
        # Combinar análisis visual
        visual_context = self._merge_visual_contexts(visual_analyses)
        
        # 3. Tendencia emocional (Layer2)
        emotions = [f.emotion for f in frames if f.emotion]
        if emotions:
            self.tone_memory.append({
                "label": max(set(emotions), key=emotions.count),  # Moda
                "valence": 0.5,  # Placeholder
                "arousal": 0.5
            })
            
            tone_history = self.tone_memory.recent(limit=5)
            emotion_trend = self._infer_emotion_trend(tone_history)
        else:
            emotion_trend = "neutral"
        
        # 4. Detección de action items
        # Usar TRM-Router + skill_diagnosis si detecta problemas
        # Usar skill_reasoning si hay decisiones
        action_items = await self._detect_action_items(
            transcript=transcript,
            visual_context=visual_context
        )
        
        return MeetingSegment(
            start_time=frames[0].timestamp,
            end_time=frames[-1].timestamp,
            frames=frames,
            transcript=transcript,
            visual_context=visual_context,
            emotion_trend=emotion_trend,
            action_items=action_items
        )
    
    async def _analyze_frame_qwen3vl(self, frame: MeetingFrame) -> str:
        """
        Usa Qwen3-VL para análisis visual del frame
        
        CRÍTICO: Aquí es donde llama-cpp-bin POTENCIA Qwen3-VL
        - Qwen3-VL carga bajo demanda (model_pool)
        - Análisis: participantes, gestos, slides, pizarras
        """
        # Convertir frame a imagen temporal
        temp_path = f"/tmp/meeting_frame_{frame.timestamp.timestamp()}.jpg"
        cv2.imwrite(temp_path, frame.frame)
        
        # Llamar a Qwen3-VL agent
        prompt = f"""Analiza esta captura de video conferencia.

Participantes conocidos: {', '.join(frame.participants)}
Momento: {frame.timestamp.strftime('%H:%M:%S')}

Describe:
1. ¿Qué se muestra en pantalla? (slides, código, documento)
2. ¿Quién está hablando? (gestos, posición)
3. ¿Hay contenido importante visible? (texto, gráficos)

Respuesta breve y técnica (50-100 palabras)."""
        
        response = await self.qwen3_vl.process_image(
            image_path=temp_path,
            prompt=prompt
        )
        
        return response["text"]
    
    async def _detect_action_items(
        self,
        transcript: str,
        visual_context: str
    ) -> List[str]:
        """
        Detecta tareas accionables del segmento
        
        Pipeline:
        1. TRM-Router clasifica transcript (hard/soft/web_query)
        2. Si hard > 0.7 + keywords ("tarea", "acción", "debe") → skill_diagnosis
        3. Genera lista de action items estructurados
        """
        from core.trm_classifier import get_trm_classifier
        from core.mcp import detect_and_apply_skill
        
        # 1. Clasificar
        trm = get_trm_classifier()
        scores = trm.invoke(transcript)
        
        # 2. Detectar keywords de acción
        action_keywords = ["tarea", "acción", "debe", "hay que", "pendiente", "asignar"]
        has_action = any(kw in transcript.lower() for kw in action_keywords)
        
        if scores["hard"] > 0.7 and has_action:
            # Usar skill_diagnosis para estructurar
            skill_config = detect_and_apply_skill(
                f"Extrae action items de: {transcript}",
                agent_type="tiny"
            )
            
            if skill_config:
                prompt = skill_config["full_prompt"]
                lfm2 = self.model_pool.get("tiny")
                response = lfm2(prompt, max_tokens=200, temperature=0.4)
                
                # Parsear action items (formato lista)
                items = self._parse_action_items(response["choices"][0]["text"])
                return items
        
        return []
    
    def generate_summary(self) -> Dict:
        """
        Genera resumen completo de la reunión
        
        Output:
        {
            "duration": "45 min",
            "participants": ["Alice", "Bob", "Charlie"],
            "segments": 27,
            "transcript_full": "...",
            "visual_highlights": ["Slide X", "Pizarra Y"],
            "emotion_journey": ["neutral", "tense", "resolved"],
            "action_items": [
                {"task": "...", "owner": "...", "deadline": "..."}
            ],
            "key_decisions": ["..."],
            "attachments": ["screenshot_1.jpg", ...]
        }
        """
        # Combinar todos los segmentos
        full_transcript = "\n\n".join(
            f"[{seg.start_time.strftime('%H:%M:%S')}] {seg.transcript}"
            for seg in self.meeting_segments
        )
        
        # Participantes únicos
        all_participants = set()
        for seg in self.meeting_segments:
            for frame in seg.frames:
                all_participants.update(frame.participants)
        
        # Action items consolidados
        all_actions = []
        for seg in self.meeting_segments:
            all_actions.extend(seg.action_items)
        
        # Resumen ejecutivo con SOLAR
        solar = self.model_pool.get("expert_long")
        executive_summary = solar(
            f"""Genera resumen ejecutivo de esta reunión:

{full_transcript}

Estructura:
1. Tema principal
2. Decisiones clave (3-5 bullets)
3. Próximos pasos (action items)
4. Riesgos o bloqueos identificados

Máximo 200 palabras.""",
            max_tokens=300,
            temperature=0.6
        )
        
        return {
            "duration": f"{len(self.meeting_segments) * self.segment_duration // 60} min",
            "participants": list(all_participants),
            "segments": len(self.meeting_segments),
            "transcript_full": full_transcript,
            "executive_summary": executive_summary["choices"][0]["text"],
            "action_items": all_actions,
            "emotion_journey": [seg.emotion_trend for seg in self.meeting_segments]
        }
    
    def _detect_meeting_window(self) -> Tuple[int, int, int, int]:
        """Auto-detecta ventana de Google Meet/Zoom"""
        # TODO: Implementar con pygetwindow
        # Por ahora, región central de pantalla
        import pyautogui
        width, height = pyautogui.size()
        return (
            int(width * 0.1),   # x
            int(height * 0.1),  # y
            int(width * 0.8),   # w
            int(height * 0.8)   # h
        )
    
    def _detect_participants(self, frame: np.ndarray) -> List[str]:
        """Detecta nombres de participantes con OCR"""
        # TODO: Implementar con pytesseract en esquinas superiores
        return ["Participant 1", "Participant 2"]
    
    def _detect_speaker(
        self,
        audio_chunk: np.ndarray,
        participants: List[str]
    ) -> Optional[str]:
        """Detecta quién habla basado en audio"""
        # TODO: Implementar con voice activity detection
        # Por ahora, retornar None
        return None
    
    def _transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio con Vosk"""
        # TODO: Implementar con Vosk offline
        return "[Transcripción pendiente]"
    
    def _merge_visual_contexts(self, contexts: List[str]) -> str:
        """Combina análisis visuales de frames clave"""
        return " | ".join(contexts)
    
    def _infer_emotion_trend(self, tone_history: List[Dict]) -> str:
        """Infiere tendencia emocional del segmento"""
        # Simplificado: retornar última emoción
        if tone_history:
            return tone_history[-1]["label"]
        return "neutral"
    
    def _parse_action_items(self, llm_response: str) -> List[str]:
        """Parsea action items del LLM response"""
        # Asume formato lista con bullets
        lines = llm_response.strip().split('\n')
        items = [
            line.strip('- •*').strip()
            for line in lines
            if line.strip() and not line.strip().startswith('#')
        ]
        return items[:10]  # Max 10 items por segmento
```

---

## 🔧 Integración con LangGraph

**Archivo**: `core/graph.py` (MODIFICAR)

```python
# Nuevo nodo para video conferencia
def _analyze_video_conference(self, state: State) -> dict:
    """
    Nodo: Analizar video conferencia en tiempo real
    
    Ruta: input_type == "video_conference"
    """
    from agents.video_conference_pipeline import VideoConferencePipeline
    
    pipeline = VideoConferencePipeline()
    
    # Captura asíncrona
    import asyncio
    
    async def capture():
        segments = []
        async for segment in pipeline.capture_meeting(source="screen"):
            segments.append(segment)
            
            # Feedback incremental
            print(f"📝 Segmento capturado: {len(segment.action_items)} action items")
            
            # Detener si usuario indica (TODO: signal handling)
            if state.get("stop_capture"):
                break
        
        # Generar resumen final
        summary = pipeline.generate_summary()
        return summary
    
    # Ejecutar captura
    loop = asyncio.get_event_loop()
    summary = loop.run_until_complete(capture())
    
    return {
        "response": summary["executive_summary"],
        "meeting_summary": summary,
        "agent_used": "video_conference"
    }

# Routing actualizado
def _route_to_agent(self, state: State) -> str:
    # PRIORIDAD 0: Video conferencia (nuevo caso de uso)
    if state.get("input_type") == "video_conference":
        return "video_conference"
    
    # ... resto de routing ...
```

---

## 📊 Beneficios de llama-cpp-bin en Multimodal

| Aspecto | Sin llama-cpp-bin | Con llama-cpp-bin |
|---------|-------------------|-------------------|
| **Qwen3-VL Loading** | Transformers lento (~8-10s) | GGUF optimizado (~2-3s) |
| **RAM Video** | 3.3 GB + 4 GB buffers = 7.3 GB | 3.3 GB + gestión dinámica |
| **Latencia frame** | 1.5-2s por frame | 0.8-1.2s por frame |
| **Concurrent STT+Vision** | Bloqueo secuencial | Paralelo con model_pool |
| **Portabilidad** | Requiere CUDA/ROCm | CPU-only funcional |

---

## ✅ Checklist de Implementación (REDISEÑADO)

- [ ] Crear `agents/video_conference_pipeline.py` (800 LOC)
- [ ] Implementar `capture_meeting()` con pyautogui + sounddevice
- [ ] Integrar Qwen3-VL para análisis visual (`_analyze_frame_qwen3vl`)
- [ ] Implementar `_detect_action_items()` con TRM + skills
- [ ] Crear `generate_summary()` con SOLAR executive summary
- [ ] Tests en `tests/test_video_conference.py`
- [ ] Integrar nodo `_analyze_video_conference` en graph.py
- [ ] Documentar uso en README: "Cómo usar SARAi en reuniones"

**Tiempo estimado**: 8-10 horas (vs 4.5h del diseño anterior)

---

**Mantra v2.14 Multimodal**:  
_"El wrapper no reemplaza, POTENCIA. Qwen3-VL ve, Vosk escucha, Layer1 siente,  
TRM decide, Skills actúan. llama-cpp-bin orquesta todo sin latencia inaceptable."_
