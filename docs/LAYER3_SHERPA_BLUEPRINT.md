# Sherpa Layer 3 Blueprint (Supervisor Conversacional)

**Fecha**: 30 de octubre de 2025  
**Versión**: Draft v0.1  
**Ámbito**: Capa 3 - Sherpa Supervisor Conversacional

---

## 1. Propósito

Sherpa es el director de escena de SARAi. Su misión principal es garantizar una experiencia conversacional continua, empática y libre de silencios incómodos, coordinando simultáneamente:
- La entrega de contenido generado por el hilo 2 (razonamiento principal, RAG, skills).
- El uso inteligente de muletillas y fillers cuando la información aún no está lista.
- Las interrupciones, reparos y confirmaciones cuando detecta fallos en STT, traducción o respuestas parciales.
- La transición fluida de tonos y contextos a través del tiempo.

Sherpa no solo suaviza emociones: controla el tempo, decide quién habla y cuándo, y protege la continuidad del diálogo.

---

## 2. Responsabilidades Clave

1. **Gestión de Turnos**
   - Decide cuándo el sistema habla y cuándo cede el turno al usuario.
   - Negocia con hilo 2 (razonamiento) el momento exacto de liberar grandes bloques de información.
   - Prioriza mensajes críticos (alertas de seguridad, fallos) sobre fillers u otros contenidos.

2. **Orquestación de Fillers y Muletillas**
   - Emite fillers contextuales cuando espera más de _X_ ms por el hilo 2.
   - Selecciona muletillas alineadas con estilo y tono actual (`tone_bridge`).
   - Se asegura de que cada filler mantenga viva la conversación sin sonar repetitivo.

3. **Backpressure y Cola de Contenido**
   - Mantiene una cola `pending_payloads` con mensajes listos del hilo 2.
   - Libera contenido solo tras obtener un `go` explícito de Sherpa.
   - Permite suspender o reordenar la cola si cambia la prioridad (ej., nueva pregunta del usuario).

4. **Vigilancia de Fallos**
   - Monitorea estados de STT, TTS, traducción y skills.
   - Si detecta error, dispara frases reparadoras (“¿podrías repetir, por favor?”) o enciende modo seguro.
   - Reintenta o redirige a rutas de fallback según políticas.

5. **Integración de Tono y Contexto**
   - Consume `ToneProfile` (valencia/arousal, hint) para ajustar fillers y ritmo.
   - Coordina con Layer 2 para mantener la narrativa coherente a través de turnos múltiples.

6. **Instrumentación y Telemetría**
   - Expone métricas de ritmo conversacional: latencia entre turnos, densidad de fillers, ratio de interrupciones.
   - Registra eventos significativos en audit log (silencios evitados, fallos resueltos).

---

## 3. Componentes Principales

| Componente | Rol | Interfaces |
|------------|-----|------------|
| `SherpaCoordinator` | Núcleo stateful. Controla turnos, colas y timers. | API interna (`signal_ready`, `request_slot`, `yield_slot`). |
| `FillerDispatcher` | Selecciona y sintetiza fillers acorde a estilo. | Usa `tone_bridge.snapshot()` y catálogo dinámico. |
| `BackpressureGate` | Controla flujo de `pending_payloads` desde hilo 2. | Comunica `hold`/`release` via canales asincrónicos. |
| `FaultSentinel` | Observa pipelines (STT, traducción, TTS). | Recibe señales de watchdogs, emite frases de reparación. |
| `RhythmMetrics` | Registra métricas y eventos. | Envía datos a Layer 4 / dashboards. |

---

## 4. Interfaces y Señales

### 4.1. Contractos Internos

```text
InputThread ──(SpeechEvent)──▶ SherpaCoordinator
SherpaCoordinator ──(TurnDecision)──▶ OutputThread
SherpaCoordinator ──(FillRequest)──▶ FillerDispatcher
Hilo 2 (Reasoner) ──(PayloadReady)──▶ BackpressureGate ──▶ SherpaCoordinator
FaultSentinel ◀──(PipelineAlerts)─── Watchdogs (STT/TTS/Translator)
```

### 4.2. Eventos Clave

- `speech_end`: detectado por Layer 1; Sherpa decide si responde o espera.
- `payload_ready`: hilo 2 indica que existe contenido listo.
- `silence_timer_hit`: Sherpa programa filler al alcanzar umbral configurable (ej., 1200 ms).
- `fault_detected`: dispara reparo y, opcionalmente, solicita repetición al usuario.

---

## 5. Máquina de Estados (Simplificada)

```text
[IDLE]
  │ speech_end
  ▼
[EVALUATE_CONTEXT]
  │ payload_ready + slot_free
  ├────────────┐
  │            ▼
  │        [DELIVER_PAYLOAD]
  │            │ done
  │            ▼
  │        [POST_DELIVERY]
  │            │ update tone, stats
  │            ▼
  │        [IDLE]
  │
  │ payload_pending + slot_busy
  ├────────────┐
  │            ▼
  │        [QUEUE_WAIT]
  │            │ silence_timer_hit
  │            ▼
  │        [DISPATCH_FILLER]
  │            │ filler_done
  │            ▼
  │        [EVALUATE_CONTEXT]
  │
  │ fault_detected
  ▼
[REPAIR]
  │ fallback_done
  ▼
[IDLE]
```

---

## 6. Gestión de Colas y Tiempos

- `pending_payloads`: cola FIFO con prioridad ajustable.
- `filler_timeout`: configurable entre 800-1500 ms según modo (normal/urgente).
- `max_fillers_in_row`: evita repetición excesiva (ej., 2 consecutivos antes de forzar resumen parcial).
- `handoff_token`: semáforo que indica que Sherpa cede el turno a hilo 2; obligatorio antes de emitir bloques largos.

---

## 7. Estrategias de Fallback

1. **Fallo de traducción/STT**: Sherpa emite “No capté bien, ¿podrías repetirlo por favor?” y reinicia timer de escucha.
2. **Tiempo de espera excedido**: ofrece cortesía (“Dame un segundo, sigo revisando esto…”) y extiende ventana.
3. **Payload excesivo**: pide confirmación (“Tengo mucha información, ¿quieres que la resuma o la detallo?”).
4. **Modo seguro activo**: notifica al usuario que opera en modo cautela y limita promesas.

---

## 8. Telemetría y Logs

- `metrics`: latencia entre turnos, fillers/minuto, reparos disparados, tiempo total con slot retenido.
- `logs`: eventos con sello temporal y hash (`pending_payload`, `filler_dispatched`, `fault_repair`).
- Todos los eventos se auditan vía `core/audit.py` con HMAC (reutiliza patrón de voz).

---

## 9. Roadmap de Implementación

1. **Fase 1 – Skeleton** (4h)
   - Crear `core/layer3_fluidity/sherpa_coordinator.py` con máquina de estados básica.
   - Definir interfaces (`signal_ready`, `register_payload`, `notify_fault`).

2. **Fase 2 – Fillers & Backpressure** (6h)
   - Implementar `FillerDispatcher` y catálogo inicial de fillers.
   - Controlar timers y emitir muletillas contextuales.
   - Integrar con OutputThread para reproducir fillers sin bloquear hilo 2.

3. **Fase 3 – FaultSentinel** (5h)
   - Añadir observadores para STT, traducción, TTS.
   - Diseñar estrategias de reparación y reintentos.

4. **Fase 4 – Métricas y Auditoría** (4h)
   - Exponer métricas Prometheus (`latency_gap_seconds`, `fillers_count`).
   - Auditar eventos relevantes con SHA-256/HMAC.

5. **Fase 5 – Pruebas E2E** (6h)
   - Crear `tests/test_layer3_sherpa.py` con escenarios: payload tardío, fallo traducción, silencios largos.
   - Validar integración con InputThread/OutputThread.

---

## 10. Dependencias

- **Layer 1**: señales de discurso (`SpeechEvent`), estado del canal `input_thread`.
- **Layer 2**: `tone_memory` y `tone_bridge` para hints estilísticos.
- **Layer 4**: políticas de prioridad y actualización del MCP.
- **Assets**: catálogo de fillers multilingüe + plantillas de reparo.

---

## 11. Métricas de Éxito

| Métrica | Objetivo |
|---------|----------|
| Silencios >1.5s durante turnos del sistema | ≤ 1 por sesión |
| Fillers contextuales acertados | ≥ 90% (según scoring heurístico) |
| Reparos efectivos tras fallo STT | ≥ 95% |
| Backpressure cumplido (payloads sin solaparse) | 100% |
| MOS percibido (UX) | ≥ 4.6/5.0 |

---

## 12. Próximos Pasos Inmediatos

- Validar con el equipo los tiempos de espera y catálogo inicial de fillers.
- Implementar la fase 1 (skeleton + contratos) y conectar señales mínimas con Layer 1.
- Preparar dataset de fillers y reparos etiquetado por tipo de situación (espera, confirmación, reparación).

---

**Fin del blueprint**. Este documento evoluciona con cada iteración; actualizar versión tras completar cada fase.
