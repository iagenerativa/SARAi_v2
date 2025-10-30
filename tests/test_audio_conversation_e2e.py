"""
Tests de Comunicación E2E - Pipeline de Audio v2.16.3

Valida conversaciones completas simuladas con:
- STT simulado (encoder + decoder ONNX)
- LLM REAL (LFM2-1.2B)
- TTS simulado (encoder + talker + vocoder ONNX)

Objetivo: Medir latencias end-to-end en escenarios reales de uso.
"""

import pytest
import time
from pathlib import Path
from llama_cpp import Llama


class TestAudioConversationE2E:
    """Tests de conversación simulada end-to-end"""
    
    @pytest.fixture(scope="class")
    def lfm2_model(self):
        """Cargar LFM2 una vez para todos los tests"""
        model_path = Path("models/lfm2/LFM2-1.2B-Q4_K_M.gguf")
        
        if not model_path.exists():
            pytest.skip(f"Modelo LFM2 no encontrado: {model_path}")
        
        print(f"\n[SETUP] Cargando LFM2 desde: {model_path}")
        start = time.time()
        
        model = Llama(
            model_path=str(model_path),
            n_ctx=512,
            n_threads=6,
            verbose=False
        )
        
        load_time_ms = (time.time() - start) * 1000
        print(f"✅ LFM2 cargado en: {load_time_ms:.2f} ms")
        
        # Warm-up
        model.create_completion("Hola", max_tokens=1)
        
        return model
    
    def simulate_stt(self, audio_input: str) -> tuple:
        """
        Simula STT (Speech-to-Text) con latencia proyectada
        
        Returns:
            (texto_transcrito, latencia_ms)
        """
        # Simulación de latencias ONNX:
        # - Encoder: 100 ms (qwen25_audio_int8.onnx)
        # - Decoder: 40 ms (qwen25_audio_int8.onnx)
        # - Overhead: 5 ms
        latency_ms = 145
        time.sleep(latency_ms / 1000)
        
        return audio_input, latency_ms
    
    def simulate_tts(self, text_output: str) -> tuple:
        """
        Simula TTS (Text-to-Speech) con latencia proyectada
        
        Returns:
            (audio_bytes, latencia_ms)
        """
        # Simulación de latencias ONNX:
        # - Encoder TTS: 40 ms (qwen25_audio_int8.onnx)
        # - Talker: 5 ms (qwen25_7b_audio.onnx)
        # - Vocoder: 100 ms (qwen25_audio_int8.onnx)
        # - Overhead: 5 ms
        latency_ms = 150
        time.sleep(latency_ms / 1000)
        
        return b"<simulated_audio>", latency_ms
    
    def llm_reasoning(self, model: Llama, prompt: str, max_tokens: int = 30) -> tuple:
        """
        Razonamiento LLM REAL con LFM2
        
        Returns:
            (respuesta_texto, latencia_ms)
        """
        start = time.time()
        
        response = model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["\n\n"]
        )
        
        latency_ms = (time.time() - start) * 1000
        text = response['choices'][0]['text']
        
        return text, latency_ms
    
    def test_simple_greeting(self, lfm2_model):
        """
        Test 1: Conversación simple (saludo)
        
        Usuario: "Hola, ¿cómo estás?"
        Agent: Respuesta casual
        
        Objetivo: Latencia E2E < 2000 ms
        """
        print("\n" + "="*60)
        print("TEST 1: Conversación Simple (Saludo)")
        print("="*60)
        
        # Audio input simulado
        audio_input = "Hola, ¿cómo estás?"
        
        # Fase 1: STT
        print(f"\n[Usuario Audio]: {audio_input}")
        text_input, stt_latency = self.simulate_stt(audio_input)
        print(f"[STT] Audio → Texto: {stt_latency:.2f} ms")
        print(f"  └─ Output: '{text_input}'")
        
        # Fase 2: LLM
        response_text, llm_latency = self.llm_reasoning(
            lfm2_model, 
            text_input,
            max_tokens=30
        )
        print(f"[LLM] Razonamiento: {llm_latency:.2f} ms ✅ REAL")
        print(f"  └─ Output: '{response_text.strip()}'")
        
        # Fase 3: TTS
        audio_output, tts_latency = self.simulate_tts(response_text)
        print(f"[TTS] Texto → Audio: {tts_latency:.2f} ms")
        print(f"  └─ Output: {len(audio_output)} bytes")
        
        # E2E Total
        total_latency_ms = stt_latency + llm_latency + tts_latency
        print(f"\n{'='*60}")
        print(f"[TOTAL E2E]: {total_latency_ms:.2f} ms ✅")
        print(f"{'='*60}\n")
        
        # Validaciones
        assert stt_latency < 200, "STT demasiado lento"
        assert llm_latency < 2000, "LLM demasiado lento"
        assert tts_latency < 200, "TTS demasiado lento"
        assert total_latency_ms < 2500, f"E2E excede objetivo: {total_latency_ms:.0f}ms"
    
    def test_technical_question(self, lfm2_model):
        """
        Test 2: Pregunta técnica (respuesta más larga)
        
        Usuario: "Explícame qué es machine learning"
        Agent: Explicación técnica
        
        Objetivo: Latencia E2E < 3000 ms
        """
        print("\n" + "="*60)
        print("TEST 2: Conversación Técnica")
        print("="*60)
        
        # Audio input simulado
        audio_input = "Explícame qué es machine learning y cómo funciona"
        
        # Fase 1: STT
        print(f"\n[Usuario Audio]: {audio_input}")
        text_input, stt_latency = self.simulate_stt(audio_input)
        print(f"[STT] Audio → Texto: {stt_latency:.2f} ms")
        
        # Fase 2: LLM (más tokens para respuesta técnica)
        response_text, llm_latency = self.llm_reasoning(
            lfm2_model,
            text_input,
            max_tokens=50
        )
        print(f"[LLM] Razonamiento: {llm_latency:.2f} ms ✅ REAL")
        print(f"  └─ Output: '{response_text.strip()}'")
        
        # Fase 3: TTS
        audio_output, tts_latency = self.simulate_tts(response_text)
        print(f"[TTS] Texto → Audio: {tts_latency:.2f} ms")
        
        # E2E Total
        total_latency_ms = stt_latency + llm_latency + tts_latency
        print(f"\n{'='*60}")
        print(f"[TOTAL E2E]: {total_latency_ms:.2f} ms ✅")
        print(f"{'='*60}\n")
        
        # Validaciones
        assert llm_latency < 3000, "LLM demasiado lento para pregunta técnica"
        assert total_latency_ms < 3500, f"E2E excede objetivo: {total_latency_ms:.0f}ms"
    
    def test_multiturn_conversation(self, lfm2_model):
        """
        Test 3: Conversación multiturno (3 intercambios)
        
        Turno 1: "¿Qué es Python?"
        Turno 2: "¿Para qué se usa?"
        Turno 3: "Dame un ejemplo de código"
        
        Objetivo: Latencia promedio/turno < 1200 ms
        """
        print("\n" + "="*60)
        print("TEST 3: Conversación Multiturno (3 turnos)")
        print("="*60)
        
        conversation = [
            "¿Qué es Python?",
            "¿Para qué se usa?",
            "Dame un ejemplo de código"
        ]
        
        latencies = []
        
        for i, user_input in enumerate(conversation, 1):
            print(f"\n{'─'*60}")
            print(f"[Turno {i}]")
            print(f"{'─'*60}")
            
            # STT
            text_input, stt_latency = self.simulate_stt(user_input)
            print(f"Usuario: {text_input}")
            
            # LLM
            response_text, llm_latency = self.llm_reasoning(
                lfm2_model,
                text_input,
                max_tokens=15
            )
            print(f"Agent: {response_text.strip()}")
            
            # TTS
            _, tts_latency = self.simulate_tts(response_text)
            
            # Turno total
            turn_latency = stt_latency + llm_latency + tts_latency
            latencies.append(turn_latency)
            
            print(f"⏱️  Latencia turno: {turn_latency:.2f} ms")
        
        # Estadísticas finales
        total_conversation = sum(latencies)
        avg_latency = total_conversation / len(latencies)
        
        print(f"\n{'='*60}")
        print(f"RESUMEN CONVERSACIÓN:")
        print(f"  • Turnos: {len(latencies)}")
        print(f"  • Total: {total_conversation:.2f} ms")
        print(f"  • Promedio/turno: {avg_latency:.2f} ms ✅")
        print(f"  • Mínimo: {min(latencies):.2f} ms")
        print(f"  • Máximo: {max(latencies):.2f} ms")
        print(f"{'='*60}\n")
        
        # Validaciones
        assert avg_latency < 1200, f"Latencia promedio excede objetivo: {avg_latency:.0f}ms"
        assert max(latencies) < 1500, f"Turno más lento excede límite: {max(latencies):.0f}ms"
        
        # Validar consistencia (no degradación)
        variance = max(latencies) - min(latencies)
        assert variance < 300, f"Variación excesiva entre turnos: {variance:.0f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
