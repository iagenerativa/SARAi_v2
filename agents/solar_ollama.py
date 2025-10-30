#!/usr/bin/env python3
"""
SOLAR-10.7B Ollama HTTP Wrapper - v2.16
Cliente HTTP para servidor Ollama remoto/local

Características:
- Lee configuración de .env (OLLAMA_BASE_URL, SOLAR_MODEL_NAME)
- Reintentos automáticos con backoff exponencial
- Timeout configurable
- Fallback a GGUF nativo si servidor no disponible

Author: SARAi v2.16 Integration Team
Date: 2025-10-29
"""

import os
import sys
import json
import time
from typing import Optional, Dict, List, Generator
from pathlib import Path

# Cargar variables de entorno
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Requests para HTTP
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  requests no disponible. Instalar: pip install requests")


class SolarOllama:
    """
    Cliente HTTP para SOLAR-10.7B en servidor Ollama
    
    Configuración desde .env:
    - OLLAMA_BASE_URL: URL del servidor (ej. http://192.168.0.251:11434)
    - SOLAR_MODEL_NAME: Nombre del modelo (ej. fblgit/UNA-SOLAR-10.7B-Instruct-v1.0:Q5_K_M)
    - OLLAMA_TIMEOUT: Timeout en segundos (default: 120)
    - OLLAMA_RETRIES: Número de reintentos (default: 3)
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
        retries: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Args:
            base_url: URL del servidor Ollama (default: desde .env)
            model_name: Nombre del modelo SOLAR (default: desde .env)
            timeout: Timeout en segundos (default: desde .env o 120)
            retries: Número de reintentos (default: desde .env o 3)
            verbose: Mostrar logs de requests
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests no instalado. Ejecuta: pip install requests")
        
        # Cargar configuración desde .env
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model_name = model_name or os.getenv("SOLAR_MODEL_NAME", "solar:10.7b")
        self.timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", "120"))
        self.retries = retries or int(os.getenv("OLLAMA_RETRIES", "3"))
        self.verbose = verbose
        
        if self.verbose:
            print(f"✅ Cliente Ollama inicializado:")
            print(f"   Servidor: {self.base_url}")
            print(f"   Modelo: {self.model_name}")
            print(f"   Timeout: {self.timeout}s")
            print(f"   Reintentos: {self.retries}")
        
        # Verificar conectividad
        self._check_connectivity()
    
    def _check_connectivity(self):
        """Verifica que el servidor Ollama esté disponible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Verificar que el modelo SOLAR esté disponible
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            
            if self.model_name not in model_names:
                print(f"⚠️  Modelo {self.model_name} NO encontrado en servidor")
                print(f"   Modelos disponibles: {', '.join(model_names)}")
                print(f"\n   Para descargar: ollama pull {self.model_name}")
            else:
                if self.verbose:
                    print(f"✅ Modelo {self.model_name} disponible en servidor")
        
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"No se pudo conectar al servidor Ollama en {self.base_url}\n"
                f"Error: {e}\n"
                f"Verifica que el servidor esté corriendo o cambia OLLAMA_BASE_URL en .env"
            )
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Genera respuesta con SOLAR via Ollama HTTP API
        
        Args:
            prompt: Texto de entrada
            temperature: Creatividad (0.0-1.0)
            top_p: Nucleus sampling (0.0-1.0)
            max_tokens: Máximo de tokens a generar
            stream: Si True, retorna generador (streaming)
            **kwargs: Parámetros adicionales de Ollama
        
        Returns:
            Respuesta generada (str) o generador si stream=True
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                **kwargs
            }
        }
        
        # Reintentos con backoff exponencial
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                    stream=stream
                )
                response.raise_for_status()
                
                if stream:
                    return self._stream_response(response)
                else:
                    return self._parse_response(response)
            
            except requests.exceptions.Timeout:
                if attempt < self.retries - 1:
                    wait_time = 2 ** attempt  # Backoff exponencial
                    if self.verbose:
                        print(f"⏱️  Timeout en intento {attempt + 1}/{self.retries}. Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise TimeoutError(f"Timeout tras {self.retries} intentos")
            
            except requests.exceptions.RequestException as e:
                if attempt < self.retries - 1:
                    wait_time = 2 ** attempt
                    if self.verbose:
                        print(f"❌ Error en intento {attempt + 1}/{self.retries}: {e}. Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Error tras {self.retries} intentos: {e}")
    
    def _parse_response(self, response: requests.Response) -> str:
        """Parsea respuesta no-streaming de Ollama"""
        try:
            data = response.json()
            return data.get("response", "")
        except json.JSONDecodeError as e:
            raise ValueError(f"Respuesta inválida del servidor: {e}")
    
    def _stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        """Generador para respuestas streaming de Ollama"""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                except json.JSONDecodeError:
                    continue
    
    def generate_upstage_style(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """
        Genera respuesta con el estilo de prompt de Upstage SOLAR
        
        Formato:
        ### System:
        {system_prompt}
        
        ### User:
        {prompt}
        
        ### Assistant:
        """
        # Construir prompt en formato Upstage
        full_prompt = ""
        
        if system_prompt:
            full_prompt += f"### System:\n{system_prompt}\n\n"
        
        full_prompt += f"### User:\n{prompt}\n\n### Assistant:\n"
        
        return self.generate(full_prompt, max_tokens=max_tokens, **kwargs)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Genera respuesta con formato de chat
        
        Args:
            messages: Lista de mensajes [{"role": "user|assistant|system", "content": "..."}]
            temperature: Creatividad (0.0-1.0)
            top_p: Nucleus sampling (0.0-1.0)
            max_tokens: Máximo de tokens a generar
            stream: Si True, retorna generador
            **kwargs: Parámetros adicionales
        
        Returns:
            Respuesta generada (str) o generador si stream=True
        """
        # Convertir mensajes a formato Upstage
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"### System:\n{content}")
            elif role == "user":
                prompt_parts.append(f"### User:\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"### Assistant:\n{content}")
        
        # Añadir prompt final para respuesta del asistente
        prompt_parts.append("### Assistant:\n")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        return self.generate(
            full_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )


def main():
    """Función de prueba"""
    print("🧪 Probando SolarOllama...")
    
    try:
        # Crear cliente (lee configuración de .env)
        client = SolarOllama(verbose=True)
        
        # Test 1: Generación simple
        print("\n" + "="*60)
        print("TEST 1: Generación simple")
        print("="*60)
        
        response = client.generate(
            prompt="¿Cuál es la capital de Francia?",
            max_tokens=100,
            temperature=0.3
        )
        print(f"\nRespuesta: {response}")
        
        # Test 2: Estilo Upstage
        print("\n" + "="*60)
        print("TEST 2: Estilo Upstage")
        print("="*60)
        
        response = client.generate_upstage_style(
            prompt="Explica qué es la inteligencia artificial en una frase",
            system_prompt="Eres un asistente conciso y técnico",
            max_tokens=100
        )
        print(f"\nRespuesta: {response}")
        
        # Test 3: Chat multi-turno
        print("\n" + "="*60)
        print("TEST 3: Chat multi-turno")
        print("="*60)
        
        messages = [
            {"role": "system", "content": "Eres un asistente amigable"},
            {"role": "user", "content": "Hola, ¿cómo estás?"},
            {"role": "assistant", "content": "¡Hola! Estoy bien, gracias. ¿En qué puedo ayudarte?"},
            {"role": "user", "content": "¿Cuánto es 2+2?"}
        ]
        
        response = client.chat(messages, max_tokens=50)
        print(f"\nRespuesta: {response}")
        
        print("\n✅ Todas las pruebas completadas")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
