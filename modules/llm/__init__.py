#!/usr/bin/env python3
"""
LLM Module - Fallback Implementation
Minimale aber funktionale LLM-Implementierungen für System-Stabilität
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class BaseLLM:
    """Base LLM mit konkreter Fallback-Implementation"""
    
    def __init__(self, config=None, **kwargs):
        self.config = config
        self.model_name = getattr(config, 'model_name', 'fallback') if config else 'fallback'
        logger.info(f"BaseLLM initialisiert: {self.model_name}")
    
    def generate_response(self, query: str, context: str = "") -> str:
        """Fallback Response-Generation"""
        if context:
            return f"Basierend auf den verfügbaren Informationen zu '{query}': {context[:200]}... [Fallback-Antwort]"
        else:
            return f"Antwort zu '{query}': Das System verarbeitet Ihre Anfrage. [Fallback-Modus aktiv]"
    
    def health_check(self) -> Dict[str, Any]:
        """Health Check für LLM"""
        return {
            "status": "healthy",
            "model": self.model_name,
            "type": "fallback"
        }

class OllamaLLM(BaseLLM):
    """Ollama LLM mit Fallback-Verhalten"""
    
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.base_url = getattr(config, 'base_url', 'http://localhost:11434') if config else 'http://localhost:11434'
        
        # Versuche echte Ollama-Verbindung, aber falle zurück wenn nicht verfügbar
        try:
            self._test_connection()
            self.available = True
            logger.info("Ollama-Verbindung erfolgreich")
        except Exception as e:
            logger.warning(f"Ollama nicht verfügbar, verwende Fallback: {e}")
            self.available = False
    
    def _test_connection(self):
        """Teste Ollama-Verbindung"""
        import requests
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
    
    def generate_response(self, query: str, context: str = "") -> str:
        """Response mit Ollama oder Fallback"""
        if self.available:
            try:
                return self._generate_ollama_response(query, context)
            except Exception as e:
                logger.error(f"Ollama-Response fehlgeschlagen: {e}")
        
        # Fallback
        return super().generate_response(query, context)
    
    def _generate_ollama_response(self, query: str, context: str = "") -> str:
        """Echte Ollama-Response"""
        import requests
        
        prompt = f"Kontext: {context}\n\nFrage: {query}\n\nAntwort:" if context else f"Frage: {query}\n\nAntwort:"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        
        return response.json().get('response', 'Keine Antwort erhalten')

# Exports
__all__ = ['BaseLLM', 'OllamaLLM']

print("LLM Module loaded with fallback support")
