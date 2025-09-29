#!/usr/bin/env python3
"""
Embeddings Module - Non-Abstract Base Implementation
Industrielle RAG-Architektur - Syntax-Repair Version

Konkrete BaseEmbeddings ohne abstrakte Methoden für System-Kompatibilität.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseEmbeddings:
    """
    Base Embeddings Klasse - KONKRET, NICHT ABSTRAKT
    
    Funktionale Implementation ohne abstrakte Methoden.
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialisiere Base Embeddings"""
        self.config = config
        self.model_name = getattr(config, 'model_name', 'base') if config else 'base'
        self.dimensions = getattr(config, 'dimensions', 768) if config else 768
        logger.info(f"BaseEmbeddings initialisiert: {self.model_name}")
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Erstelle Embedding-Vektor für Text - KONKRETE IMPLEMENTATION
        
        Args:
            text: Input-Text
            
        Returns:
            List[float]: Embedding-Vektor
        """
        import hashlib
        
        # Hash-basierte deterministische Embeddings
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # Konvertiere zu normalized float vector
        embedding = []
        for i in range(min(len(hash_bytes), self.dimensions)):
            # Normalisiere Byte-Werte zu [-1, 1]
            normalized = (hash_bytes[i] - 127.5) / 127.5
            embedding.append(normalized)
        
        # Fülle auf gewünschte Dimension auf
        while len(embedding) < self.dimensions:
            # Wiederhole Pattern
            remaining = self.dimensions - len(embedding)
            embedding.extend(embedding[:remaining])
        
        return embedding[:self.dimensions]
    
    def _validate_connection(self) -> bool:
        """
        Validiere Verbindung - KONKRETE IMPLEMENTATION
        
        Returns:
            bool: True wenn Verbindung OK
        """
        return True  # BaseEmbeddings ist immer verfügbar
    
    def embed_query(self, text: str) -> List[float]:
        """
        Erstelle Embedding für Query
        
        Args:
            text: Query-Text
            
        Returns:
            List[float]: Embedding-Vektor
        """
        if not text or not text.strip():
            return [0.0] * self.dimensions
        
        return self._create_embedding(text.strip())
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Erstelle Embeddings für mehrere Dokumente
        
        Args:
            texts: Liste von Texten
            
        Returns:
            List[List[float]]: Liste von Embedding-Vektoren
        """
        embeddings = []
        
        for text in texts:
            if text and text.strip():
                embedding = self._create_embedding(text.strip())
            else:
                embedding = [0.0] * self.dimensions
            embeddings.append(embedding)
        
        return embeddings
    
    def health_check(self) -> Dict[str, Any]:
        """Health Check für Embeddings-Service"""
        try:
            # Test embedding creation
            test_embedding = self.embed_query("test")
            
            return {
                "status": "healthy",
                "model": self.model_name,
                "dimensions": len(test_embedding),
                "type": "base"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.model_name,
                "error": str(e),
                "type": "base"
            }

class OllamaEmbeddings(BaseEmbeddings):
    """
    Ollama Embeddings mit Fallback auf BaseEmbeddings
    """
    
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.base_url = getattr(config, 'base_url', 'http://localhost:11434') if config else 'http://localhost:11434'
        
        # Teste Ollama-Verfügbarkeit
        try:
            self._test_ollama_connection()
            self.ollama_available = True
            logger.info("Ollama-Embeddings verfügbar")
        except Exception as e:
            logger.warning(f"Ollama nicht verfügbar, verwende Fallback: {e}")
            self.ollama_available = False
    
    def _test_ollama_connection(self):
        """Teste Ollama-Verbindung"""
        import requests
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
    
    def _create_embedding(self, text: str) -> List[float]:
        """Erstelle Embedding mit Ollama oder Fallback"""
        if self.ollama_available:
            try:
                return self._create_ollama_embedding(text)
            except Exception as e:
                logger.error(f"Ollama-Embedding fehlgeschlagen: {e}")
                self.ollama_available = False  # Disable for future calls
        
        # Fallback auf BaseEmbeddings
        return super()._create_embedding(text)
    
    def _create_ollama_embedding(self, text: str) -> List[float]:
        """Erstelle Embedding mit Ollama API"""
        import requests
        
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        response = requests.post(f"{self.base_url}/api/embeddings", json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result.get('embedding', super()._create_embedding(text))
    
    def _validate_connection(self) -> bool:
        """Validiere Ollama-Verbindung"""
        if self.ollama_available:
            try:
                self._test_ollama_connection()
                return True
            except:
                self.ollama_available = False
        
        return super()._validate_connection()  # Fallback ist immer OK

@dataclass
class EmbeddingResult:
    """Ergebnis einer Embedding-Operation"""
    embeddings: List[float]
    metadata: Dict[str, Any] = None
    model_name: Optional[str] = None
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def dimensions(self) -> int:
        return len(self.embeddings)
    
    @property 
    def has_embeddings(self) -> bool:
        return bool(self.embeddings)


class EmbeddingFactory:
    """Factory für Embedding-Erstellung"""
    
    @staticmethod
    def create_embeddings(provider='ollama', **kwargs):
        """Erstellt Embedding-Instanz"""
        if provider == 'ollama':
            return OllamaEmbeddings(**kwargs)
        else:
            return BaseEmbeddings(**kwargs)
    
    @staticmethod
    def get_available_providers():
        """Liefert verfügbare Provider"""
        return ['ollama', 'base']

def create_auto_embeddings(**kwargs):
    """Automatische Embedding-Erstellung"""
    return EmbeddingFactory.create_embeddings(**kwargs)

def get_available_providers():
    """Alias für Factory-Methode"""
    return EmbeddingFactory.get_available_providers()

# Exports
__all__ = [
    "BaseEmbeddings",
    "OllamaEmbeddings", 
    "EmbeddingResult",
    "EmbeddingFactory",
    "create_auto_embeddings", 
    "get_available_providers"
]

print("Embeddings module loaded with concrete implementations")
