#!/usr/bin/env python3
"""
Base Embeddings für RAG Chatbot Industrial

Abstract Base Class für alle Embedding-Provider mit standardisierten Schnittstellen
für lokale und cloud-basierte Embedding-Modelle.

Features:
- Standardisierte Embedding-Interface für Provider-Unabhängigkeit
- Batch-Processing für Performance-optimierte Embedding-Erzeugung
- Caching-Mechanismen für wiederkehrende Texte
- Embedding-Validierung und Normalisierung
- Performance-Monitoring und Statistiken

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import numpy as np

# Core-Komponenten
from core import (
    get_logger, RAGConfig, get_current_config,
    DocumentProcessingError, ValidationError,
    create_error_context, log_performance
)


# =============================================================================
# EMBEDDING-DATENSTRUKTUREN UND ENUMS
# =============================================================================

class EmbeddingProvider(str, Enum):
    """Verfügbare Embedding-Provider"""
    OLLAMA = "ollama"                    # Lokale Ollama-Modelle
    OPENAI = "openai"                    # OpenAI Embeddings API
    HUGGINGFACE = "huggingface"          # HuggingFace Transformers
    SENTENCE_TRANSFORMERS = "sentence_transformers"  # SentenceTransformers
    CUSTOM = "custom"                    # Benutzerdefinierte Implementierung


class EmbeddingStatus(str, Enum):
    """Status der Embedding-Erzeugung"""
    SUCCESS = "success"                  # Erfolgreich erstellt
    CACHED = "cached"                   # Aus Cache geladen
    FAILED = "failed"                   # Fehlgeschlagen
    PARTIAL = "partial"                 # Teilweise erfolgreich (Batch)


@dataclass
class EmbeddingRequest:
    """
    Anfrage für Embedding-Erzeugung
    
    Attributes:
        text (str): Zu verarbeitender Text
        request_id (str): Eindeutige Request-ID
        metadata (Dict[str, Any]): Zusätzliche Metadaten
        cache_key (Optional[str]): Cache-Schlüssel (wird automatisch generiert)
        priority (int): Priorität (1=niedrig, 5=hoch)
    """
    text: str
    request_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_key: Optional[str] = None
    priority: int = 3
    
    def __post_init__(self):
        """Generiert Cache-Key nach Initialisierung"""
        if self.cache_key is None:
            self.cache_key = self._generate_cache_key()
    
    def _generate_cache_key(self) -> str:
        """Generiert deterministischen Cache-Key"""
        content = f"{self.text}:{self.metadata.get('model', '')}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


@dataclass
class EmbeddingResult:
    """
    Ergebnis der Embedding-Erzeugung
    
    Attributes:
        request_id (str): Ursprüngliche Request-ID
        embedding (List[float]): Embedding-Vektor
        status (EmbeddingStatus): Status der Verarbeitung
        model_name (str): Verwendetes Modell
        dimension (int): Dimensionalität des Embeddings
        processing_time_ms (float): Verarbeitungszeit in Millisekunden
        metadata (Dict[str, Any]): Zusätzliche Ergebnis-Metadaten
        error_message (Optional[str]): Fehlermeldung bei Problemen
    """
    request_id: str
    embedding: List[float]
    status: EmbeddingStatus
    model_name: str
    dimension: int
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """Prüft ob Embedding gültig ist"""
        return (
            self.status == EmbeddingStatus.SUCCESS and
            self.embedding is not None and
            len(self.embedding) == self.dimension and
            all(isinstance(x, (int, float)) and not np.isnan(x) for x in self.embedding)
        )
    
    def normalize(self) -> 'EmbeddingResult':
        """Normalisiert Embedding-Vektor (L2-Norm)"""
        if not self.embedding:
            return self
        
        embedding_array = np.array(self.embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding_array)
        
        if norm > 0:
            normalized_embedding = (embedding_array / norm).tolist()
            
            return EmbeddingResult(
                request_id=self.request_id,
                embedding=normalized_embedding,
                status=self.status,
                model_name=self.model_name,
                dimension=self.dimension,
                processing_time_ms=self.processing_time_ms,
                metadata={**self.metadata, 'normalized': True},
                error_message=self.error_message
            )
        
        return self


@dataclass
class BatchEmbeddingResult:
    """Ergebnis einer Batch-Embedding-Operation"""
    results: List[EmbeddingResult]
    total_requests: int
    successful_requests: int
    failed_requests: int
    cached_requests: int
    total_processing_time_ms: float
    batch_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Erfolgsrate der Batch-Operation"""
        return self.successful_requests / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def average_processing_time_ms(self) -> float:
        """Durchschnittliche Verarbeitungszeit"""
        return self.total_processing_time_ms / self.total_requests if self.total_requests > 0 else 0.0


# =============================================================================
# EMBEDDING-CACHE
# =============================================================================

class EmbeddingCache:
    """
    In-Memory Cache für Embeddings mit LRU-Eviction
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialisiert Embedding-Cache
        
        Args:
            max_size (int): Maximale Anzahl cached Embeddings
        """
        self.max_size = max_size
        self._cache: Dict[str, Tuple[EmbeddingResult, float]] = {}  # key -> (result, timestamp)
        self._access_order: List[str] = []  # LRU-Ordnung
        self.logger = get_logger("embedding_cache", "modules.embeddings")
        
        # Cache-Statistiken
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
    
    def get(self, cache_key: str) -> Optional[EmbeddingResult]:
        """
        Holt Embedding aus Cache
        
        Args:
            cache_key (str): Cache-Schlüssel
            
        Returns:
            Optional[EmbeddingResult]: Cached Embedding oder None
        """
        self._stats['total_requests'] += 1
        
        if cache_key in self._cache:
            # Cache-Hit
            embedding_result, _ = self._cache[cache_key]
            self._update_access_order(cache_key)
            
            # Neues EmbeddingResult mit CACHED Status
            cached_result = EmbeddingResult(
                request_id=embedding_result.request_id,
                embedding=embedding_result.embedding,
                status=EmbeddingStatus.CACHED,
                model_name=embedding_result.model_name,
                dimension=embedding_result.dimension,
                processing_time_ms=0.0,  # Cache-Zugriff ist instant
                metadata={**embedding_result.metadata, 'cached_at': time.time()},
                error_message=embedding_result.error_message
            )
            
            self._stats['hits'] += 1
            return cached_result
        
        # Cache-Miss
        self._stats['misses'] += 1
        return None
    
    def put(self, cache_key: str, embedding_result: EmbeddingResult) -> None:
        """
        Speichert Embedding in Cache
        
        Args:
            cache_key (str): Cache-Schlüssel
            embedding_result (EmbeddingResult): Zu cachendes Embedding
        """
        # Nur erfolgreiche Embeddings cachen
        if embedding_result.status != EmbeddingStatus.SUCCESS:
            return
        
        # Cache-Größe prüfen und ggf. evicten
        if len(self._cache) >= self.max_size and cache_key not in self._cache:
            self._evict_lru()
        
        # Embedding cachen
        self._cache[cache_key] = (embedding_result, time.time())
        self._update_access_order(cache_key)
    
    def _update_access_order(self, cache_key: str) -> None:
        """Aktualisiert LRU-Zugriffs-Reihenfolge"""
        if cache_key in self._access_order:
            self._access_order.remove(cache_key)
        self._access_order.append(cache_key)
    
    def _evict_lru(self) -> None:
        """Entfernt least-recently-used Embedding aus Cache"""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]
                self._stats['evictions'] += 1
    
    def clear(self) -> None:
        """Leert kompletten Cache"""
        self._cache.clear()
        self._access_order.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Holt Cache-Statistiken"""
        total_requests = self._stats['total_requests']
        hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_size': len(self._cache),
            'max_size': self.max_size,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            **self._stats
        }


# =============================================================================
# ABSTRACT BASE EMBEDDING CLASS
# =============================================================================

class BaseEmbeddings(ABC):
    """
    Abstract Base Class für alle Embedding-Provider
    
    Definiert standardisierte Schnittstelle für lokale und cloud-basierte
    Embedding-Modelle mit Caching und Performance-Monitoring.
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialisiert Base Embeddings
        
        Args:
            config (RAGConfig): Konfiguration
        """
        self.config = config or get_current_config()
        self.logger = get_logger(self.__class__.__name__.lower(), "modules.embeddings")
        
        # Provider-spezifische Eigenschaften (von Subklassen gesetzt)
        self.provider = EmbeddingProvider.CUSTOM
        self.model_name = "unknown"
        self.dimension = 0
        self.max_tokens = 512
        
        # Cache und Performance-Tracking
        self.cache = EmbeddingCache(max_size=self.config.embeddings.cache_size)
        
        self._performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time_ms': 0.0,
            'average_embedding_time_ms': 0.0,
            'cache_hit_rate': 0.0
        }
    
    # =============================================================================
    # ABSTRACT METHODS (müssen von Subklassen implementiert werden)
    # =============================================================================
    
    @abstractmethod
    def _create_embedding(self, text: str) -> List[float]:
        """
        Erstellt Embedding für einzelnen Text (Provider-spezifisch)
        
        Args:
            text (str): Zu verarbeitender Text
            
        Returns:
            List[float]: Embedding-Vektor
            
        Raises:
            NotImplementedError: Muss von Subklasse implementiert werden
        """
        raise NotImplementedError("Subklasse muss _create_embedding implementieren")
    
    @abstractmethod
    def _validate_connection(self) -> bool:
        """
        Validiert Verbindung zum Embedding-Provider
        
        Returns:
            bool: True wenn Verbindung funktioniert
        """
        raise NotImplementedError("Subklasse muss _validate_connection implementieren")
    
    # =============================================================================
    # PUBLIC API METHODS
    # =============================================================================
    
    @log_performance()
    def embed_text(self, text: str, request_id: str = None) -> EmbeddingResult:
        """
        Erstellt Embedding für einzelnen Text
        
        Args:
            text (str): Zu verarbeitender Text
            request_id (str): Optional Request-ID
            
        Returns:
            EmbeddingResult: Embedding-Ergebnis
        """
        if not text or len(text.strip()) == 0:
            raise ValidationError("Text für Embedding darf nicht leer sein", field_name="text")
        
        # Request-ID generieren falls nicht vorhanden
        if request_id is None:
            request_id = f"{self.provider.value}_{int(time.time() * 1000)}"
        
        # EmbeddingRequest erstellen
        embedding_request = EmbeddingRequest(
            text=text,
            request_id=request_id,
            metadata={'model': self.model_name, 'provider': self.provider.value}
        )
        
        start_time = time.time()
        
        try:
            # Cache prüfen
            cached_result = self.cache.get(embedding_request.cache_key)
            if cached_result:
                self.logger.debug(f"Embedding aus Cache geladen: {request_id}")
                self._update_performance_stats(cached_result)
                return cached_result
            
            # Text validieren und vorverarbeiten
            processed_text = self._preprocess_text(text)
            
            # Embedding erstellen (Provider-spezifisch)
            embedding_vector = self._create_embedding(processed_text)
            
            # Verarbeitungszeit berechnen
            processing_time_ms = (time.time() - start_time) * 1000
            
            # EmbeddingResult erstellen
            result = EmbeddingResult(
                request_id=request_id,
                embedding=embedding_vector,
                status=EmbeddingStatus.SUCCESS,
                model_name=self.model_name,
                dimension=len(embedding_vector) if embedding_vector else 0,
                processing_time_ms=processing_time_ms,
                metadata={
                    'provider': self.provider.value,
                    'text_length': len(processed_text),
                    'created_at': datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Validierung
            if not result.is_valid:
                raise DocumentProcessingError(
                    f"Ungültiges Embedding erstellt: Dimension={result.dimension}",
                    processing_stage="embedding_creation"
                )
            
            # Cache speichern
            self.cache.put(embedding_request.cache_key, result)
            
            # Statistiken aktualisieren
            self._update_performance_stats(result)
            
            self.logger.debug(
                f"Embedding erstellt: {request_id} (Dim: {result.dimension}, Zeit: {processing_time_ms:.1f}ms)"
            )
            
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            
            error_result = EmbeddingResult(
                request_id=request_id,
                embedding=[],
                status=EmbeddingStatus.FAILED,
                model_name=self.model_name,
                dimension=0,
                processing_time_ms=processing_time_ms,
                error_message=str(e)
            )
            
            self._update_performance_stats(error_result)
            
            error_context = create_error_context(
                component=f"modules.embeddings.{self.__class__.__name__.lower()}",
                operation="embed_text",
                request_id=request_id,
                text_length=len(text)
            )
            
            raise DocumentProcessingError(
                message=f"Fehler bei Embedding-Erstellung: {str(e)}",
                processing_stage="embedding_creation",
                context=error_context,
                original_exception=e
            )
    
    @log_performance()
    def embed_batch(self, 
                   texts: List[str], 
                   batch_size: int = None) -> BatchEmbeddingResult:
        """
        Erstellt Embeddings für Text-Batch
        
        Args:
            texts (List[str]): Liste zu verarbeitender Texte
            batch_size (int): Batch-Größe (None = alle auf einmal)
            
        Returns:
            BatchEmbeddingResult: Batch-Ergebnis
        """
        if not texts:
            return BatchEmbeddingResult(
                results=[],
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                cached_requests=0,
                total_processing_time_ms=0.0
            )
        
        # Batch-Größe aus Konfiguration falls nicht gesetzt
        if batch_size is None:
            batch_size = getattr(self.config.embeddings, 'batch_size', 50)
        
        start_time = time.time()
        all_results = []
        successful_count = 0
        failed_count = 0
        cached_count = 0
        
        # Texte in Batches verarbeiten
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for j, text in enumerate(batch):
                request_id = f"batch_{i+j}_{int(time.time() * 1000)}"
                
                try:
                    result = self.embed_text(text, request_id)
                    all_results.append(result)
                    
                    if result.status == EmbeddingStatus.SUCCESS:
                        successful_count += 1
                    elif result.status == EmbeddingStatus.CACHED:
                        cached_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Batch-Embedding fehlgeschlagen für Text {i+j}: {str(e)}")
                    
                    error_result = EmbeddingResult(
                        request_id=request_id,
                        embedding=[],
                        status=EmbeddingStatus.FAILED,
                        model_name=self.model_name,
                        dimension=0,
                        processing_time_ms=0.0,
                        error_message=str(e)
                    )
                    all_results.append(error_result)
                    failed_count += 1
        
        total_processing_time_ms = (time.time() - start_time) * 1000
        
        batch_result = BatchEmbeddingResult(
            results=all_results,
            total_requests=len(texts),
            successful_requests=successful_count,
            failed_requests=failed_count,
            cached_requests=cached_count,
            total_processing_time_ms=total_processing_time_ms,
            batch_metadata={
                'batch_size': batch_size,
                'provider': self.provider.value,
                'model_name': self.model_name
            }
        )
        
        self.logger.info(
            f"Batch-Embedding abgeschlossen: {len(texts)} Texte, "
            f"{successful_count} erfolgreich, {failed_count} fehlgeschlagen, "
            f"{cached_count} aus Cache ({total_processing_time_ms:.1f}ms)",
            extra={
                'extra_data': {
                    'total_requests': len(texts),
                    'success_rate': batch_result.success_rate,
                    'processing_time_ms': total_processing_time_ms
                }
            }
        )
        
        return batch_result
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _preprocess_text(self, text: str) -> str:
        """
        Vorverarbeitung von Text für Embeddings
        
        Args:
            text (str): Ursprünglicher Text
            
        Returns:
            str: Vorverarbeiteter Text
        """
        # Basis-Preprocessing
        processed = text.strip()
        
        # Text-Länge begrenzen (Token-Schätzung: ~4 Zeichen pro Token)
        max_chars = self.max_tokens * 4
        if len(processed) > max_chars:
            processed = processed[:max_chars]
            self.logger.debug(f"Text gekürzt: {len(text)} -> {len(processed)} Zeichen")
        
        return processed
    
    def _update_performance_stats(self, result: EmbeddingResult) -> None:
        """Aktualisiert Performance-Statistiken"""
        self._performance_stats['total_requests'] += 1
        
        if result.status == EmbeddingStatus.SUCCESS:
            self._performance_stats['successful_requests'] += 1
        elif result.status == EmbeddingStatus.FAILED:
            self._performance_stats['failed_requests'] += 1
        
        # Verarbeitungszeit (nur für neue Embeddings, nicht Cache)
        if result.status != EmbeddingStatus.CACHED:
            self._performance_stats['total_processing_time_ms'] += result.processing_time_ms
        
        # Durchschnittliche Zeit neu berechnen
        if self._performance_stats['successful_requests'] > 0:
            self._performance_stats['average_embedding_time_ms'] = (
                self._performance_stats['total_processing_time_ms'] / 
                self._performance_stats['successful_requests']
            )
        
        # Cache-Hit-Rate aktualisieren
        cache_stats = self.cache.get_statistics()
        self._performance_stats['cache_hit_rate'] = cache_stats['hit_rate']
    
    def validate_model(self) -> bool:
        """
        Validiert Modell-Verfügbarkeit und -Funktion
        
        Returns:
            bool: True wenn Modell funktioniert
        """
        try:
            # Verbindung prüfen
            if not self._validate_connection():
                return False
            
            # Test-Embedding erstellen
            test_result = self.embed_text("Test embedding validation", "validation_test")
            return test_result.is_valid
            
        except Exception as e:
            self.logger.error(f"Modell-Validierung fehlgeschlagen: {str(e)}")
            return False
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Holt Performance-Statistiken
        
        Returns:
            Dict[str, Any]: Detaillierte Performance-Daten
        """
        stats = self._performance_stats.copy()
        
        # Cache-Statistiken hinzufügen
        cache_stats = self.cache.get_statistics()
        stats['cache'] = cache_stats
        
        # Provider-Informationen
        stats['provider_info'] = {
            'provider': self.provider.value,
            'model_name': self.model_name,
            'dimension': self.dimension,
            'max_tokens': self.max_tokens
        }
        
        return stats
    
    def clear_cache(self) -> None:
        """Leert Embedding-Cache"""
        self.cache.clear()
        self.logger.info("Embedding-Cache geleert")
    
    def __str__(self) -> str:
        """String-Repräsentation"""
        return f"{self.__class__.__name__}(provider={self.provider.value}, model={self.model_name})"


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums und Datenstrukturen
    'EmbeddingProvider', 'EmbeddingStatus',
    'EmbeddingRequest', 'EmbeddingResult', 'BatchEmbeddingResult',
    
    # Cache
    'EmbeddingCache',
    
    # Base Class
    'BaseEmbeddings'
]


if __name__ == "__main__":
    # Testing der Base Class (abstrakt - kann nicht direkt instantiiert werden)
    print("Base Embeddings - Abstract Base Class")
    print("====================================")
    
    # Beispiel-Implementation für Testing
    class TestEmbeddings(BaseEmbeddings):
        def __init__(self):
            super().__init__()
            self.provider = EmbeddingProvider.CUSTOM
            self.model_name = "test_model"
            self.dimension = 384
            self.max_tokens = 512
        
        def _create_embedding(self, text: str) -> List[float]:
            # Test-Embedding (Random-Vektor)
            import random
            return [random.uniform(-1, 1) for _ in range(self.dimension)]
        
        def _validate_connection(self) -> bool:
            return True
    
    # Test-Embeddings erstellen
    test_embeddings = TestEmbeddings()
    
    print(f"Test-Embeddings: {test_embeddings}")
    print(f"Modell-Validierung: {test_embeddings.validate_model()}")
    
    # Einzelnes Embedding testen
    test_text = "Dies ist ein Test-Text für Embeddings."
    result = test_embeddings.embed_text(test_text)
    
    print(f"\nEinzel-Embedding:")
    print(f"  Status: {result.status}")
    print(f"  Dimension: {result.dimension}")
    print(f"  Verarbeitungszeit: {result.processing_time_ms:.1f}ms")
    print(f"  Gültig: {result.is_valid}")
    
    # Cache-Test
    cached_result = test_embeddings.embed_text(test_text)  # Sollte aus Cache kommen
    print(f"  Cached Status: {cached_result.status}")
    
    # Batch-Test
    test_texts = [
        "Erster Test-Text",
        "Zweiter Test-Text", 
        "Dritter Test-Text"
    ]
    
    batch_result = test_embeddings.embed_batch(test_texts)
    print(f"\nBatch-Embedding:")
    print(f"  Gesamt Requests: {batch_result.total_requests}")
    print(f"  Erfolgreich: {batch_result.successful_requests}")
    print(f"  Erfolgsrate: {batch_result.success_rate:.1%}")
    print(f"  Durchschnittliche Zeit: {batch_result.average_processing_time_ms:.1f}ms")
    
    # Performance-Statistiken
    stats = test_embeddings.get_performance_statistics()
    print(f"\nPerformance-Statistiken:")
    print(f"  Gesamt Requests: {stats['total_requests']}")
    print(f"  Cache Hit-Rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Durchschnittliche Embedding-Zeit: {stats['average_embedding_time_ms']:.1f}ms")
    
    print("\n✅ Base Embeddings erfolgreich getestet")
