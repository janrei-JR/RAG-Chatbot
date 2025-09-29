#!/usr/bin/env python3
"""
Base Retriever - Abstract Base Class
Industrielle RAG-Architektur - Module Layer

Abstract Base Class für alle Retrieval-Implementierungen mit standardisierten
Schnittstellen, Performance-Monitoring und Production-Features für 
industrielle RAG-Anwendungen.

Features:
- Standardisierte Retrieval-Interface für alle Implementierungen
- Performance-Monitoring und Caching-Unterstützung
- Flexible Konfiguration und Metadaten-Handling
- Production-Features: Health-Checks, Error-Recovery, Logging
- Plugin-Architektur für einfache Erweiterung

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

import time
from typing import Dict, Any, List, Optional, Tuple, Union, Protocol
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from abc import ABC, abstractmethod

# Core-Komponenten
from core import get_logger, RAGConfig, ValidationError, create_error_context


# =============================================================================
# RETRIEVER BASE DATENSTRUKTUREN
# =============================================================================

class RetrievalMode(str, Enum):
    """Retrieval-Modi für verschiedene Anwendungsfälle"""
    PRECISION = "precision"        # Hohe Präzision, weniger Ergebnisse
    RECALL = "recall"              # Hohe Vollständigkeit, mehr Ergebnisse
    BALANCED = "balanced"          # Ausgewogen
    SPEED = "speed"                # Schnelle Antworten
    QUALITY = "quality"            # Beste Relevanz


@dataclass
class Document:
    """Standardisierte Dokument-Struktur"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    source: Optional[str] = None
    
    def __post_init__(self):
        """Post-Initialisierung mit Validierung"""
        if not self.content or not self.content.strip():
            raise ValidationError("Dokument-Content darf nicht leer sein")


@dataclass  
class RetrievalQuery:
    """Standardisierte Query-Struktur für Retriever"""
    text: str
    k: int = 5
    filters: Optional[Dict[str, Any]] = None
    mode: RetrievalMode = RetrievalMode.BALANCED
    score_threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validierung der Query-Parameter"""
        if not self.text or not self.text.strip():
            raise ValidationError("Query-Text darf nicht leer sein")
        
        if self.k < 1 or self.k > 1000:
            raise ValidationError("k muss zwischen 1 und 1000 liegen")


@dataclass
class RetrievalResult:
    """Standardisierte Ergebnis-Struktur"""
    documents: List[Tuple[Document, float]] = field(default_factory=list)
    query: Optional[RetrievalQuery] = None
    total_found: int = 0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def scores(self) -> List[float]:
        """Extrahiert alle Relevanz-Scores"""
        return [score for _, score in self.documents]
    
    @property
    def average_score(self) -> float:
        """Durchschnittlicher Relevanz-Score"""
        scores = self.scores
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_documents_only(self) -> List[Document]:
        """Extrahiert nur die Dokumente ohne Scores"""
        return [doc for doc, _ in self.documents]


@dataclass
class RetrieverConfig:
    """Basis-Konfiguration für alle Retriever"""
    name: str
    description: str = ""
    cache_enabled: bool = True
    cache_size: int = 1000
    performance_monitoring: bool = True
    health_checks_enabled: bool = True
    default_k: int = 5
    max_k: int = 100
    default_score_threshold: Optional[float] = None
    
    # Erweiterte Konfiguration
    custom_params: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ABSTRACT BASE RETRIEVER
# =============================================================================

class BaseRetriever(ABC):
    """
    Abstract Base Class für alle Retrieval-Implementierungen
    
    Definiert standardisierte Schnittstelle für:
    - Dokumenten-Suche mit konfigurierbaren Parametern
    - Performance-Monitoring und Caching
    - Health-Checks und Error-Recovery
    - Metadaten-Handling und Logging
    """
    
    def __init__(self, config: RetrieverConfig):
        """
        Initialisiert Base Retriever
        
        Args:
            config: Retriever-Konfiguration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{config.name}")
        
        # Performance-Monitoring
        self._total_queries = 0
        self._total_processing_time_ms = 0.0
        self._successful_queries = 0
        self._error_count = 0
        self._last_query_time = None
        
        # Simple LRU Cache (wenn aktiviert)
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Initialisierung
        self._initialize()
        
        self.logger.info(f"BaseRetriever '{config.name}' initialisiert")
    
    def _initialize(self):
        """Template Method für spezifische Initialisierung in Subklassen"""
        pass
    
    @abstractmethod
    def _retrieve_documents(self, query: RetrievalQuery) -> List[Tuple[Document, float]]:
        """
        Abstrakte Methode für Dokumenten-Retrieval
        
        Args:
            query: Retrieval-Query mit Parametern
            
        Returns:
            List[Tuple[Document, float]]: Dokumente mit Relevanz-Scores
        """
        pass
    
    def retrieve(self, 
                query: Union[str, RetrievalQuery],
                k: Optional[int] = None,
                filters: Optional[Dict[str, Any]] = None,
                mode: Optional[RetrievalMode] = None,
                score_threshold: Optional[float] = None) -> RetrievalResult:
        """
        Hauptmethode für Dokumenten-Retrieval mit Monitoring und Caching
        
        Args:
            query: Query-Text oder RetrievalQuery-Objekt
            k: Anzahl Ergebnisse (überschreibt Query-Parameter)
            filters: Metadaten-Filter (überschreibt Query-Parameter)
            mode: Retrieval-Modus (überschreibt Query-Parameter)
            score_threshold: Mindest-Relevanz-Score
            
        Returns:
            RetrievalResult: Retrieval-Ergebnisse mit Metadaten
        """
        start_time = time.time()
        
        try:
            # Query normalisieren
            if isinstance(query, str):
                query_obj = RetrievalQuery(
                    text=query,
                    k=k or self.config.default_k,
                    filters=filters,
                    mode=mode or RetrievalMode.BALANCED,
                    score_threshold=score_threshold or self.config.default_score_threshold
                )
            else:
                query_obj = query
                # Parameter überschreiben wenn explizit angegeben
                if k is not None:
                    query_obj.k = k
                if filters is not None:
                    query_obj.filters = filters
                if mode is not None:
                    query_obj.mode = mode
                if score_threshold is not None:
                    query_obj.score_threshold = score_threshold
            
            # Cache-Check
            if self.config.cache_enabled:
                cache_key = self._generate_cache_key(query_obj)
                cached_result = self._get_from_cache(cache_key)
                if cached_result:
                    self._cache_hits += 1
                    self.logger.debug(f"Cache-Hit für Query: {query_obj.text[:50]}...")
                    return cached_result
                else:
                    self._cache_misses += 1
            
            # Retrieval ausführen
            documents = self._retrieve_documents(query_obj)
            
            # Score-Filtering anwenden
            if query_obj.score_threshold:
                documents = [
                    (doc, score) for doc, score in documents 
                    if score >= query_obj.score_threshold
                ]
            
            # Ergebnisse auf k begrenzen
            documents = documents[:query_obj.k]
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Ergebnis zusammenstellen
            result = RetrievalResult(
                documents=documents,
                query=query_obj,
                total_found=len(documents),
                processing_time_ms=processing_time_ms,
                metadata={
                    'retriever': self.config.name,
                    'cache_hit': False,
                    'filters_applied': query_obj.filters is not None,
                    'score_threshold_applied': query_obj.score_threshold is not None
                }
            )
            
            # Performance-Stats aktualisieren
            self._update_performance_stats(result, success=True)
            
            # Cache-Update
            if self.config.cache_enabled:
                self._update_cache(cache_key, result)
            
            self.logger.debug(
                f"Retrieval abgeschlossen: {len(documents)} Dokumente "
                f"in {processing_time_ms:.1f}ms (Query: {query_obj.text[:50]}...)"
            )
            
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Retrieval fehlgeschlagen: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Performance-Stats für Fehler
            self._error_count += 1
            self._total_queries += 1
            self._total_processing_time_ms += processing_time_ms
            self._last_query_time = datetime.now(timezone.utc)
            
            # Leeres Ergebnis mit Fehler-Info zurückgeben
            return RetrievalResult(
                documents=[],
                query=query_obj if 'query_obj' in locals() else None,
                processing_time_ms=processing_time_ms,
                metadata={'error': error_msg, 'retriever': self.config.name}
            )
    
    def _generate_cache_key(self, query: RetrievalQuery) -> str:
        """Generiert Cache-Key für Query"""
        import hashlib
        
        key_components = [
            query.text,
            str(query.k),
            str(query.filters),
            query.mode.value,
            str(query.score_threshold)
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[RetrievalResult]:
        """Holt Ergebnis aus Cache"""
        if cache_key in self._cache:
            cached_result = self._cache[cache_key]
            # Cache-Hit Metadaten setzen
            cached_result.metadata['cache_hit'] = True
            return cached_result
        return None
    
    def _update_cache(self, cache_key: str, result: RetrievalResult):
        """Aktualisiert Cache mit neuem Ergebnis"""
        # LRU-Cache-Logik (vereinfacht)
        if len(self._cache) >= self.config.cache_size:
            # Entferne älteste Einträge (vereinfacht: erste N Einträge)
            oldest_keys = list(self._cache.keys())[:self.config.cache_size // 4]
            for old_key in oldest_keys:
                del self._cache[old_key]
        
        self._cache[cache_key] = result
    
    def _update_performance_stats(self, result: RetrievalResult, success: bool = True):
        """Aktualisiert Performance-Statistiken"""
        if not self.config.performance_monitoring:
            return
        
        self._total_queries += 1
        self._total_processing_time_ms += result.processing_time_ms
        self._last_query_time = datetime.now(timezone.utc)
        
        if success:
            self._successful_queries += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Holt Performance-Statistiken
        
        Returns:
            Dict: Performance-Metriken
        """
        if self._total_queries == 0:
            return {
                'retriever': self.config.name,
                'total_queries': 0,
                'status': 'no_queries_yet'
            }
        
        stats = {
            'retriever': self.config.name,
            'total_queries': self._total_queries,
            'successful_queries': self._successful_queries,
            'error_count': self._error_count,
            'success_rate_percent': (self._successful_queries / self._total_queries) * 100,
            'average_processing_time_ms': self._total_processing_time_ms / self._total_queries,
            'total_processing_time_ms': self._total_processing_time_ms,
            'last_query_time': self._last_query_time.isoformat() if self._last_query_time else None
        }
        
        # Cache-Statistiken hinzufügen wenn aktiviert
        if self.config.cache_enabled:
            total_cache_requests = self._cache_hits + self._cache_misses
            hit_rate = (self._cache_hits / total_cache_requests) * 100 if total_cache_requests > 0 else 0
            
            stats.update({
                'cache_enabled': True,
                'cache_size': len(self._cache),
                'cache_capacity': self.config.cache_size,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'cache_hit_rate_percent': hit_rate
            })
        else:
            stats['cache_enabled'] = False
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        Führt Health-Check durch
        
        Returns:
            Dict: Health-Status und Diagnostik-Informationen
        """
        try:
            health_status = {
                'retriever': self.config.name,
                'status': 'healthy',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'config_valid': self._validate_config(),
                'performance_stats': self.get_performance_stats() if self.config.performance_monitoring else {}
            }
            
            # Zusätzliche Health-Checks in Subklassen
            custom_health = self._custom_health_check()
            if custom_health:
                health_status.update(custom_health)
            
            return health_status
            
        except Exception as e:
            return {
                'retriever': self.config.name,
                'status': 'error',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'error': str(e)
            }
    
    def _validate_config(self) -> bool:
        """Validiert Retriever-Konfiguration"""
        try:
            if not self.config.name:
                return False
            if self.config.cache_size <= 0:
                return False
            if self.config.default_k <= 0:
                return False
            if self.config.max_k <= self.config.default_k:
                return False
            return True
        except:
            return False
    
    def _custom_health_check(self) -> Optional[Dict[str, Any]]:
        """Template Method für spezifische Health-Checks in Subklassen"""
        return None
    
    def clear_cache(self):
        """Leert den Cache"""
        if self.config.cache_enabled:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self.logger.info(f"Cache für {self.config.name} geleert")
    
    def reset_performance_stats(self):
        """Setzt Performance-Statistiken zurück"""
        self._total_queries = 0
        self._total_processing_time_ms = 0.0
        self._successful_queries = 0
        self._error_count = 0
        self._last_query_time = None
        self.logger.info(f"Performance-Stats für {self.config.name} zurückgesetzt")
    
    @property
    def name(self) -> str:
        """Name des Retrievers"""
        return self.config.name
    
    @property
    def is_healthy(self) -> bool:
        """Prüft ob Retriever funktionsfähig ist"""
        try:
            health = self.health_check()
            return health.get('status') == 'healthy'
        except:
            return False


# =============================================================================
# RETRIEVER REGISTRY UND FACTORY
# =============================================================================

class RetrieverRegistry:
    """
    Registry für verfügbare Retriever-Implementierungen
    
    Ermöglicht Plugin-basierte Architektur wo neue Retriever-Typen
    dynamisch registriert und erstellt werden können.
    """
    
    _retrievers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, retriever_class: type):
        """
        Registriert neue Retriever-Implementierung
        
        Args:
            name: Eindeutiger Name des Retrievers
            retriever_class: Retriever-Klasse (muss BaseRetriever erweitern)
        """
        if not issubclass(retriever_class, BaseRetriever):
            raise ValidationError(f"Retriever-Klasse muss BaseRetriever erweitern")
        
        cls._retrievers[name] = retriever_class
        logger = get_logger(__name__)
        logger.info(f"Retriever '{name}' registriert: {retriever_class.__name__}")
    
    @classmethod
    def get_available_retrievers(cls) -> List[str]:
        """
        Holt Liste verfügbarer Retriever-Namen
        
        Returns:
            List[str]: Verfügbare Retriever-Namen
        """
        return list(cls._retrievers.keys())
    
    @classmethod
    def create_retriever(cls, name: str, config: RetrieverConfig, **kwargs) -> BaseRetriever:
        """
        Erstellt Retriever-Instanz nach Name
        
        Args:
            name: Name des zu erstellenden Retrievers
            config: Retriever-Konfiguration
            **kwargs: Zusätzliche Parameter für Retriever-Konstruktor
            
        Returns:
            BaseRetriever: Retriever-Instanz
        """
        if name not in cls._retrievers:
            available = ', '.join(cls._retrievers.keys())
            raise ValidationError(f"Unbekannter Retriever '{name}'. Verfügbar: {available}")
        
        retriever_class = cls._retrievers[name]
        return retriever_class(config, **kwargs)
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Prüft ob Retriever registriert ist"""
        return name in cls._retrievers


# =============================================================================
# RETRIEVER PROTOCOL FÜR DUCK TYPING
# =============================================================================

class RetrieverProtocol(Protocol):
    """
    Protocol für Duck Typing von Retriever-ähnlichen Objekten
    
    Ermöglicht Verwendung von Objekten als Retriever auch ohne
    explizite BaseRetriever-Vererbung, solange sie die benötigten
    Methoden implementieren.
    """
    
    def retrieve(self, 
                query: Union[str, RetrievalQuery],
                k: Optional[int] = None,
                filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """Retrieval-Methode"""
        ...
    
    def health_check(self) -> Dict[str, Any]:
        """Health-Check-Methode"""
        ...
    
    @property
    def name(self) -> str:
        """Name-Property"""
        ...


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_retriever_config(name: str,
                           cache_enabled: bool = True,
                           performance_monitoring: bool = True,
                           **kwargs) -> RetrieverConfig:
    """
    Convenience-Funktion für Retriever-Config-Erstellung
    
    Args:
        name: Name des Retrievers
        cache_enabled: Cache aktivieren
        performance_monitoring: Performance-Monitoring aktivieren
        **kwargs: Zusätzliche Config-Parameter
        
    Returns:
        RetrieverConfig: Konfigurierte RetrieverConfig-Instanz
    """
    return RetrieverConfig(
        name=name,
        cache_enabled=cache_enabled,
        performance_monitoring=performance_monitoring,
        **kwargs
    )


def validate_retrieval_result(result: RetrievalResult) -> bool:
    """
    Validiert Retrieval-Result auf Korrektheit
    
    Args:
        result: Zu validierendes RetrievalResult
        
    Returns:
        bool: True wenn valid
    """
    try:
        # Basis-Validierung
        if not isinstance(result, RetrievalResult):
            return False
        
        if not isinstance(result.documents, list):
            return False
        
        # Dokument-Score-Validierung
        for doc, score in result.documents:
            if not isinstance(doc, Document):
                return False
            if not isinstance(score, (int, float)):
                return False
            if score < 0.0:
                return False
        
        # Processing-Time sollte >= 0 sein
        if result.processing_time_ms < 0:
            return False
        
        return True
        
    except Exception:
        return False


# =============================================================================
# ABSTRACT BATCH RETRIEVER
# =============================================================================

class BatchRetriever(BaseRetriever):
    """
    Abstract Batch Retriever für effiziente Multi-Query-Verarbeitung
    
    Erweitert BaseRetriever um Batch-Processing-Fähigkeiten für
    Anwendungsfälle wo mehrere Queries gleichzeitig verarbeitet werden müssen.
    """
    
    def __init__(self, config: RetrieverConfig):
        """Initialisiert Batch Retriever"""
        super().__init__(config)
        self._batch_stats = {
            'total_batches': 0,
            'total_queries_in_batches': 0,
            'average_batch_size': 0.0,
            'batch_processing_time_ms': 0.0
        }
    
    @abstractmethod
    def _retrieve_documents_batch(self, queries: List[RetrievalQuery]) -> List[List[Tuple[Document, float]]]:
        """
        Abstrakte Methode für Batch-Retrieval
        
        Args:
            queries: Liste von Retrieval-Queries
            
        Returns:
            List[List[Tuple[Document, float]]]: Ergebnisse für jede Query
        """
        pass
    
    def retrieve_batch(self, 
                      queries: List[Union[str, RetrievalQuery]],
                      k: Optional[int] = None,
                      filters: Optional[Dict[str, Any]] = None,
                      mode: Optional[RetrievalMode] = None) -> List[RetrievalResult]:
        """
        Batch-Retrieval für multiple Queries
        
        Args:
            queries: Liste von Query-Texten oder RetrievalQuery-Objekten
            k: Anzahl Ergebnisse pro Query
            filters: Metadaten-Filter für alle Queries
            mode: Retrieval-Modus für alle Queries
            
        Returns:
            List[RetrievalResult]: Ergebnisse für jede Query
        """
        start_time = time.time()
        
        try:
            # Queries normalisieren
            normalized_queries = []
            for query in queries:
                if isinstance(query, str):
                    query_obj = RetrievalQuery(
                        text=query,
                        k=k or self.config.default_k,
                        filters=filters,
                        mode=mode or RetrievalMode.BALANCED
                    )
                else:
                    query_obj = query
                    # Parameter überschreiben wenn explizit angegeben
                    if k is not None:
                        query_obj.k = k
                    if filters is not None:
                        query_obj.filters = filters
                    if mode is not None:
                        query_obj.mode = mode
                
                normalized_queries.append(query_obj)
            
            # Batch-Retrieval ausführen
            batch_results = self._retrieve_documents_batch(normalized_queries)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Einzelne RetrievalResult-Objekte erstellen
            results = []
            for i, (query_obj, documents) in enumerate(zip(normalized_queries, batch_results)):
                # Score-Filtering anwenden
                if query_obj.score_threshold:
                    documents = [
                        (doc, score) for doc, score in documents 
                        if score >= query_obj.score_threshold
                    ]
                
                # Ergebnisse auf k begrenzen
                documents = documents[:query_obj.k]
                
                result = RetrievalResult(
                    documents=documents,
                    query=query_obj,
                    total_found=len(documents),
                    processing_time_ms=processing_time_ms / len(normalized_queries),  # Aufgeteilt
                    metadata={
                        'retriever': self.config.name,
                        'batch_processing': True,
                        'batch_index': i,
                        'batch_size': len(normalized_queries)
                    }
                )
                
                results.append(result)
            
            # Batch-Statistiken aktualisieren
            self._update_batch_stats(len(normalized_queries), processing_time_ms)
            
            self.logger.debug(
                f"Batch-Retrieval abgeschlossen: {len(normalized_queries)} Queries "
                f"in {processing_time_ms:.1f}ms ({processing_time_ms/len(normalized_queries):.1f}ms/query)"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch-Retrieval fehlgeschlagen: {str(e)}", exc_info=True)
            
            # Fallback: Einzelne Queries verarbeiten
            return [self.retrieve(query, k, filters, mode) for query in queries]
    
    def _update_batch_stats(self, batch_size: int, processing_time_ms: float):
        """Aktualisiert Batch-Processing-Statistiken"""
        self._batch_stats['total_batches'] += 1
        self._batch_stats['total_queries_in_batches'] += batch_size
        self._batch_stats['batch_processing_time_ms'] += processing_time_ms
        
        # Durchschnittliche Batch-Größe neu berechnen
        self._batch_stats['average_batch_size'] = (
            self._batch_stats['total_queries_in_batches'] / 
            self._batch_stats['total_batches']
        )
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """
        Holt Batch-Processing-Statistiken
        
        Returns:
            Dict: Batch-Performance-Metriken
        """
        stats = self._batch_stats.copy()
        
        if stats['total_batches'] > 0:
            stats['average_processing_time_per_batch_ms'] = (
                stats['batch_processing_time_ms'] / stats['total_batches']
            )
            stats['average_processing_time_per_query_in_batch_ms'] = (
                stats['batch_processing_time_ms'] / stats['total_queries_in_batches']
            )
        
        return stats


# =============================================================================
# MOCK RETRIEVER FÜR TESTING
# =============================================================================

class MockRetriever(BaseRetriever):
    """
    Mock Retriever für Testing und Entwicklung
    
    Implementiert BaseRetriever mit simulierten Ergebnissen für
    Unit-Tests und Entwicklung ohne echte Datenquellen.
    """
    
    def __init__(self, config: RetrieverConfig, mock_documents: Optional[List[Document]] = None):
        """
        Initialisiert Mock Retriever
        
        Args:
            config: Retriever-Konfiguration
            mock_documents: Liste von Mock-Dokumenten (optional)
        """
        super().__init__(config)
        
        # Mock-Dokumente
        self.mock_documents = mock_documents or self._create_default_mock_documents()
        
        self.logger.info(f"MockRetriever mit {len(self.mock_documents)} Mock-Dokumenten initialisiert")
    
    def _create_default_mock_documents(self) -> List[Document]:
        """Erstellt Standard-Mock-Dokumente"""
        return [
            Document(
                content="Dies ist ein Mock-Dokument über Industrieanlagen. Es enthält Informationen über Sicherheitsrichtlinien und Wartungsverfahren.",
                metadata={"title": "Sicherheitsrichtlinien", "page": 1, "source": "manual_001.pdf"},
                doc_id="mock_doc_1",
                source="mock_manual_001.pdf"
            ),
            Document(
                content="Technische Spezifikationen für Pumpen und Motoren. Betriebsspannung 400V AC, 50Hz. Schutzart IP65.",
                metadata={"title": "Technische Daten", "page": 5, "source": "specs_001.pdf"},
                doc_id="mock_doc_2", 
                source="mock_specs_001.pdf"
            ),
            Document(
                content="Anleitung zur Installation und Konfiguration. Schritt 1: Spannungsversorgung prüfen. Schritt 2: Anschlüsse kontrollieren.",
                metadata={"title": "Installation", "page": 12, "source": "install_001.pdf"},
                doc_id="mock_doc_3",
                source="mock_install_001.pdf"
            ),
            Document(
                content="Fehlerbehebung und Diagnose. Bei Störungen prüfen Sie zunächst die Sicherungen und Verbindungen.",
                metadata={"title": "Troubleshooting", "page": 25, "source": "troubleshoot_001.pdf"},
                doc_id="mock_doc_4",
                source="mock_troubleshoot_001.pdf"
            ),
            Document(
                content="Wartungsplan und Inspektionsintervalle. Monatliche Kontrolle erforderlich. Jährliche Hauptwartung.",
                metadata={"title": "Wartung", "page": 8, "source": "maintenance_001.pdf"}, 
                doc_id="mock_doc_5",
                source="mock_maintenance_001.pdf"
            )
        ]
    
    def _retrieve_documents(self, query: RetrievalQuery) -> List[Tuple[Document, float]]:
        """
        Simuliert Dokumenten-Retrieval mit Mock-Daten
        
        Args:
            query: Retrieval-Query
            
        Returns:
            List[Tuple[Document, float]]: Mock-Dokumente mit simulierten Scores
        """
        import random
        
        # Simuliere Verarbeitungszeit
        import time
        time.sleep(0.001)  # 1ms Delay
        
        # Einfache Keyword-Matching-Simulation
        query_tokens = set(query.text.lower().split())
        
        scored_documents = []
        for doc in self.mock_documents:
            # Simuliere Relevanz-Score basierend auf Token-Überlappung
            doc_tokens = set(doc.content.lower().split())
            overlap = len(query_tokens.intersection(doc_tokens))
            
            if overlap > 0:
                # Base-Score aus Token-Überlappung
                base_score = min(overlap / len(query_tokens), 1.0)
                
                # Füge etwas Zufälligkeit hinzu für Realismus
                noise = random.uniform(-0.1, 0.1)
                final_score = max(0.0, min(1.0, base_score + noise))
                
                # Filter anwenden wenn vorhanden
                if query.filters:
                    matches_filter = True
                    for filter_key, filter_value in query.filters.items():
                        if filter_key in doc.metadata:
                            if doc.metadata[filter_key] != filter_value:
                                matches_filter = False
                                break
                    
                    if not matches_filter:
                        continue
                
                scored_documents.append((doc, final_score))
        
        # Nach Score sortieren (höchste zuerst)
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        
        return scored_documents
    
    def add_mock_document(self, document: Document):
        """Fügt Mock-Dokument hinzu"""
        self.mock_documents.append(document)
        self.logger.debug(f"Mock-Dokument hinzugefügt: {document.doc_id}")
    
    def set_mock_documents(self, documents: List[Document]):
        """Setzt Mock-Dokumente"""
        self.mock_documents = documents
        self.logger.info(f"Mock-Dokumente gesetzt: {len(documents)} Dokumente")
    
    def _custom_health_check(self) -> Optional[Dict[str, Any]]:
        """Custom Health-Check für Mock Retriever"""
        return {
            'mock_documents_count': len(self.mock_documents),
            'mock_mode': True,
            'ready_for_testing': True
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core Classes
    'BaseRetriever',
    'BatchRetriever',
    'MockRetriever',
    
    # Data Structures
    'Document',
    'RetrievalQuery',
    'RetrievalResult',
    'RetrieverConfig',
    'RetrievalMode',
    
    # Registry und Factory
    'RetrieverRegistry',
    'RetrieverProtocol',
    
    # Utility Functions
    'create_retriever_config',
    'validate_retrieval_result',
]