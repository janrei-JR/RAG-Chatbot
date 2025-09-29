#!/usr/bin/env python3
"""
Search Service - Controller-Layer Kompatibilitäts-Wrapper
Industrielle RAG-Architektur - Phase 1 Migration

Alias/Wrapper für RetrievalService um Controller-Layer Import-Kompatibilität
zu gewährleisten. Bietet vereinfachte API für Pipeline-Controller Integration.

Controller-orientierte Features:
- Vereinfachte Search-API für Pipeline-Controller
- Result-Format-Anpassung für Controller-Erwartungen
- Error-Handling mit Controller-spezifischen Exception-Codes
- Backward-Kompatibilität für bestehende Controller-Integrationen

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

import time
import threading
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime, timezone

# Core-Komponenten
from core import (
    get_logger, get_config, RAGConfig,
    ServiceError, ValidationError, create_error_context,
    log_performance, log_method_calls
)

# Service-Integrationen
from services.retrieval_service import (
    RetrievalService, get_retrieval_service, create_retrieval_service,
    RetrievalRequest, RetrievalResult, RetrievedDocument,
    QueryType, RetrievalStrategy, RetrievalMode
)
from services.vector_store_service import VectorStoreService
from services.embedding_service import EmbeddingService


# =============================================================================
# SEARCH SERVICE DATENSTRUKTUREN
# =============================================================================

@dataclass
class SearchRequest:
    """Vereinfachte Search-Request für Controller-Integration"""
    query: str
    collection_name: str = "default"
    max_results: int = 5
    strategy: str = "adaptive"  # String statt Enum für Controller-Kompatibilität
    score_threshold: float = 0.0
    include_metadata: bool = True
    
    def to_retrieval_request(self) -> RetrievalRequest:
        """Konvertiert zu RetrievalRequest"""
        # String zu Enum-Mapping
        strategy_mapping = {
            "semantic": RetrievalStrategy.SEMANTIC,
            "keyword": RetrievalStrategy.KEYWORD,
            "hybrid": RetrievalStrategy.HYBRID,
            "mmr": RetrievalStrategy.MMR,
            "adaptive": RetrievalStrategy.ADAPTIVE,
            "multi_strategy": RetrievalStrategy.MULTI_STRATEGY
        }
        
        strategy_enum = strategy_mapping.get(self.strategy.lower(), RetrievalStrategy.ADAPTIVE)
        
        return RetrievalRequest(
            query=self.query,
            collection_name=self.collection_name,
            k=self.max_results,
            strategy=strategy_enum,
            score_threshold=self.score_threshold
        )


@dataclass 
class SearchResult:
    """Controller-kompatibles Search-Ergebnis"""
    success: bool
    results: List[Dict[str, Any]]
    total_found: int
    query: str
    processing_time_ms: float
    strategy_used: str
    error_message: Optional[str] = None
    
    @classmethod
    def from_retrieval_result(cls, retrieval_result: RetrievalResult) -> 'SearchResult':
        """Erstellt SearchResult aus RetrievalResult"""
        results = []
        
        for doc in retrieval_result.documents:
            result_dict = {
                'content': doc.content,
                'score': doc.relevance_score,
                'source': doc.metadata.get('source', ''),
                'page': doc.metadata.get('page', 0),
                'document_id': doc.document_id
            }
            
            # Optional: Erweiterte Metadaten
            if doc.metadata:
                result_dict['metadata'] = doc.metadata
            
            results.append(result_dict)
        
        return cls(
            success=retrieval_result.success,
            results=results,
            total_found=retrieval_result.total_found,
            query=retrieval_result.query,
            processing_time_ms=retrieval_result.processing_time_ms,
            strategy_used=retrieval_result.strategy_used.value if retrieval_result.strategy_used else "unknown",
            error_message=retrieval_result.error_message
        )


# =============================================================================
# SEARCH SERVICE IMPLEMENTIERUNG
# =============================================================================

class SearchService:
    """
    Search Service - Controller-Layer Kompatibilitäts-Wrapper
    
    Vereinfachte Schnittstelle für Controller-Integration mit:
    - String-basierte API (keine Enums für Controller-Kompatibilität)
    - Vereinfachte Result-Formate
    - Controller-spezifisches Error-Handling
    - Backward-Kompatibilität für bestehende Integrationen
    """
    
    def __init__(self, 
                 retrieval_service: RetrievalService = None,
                 vector_store_service: VectorStoreService = None,
                 embedding_service: EmbeddingService = None,
                 config: RAGConfig = None):
        """
        Initialisiert Search Service
        
        Args:
            retrieval_service: RetrievalService-Instanz (optional, wird automatisch erstellt)
            vector_store_service: Vector Store Service (für RetrievalService-Erstellung)
            embedding_service: Embedding Service (optional)
            config: RAG-Konfiguration (optional)
        """
        self.logger = get_logger(__name__)
        self.config = config or get_config()
        
        # RetrievalService initialisieren oder verwenden
        if retrieval_service:
            self.retrieval_service = retrieval_service
        else:
            if not vector_store_service:
                raise ValueError("vector_store_service ist erforderlich wenn kein retrieval_service übergeben wird")
            
            self.retrieval_service = create_retrieval_service(
                vector_store_service=vector_store_service,
                embedding_service=embedding_service,
                config=config
            )
        
        # Service-Statistiken
        self._search_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'total_time_ms': 0.0,
            'avg_time_ms': 0.0
        }
        self._stats_lock = threading.Lock()
        
        self.logger.info("Search Service als RetrievalService-Wrapper initialisiert")
    
    # =============================================================================
    # PUBLIC API - VEREINFACHTE SEARCH-METHODEN
    # =============================================================================
    
    @log_method_calls
    @log_performance
    def search(self, 
               query: str,
               collection_name: str = "default", 
               max_results: int = 5,
               strategy: str = "adaptive") -> SearchResult:
        """
        Hauptsuchmethode mit vereinfachter API
        
        Args:
            query: Suchquery
            collection_name: Name der Collection
            max_results: Maximale Anzahl Ergebnisse
            strategy: Such-Strategie ("semantic", "keyword", "hybrid", "mmr", "adaptive")
            
        Returns:
            SearchResult: Suchergebnisse
        """
        start_time = time.time()
        
        try:
            # SearchRequest erstellen
            search_request = SearchRequest(
                query=query,
                collection_name=collection_name,
                max_results=max_results,
                strategy=strategy
            )
            
            # An RetrievalService delegieren
            retrieval_request = search_request.to_retrieval_request()
            retrieval_result = self.retrieval_service.retrieve_documents(retrieval_request)
            
            # Result konvertieren
            search_result = SearchResult.from_retrieval_result(retrieval_result)
            
            # Statistiken aktualisieren
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(success=search_result.success, time_ms=processing_time)
            
            return search_result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self._update_stats(success=False, time_ms=processing_time)
            
            error_context = create_error_context(
                component="services.search_service",
                operation="search",
                query=query
            )
            
            self.logger.error(f"Search fehlgeschlagen: {str(e)}")
            
            return SearchResult(
                success=False,
                results=[],
                total_found=0,
                query=query,
                processing_time_ms=processing_time,
                strategy_used=strategy,
                error_message=str(e)
            )
    
    def search_advanced(self, search_request: SearchRequest) -> SearchResult:
        """
        Erweiterte Suchmethode mit SearchRequest-Objekt
        
        Args:
            search_request: Detaillierte Suchparameter
            
        Returns:
            SearchResult: Suchergebnisse
        """
        return self.search(
            query=search_request.query,
            collection_name=search_request.collection_name,
            max_results=search_request.max_results,
            strategy=search_request.strategy
        )
    
    def semantic_search(self, 
                       query: str, 
                       collection_name: str = "default", 
                       max_results: int = 5) -> SearchResult:
        """Semantische Suche (Shortcut-Methode)"""
        return self.search(query, collection_name, max_results, "semantic")
    
    def keyword_search(self, 
                      query: str, 
                      collection_name: str = "default", 
                      max_results: int = 5) -> SearchResult:
        """Keyword-basierte Suche (Shortcut-Methode)"""
        return self.search(query, collection_name, max_results, "keyword")
    
    def hybrid_search(self, 
                     query: str, 
                     collection_name: str = "default", 
                     max_results: int = 5) -> SearchResult:
        """Hybrid-Suche (Shortcut-Methode)"""
        return self.search(query, collection_name, max_results, "hybrid")
    
    # =============================================================================
    # CONTROLLER-KOMPATIBLE METHODEN
    # =============================================================================
    
    def execute_search(self, 
                      query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Controller-kompatible execute_search Methode
        
        Für Integration mit controllers/pipeline_controller.py
        
        Args:
            query_params: Dictionary mit Suchparametern
            
        Returns:
            Dict: Suchergebnisse im Controller-erwarteten Format
        """
        try:
            # Parameter extrahieren
            query = query_params.get('query', '')
            collection = query_params.get('collection_name', 'default')
            k = query_params.get('k', 5)
            strategy = query_params.get('strategy', 'adaptive')
            
            if not query:
                return {
                    'success': False,
                    'error': 'Query-Parameter fehlt',
                    'results': [],
                    'total_found': 0
                }
            
            # Suche ausführen
            result = self.search(
                query=query,
                collection_name=collection,
                max_results=k,
                strategy=strategy
            )
            
            # Controller-Format
            return {
                'success': result.success,
                'results': result.results,
                'total_found': result.total_found,
                'processing_time_ms': result.processing_time_ms,
                'strategy_used': result.strategy_used,
                'error': result.error_message
            }
            
        except Exception as e:
            self.logger.error(f"Controller-Search fehlgeschlagen: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'results': [],
                'total_found': 0
            }
    
    def search_documents(self, 
                        query: str, 
                        **kwargs) -> List[Dict[str, Any]]:
        """
        Legacy-kompatible search_documents Methode
        
        Für Backward-Kompatibilität mit bestehenden Controller-Integrationen
        
        Args:
            query: Suchquery
            **kwargs: Zusätzliche Parameter
            
        Returns:
            List[Dict]: Liste der gefundenen Dokumente
        """
        try:
            collection_name = kwargs.get('collection_name', 'default')
            max_results = kwargs.get('max_results', kwargs.get('k', 5))
            strategy = kwargs.get('strategy', 'adaptive')
            
            result = self.search(
                query=query,
                collection_name=collection_name,
                max_results=max_results,
                strategy=strategy
            )
            
            return result.results if result.success else []
            
        except Exception as e:
            self.logger.error(f"Legacy search_documents fehlgeschlagen: {str(e)}")
            return []
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _update_stats(self, success: bool, time_ms: float) -> None:
        """Aktualisiert Service-Statistiken"""
        with self._stats_lock:
            self._search_stats['total_searches'] += 1
            self._search_stats['total_time_ms'] += time_ms
            
            if success:
                self._search_stats['successful_searches'] += 1
            else:
                self._search_stats['failed_searches'] += 1
            
            # Durchschnittszeit berechnen
            if self._search_stats['total_searches'] > 0:
                self._search_stats['avg_time_ms'] = (
                    self._search_stats['total_time_ms'] / 
                    self._search_stats['total_searches']
                )
    
    def get_available_strategies(self) -> List[str]:
        """Holt verfügbare Such-Strategien"""
        return ["semantic", "keyword", "hybrid", "mmr", "adaptive", "multi_strategy"]
    
    def validate_strategy(self, strategy: str) -> bool:
        """Validiert Such-Strategie"""
        return strategy.lower() in self.get_available_strategies()
    
    # =============================================================================
    # SERVICE MANAGEMENT
    # =============================================================================
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Holt Service-Statistiken"""
        with self._stats_lock:
            stats = self._search_stats.copy()
        
        # RetrievalService-Stats hinzufügen
        retrieval_stats = self.retrieval_service.get_service_stats()
        stats['retrieval_service_stats'] = retrieval_stats
        
        return stats
    
    def get_service_health(self) -> Dict[str, Any]:
        """Holt Service-Health-Status"""
        try:
            # Eigene Health-Daten
            health_data = {
                'status': 'healthy',
                'search_service_available': True,
                'retrieval_service_available': bool(self.retrieval_service),
                'service_stats': self.get_service_stats(),
                'last_check': datetime.now(timezone.utc).isoformat()
            }
            
            # RetrievalService Health prüfen
            try:
                retrieval_health = self.retrieval_service.get_service_health()
                health_data['retrieval_service_health'] = retrieval_health
                
                if retrieval_health.get('status') != 'healthy':
                    health_data['status'] = 'degraded'
            except:
                health_data['retrieval_service_health'] = {'status': 'error'}
                health_data['status'] = 'degraded'
            
            return health_data
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now(timezone.utc).isoformat()
            }
    
    def reset_stats(self) -> None:
        """Setzt Service-Statistiken zurück"""
        with self._stats_lock:
            self._search_stats = {
                'total_searches': 0,
                'successful_searches': 0,
                'failed_searches': 0,
                'total_time_ms': 0.0,
                'avg_time_ms': 0.0
            }
        self.logger.info("Search Service Statistiken zurückgesetzt")
    
    # =============================================================================
    # DELEGATION AN RETRIEVAL SERVICE
    # =============================================================================
    
    def clear_cache(self) -> None:
        """Delegiert Cache-Clearing an RetrievalService"""
        self.retrieval_service.clear_cache()
    
    def get_query_analyzer(self):
        """Holt QueryAnalyzer vom RetrievalService"""
        return getattr(self.retrieval_service, 'query_analyzer', None)
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analysiert Query und gibt Ergebnis zurück
        
        Args:
            query: Zu analysierende Query
            
        Returns:
            Dict: Query-Analyse-Ergebnis
        """
        try:
            analyzer = self.get_query_analyzer()
            if analyzer:
                analysis = analyzer.analyze_query(query)
                return {
                    'query_type': analysis.query_type.value,
                    'suggested_strategy': analysis.suggested_strategy.value,
                    'expected_results': analysis.expected_result_count,
                    'domain_keywords': analysis.domain_keywords,
                    'technical_terms': analysis.technical_terms,
                    'is_technical': analysis.is_technical_query
                }
            else:
                return {'error': 'QueryAnalyzer nicht verfügbar'}
                
        except Exception as e:
            self.logger.error(f"Query-Analyse fehlgeschlagen: {str(e)}")
            return {'error': str(e)}


# =============================================================================
# FACTORY FUNCTIONS UND SINGLETON
# =============================================================================

_search_service_instance: Optional[SearchService] = None
_service_lock = threading.Lock()

def get_search_service(
    retrieval_service: RetrievalService = None,
    vector_store_service: VectorStoreService = None,
    embedding_service: EmbeddingService = None,
    config: RAGConfig = None
) -> SearchService:
    """
    Holt Search Service Singleton-Instanz
    
    Args:
        retrieval_service: RetrievalService-Instanz (optional)
        vector_store_service: Vector Store Service (für RetrievalService-Erstellung)
        embedding_service: Embedding Service (optional)
        config: RAG-Konfiguration (optional)
        
    Returns:
        SearchService: Service-Instanz
    """
    global _search_service_instance
    
    if _search_service_instance is None:
        with _service_lock:
            if _search_service_instance is None:
                _search_service_instance = SearchService(
                    retrieval_service=retrieval_service,
                    vector_store_service=vector_store_service,
                    embedding_service=embedding_service,
                    config=config
                )
    
    return _search_service_instance


def create_search_service(
    retrieval_service: RetrievalService = None,
    vector_store_service: VectorStoreService = None,
    embedding_service: EmbeddingService = None,
    config: RAGConfig = None
) -> SearchService:
    """
    Erstellt neue Search Service Instanz (für Testing/Multi-Instance)
    
    Args:
        retrieval_service: RetrievalService-Instanz (optional)
        vector_store_service: Vector Store Service (für RetrievalService-Erstellung)
        embedding_service: Embedding Service (optional)
        config: RAG-Konfiguration (optional)
        
    Returns:
        SearchService: Neue Service-Instanz
    """
    return SearchService(
        retrieval_service=retrieval_service,
        vector_store_service=vector_store_service,
        embedding_service=embedding_service,
        config=config
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Haupt-Service-Klasse
    'SearchService',
    
    # Datenstrukturen
    'SearchRequest', 'SearchResult',
    
    # Factory-Funktionen
    'get_search_service', 'create_search_service'
]
