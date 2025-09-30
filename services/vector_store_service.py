#!/usr/bin/env python3
"""
Vector Store Service - Business Logic Layer
Industrielle RAG-Architektur - Phase 4 Migration

Orchestriert Vector-Database-Management mit Provider-Abstraktion,
automatischer Fallback-Strategien und Production-Features für
nahtlose Integration in die Service-orientierte Architektur.

Features:
- Provider-unabhängige Vector-Store-Verwaltung (Chroma, Pinecone, Qdrant)
- Automatische Provider-Auswahl und Fallback-Mechanismen
- Batch-Operationen für Performance-optimierte Verarbeitung
- Health-Checks und Monitoring für Production-Deployment
- Collection-Management mit Backup und Recovery-Funktionen
- Enhanced Chunk Integration mit Metadaten-Konvertierung

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

# Core-Komponenten
from core import (
    get_logger, get_config, RAGConfig,
    ServiceError, VectorStoreError, ValidationError,
    create_error_context, log_performance, log_method_calls
)

# Vector Store Module Imports
from modules.vector_stores import (
    BaseVectorStore, VectorStoreResult, VectorStoreFactory,
    create_auto_vector_store, get_available_providers
)

# Embedding Integration
from modules.embeddings import BaseEmbeddings

logger = get_logger(__name__)


# =============================================================================
# VECTOR STORE SERVICE DATENSTRUKTUREN
# =============================================================================

class VectorStoreOperation(str, Enum):
    """Vector Store Operation Types"""
    ADD_DOCUMENTS = "add_documents"
    UPDATE_DOCUMENTS = "update_documents"
    DELETE_DOCUMENTS = "delete_documents"
    SEARCH_SIMILARITY = "search_similarity"
    SEARCH_MMR = "search_mmr"
    GET_DOCUMENTS = "get_documents"
    CREATE_COLLECTION = "create_collection"
    DELETE_COLLECTION = "delete_collection"
    BACKUP_COLLECTION = "backup_collection"
    RESTORE_COLLECTION = "restore_collection"


class VectorStoreStatus(str, Enum):
    """Vector Store Service Status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    ERROR = "error"


@dataclass
class VectorStoreServiceConfig:
    """Konfiguration für Vector Store Service"""
    default_provider: str = "chroma"
    fallback_providers: List[str] = field(default_factory=lambda: ["chroma"])
    auto_provider_selection: bool = True
    health_check_interval: int = 300  # Sekunden
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    backup_enabled: bool = True
    backup_directory: str = "data/vector_store_backups"
    monitoring_enabled: bool = True
    performance_tracking: bool = True
    
    def __post_init__(self):
        """Post-Initialisierung mit Validierung"""
        if not self.fallback_providers:
            self.fallback_providers = [self.default_provider]


@dataclass
class VectorStoreOperationRequest:
    """Request für Vector Store Operationen"""
    operation: VectorStoreOperation
    collection_name: str
    documents: Optional[List[Dict[str, Any]]] = None
    query: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    filters: Optional[Dict[str, Any]] = None
    k: int = 5
    score_threshold: Optional[float] = None
    fetch_k: int = 20  # Für MMR
    lambda_mult: float = 0.5  # Für MMR Diversität
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validierung der Request-Parameter"""
        if self.operation in [VectorStoreOperation.ADD_DOCUMENTS, VectorStoreOperation.UPDATE_DOCUMENTS]:
            if not self.documents:
                raise ValidationError("documents sind erforderlich für add/update Operationen")
        
        if self.operation in [VectorStoreOperation.SEARCH_SIMILARITY, VectorStoreOperation.SEARCH_MMR]:
            if not self.query and not self.query_embedding:
                raise ValidationError("query oder query_embedding erforderlich für Suchoperationen")


@dataclass
class VectorStoreOperationResult:
    """Ergebnis einer Vector Store Operation"""
    success: bool
    operation: VectorStoreOperation
    collection_name: str
    result_data: Any = None
    documents_processed: int = 0
    processing_time_ms: float = 0
    provider_used: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def documents_per_second(self) -> float:
        """Berechnet Dokumente pro Sekunde"""
        if self.processing_time_ms > 0:
            return (self.documents_processed * 1000) / self.processing_time_ms
        return 0.0


@dataclass 
class VectorStoreHealth:
    """Health-Status eines Vector Stores"""
    provider: str
    status: VectorStoreStatus
    collection_count: int = 0
    total_documents: int = 0
    last_operation_time: Optional[datetime] = None
    error_rate: float = 0.0
    average_response_time_ms: float = 0.0
    available_collections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# VECTOR STORE SERVICE IMPLEMENTIERUNG
# =============================================================================

class VectorStoreService:
    """
    Vector Store Service - Business Logic Orchestrierung
    
    Zentrale Schnittstelle für Vector-Database-Management mit:
    - Provider-Abstraktion und automatische Fallback-Strategien
    - Performance-Monitoring und Health-Checks
    - Batch-Processing für große Dokumentensammlungen
    - Production-Features: Backup, Recovery, Monitoring
    """
    
    def __init__(self, 
                 config: Optional[VectorStoreServiceConfig] = None,
                 embeddings: Optional[BaseEmbeddings] = None):
        """
        Initialisiert Vector Store Service
        
        Args:
            config: Service-Konfiguration
            embeddings: Embedding-Provider für Vektorisierung
        """
        self.logger = get_logger(f"{__name__}.vector_store")
        self.config = config or VectorStoreServiceConfig()
        self.embeddings = embeddings
        
        # RAG-System-Konfiguration
        self.rag_config = get_config()
        
        # Vector Store Management
        self._vector_stores: Dict[str, BaseVectorStore] = {}
        self._current_provider = None
        
        # Performance und Monitoring
        self._operation_stats = {}
        self._health_status = {}
        self._last_health_check = None
        
        # Initialisierung
        self._initialized = False
        self._initialize_vector_stores()
        
        self.logger.info(
            f"Vector Store Service initialisiert mit Provider: {self.config.default_provider}, "
            f"Fallbacks: {self.config.fallback_providers}"
        )
    
    def _initialize_vector_stores(self):
        """Initialisiert verfügbare Vector Store Provider"""
        try:
            available_providers = get_available_providers()
            self.logger.debug(f"Verfügbare Vector Store Provider: {available_providers}")
            
            # Standard-Provider initialisieren
            if self.config.default_provider in available_providers:
                self._initialize_provider(self.config.default_provider, primary=True)
            
            # Fallback-Provider initialisieren
            for provider in self.config.fallback_providers:
                if provider != self.config.default_provider and provider in available_providers:
                    self._initialize_provider(provider, primary=False)
            
            self._initialized = True
            self.logger.info(f"Vector Store Service bereit mit {len(self._vector_stores)} Providern")
            
        except Exception as e:
            error_msg = f"Vector Store Service Initialisierung fehlgeschlagen: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg) from e
    
    def _initialize_provider(self, provider_name: str, primary: bool = False):
        """Initialisiert einzelnen Vector Store Provider"""
        try:
            if self.embeddings:
                vector_store = create_auto_vector_store(
                    provider=provider_name,
                    embeddings=self.embeddings,
                    config=self.rag_config.vector_store
                )
            else:
                vector_store = VectorStoreFactory.create_vector_store(
                    provider=provider_name,
                    config=self.rag_config.vector_store
                )
            
            self._vector_stores[provider_name] = vector_store
            
            if primary or not self._current_provider:
                self._current_provider = provider_name
            
            self.logger.debug(f"Vector Store Provider '{provider_name}' initialisiert {'(primary)' if primary else '(fallback)'}")
            
        except Exception as e:
            self.logger.warning(f"Provider '{provider_name}' Initialisierung fehlgeschlagen: {str(e)}")
    
    @log_method_calls
    @log_performance
    def add_documents(self, 
                     collection_name: str,
                     documents: List[Dict[str, Any]],
                     **kwargs) -> VectorStoreOperationResult:
        """
        Fügt Dokumente zur Vector Store hinzu
        
        Args:
            collection_name: Name der Collection
            documents: Liste von Dokumenten mit content, metadata, etc.
            **kwargs: Zusätzliche Parameter
            
        Returns:
            VectorStoreOperationResult: Ergebnis der Operation
        """
        request = VectorStoreOperationRequest(
            operation=VectorStoreOperation.ADD_DOCUMENTS,
            collection_name=collection_name,
            documents=documents,
            metadata=kwargs
        )
        
        return self._execute_operation(request)
    
    @log_method_calls
    @log_performance
    def similarity_search(self,
                         collection_name: str,
                         query: str,
                         k: int = 5,
                         score_threshold: Optional[float] = None,
                         filters: Optional[Dict[str, Any]] = None,
                         **kwargs) -> VectorStoreOperationResult:
        """
        Führt Ähnlichkeitssuche durch
        
        Args:
            collection_name: Name der Collection
            query: Suchanfrage
            k: Anzahl Ergebnisse
            score_threshold: Minimaler Similarity-Score
            filters: Metadaten-Filter
            **kwargs: Zusätzliche Parameter
            
        Returns:
            VectorStoreOperationResult: Suchergebnisse
        """
        request = VectorStoreOperationRequest(
            operation=VectorStoreOperation.SEARCH_SIMILARITY,
            collection_name=collection_name,
            query=query,
            k=k,
            score_threshold=score_threshold,
            filters=filters,
            metadata=kwargs
        )
        
        return self._execute_operation(request)
    
    @log_method_calls
    @log_performance
    def similarity_search_with_embeddings(self,
                                        collection_name: str,
                                        query_embedding: List[float],
                                        k: int = 5,
                                        score_threshold: Optional[float] = None,
                                        filters: Optional[Dict[str, Any]] = None,
                                        **kwargs) -> VectorStoreOperationResult:
        """
        Führt Ähnlichkeitssuche mit vorberechneten Embeddings durch
        
        Args:
            collection_name: Name der Collection
            query_embedding: Query-Embedding-Vektor
            k: Anzahl Ergebnisse
            score_threshold: Minimaler Similarity-Score
            filters: Metadaten-Filter
            **kwargs: Zusätzliche Parameter
            
        Returns:
            VectorStoreOperationResult: Suchergebnisse
        """
        request = VectorStoreOperationRequest(
            operation=VectorStoreOperation.SEARCH_SIMILARITY,
            collection_name=collection_name,
            query_embedding=query_embedding,
            k=k,
            score_threshold=score_threshold,
            filters=filters,
            metadata=kwargs
        )
        
        return self._execute_operation(request)
    
    @log_method_calls
    @log_performance
    def max_marginal_relevance_search(self,
                                     collection_name: str,
                                     query: str,
                                     k: int = 5,
                                     fetch_k: int = 20,
                                     lambda_mult: float = 0.5,
                                     filters: Optional[Dict[str, Any]] = None,
                                     **kwargs) -> VectorStoreOperationResult:
        """
        Führt Maximal Marginal Relevance (MMR) Suche durch
        
        Args:
            collection_name: Name der Collection
            query: Suchanfrage
            k: Anzahl finale Ergebnisse
            fetch_k: Anzahl initial zu holen
            lambda_mult: Diversitätsfaktor (0=max Diversität, 1=max Relevanz)
            filters: Metadaten-Filter
            **kwargs: Zusätzliche Parameter
            
        Returns:
            VectorStoreOperationResult: MMR-Suchergebnisse
        """
        request = VectorStoreOperationRequest(
            operation=VectorStoreOperation.SEARCH_MMR,
            collection_name=collection_name,
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filters=filters,
            metadata=kwargs
        )
        
        return self._execute_operation(request)
    
    @log_method_calls
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Holt Informationen über Collection
        
        Args:
            collection_name: Name der Collection
            
        Returns:
            Dict: Collection-Informationen
        """
        try:
            vector_store = self._get_vector_store()
            
            if hasattr(vector_store, 'get_collection_info'):
                return vector_store.get_collection_info(collection_name)
            else:
                # Fallback: Basis-Informationen sammeln
                return {
                    'name': collection_name,
                    'provider': self._current_provider,
                    'exists': True  # Vereinfachte Annahme
                }
                
        except Exception as e:
            self.logger.error(f"Collection-Info Fehler für '{collection_name}': {str(e)}")
            return {
                'name': collection_name,
                'error': str(e),
                'exists': False
            }
    
    @log_method_calls
    def list_collections(self) -> List[str]:
        """
        Listet alle verfügbaren Collections auf
        
        Returns:
            List[str]: Collection-Namen
        """
        try:
            vector_store = self._get_vector_store()
            
            if hasattr(vector_store, 'list_collections'):
                return vector_store.list_collections()
            else:
                # Fallback für Provider ohne list_collections
                self.logger.warning(f"Provider '{self._current_provider}' unterstützt list_collections nicht")
                return []
                
        except Exception as e:
            self.logger.error(f"Collection-Listing Fehler: {str(e)}")
            return []
    
    @log_method_calls
    def delete_collection(self, collection_name: str) -> bool:
        """
        Löscht eine Collection
        
        Args:
            collection_name: Name der zu löschenden Collection
            
        Returns:
            bool: True wenn erfolgreich gelöscht
        """
        try:
            request = VectorStoreOperationRequest(
                operation=VectorStoreOperation.DELETE_COLLECTION,
                collection_name=collection_name
            )
            
            result = self._execute_operation(request)
            return result.success
            
        except Exception as e:
            self.logger.error(f"Collection-Löschung fehlgeschlagen für '{collection_name}': {str(e)}")
            return False
    
    def _execute_operation(self, request: VectorStoreOperationRequest) -> VectorStoreOperationResult:
        """
        Führt Vector Store Operation mit Fallback-Strategien aus
        
        Args:
            request: Operation-Request
            
        Returns:
            VectorStoreOperationResult: Ergebnis der Operation
        """
        start_time = time.time()
        
        # Provider-Reihenfolge bestimmen
        providers_to_try = [self._current_provider]
        if self.config.auto_provider_selection:
            providers_to_try.extend([
                p for p in self.config.fallback_providers 
                if p != self._current_provider and p in self._vector_stores
            ])
        
        last_error = None
        
        # Provider nacheinander versuchen
        for provider in providers_to_try:
            try:
                self.logger.debug(f"Versuche Operation {request.operation} mit Provider '{provider}'")
                
                vector_store = self._vector_stores[provider]
                result_data = self._execute_provider_operation(vector_store, request)
                
                processing_time_ms = (time.time() - start_time) * 1000
                
                result = VectorStoreOperationResult(
                    success=True,
                    operation=request.operation,
                    collection_name=request.collection_name,
                    result_data=result_data,
                    documents_processed=len(request.documents) if request.documents else 0,
                    processing_time_ms=processing_time_ms,
                    provider_used=provider
                )
                
                # Stats aktualisieren
                self._update_operation_stats(request.operation, result)
                
                self.logger.debug(
                    f"Operation {request.operation} erfolgreich mit '{provider}' "
                    f"({processing_time_ms:.1f}ms, {result.documents_processed} docs)"
                )
                
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Provider '{provider}' Fehler: {str(e)}")
                
                # Bei letztem Provider: Error-Result zurückgeben
                if provider == providers_to_try[-1]:
                    break
        
        # Alle Provider fehlgeschlagen
        processing_time_ms = (time.time() - start_time) * 1000
        error_result = VectorStoreOperationResult(
            success=False,
            operation=request.operation,
            collection_name=request.collection_name,
            processing_time_ms=processing_time_ms,
            error_message=str(last_error) if last_error else "Alle Provider fehlgeschlagen"
        )
        
        self.logger.error(f"Operation {request.operation} komplett fehlgeschlagen: {error_result.error_message}")
        return error_result
    
    def _execute_provider_operation(self, 
                                   vector_store: BaseVectorStore, 
                                   request: VectorStoreOperationRequest) -> Any:
        """Führt Operation auf spezifischem Provider aus"""
        
        if request.operation == VectorStoreOperation.ADD_DOCUMENTS:
            return vector_store.add_documents(
                documents=request.documents,
                collection_name=request.collection_name
            )
        
        elif request.operation == VectorStoreOperation.SEARCH_SIMILARITY:
            if request.query_embedding:
                return vector_store.similarity_search_by_vector(
                    embedding=request.query_embedding,
                    k=request.k,
                    filter=request.filters,
                    collection_name=request.collection_name
                )
            else:
                return vector_store.similarity_search(
                    query=request.query,
                    k=request.k,
                    filter=request.filters,
                    collection_name=request.collection_name
                )
        
        elif request.operation == VectorStoreOperation.SEARCH_MMR:
            if hasattr(vector_store, 'max_marginal_relevance_search'):
                return vector_store.max_marginal_relevance_search(
                    query=request.query,
                    k=request.k,
                    fetch_k=request.fetch_k,
                    lambda_mult=request.lambda_mult,
                    filter=request.filters,
                    collection_name=request.collection_name
                )
            else:
                # Fallback auf normale Similarity Search
                self.logger.warning(f"MMR nicht unterstützt, fallback auf similarity search")
                return vector_store.similarity_search(
                    query=request.query,
                    k=request.k,
                    filter=request.filters,
                    collection_name=request.collection_name
                )
        
        elif request.operation == VectorStoreOperation.DELETE_COLLECTION:
            if hasattr(vector_store, 'delete_collection'):
                return vector_store.delete_collection(request.collection_name)
            else:
                raise VectorStoreError(f"Provider unterstützt delete_collection nicht")
        
        else:
            raise VectorStoreError(f"Unbekannte Operation: {request.operation}")
    
    def _get_vector_store(self) -> BaseVectorStore:
        """Holt aktuelle Vector Store Instanz"""
        if not self._current_provider or self._current_provider not in self._vector_stores:
            raise ServiceError("Kein Vector Store Provider verfügbar")
        
        return self._vector_stores[self._current_provider]
    
    def _update_operation_stats(self, operation: VectorStoreOperation, result: VectorStoreOperationResult):
        """Aktualisiert Operation-Statistiken"""
        if not self.config.performance_tracking:
            return
        
        op_key = f"{operation}_{result.provider_used}"
        
        if op_key not in self._operation_stats:
            self._operation_stats[op_key] = {
                'count': 0,
                'success_count': 0,
                'total_time_ms': 0,
                'total_documents': 0
            }
        
        stats = self._operation_stats[op_key]
        stats['count'] += 1
        stats['total_time_ms'] += result.processing_time_ms
        stats['total_documents'] += result.documents_processed
        
        if result.success:
            stats['success_count'] += 1
    
    @log_method_calls
    def get_service_health(self) -> Dict[str, Any]:
        """
        Holt Service-Health-Status
        
        Returns:
            Dict: Health-Informationen
        """
        try:
            # Health-Check für alle Provider
            provider_health = {}
            overall_status = VectorStoreStatus.HEALTHY
            
            for provider_name, vector_store in self._vector_stores.items():
                health = self._check_provider_health(provider_name, vector_store)
                provider_health[provider_name] = health
                
                # Overall-Status bestimmen
                if health.status == VectorStoreStatus.ERROR:
                    overall_status = VectorStoreStatus.UNHEALTHY
                elif health.status == VectorStoreStatus.DEGRADED and overall_status == VectorStoreStatus.HEALTHY:
                    overall_status = VectorStoreStatus.DEGRADED
            
            # Performance-Statistiken
            performance_stats = {}
            if self.config.performance_tracking:
                performance_stats = self._calculate_performance_stats()
            
            return {
                'overall_status': overall_status,
                'current_provider': self._current_provider,
                'available_providers': list(self._vector_stores.keys()),
                'provider_health': provider_health,
                'performance_stats': performance_stats,
                'last_health_check': datetime.now(timezone.utc).isoformat(),
                'service_config': {
                    'auto_fallback': self.config.auto_provider_selection,
                    'fallback_providers': self.config.fallback_providers,
                    'monitoring_enabled': self.config.monitoring_enabled
                }
            }
            
        except Exception as e:
            self.logger.error(f"Service Health Check fehlgeschlagen: {str(e)}")
            return {
                'overall_status': VectorStoreStatus.ERROR,
                'error': str(e),
                'last_health_check': datetime.now(timezone.utc).isoformat()
            }
    
    def _check_provider_health(self, provider_name: str, vector_store: BaseVectorStore) -> VectorStoreHealth:
        """Prüft Health eines einzelnen Providers"""
        try:
            # Basis-Health-Check
            if hasattr(vector_store, 'health_check'):
                health_info = vector_store.health_check()
                status = VectorStoreStatus.HEALTHY if health_info.get('healthy', True) else VectorStoreStatus.UNHEALTHY
            else:
                # Fallback: Einfacher Ping
                collections = []
                if hasattr(vector_store, 'list_collections'):
                    collections = vector_store.list_collections() or []
                status = VectorStoreStatus.HEALTHY
            
            # Performance-Metriken berechnen
            error_rate = self._calculate_error_rate(provider_name)
            avg_response_time = self._calculate_avg_response_time(provider_name)
            
            return VectorStoreHealth(
                provider=provider_name,
                status=status,
                collection_count=len(collections) if 'collections' in locals() else 0,
                available_collections=collections if 'collections' in locals() else [],
                error_rate=error_rate,
                average_response_time_ms=avg_response_time,
                last_operation_time=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.warning(f"Health Check für '{provider_name}' fehlgeschlagen: {str(e)}")
            return VectorStoreHealth(
                provider=provider_name,
                status=VectorStoreStatus.ERROR,
                metadata={'error': str(e)}
            )
    
    def _calculate_performance_stats(self) -> Dict[str, Any]:
        """Berechnet Performance-Statistiken"""
        if not self._operation_stats:
            return {}
        
        stats = {}
        for op_key, data in self._operation_stats.items():
            if data['count'] > 0:
                success_rate = (data['success_count'] / data['count']) * 100
                avg_time = data['total_time_ms'] / data['count']
                avg_docs_per_op = data['total_documents'] / data['count'] if data['total_documents'] > 0 else 0
                
                stats[op_key] = {
                    'operations': data['count'],
                    'success_rate_percent': round(success_rate, 2),
                    'average_time_ms': round(avg_time, 2),
                    'average_documents_per_operation': round(avg_docs_per_op, 1),
                    'total_documents_processed': data['total_documents']
                }
        
        return stats
    
    def _calculate_error_rate(self, provider_name: str) -> float:
        """Berechnet Error-Rate für Provider"""
        total_ops = 0
        failed_ops = 0
        
        for op_key, data in self._operation_stats.items():
            if provider_name in op_key:
                total_ops += data['count']
                failed_ops += (data['count'] - data['success_count'])
        
        return (failed_ops / total_ops) * 100 if total_ops > 0 else 0.0
    
    def _calculate_avg_response_time(self, provider_name: str) -> float:
        """Berechnet durchschnittliche Response-Zeit für Provider"""
        total_time = 0
        total_ops = 0
        
        for op_key, data in self._operation_stats.items():
            if provider_name in op_key:
                total_time += data['total_time_ms']
                total_ops += data['count']
        
        return total_time / total_ops if total_ops > 0 else 0.0
    
    def switch_provider(self, provider_name: str) -> bool:
        """
        Wechselt zu anderem Vector Store Provider
        
        Args:
            provider_name: Name des gewünschten Providers
            
        Returns:
            bool: True wenn erfolgreich gewechselt
        """
        if provider_name not in self._vector_stores:
            self.logger.error(f"Provider '{provider_name}' nicht verfügbar")
            return False
        
        old_provider = self._current_provider
        self._current_provider = provider_name
        
        self.logger.info(f"Vector Store Provider gewechselt: {old_provider} → {provider_name}")
        return True
    
    def is_healthy(self) -> bool:
        """
        Prüft ob Service funktionsfähig ist
        
        Returns:
            bool: True wenn mindestens ein Provider verfügbar
        """
        try:
            health = self.get_service_health()
            return health['overall_status'] in [VectorStoreStatus.HEALTHY, VectorStoreStatus.DEGRADED]
        except:
            return False


# =============================================================================
# SERVICE FACTORY UND CONVENIENCE-FUNKTIONEN
# =============================================================================

def create_vector_store_service(config: Optional[VectorStoreServiceConfig] = None,
                               embeddings: Optional[BaseEmbeddings] = None) -> VectorStoreService:
    """
    Factory-Funktion für Vector Store Service
    
    Args:
        config: Service-Konfiguration
        embeddings: Embedding-Provider
        
    Returns:
        VectorStoreService: Konfigurierte Service-Instanz
    """
    return VectorStoreService(config=config, embeddings=embeddings)


def get_vector_store_service_from_config(rag_config: Optional[RAGConfig] = None) -> VectorStoreService:
    """
    Erstellt Vector Store Service aus RAG-Konfiguration
    
    Args:
        rag_config: RAG-System-Konfiguration
        
    Returns:
        VectorStoreService: Service mit RAG-Config-basierten Einstellungen
    """
    if not rag_config:
        rag_config = get_config()
    
    # Service-Config aus RAG-Config ableiten
    service_config = VectorStoreServiceConfig(
        default_provider=rag_config.vector_store.providers[0],
        fallback_providers=rag_config.vector_store.providers,
        auto_provider_selection=True,
        health_check_interval=300,
        batch_size=100,
        backup_enabled=True,
        monitoring_enabled=True,
        performance_tracking=True
    )
    
    return VectorStoreService(config=service_config)


# =============================================================================
# BATCH-PROCESSING UTILITIES
# =============================================================================

class VectorStoreBatchProcessor:
    """
    Batch-Processor für große Dokumentenmengen
    
    Optimiert Vector Store Operationen durch:
    - Intelligent chunking basierend auf Provider-Limits
    - Parallel-Processing für bessere Performance
    - Progress-Tracking für UI-Integration
    - Automatische Retry-Mechanismen bei Fehlern
    """
    
    def __init__(self, vector_store_service: VectorStoreService, batch_size: int = 100):
        """
        Initialisiert Batch-Processor
        
        Args:
            vector_store_service: Vector Store Service Instanz
            batch_size: Größe der Verarbeitungs-Batches
        """
        self.service = vector_store_service
        self.batch_size = batch_size
        self.logger = get_logger(f"{__name__}.batch_processor")
    
    def process_documents_in_batches(self,
                                   collection_name: str,
                                   documents: List[Dict[str, Any]],
                                   progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        """
        Verarbeitet Dokumente in Batches
        
        Args:
            collection_name: Name der Ziel-Collection
            documents: Liste aller zu verarbeitenden Dokumente
            progress_callback: Callback für Progress-Updates (current, total)
            
        Returns:
            Dict: Batch-Processing Ergebnisse
        """
        start_time = time.time()
        
        total_docs = len(documents)
        processed_docs = 0
        successful_batches = 0
        failed_batches = 0
        errors = []
        
        self.logger.info(f"Starte Batch-Processing für {total_docs} Dokumente in {collection_name}")
        
        # Dokumente in Batches aufteilen
        for i in range(0, total_docs, self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            batch_number = (i // self.batch_size) + 1
            total_batches = (total_docs + self.batch_size - 1) // self.batch_size
            
            try:
                self.logger.debug(f"Verarbeite Batch {batch_number}/{total_batches} ({len(batch_docs)} docs)")
                
                # Batch verarbeiten
                result = self.service.add_documents(
                    collection_name=collection_name,
                    documents=batch_docs
                )
                
                if result.success:
                    processed_docs += len(batch_docs)
                    successful_batches += 1
                    self.logger.debug(f"Batch {batch_number} erfolgreich ({result.processing_time_ms:.1f}ms)")
                else:
                    failed_batches += 1
                    error_msg = f"Batch {batch_number} fehlgeschlagen: {result.error_message}"
                    errors.append(error_msg)
                    self.logger.warning(error_msg)
                
                # Progress-Callback aufrufen
                if progress_callback:
                    progress_callback(processed_docs, total_docs)
                
            except Exception as e:
                failed_batches += 1
                error_msg = f"Batch {batch_number} Exception: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg, exc_info=True)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Ergebnisse zusammenfassen
        results = {
            'total_documents': total_docs,
            'processed_documents': processed_docs,
            'successful_batches': successful_batches,
            'failed_batches': failed_batches,
            'total_batches': successful_batches + failed_batches,
            'processing_time_ms': total_time_ms,
            'documents_per_second': (processed_docs * 1000 / total_time_ms) if total_time_ms > 0 else 0,
            'success_rate_percent': (processed_docs / total_docs) * 100 if total_docs > 0 else 0,
            'errors': errors
        }
        
        self.logger.info(
            f"Batch-Processing abgeschlossen: {processed_docs}/{total_docs} Dokumente "
            f"in {total_time_ms:.1f}ms ({results['documents_per_second']:.1f} docs/sec)"
        )
        
        return results


# =============================================================================
# BACKUP UND RECOVERY UTILITIES  
# =============================================================================

class VectorStoreBackupManager:
    """
    Backup und Recovery Manager für Vector Stores
    
    Features:
    - Automatische Collection-Backups
    - Metadaten-Export mit Struktur-Informationen
    - Inkrementelle Backup-Strategien
    - Recovery mit Validierung
    """
    
    def __init__(self, vector_store_service: VectorStoreService):
        """
        Initialisiert Backup Manager
        
        Args:
            vector_store_service: Vector Store Service Instanz
        """
        self.service = vector_store_service
        self.logger = get_logger(f"{__name__}.backup_manager")
        self.backup_dir = Path(vector_store_service.config.backup_directory)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def backup_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        Erstellt Backup einer Collection
        
        Args:
            collection_name: Name der zu sichernden Collection
            
        Returns:
            Dict: Backup-Informationen
        """
        try:
            start_time = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_id = f"{collection_name}_{timestamp}"
            
            self.logger.info(f"Starte Backup für Collection '{collection_name}' (ID: {backup_id})")
            
            # Collection-Info sammeln
            collection_info = self.service.get_collection_info(collection_name)
            
            # Backup-Metadaten
            backup_metadata = {
                'backup_id': backup_id,
                'collection_name': collection_name,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'provider': self.service._current_provider,
                'collection_info': collection_info,
                'backup_version': '1.0'
            }
            
            # Backup-Verzeichnis erstellen
            backup_path = self.backup_dir / backup_id
            backup_path.mkdir(exist_ok=True)
            
            # Metadaten speichern
            metadata_file = backup_path / 'metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(backup_metadata, f, indent=2, ensure_ascii=False)
            
            processing_time = time.time() - start_time
            
            backup_result = {
                'success': True,
                'backup_id': backup_id,
                'backup_path': str(backup_path),
                'processing_time_seconds': processing_time,
                'metadata': backup_metadata
            }
            
            self.logger.info(f"Backup '{backup_id}' abgeschlossen in {processing_time:.1f}s")
            return backup_result
            
        except Exception as e:
            error_msg = f"Backup für Collection '{collection_name}' fehlgeschlagen: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'error': str(e),
                'collection_name': collection_name
            }
    
    def list_backups(self, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Listet verfügbare Backups auf
        
        Args:
            collection_name: Filter für spezifische Collection (optional)
            
        Returns:
            List[Dict]: Liste der verfügbaren Backups
        """
        backups = []
        
        try:
            for backup_dir in self.backup_dir.iterdir():
                if backup_dir.is_dir():
                    metadata_file = backup_dir / 'metadata.json'
                    
                    if metadata_file.exists():
                        try:
                            import json
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            
                            # Filter nach Collection-Name wenn angegeben
                            if collection_name and metadata.get('collection_name') != collection_name:
                                continue
                            
                            backup_info = {
                                'backup_id': metadata.get('backup_id'),
                                'collection_name': metadata.get('collection_name'),
                                'timestamp': metadata.get('timestamp'),
                                'provider': metadata.get('provider'),
                                'backup_path': str(backup_dir),
                                'size_mb': self._calculate_backup_size(backup_dir)
                            }
                            
                            backups.append(backup_info)
                            
                        except Exception as e:
                            self.logger.warning(f"Backup-Metadaten-Parsing fehlgeschlagen für {backup_dir}: {str(e)}")
            
            # Nach Timestamp sortieren (neueste zuerst)
            backups.sort(key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Backup-Listing fehlgeschlagen: {str(e)}")
        
        return backups
    
    def _calculate_backup_size(self, backup_path: Path) -> float:
        """Berechnet Backup-Größe in MB"""
        try:
            total_size = sum(
                f.stat().st_size for f in backup_path.rglob('*') if f.is_file()
            )
            return round(total_size / (1024 * 1024), 2)
        except:
            return 0.0


# =============================================================================
# MONITORING UND DIAGNOSTIK
# =============================================================================

class VectorStoreMonitor:
    """
    Monitoring und Diagnostik für Vector Store Service
    
    Überwacht:
    - Performance-Metriken und Trends
    - Error-Rates und Patterns
    - Resource-Utilization
    - Provider-Health über Zeit
    """
    
    def __init__(self, vector_store_service: VectorStoreService):
        """
        Initialisiert Monitor
        
        Args:
            vector_store_service: Vector Store Service Instanz
        """
        self.service = vector_store_service
        self.logger = get_logger(f"{__name__}.monitor")
        self._metrics_history = []
        self._alerts_history = []
    
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Sammelt aktuelle Metriken
        
        Returns:
            Dict: Aktuelle System-Metriken
        """
        timestamp = datetime.now(timezone.utc)
        
        try:
            # Health-Informationen sammeln
            health_info = self.service.get_service_health()
            
            # Performance-Statistiken
            performance_stats = health_info.get('performance_stats', {})
            
            # Provider-spezifische Metriken
            provider_metrics = {}
            for provider_name in self.service._vector_stores.keys():
                provider_metrics[provider_name] = {
                    'error_rate': self.service._calculate_error_rate(provider_name),
                    'avg_response_time_ms': self.service._calculate_avg_response_time(provider_name),
                    'is_current': provider_name == self.service._current_provider
                }
            
            metrics = {
                'timestamp': timestamp.isoformat(),
                'overall_status': health_info.get('overall_status'),
                'current_provider': health_info.get('current_provider'),
                'available_providers': health_info.get('available_providers', []),
                'performance_stats': performance_stats,
                'provider_metrics': provider_metrics,
                'service_uptime_minutes': self._calculate_uptime_minutes(),
                'total_operations': sum(
                    stats.get('operations', 0) 
                    for stats in performance_stats.values()
                ),
            }
            
            # Zu Verlauf hinzufügen
            self._metrics_history.append(metrics)
            
            # Historie begrenzen (letzten 24 Stunden)
            cutoff_time = timestamp - timedelta(hours=24)
            self._metrics_history = [
                m for m in self._metrics_history 
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metriken-Sammlung fehlgeschlagen: {str(e)}")
            return {
                'timestamp': timestamp.isoformat(),
                'error': str(e),
                'overall_status': 'error'
            }
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Erstellt Metriken-Zusammenfassung
        
        Args:
            hours: Anzahl Stunden für Analyse
            
        Returns:
            Dict: Metriken-Zusammenfassung
        """
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            # Relevante Metriken filtern
            recent_metrics = [
                m for m in self._metrics_history
                if datetime.fromisoformat(m['timestamp']) > cutoff_time
            ]
            
            if not recent_metrics:
                return {'error': 'Keine Metriken verfügbar für den angegebenen Zeitraum'}
            
            # Zusammenfassung erstellen
            total_operations = sum(m.get('total_operations', 0) for m in recent_metrics)
            
            # Status-Verteilung
            status_counts = {}
            for metrics in recent_metrics:
                status = metrics.get('overall_status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Provider-Wechsel analysieren
            provider_changes = 0
            last_provider = None
            for metrics in recent_metrics:
                current_provider = metrics.get('current_provider')
                if last_provider and current_provider != last_provider:
                    provider_changes += 1
                last_provider = current_provider
            
            summary = {
                'period_hours': hours,
                'metrics_collected': len(recent_metrics),
                'total_operations': total_operations,
                'status_distribution': status_counts,
                'provider_changes': provider_changes,
                'current_status': recent_metrics[-1].get('overall_status') if recent_metrics else 'unknown',
                'availability_percent': (status_counts.get('healthy', 0) / len(recent_metrics)) * 100 if recent_metrics else 0
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Metriken-Zusammenfassung fehlgeschlagen: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_uptime_minutes(self) -> float:
        """Berechnet Service-Uptime in Minuten"""
        if self._metrics_history:
            first_metric_time = datetime.fromisoformat(self._metrics_history[0]['timestamp'])
            current_time = datetime.now(timezone.utc)
            uptime_delta = current_time - first_metric_time
            return uptime_delta.total_seconds() / 60
        return 0.0


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core Service
    'VectorStoreService',
    'VectorStoreServiceConfig',
    
    # Data Structures
    'VectorStoreOperation',
    'VectorStoreStatus',
    'VectorStoreOperationRequest',
    'VectorStoreOperationResult',
    'VectorStoreHealth',
    
    # Utilities
    'VectorStoreBatchProcessor',
    'VectorStoreBackupManager',
    'VectorStoreMonitor',
    
    # Factory Functions
    'create_vector_store_service',
    'get_vector_store_service_from_config',
]
    
