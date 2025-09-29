# controllers/pipeline_controller.py
"""
Pipeline Controller - RAG-Pipeline Orchestrierung
Industrielle RAG-Architektur - Phase 3 Migration

Zentrale Steuerung der RAG-Pipeline mit Service-Orchestrierung,
Request-Routing und robustem State-Management für industrielle Anwendungen.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
from pathlib import Path

from core.logger import get_logger
from core.exceptions import PipelineException, ServiceException, ConfigurationException
from core.container import get_container
from core.config import get_config

# Service Layer Imports
from services.document_service import DocumentService
from services.embedding_service import EmbeddingService
from services.search_service import SearchService
from services.chat_service import ChatService

logger = get_logger(__name__)


class PipelineStage(Enum):
    """Pipeline-Stufen für Status-Tracking"""
    INITIALIZED = "initialized"
    DOCUMENT_PROCESSING = "document_processing"
    EMBEDDING_CREATION = "embedding_creation"
    INDEXING = "indexing"
    QUERY_PROCESSING = "query_processing"
    RETRIEVAL = "retrieval"
    RESPONSE_GENERATION = "response_generation"
    COMPLETED = "completed"
    ERROR = "error"


class PipelineMode(Enum):
    """Pipeline-Betriebsmodi"""
    INDEXING = "indexing"      # Dokumente verarbeiten und indexieren
    QUERYING = "querying"      # Anfragen beantworten
    HYBRID = "hybrid"          # Beides gleichzeitig (für Live-Updates)


@dataclass
class PipelineRequest:
    """Strukturierte Pipeline-Anfrage"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_type: str = ""                    # "index_documents", "process_query", "health_check"
    timestamp: float = field(default_factory=time.time)
    
    # Document Processing
    documents: List[Dict[str, Any]] = field(default_factory=list)
    document_source: str = "unknown"
    
    # Query Processing  
    query: str = ""
    query_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Pipeline Configuration
    pipeline_config: Dict[str, Any] = field(default_factory=dict)
    user_context: Dict[str, Any] = field(default_factory=dict)
    
    # Processing Options
    async_processing: bool = False
    priority: int = 5                         # 1-10 (10 = höchste Priorität)
    timeout_seconds: int = 300               # 5 Minuten Standard-Timeout


@dataclass
class PipelineResponse:
    """Strukturierte Pipeline-Antwort"""
    request_id: str
    success: bool
    stage: PipelineStage
    processing_time: float
    
    # Results
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    # Processing Details
    stages_completed: List[str] = field(default_factory=list)
    service_statistics: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    pipeline_version: str = "4.0.0"
    timestamp: float = field(default_factory=time.time)


@dataclass
class PipelineConfig:
    """Pipeline-Konfiguration"""
    # Service-Konfiguration
    document_service_config: Dict[str, Any] = field(default_factory=dict)
    embedding_service_config: Dict[str, Any] = field(default_factory=dict)
    search_service_config: Dict[str, Any] = field(default_factory=dict)
    chat_service_config: Dict[str, Any] = field(default_factory=dict)
    
    # Pipeline-Verhalten
    pipeline_mode: PipelineMode = PipelineMode.HYBRID
    max_concurrent_requests: int = 5
    request_queue_size: int = 100
    default_timeout: int = 300
    
    # Performance-Optimierungen
    batch_processing: bool = True
    parallel_processing: bool = True
    cache_enabled: bool = True
    performance_monitoring: bool = True
    
    # Error Handling
    retry_failed_requests: bool = True
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 2.0
    
    # Health Monitoring
    health_check_interval: int = 60
    performance_alert_threshold: float = 0.8  # 80% Fehlerrate löst Alert aus


class PipelineController:
    """
    RAG Pipeline Controller - Zentrale Pipeline-Orchestrierung
    
    Verantwortlichkeiten:
    - Service-zu-Service Koordination
    - Request-Routing und Load-Balancing
    - Pipeline State Management
    - Performance-Monitoring und Health-Checks
    - Robuste Fehlerbehandlung und Recovery
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialisiert RAG Pipeline Controller
        
        Args:
            config: PipelineConfig mit Pipeline-Einstellungen
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.controller")
        
        # Service Dependencies (via DI Container)
        self._container = get_container()
        self._services = {}
        
        # Pipeline State
        self._active_requests: Dict[str, PipelineRequest] = {}
        self._request_history: List[PipelineResponse] = []
        self._pipeline_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'requests_by_type': {},
            'service_call_count': {}
        }
        
        # Initialize Services
        self._initialize_services()
        
        self.logger.info(
            "RAG Pipeline Controller initialisiert",
            extra={
                'pipeline_mode': config.pipeline_mode.value,
                'max_concurrent': config.max_concurrent_requests,
                'services_loaded': len(self._services)
            }
        )

    def _initialize_services(self):
        """Initialisiert alle benötigten Services"""
        try:
            # Document Service
            self._services['document'] = self._container.get_service(DocumentService)
            if not self._services['document']:
                raise ServiceException("DocumentService nicht im Container verfügbar")
            
            # Embedding Service
            self._services['embedding'] = self._container.get_service(EmbeddingService)
            if not self._services['embedding']:
                raise ServiceException("EmbeddingService nicht im Container verfügbar")
            
            # Search Service
            self._services['search'] = self._container.get_service(SearchService)
            if not self._services['search']:
                raise ServiceException("SearchService nicht im Container verfügbar")
            
            # Chat Service
            self._services['chat'] = self._container.get_service(ChatService)
            if not self._services['chat']:
                raise ServiceException("ChatService nicht im Container verfügbar")
            
            self.logger.info("Alle Pipeline-Services erfolgreich geladen")
            
        except Exception as e:
            raise PipelineException(f"Service-Initialisierung fehlgeschlagen: {str(e)}")

    def process_request(self, request: PipelineRequest) -> PipelineResponse:
        """
        Hauptschnittstelle für Pipeline-Requests
        
        Args:
            request: PipelineRequest mit Verarbeitungsanfrage
            
        Returns:
            PipelineResponse: Ergebnis der Pipeline-Verarbeitung
        """
        start_time = time.time()
        self._pipeline_stats['total_requests'] += 1
        
        # Request-Typ in Statistiken tracken
        req_type = request.request_type
        if req_type not in self._pipeline_stats['requests_by_type']:
            self._pipeline_stats['requests_by_type'][req_type] = 0
        self._pipeline_stats['requests_by_type'][req_type] += 1
        
        self.logger.info(
            "Pipeline-Request gestartet",
            extra={
                'request_id': request.request_id,
                'request_type': request.request_type,
                'priority': request.priority
            }
        )
        
        try:
            # Request zur aktiven Liste hinzufügen
            self._active_requests[request.request_id] = request
            
            # Route Request basierend auf Typ
            if request.request_type == "index_documents":
                response = self._process_document_indexing(request)
            elif request.request_type == "process_query":
                response = self._process_query_request(request)
            elif request.request_type == "health_check":
                response = self._process_health_check(request)
            elif request.request_type == "pipeline_status":
                response = self._process_status_request(request)
            else:
                raise PipelineException(f"Unbekannter Request-Typ: {request.request_type}")
            
            # Erfolg statistiken updaten
            processing_time = time.time() - start_time
            self._update_success_statistics(processing_time)
            
            # Response finalisieren
            response.processing_time = processing_time
            response.performance_metrics = self._get_performance_metrics()
            
            self.logger.info(
                "Pipeline-Request erfolgreich abgeschlossen",
                extra={
                    'request_id': request.request_id,
                    'processing_time': processing_time,
                    'stage': response.stage.value
                }
            )
            
            return response
            
        except Exception as e:
            # Fehler-Statistiken updaten
            processing_time = time.time() - start_time
            self._update_error_statistics(processing_time)
            
            self.logger.error(
                "Pipeline-Request fehlgeschlagen",
                extra={
                    'request_id': request.request_id,
                    'error': str(e),
                    'processing_time': processing_time
                },
                exc_info=True
            )
            
            # Fehler-Response erstellen
            return PipelineResponse(
                request_id=request.request_id,
                success=False,
                stage=PipelineStage.ERROR,
                processing_time=processing_time,
                error_message=str(e),
                service_statistics=self._get_service_statistics()
            )
            
        finally:
            # Request aus aktiver Liste entfernen
            if request.request_id in self._active_requests:
                del self._active_requests[request.request_id]

    def _process_document_indexing(self, request: PipelineRequest) -> PipelineResponse:
        """
        Verarbeitet Dokument-Indexierung durch die komplette Pipeline
        
        Args:
            request: PipelineRequest mit Dokumenten
            
        Returns:
            PipelineResponse: Indexierung-Ergebnis
        """
        response = PipelineResponse(
            request_id=request.request_id,
            success=True,
            stage=PipelineStage.INITIALIZED
        )
        
        try:
            # Stage 1: Document Processing
            response.stage = PipelineStage.DOCUMENT_PROCESSING
            response.stages_completed.append("document_processing_started")
            
            document_service = self._services['document']
            processed_docs = document_service.process_documents(
                documents=request.documents,
                source=request.document_source,
                metadata=request.user_context
            )
            
            self._track_service_call('document_service', 'process_documents')
            response.stages_completed.append("document_processing_completed")
            
            if not processed_docs or not processed_docs.success:
                raise PipelineException(f"Dokument-Verarbeitung fehlgeschlagen: {processed_docs.error if processed_docs else 'Unbekannter Fehler'}")
            
            # Stage 2: Embedding Creation  
            response.stage = PipelineStage.EMBEDDING_CREATION
            response.stages_completed.append("embedding_creation_started")
            
            embedding_service = self._services['embedding']
            
            # Texte für Embedding extrahieren
            texts_to_embed = []
            for doc in processed_docs.processed_documents:
                if hasattr(doc, 'content') and doc.content:
                    texts_to_embed.append(doc.content)
            
            if not texts_to_embed:
                raise PipelineException("Keine Texte für Embedding-Erstellung gefunden")
            
            embeddings_result = embedding_service.create_embeddings(
                texts=texts_to_embed,
                source=f"indexing_{request.document_source}",
                metadata={'request_id': request.request_id}
            )
            
            self._track_service_call('embedding_service', 'create_embeddings')
            response.stages_completed.append("embedding_creation_completed")
            
            if not embeddings_result.success:
                raise PipelineException(f"Embedding-Erstellung fehlgeschlagen: {embeddings_result.error}")
            
            # Stage 3: Vector Store Indexing
            response.stage = PipelineStage.INDEXING
            response.stages_completed.append("indexing_started")
            
            search_service = self._services['search']
            
            # Dokumente mit Embeddings für Indexierung vorbereiten
            docs_with_embeddings = []
            for i, doc in enumerate(processed_docs.processed_documents):
                if i < len(embeddings_result.embeddings):
                    doc_with_embedding = {
                        'content': doc.content,
                        'metadata': doc.metadata,
                        'embedding': embeddings_result.embeddings[i],
                        'source': request.document_source
                    }
                    docs_with_embeddings.append(doc_with_embedding)
            
            index_result = search_service.index_documents(
                documents=docs_with_embeddings,
                index_name=request.pipeline_config.get('index_name', 'default'),
                metadata={'request_id': request.request_id}
            )
            
            self._track_service_call('search_service', 'index_documents')
            response.stages_completed.append("indexing_completed")
            
            if not index_result.success:
                raise PipelineException(f"Indexierung fehlgeschlagen: {index_result.error}")
            
            # Pipeline erfolgreich abgeschlossen
            response.stage = PipelineStage.COMPLETED
            response.result_data = {
                'documents_processed': len(processed_docs.processed_documents),
                'embeddings_created': len(embeddings_result.embeddings),
                'documents_indexed': index_result.documents_indexed,
                'index_name': index_result.index_name,
                'processing_details': {
                    'document_stats': processed_docs.processing_stats,
                    'embedding_stats': embeddings_result.processing_stats,
                    'index_stats': index_result.processing_stats
                }
            }
            
            return response
            
        except Exception as e:
            response.success = False
            response.stage = PipelineStage.ERROR
            response.error_message = str(e)
            return response

    def _process_query_request(self, request: PipelineRequest) -> PipelineResponse:
        """
        Verarbeitet Query-Anfrage durch Retrieval und Response-Generation
        
        Args:
            request: PipelineRequest mit Query
            
        Returns:
            PipelineResponse: Query-Ergebnis
        """
        response = PipelineResponse(
            request_id=request.request_id,
            success=True,
            stage=PipelineStage.INITIALIZED
        )
        
        try:
            if not request.query.strip():
                raise PipelineException("Leere Query-Anfrage")
            
            # Stage 1: Query Processing & Embedding
            response.stage = PipelineStage.QUERY_PROCESSING
            response.stages_completed.append("query_processing_started")
            
            embedding_service = self._services['embedding']
            
            query_embedding_result = embedding_service.create_embeddings(
                texts=[request.query],
                source=f"query_{request.request_id}",
                metadata={'query_type': request.query_metadata.get('type', 'unknown')}
            )
            
            self._track_service_call('embedding_service', 'create_embeddings')
            response.stages_completed.append("query_embedding_completed")
            
            if not query_embedding_result.success:
                raise PipelineException(f"Query-Embedding fehlgeschlagen: {query_embedding_result.error}")
            
            query_embedding = query_embedding_result.embeddings[0]
            
            # Stage 2: Document Retrieval
            response.stage = PipelineStage.RETRIEVAL
            response.stages_completed.append("retrieval_started")
            
            search_service = self._services['search']
            
            retrieval_config = request.pipeline_config.get('retrieval', {})
            retrieval_result = search_service.search_documents(
                query_embedding=query_embedding,
                query_text=request.query,
                index_name=retrieval_config.get('index_name', 'default'),
                top_k=retrieval_config.get('top_k', 5),
                metadata_filters=request.query_metadata.get('filters', {}),
                search_params=retrieval_config
            )
            
            self._track_service_call('search_service', 'search_documents')
            response.stages_completed.append("retrieval_completed")
            
            if not retrieval_result.success:
                raise PipelineException(f"Dokument-Retrieval fehlgeschlagen: {retrieval_result.error}")
            
            # Stage 3: Response Generation
            response.stage = PipelineStage.RESPONSE_GENERATION
            response.stages_completed.append("response_generation_started")
            
            chat_service = self._services['chat']
            
            chat_config = request.pipeline_config.get('chat', {})
            chat_result = chat_service.generate_response(
                query=request.query,
                retrieved_documents=retrieval_result.retrieved_documents,
                context=request.user_context,
                chat_config=chat_config,
                metadata={'request_id': request.request_id}
            )
            
            self._track_service_call('chat_service', 'generate_response')
            response.stages_completed.append("response_generation_completed")
            
            if not chat_result.success:
                raise PipelineException(f"Response-Generierung fehlgeschlagen: {chat_result.error}")
            
            # Pipeline erfolgreich abgeschlossen
            response.stage = PipelineStage.COMPLETED
            response.result_data = {
                'query': request.query,
                'response': chat_result.response,
                'sources': chat_result.sources,
                'confidence': chat_result.confidence,
                'retrieved_count': len(retrieval_result.retrieved_documents),
                'processing_details': {
                    'query_embedding_stats': query_embedding_result.processing_stats,
                    'retrieval_stats': retrieval_result.processing_stats,
                    'chat_stats': chat_result.processing_stats
                }
            }
            
            return response
            
        except Exception as e:
            response.success = False
            response.stage = PipelineStage.ERROR
            response.error_message = str(e)
            return response

    def _process_health_check(self, request: PipelineRequest) -> PipelineResponse:
        """Führt umfassenden Health-Check aller Services durch"""
        response = PipelineResponse(
            request_id=request.request_id,
            success=True,
            stage=PipelineStage.COMPLETED
        )
        
        health_results = {}
        overall_healthy = True
        
        try:
            # Health-Check für alle Services
            for service_name, service in self._services.items():
                try:
                    if hasattr(service, 'get_service_health'):
                        service_health = service.get_service_health()
                        health_results[service_name] = service_health
                        
                        if service_health.get('service_status') not in ['healthy', 'degraded']:
                            overall_healthy = False
                    else:
                        health_results[service_name] = {'status': 'no_health_check', 'available': True}
                        
                except Exception as e:
                    health_results[service_name] = {
                        'status': 'error',
                        'error': str(e),
                        'available': False
                    }
                    overall_healthy = False
            
            # Pipeline-spezifische Health-Metriken
            pipeline_health = {
                'active_requests': len(self._active_requests),
                'total_requests_processed': self._pipeline_stats['total_requests'],
                'success_rate': self._calculate_success_rate(),
                'avg_processing_time': self._pipeline_stats['avg_processing_time'],
                'services_healthy': sum(1 for h in health_results.values() if h.get('status') in ['healthy', 'available']),
                'services_total': len(health_results)
            }
            
            response.result_data = {
                'overall_status': 'healthy' if overall_healthy else 'unhealthy',
                'pipeline_health': pipeline_health,
                'service_health': health_results,
                'timestamp': time.time()
            }
            
            response.success = overall_healthy
            
        except Exception as e:
            response.success = False
            response.error_message = str(e)
            response.result_data = {'error': 'Health-Check fehlgeschlagen'}
        
        return response

    def _process_status_request(self, request: PipelineRequest) -> PipelineResponse:
        """Liefert detaillierte Pipeline-Statistiken"""
        return PipelineResponse(
            request_id=request.request_id,
            success=True,
            stage=PipelineStage.COMPLETED,
            processing_time=0.0,
            result_data={
                'pipeline_statistics': self._pipeline_stats,
                'active_requests': len(self._active_requests),
                'service_statistics': self._get_service_statistics(),
                'performance_metrics': self._get_performance_metrics(),
                'config': {
                    'pipeline_mode': self.config.pipeline_mode.value,
                    'max_concurrent_requests': self.config.max_concurrent_requests,
                    'performance_monitoring': self.config.performance_monitoring
                }
            }
        )

    def _track_service_call(self, service_name: str, method_name: str):
        """Trackt Service-Aufrufe für Monitoring"""
        key = f"{service_name}.{method_name}"
        if key not in self._pipeline_stats['service_call_count']:
            self._pipeline_stats['service_call_count'][key] = 0
        self._pipeline_stats['service_call_count'][key] += 1

    def _update_success_statistics(self, processing_time: float):
        """Aktualisiert Erfolgs-Statistiken"""
        self._pipeline_stats['successful_requests'] += 1
        self._update_avg_processing_time(processing_time)

    def _update_error_statistics(self, processing_time: float):
        """Aktualisiert Fehler-Statistiken"""
        self._pipeline_stats['failed_requests'] += 1
        self._update_avg_processing_time(processing_time)

    def _update_avg_processing_time(self, processing_time: float):
        """Aktualisiert durchschnittliche Verarbeitungszeit"""
        total_requests = self._pipeline_stats['total_requests']
        current_avg = self._pipeline_stats['avg_processing_time']
        
        self._pipeline_stats['avg_processing_time'] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )

    def _calculate_success_rate(self) -> float:
        """Berechnet aktuelle Success Rate"""
        total = self._pipeline_stats['total_requests']
        if total == 0:
            return 1.0
        return self._pipeline_stats['successful_requests'] / total

    def _get_service_statistics(self) -> Dict[str, Any]:
        """Sammelt Statistiken von allen Services"""
        service_stats = {}
        
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'get_service_statistics'):
                    service_stats[service_name] = service.get_service_statistics()
                elif hasattr(service, 'get_statistics'):
                    service_stats[service_name] = service.get_statistics()
                else:
                    service_stats[service_name] = {'status': 'no_statistics_available'}
            except Exception as e:
                service_stats[service_name] = {'error': str(e)}
        
        return service_stats

    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Erstellt Performance-Metriken für Monitoring"""
        return {
            'success_rate': self._calculate_success_rate(),
            'avg_processing_time_seconds': self._pipeline_stats['avg_processing_time'],
            'requests_per_minute': self._calculate_requests_per_minute(),
            'active_requests_count': len(self._active_requests),
            'service_call_distribution': self._pipeline_stats['service_call_count'],
            'request_type_distribution': self._pipeline_stats['requests_by_type'],
            'error_rate': 1.0 - self._calculate_success_rate(),
            'throughput_score': self._calculate_throughput_score()
        }

    def _calculate_requests_per_minute(self) -> float:
        """Berechnet Requests pro Minute (vereinfacht)"""
        # Vereinfachte Berechnung - in Produktion würde man ein Zeitfenster verwenden
        if not hasattr(self, '_start_time'):
            self._start_time = time.time()
        
        runtime_minutes = (time.time() - self._start_time) / 60.0
        if runtime_minutes < 1.0:
            return float(self._pipeline_stats['total_requests'])
        
        return self._pipeline_stats['total_requests'] / runtime_minutes

    def _calculate_throughput_score(self) -> float:
        """Berechnet Throughput-Score (0-100)"""
        success_rate = self._calculate_success_rate()
        avg_time = self._pipeline_stats['avg_processing_time']
        
        # Score basierend auf Success Rate und Geschwindigkeit
        time_score = max(0, 100 - (avg_time * 10))  # Penalty für langsame Requests
        success_score = success_rate * 100
        
        return (time_score + success_score) / 2

    def get_active_requests(self) -> List[Dict[str, Any]]:
        """Liefert Informationen über aktive Requests"""
        active_info = []
        
        for request_id, request in self._active_requests.items():
            info = {
                'request_id': request_id,
                'request_type': request.request_type,
                'started_at': request.timestamp,
                'running_seconds': time.time() - request.timestamp,
                'priority': request.priority,
                'timeout_seconds': request.timeout_seconds,
                'user_context': request.user_context
            }
            active_info.append(info)
        
        # Sortiert nach Laufzeit (längste zuerst)
        active_info.sort(key=lambda x: x['running_seconds'], reverse=True)
        
        return active_info

    def cancel_request(self, request_id: str) -> bool:
        """
        Bricht aktive Anfrage ab (soweit möglich)
        
        Args:
            request_id: ID der abzubrechenden Anfrage
            
        Returns:
            bool: True wenn Anfrage abgebrochen wurde
        """
        if request_id in self._active_requests:
            # In einer vollständigen Implementierung würde hier
            # die Service-Aufrufe abgebrochen werden
            del self._active_requests[request_id]
            
            self.logger.info(f"Request {request_id} manuell abgebrochen")
            return True
        
        return False

    def optimize_pipeline_performance(self) -> Dict[str, Any]:
        """
        Optimiert Pipeline-Performance basierend auf Metriken
        
        Returns:
            Dict mit Optimierungsempfehlungen
        """
        optimization_results = {
            'current_performance': self._get_performance_metrics(),
            'recommendations': [],
            'optimizations_applied': [],
            'service_optimizations': {}
        }
        
        try:
            # Performance-Analyse
            success_rate = self._calculate_success_rate()
            avg_time = self._pipeline_stats['avg_processing_time']
            active_count = len(self._active_requests)
            
            # Empfehlungen basierend auf Metriken
            if success_rate < 0.8:  # <80% Success Rate
                optimization_results['recommendations'].append(
                    f"Niedrige Success Rate ({success_rate:.1%}) - Service-Health prüfen"
                )
            
            if avg_time > 30.0:  # >30s durchschnittliche Verarbeitungszeit
                optimization_results['recommendations'].append(
                    f"Hohe Verarbeitungszeit ({avg_time:.1f}s) - Batch-Größen optimieren"
                )
            
            if active_count > self.config.max_concurrent_requests * 0.8:
                optimization_results['recommendations'].append(
                    f"Hohe Concurrent Load ({active_count}/{self.config.max_concurrent_requests}) - Scaling erwägen"
                )
            
            # Service-spezifische Optimierungen
            for service_name, service in self._services.items():
                try:
                    if hasattr(service, 'optimize_service_performance'):
                        service_opt = service.optimize_service_performance()
                        optimization_results['service_optimizations'][service_name] = service_opt
                        
                        # Service-Empfehlungen zu Pipeline-Empfehlungen hinzufügen
                        service_recommendations = service_opt.get('recommendations', [])
                        for rec in service_recommendations:
                            optimization_results['recommendations'].append(f"{service_name}: {rec}")
                            
                except Exception as e:
                    optimization_results['service_optimizations'][service_name] = {'error': str(e)}
            
            # Automatische Optimierungen anwenden
            if self.config.performance_monitoring:
                applied_optimizations = self._apply_automatic_optimizations()
                optimization_results['optimizations_applied'] = applied_optimizations
            
        except Exception as e:
            optimization_results['error'] = str(e)
        
        return optimization_results

    def _apply_automatic_optimizations(self) -> List[str]:
        """Wendet automatische Performance-Optimierungen an"""
        applied = []
        
        try:
            # Cache-Cleanup bei hoher Memory-Usage
            if hasattr(self, '_last_cache_cleanup'):
                time_since_cleanup = time.time() - self._last_cache_cleanup
                if time_since_cleanup > 3600:  # 1 Stunde
                    self._cleanup_service_caches()
                    applied.append("Service-Caches bereinigt")
                    self._last_cache_cleanup = time.time()
            else:
                self._last_cache_cleanup = time.time()
            
            # Request-History cleanup
            if len(self._request_history) > 1000:
                self._request_history = self._request_history[-500:]  # Behalte nur letzten 500
                applied.append("Request-History bereinigt")
            
        except Exception as e:
            applied.append(f"Optimierung fehlgeschlagen: {str(e)}")
        
        return applied

    def _cleanup_service_caches(self):
        """Bereinigt Caches aller Services"""
        for service_name, service in self._services.items():
            try:
                if hasattr(service, 'clear_cache'):
                    service.clear_cache()
                    self.logger.info(f"Cache für {service_name} bereinigt")
            except Exception as e:
                self.logger.warning(f"Cache-Cleanup für {service_name} fehlgeschlagen: {str(e)}")

    def get_pipeline_health(self) -> Dict[str, Any]:
        """
        Liefert umfassende Pipeline-Gesundheitsinformationen
        
        Returns:
            Dict mit detailliertem Health-Status
        """
        health_request = PipelineRequest(
            request_type="health_check",
            request_id=f"health_{int(time.time())}"
        )
        
        health_response = self._process_health_check(health_request)
        
        # Zusätzliche Pipeline-spezifische Health-Informationen
        pipeline_health = health_response.result_data.copy()
        
        pipeline_health.update({
            'controller_status': {
                'active_requests': len(self._active_requests),
                'request_queue_utilization': len(self._active_requests) / self.config.max_concurrent_requests,
                'performance_score': self._calculate_throughput_score(),
                'uptime_status': 'operational',  # Vereinfacht
                'config_valid': True
            },
            'performance_alerts': self._check_performance_alerts()
        })
        
        return pipeline_health

    def _check_performance_alerts(self) -> List[Dict[str, Any]]:
        """Prüft Performance-Metriken auf Alerts"""
        alerts = []
        
        try:
            # Success Rate Alert
            success_rate = self._calculate_success_rate()
            if success_rate < self.config.performance_alert_threshold:
                alerts.append({
                    'type': 'performance_degradation',
                    'severity': 'high' if success_rate < 0.5 else 'medium',
                    'message': f"Success Rate niedrig: {success_rate:.1%}",
                    'metric': 'success_rate',
                    'value': success_rate,
                    'threshold': self.config.performance_alert_threshold
                })
            
            # Response Time Alert
            avg_time = self._pipeline_stats['avg_processing_time']
            if avg_time > 60.0:  # >1 Minute
                alerts.append({
                    'type': 'slow_response',
                    'severity': 'high' if avg_time > 180.0 else 'medium',
                    'message': f"Hohe Verarbeitungszeit: {avg_time:.1f}s",
                    'metric': 'avg_processing_time',
                    'value': avg_time,
                    'threshold': 60.0
                })
            
            # Concurrent Load Alert
            active_ratio = len(self._active_requests) / self.config.max_concurrent_requests
            if active_ratio > 0.9:
                alerts.append({
                    'type': 'high_load',
                    'severity': 'high',
                    'message': f"Hohe Concurrent Load: {active_ratio:.1%}",
                    'metric': 'concurrent_requests_ratio',
                    'value': active_ratio,
                    'threshold': 0.9
                })
            
        except Exception as e:
            alerts.append({
                'type': 'monitoring_error',
                'severity': 'medium',
                'message': f"Alert-Prüfung fehlgeschlagen: {str(e)}"
            })
        
        return alerts

    def cleanup(self):
        """Cleanup-Routine für Controller"""
        try:
            # Service-Cleanup
            for service_name, service in self._services.items():
                try:
                    if hasattr(service, 'cleanup'):
                        service.cleanup()
                        self.logger.info(f"Service {service_name} cleanup abgeschlossen")
                except Exception as e:
                    self.logger.error(f"Service {service_name} cleanup fehlgeschlagen: {str(e)}")
            
            # Controller-State cleanup
            self._active_requests.clear()
            self._request_history.clear()
            
            self.logger.info("Pipeline Controller cleanup abgeschlossen")
            
        except Exception as e:
            self.logger.error(f"Controller cleanup fehlgeschlagen: {str(e)}")


# Factory-Funktionen für Controller-Erstellung
def create_pipeline_controller(config: Optional[Dict[str, Any]] = None) -> PipelineController:
    """
    Factory-Funktion für RAG Pipeline Controller
    
    Args:
        config: Optional Controller-Konfiguration
        
    Returns:
        PipelineController: Konfigurierter Controller
    """
    if config is None:
        # Standard-Konfiguration aus globalem Config
        app_config = get_config()
        pipeline_config_dict = app_config.get('pipeline', {})
    else:
        pipeline_config_dict = config
    
    # PipelineConfig erstellen
    pipeline_config = PipelineConfig(**pipeline_config_dict)
    
    return PipelineController(pipeline_config)


def create_indexing_request(
    documents: List[Dict[str, Any]],
    document_source: str = "unknown",
    index_name: str = "default",
    **kwargs
) -> PipelineRequest:
    """
    Convenience-Funktion für Indexing-Requests
    
    Args:
        documents: Liste von Dokumenten zum Indexieren
        document_source: Quellenbezeichnung
        index_name: Ziel-Index Name
        **kwargs: Zusätzliche Request-Parameter
        
    Returns:
        PipelineRequest: Konfigurierter Indexing-Request
    """
    pipeline_config = kwargs.get('pipeline_config', {})
    pipeline_config['index_name'] = index_name
    
    return PipelineRequest(
        request_type="index_documents",
        documents=documents,
        document_source=document_source,
        pipeline_config=pipeline_config,
        **{k: v for k, v in kwargs.items() if k != 'pipeline_config'}
    )


def create_query_request(
    query: str,
    index_name: str = "default",
    top_k: int = 5,
    **kwargs
) -> PipelineRequest:
    """
    Convenience-Funktion für Query-Requests
    
    Args:
        query: Suchanfrage
        index_name: Quell-Index Name
        top_k: Anzahl Ergebnisse
        **kwargs: Zusätzliche Request-Parameter
        
    Returns:
        PipelineRequest: Konfigurierter Query-Request
    """
    pipeline_config = kwargs.get('pipeline_config', {})
    pipeline_config['retrieval'] = {
        'index_name': index_name,
        'top_k': top_k,
        **pipeline_config.get('retrieval', {})
    }
    
    return PipelineRequest(
        request_type="process_query",
        query=query,
        pipeline_config=pipeline_config,
        **{k: v for k, v in kwargs.items() if k != 'pipeline_config'}
    )


# Container-Registrierung
def register_pipeline_controller():
    """Registriert Pipeline Controller im DI-Container"""
    container = get_container()
    
    container.register_singleton(
        PipelineController,
        lambda: create_pipeline_controller()
    )
    
    logger.info("Pipeline Controller im Container registriert")


# Beispiel-Usage für Testing
if __name__ == '__main__':
    # Beispiel-Konfiguration
    example_config = PipelineConfig(
        pipeline_mode=PipelineMode.HYBRID,
        max_concurrent_requests=3,
        performance_monitoring=True,
        batch_processing=True,
        parallel_processing=True
    )
    
    try:
        # Controller erstellen
        controller = PipelineController(example_config)
        
        # Health-Check Request
        health_request = PipelineRequest(
            request_type="health_check",
            request_id="test_health"
        )
        
        health_response = controller.process_request(health_request)
        
        if health_response.success:
            print("✅ Pipeline Controller Health-Check erfolgreich")
            print(f"🏥 Status: {health_response.result_data.get('overall_status')}")
            print(f"🔧 Services: {health_response.result_data.get('pipeline_health', {}).get('services_healthy', 0)}/{health_response.result_data.get('pipeline_health', {}).get('services_total', 0)}")
        else:
            print(f"❌ Health-Check fehlgeschlagen: {health_response.error_message}")
        
        # Performance-Optimierung testen
        optimization = controller.optimize_pipeline_performance()
        print(f"⚡ Performance-Score: {optimization['current_performance'].get('throughput_score', 0):.1f}")
        
        if optimization['recommendations']:
            print("📋 Empfehlungen:")
            for rec in optimization['recommendations'][:3]:  # Nur ersten 3
                print(f"  • {rec}")
        
        # Cleanup
        controller.cleanup()
        print("🧹 Controller cleanup abgeschlossen")
        
    except Exception as e:
        print(f"❌ Controller-Test fehlgeschlagen: {str(e)}")
        import traceback
        traceback.print_exc()