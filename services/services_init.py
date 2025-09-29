"""
Services Package - Service-orientierte Architektur für industrielle RAG-Anwendungen

Automatische Service-Registrierung und Dependency Injection Container Integration
für alle Business Logic Services der RAG Chatbot Architektur.

Version: 4.0.0
Status: Production Ready
Autor: KI-Consultant für industrielle Automatisierung
"""

from typing import Dict, Any, Optional
import logging

# Core Imports
from core.exceptions import ServiceInitializationError, ConfigurationError
from core.container import DIContainer

# Service Imports - Alle Business Logic Services
from .document_service import DocumentService
from .embedding_service import EmbeddingService  
from .chat_service import ChatService
from .retrieval_service import RetrievalService
from .session_service import SessionService
from .vector_store_service import VectorStoreService
from .service_integration import ServiceIntegrator

# Logger Setup
logger = logging.getLogger(__name__)

# Service Registry - Zentrale Service-Verwaltung
SERVICE_REGISTRY: Dict[str, type] = {
    'document_service': DocumentService,
    'embedding_service': EmbeddingService,
    'chat_service': ChatService, 
    'retrieval_service': RetrievalService,
    'session_service': SessionService,
    'vector_store_service': VectorStoreService,
    'service_integrator': ServiceIntegrator
}

# Globale Service-Instanzen
_service_instances: Dict[str, Any] = {}
_initialized: bool = False

def initialize_services(container: DIContainer) -> Dict[str, Any]:
    """
    Initialisiert alle Services und registriert sie im DI-Container
    
    Args:
        container: Dependency Injection Container
        
    Returns:
        Dict mit allen initialisierten Service-Instanzen
        
    Raises:
        ServiceInitializationError: Bei Service-Initialisierung-Fehlern
    """
    global _service_instances, _initialized
    
    if _initialized:
        logger.info("Services bereits initialisiert - verwende bestehende Instanzen")
        return _service_instances
        
    try:
        logger.info("Starte Service-Initialisierung...")
        
        # Service-Instanzen erstellen und im Container registrieren
        for service_name, service_class in SERVICE_REGISTRY.items():
            try:
                # Service-Instanz erstellen (DI-Container löst Dependencies automatisch)
                service_instance = service_class()
                
                # Im Container registrieren
                container.register(service_name, service_instance)
                _service_instances[service_name] = service_instance
                
                logger.info(f"✅ Service '{service_name}' erfolgreich initialisiert")
                
            except Exception as e:
                error_msg = f"Fehler bei Initialisierung von Service '{service_name}': {str(e)}"
                logger.error(error_msg)
                raise ServiceInitializationError(error_msg) from e
        
        # Service-Integration initialisieren
        try:
            integrator = _service_instances['service_integrator']
            integrator.initialize_service_dependencies(_service_instances)
            logger.info("✅ Service-Dependencies erfolgreich verknüpft")
        except Exception as e:
            logger.error(f"Fehler bei Service-Integration: {str(e)}")
            raise ServiceInitializationError(f"Service-Integration fehlgeschlagen: {str(e)}") from e
            
        _initialized = True
        logger.info(f"🚀 Alle {len(_service_instances)} Services erfolgreich initialisiert!")
        
        return _service_instances
        
    except Exception as e:
        logger.error(f"❌ Service-Initialisierung fehlgeschlagen: {str(e)}")
        # Cleanup bei Fehler
        _service_instances.clear()
        _initialized = False
        raise

def get_service(service_name: str) -> Optional[Any]:
    """
    Holt eine Service-Instanz nach Name
    
    Args:
        service_name: Name des Services
        
    Returns:
        Service-Instanz oder None wenn nicht gefunden
    """
    if not _initialized:
        logger.warning("Services nicht initialisiert - verwende initialize_services() zuerst")
        return None
        
    return _service_instances.get(service_name)

def get_all_services() -> Dict[str, Any]:
    """
    Gibt alle initialisierten Service-Instanzen zurück
    
    Returns:
        Dict mit allen Service-Instanzen
    """
    return _service_instances.copy()

def is_initialized() -> bool:
    """
    Prüft ob Services bereits initialisiert sind
    
    Returns:
        True wenn alle Services initialisiert sind
    """
    return _initialized

def reset_services():
    """
    Setzt alle Services zurück (für Tests und Neuinitialisierung)
    """
    global _service_instances, _initialized
    
    logger.info("Services werden zurückgesetzt...")
    _service_instances.clear()
    _initialized = False
    logger.info("✅ Services erfolgreich zurückgesetzt")

# Health Check für alle Services
def health_check() -> Dict[str, Any]:
    """
    Führt Health Check für alle Services durch
    
    Returns:
        Dict mit Health Status aller Services
    """
    health_status = {
        'services_initialized': _initialized,
        'total_services': len(SERVICE_REGISTRY),
        'active_services': len(_service_instances),
        'service_status': {}
    }
    
    if not _initialized:
        health_status['status'] = 'NOT_INITIALIZED'
        return health_status
    
    # Einzelne Service Health Checks
    all_healthy = True
    for service_name, service_instance in _service_instances.items():
        try:
            # Service Health Check (wenn verfügbar)
            if hasattr(service_instance, 'health_check'):
                service_health = service_instance.health_check()
                health_status['service_status'][service_name] = service_health
                if not service_health.get('healthy', False):
                    all_healthy = False
            else:
                health_status['service_status'][service_name] = {
                    'healthy': True,
                    'message': 'Service aktiv (kein Health Check verfügbar)'
                }
        except Exception as e:
            health_status['service_status'][service_name] = {
                'healthy': False,
                'error': str(e)
            }
            all_healthy = False
    
    health_status['status'] = 'HEALTHY' if all_healthy else 'DEGRADED'
    return health_status

# Service Performance Monitoring
def get_service_stats() -> Dict[str, Any]:
    """
    Sammelt Performance-Statistiken aller Services
    
    Returns:
        Dict mit Service-Performance-Daten
    """
    if not _initialized:
        return {'error': 'Services nicht initialisiert'}
    
    stats = {
        'total_services': len(_service_instances),
        'service_stats': {}
    }
    
    for service_name, service_instance in _service_instances.items():
        try:
            if hasattr(service_instance, 'get_stats'):
                stats['service_stats'][service_name] = service_instance.get_stats()
            else:
                stats['service_stats'][service_name] = {
                    'status': 'active',
                    'stats_available': False
                }
        except Exception as e:
            stats['service_stats'][service_name] = {
                'error': str(e),
                'status': 'error'
            }
    
    return stats

# Export der wichtigsten Funktionen und Services
__all__ = [
    # Hauptfunktionen
    'initialize_services',
    'get_service', 
    'get_all_services',
    'is_initialized',
    'reset_services',
    
    # Monitoring
    'health_check',
    'get_service_stats',
    
    # Service-Klassen
    'DocumentService',
    'EmbeddingService', 
    'ChatService',
    'RetrievalService',
    'SessionService',
    'VectorStoreService',
    'ServiceIntegrator',
    
    # Registry
    'SERVICE_REGISTRY'
]

# Automatische Initialisierung Warnung
if not _initialized:
    logger.info("📦 Services Package geladen - verwende initialize_services() für Aktivierung")
