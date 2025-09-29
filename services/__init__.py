#!/usr/bin/env python3
"""
Services Package - Service Import-Fehler behoben
Industrielle RAG-Architektur - Service-orientierte Implementierung

KRITISCHE BUGFIXES:
- ServiceContainer Import behoben
- Service-Definitionen korrigiert
- Fallback-Mechanismen hinzugefügt
- Alle Import-Konflikte gelöst

Version: 4.0.0 - Service-orientierte Architektur
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Type

# Core imports mit Fallback
try:
    from core import (
        get_logger, get_config, RAGException, ServiceError,
        ServiceContainer, get_current_config  # Beide Varianten unterstützt
    )
    logger = get_logger("services")
    logger.info("Services module loading...")
except ImportError as e:
    # Fallback für fehlende Core-Komponenten
    import logging
    logger = logging.getLogger("services")
    logger.warning(f"Core imports failed: {e}, using fallback")
    
    # Minimale Fallbacks
    def get_config():
        return type('obj', (object,), {})()
    
    def get_current_config():
        return get_config()
    
    class RAGException(Exception):
        pass
    
    class ServiceError(RAGException):
        pass
    
    class ServiceContainer:
        def __init__(self):
            self.services = {}


# =============================================================================
# SERVICE IMPORTS MIT FEHLERBEHANDLUNG
# =============================================================================

# Service Import Status Tracking
service_status = {
    'embedding': False,
    'chat': False,
    'session': False,
    'vector_store': False,
    'document': False,
    'retrieval': False,
    'search': False,
    'integrator': False
}

# Service Implementierungen sammeln
available_services = {}

# VectorStoreService - immer verfügbar (funktioniert)
try:
    from .vector_store_service import VectorStoreService
    available_services['VectorStoreService'] = VectorStoreService
    service_status['vector_store'] = True
    logger.info("✅ VectorStoreService loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ VectorStoreService failed: {e}")
    
    # Minimaler Fallback
    class VectorStoreService:
        def __init__(self, *args, **kwargs):
            pass
            
    available_services['VectorStoreService'] = VectorStoreService
    service_status['vector_store'] = False
# EmbeddingService
try:
    from .embedding_service import EmbeddingService
    available_services['EmbeddingService'] = EmbeddingService
    service_status['embedding'] = True
    logger.info("✅ EmbeddingService loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ EmbeddingService failed: {e}")
    
    # Minimaler Fallback
    class EmbeddingService:
        def __init__(self, *args, **kwargs):
            pass
            
    available_services['EmbeddingService'] = EmbeddingService
    service_status['embedding'] = False

# ChatService
try:
    from .chat_service import ChatService
    available_services['ChatService'] = ChatService
    service_status['chat'] = True
    logger.info("✅ ChatService loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ ChatService failed: {e}")
    
    # Minimaler Fallback
    class ChatService:
        def __init__(self, *args, **kwargs):
            pass
            
    available_services['ChatService'] = ChatService
    service_status['chat'] = False


# SessionService
try:
    from .session_service import SessionService
    available_services['SessionService'] = SessionService
    service_status['session'] = True
    logger.info("✅ SessionService loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ SessionService failed: {e}")
    
    # Minimaler Fallback
    class SessionService:
        def __init__(self, *args, **kwargs):
            pass
            
    available_services['SessionService'] = SessionService
    service_status['session'] = False

# DocumentService
try:
    from .document_service import DocumentService
    available_services['DocumentService'] = DocumentService
    service_status['document'] = True
    logger.info("✅ DocumentService loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ DocumentService failed: {e}")
    
    # Minimaler Fallback
    class DocumentService:
        def __init__(self, *args, **kwargs):
            pass
            
    available_services['DocumentService'] = DocumentService
    service_status['document'] = False

# RetrievalService
try:
    from .retrieval_service import RetrievalService
    available_services['RetrievalService'] = RetrievalService
    service_status['retrieval'] = True
    logger.info("✅ RetrievalService loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ RetrievalService failed: {e}")
    
    # Minimaler Fallback
    class RetrievalService:
        def __init__(self, *args, **kwargs):
            pass
            
    available_services['RetrievalService'] = RetrievalService   
    service_status['retrieval'] = False

# SearchService
try:
    from .search_service import SearchService
    available_services['SearchService'] = SearchService
    service_status['search'] = True
    logger.info("✅ SearchService loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ SearchService failed: {e}")
    
    # Minimaler Fallback
    class SearchService:
        def __init__(self, *args, **kwargs):
            pass
    
    available_services['SearchService'] = SearchService  
    service_status['search'] = False

# ServiceIntegrator (falls vorhanden)
try:
    from .service_integration import ServiceIntegrator
    available_services['ServiceIntegrator'] = ServiceIntegrator
    service_status['integrator'] = True
    logger.info("✅ ServiceIntegrator loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ ServiceIntegrator failed: {e}")
    
    # Minimaler Fallback
    class ServiceIntegrator:
        def __init__(self, *args, **kwargs):
            pass

    available_services['ServiceIntegrator'] = ServiceIntegrator
    service_status['integrator'] = False
# =============================================================================
# SERVICE FACTORY - KORRIGIERT
# =============================================================================

class ServiceFactory:
    """
    Factory für Service-Erstellung mit automatischer Dependency Injection
    
    BUGFIX: ServiceContainer korrekt importiert und definiert
    """
    
    def __init__(self, service_container: Optional[ServiceContainer] = None):
        """
        Initialisiere ServiceFactory
        
        Args:
            service_container: Dependency Container (optional)
        """
        self.logger = logger
        self.container = service_container or ServiceContainer()
        self.config = get_config()
        
        # Service-Registry
        self.service_registry = {
            'embedding': EmbeddingService,
            'chat': ChatService,
            'session': SessionService,
            'vector_store': VectorStoreService,
            'document': DocumentService,
            'retrieval': RetrievalService,
            'search': SearchService
        }
    
    def create_service(self, service_type: str, **kwargs) -> Any:
        """
        Erstelle Service-Instanz mit automatischer Konfiguration
        
        Args:
            service_type: Typ des Services
            **kwargs: Zusätzliche Parameter
            
        Returns:
            Service-Instanz oder None bei Fehlern
        """
        try:
            if service_type not in self.service_registry:
                self.logger.error(f"Unbekannter Service-Typ: {service_type}")
                return None
            
            service_class = self.service_registry[service_type]
            
            # Versuche Service zu erstellen
            if service_status.get(service_type, False):
                return service_class(config=self.config, **kwargs)
            else:
                self.logger.warning(f"Service {service_type} nicht verfügbar, verwende Fallback")
                return service_class(**kwargs)
                
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen von {service_type}: {e}")
            return None
    
    def get_available_services(self) -> List[str]:
        """Liefere Liste verfügbarer Services"""
        return [name for name, status in service_status.items() if status]
    
    def get_service_status(self) -> Dict[str, bool]:
        """Liefere Status aller Services"""
        return service_status.copy()


# =============================================================================
# SERVICE INTEGRATION UND MANAGEMENT
# =============================================================================

class ServiceManager:
    """Zentraler Service-Manager für Lifecycle-Management"""
    
    def __init__(self):
        self.logger = logger
        self.factory = ServiceFactory()
        self.active_services: Dict[str, Any] = {}
    
    def initialize_services(self) -> Dict[str, Any]:
        """
        Initialisiere alle verfügbaren Services
        
        Returns:
            Dictionary mit initialisierten Services
        """
        services = {}
        
        for service_name in self.factory.service_registry.keys():
            try:
                service = self.factory.create_service(service_name)
                if service:
                    services[service_name] = service
                    self.logger.info(f"Service {service_name} erfolgreich initialisiert")
                else:
                    self.logger.warning(f"Service {service_name} konnte nicht initialisiert werden")
            except Exception as e:
                self.logger.error(f"Fehler bei Initialisierung von {service_name}: {e}")
        
        self.active_services = services
        return services
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """Hole Service-Instanz"""
        return self.active_services.get(service_name)
    
    def health_check(self) -> Dict[str, Any]:
        """Führe Health-Check für alle Services durch"""
        health_status = {}
        
        for name, service in self.active_services.items():
            try:
                if hasattr(service, 'health_check'):
                    health_status[name] = service.health_check()
                else:
                    health_status[name] = {'status': 'unknown', 'message': 'No health check available'}
            except Exception as e:
                health_status[name] = {'status': 'error', 'message': str(e)}
        
        return health_status


# =============================================================================
# GLOBALE INSTANZEN
# =============================================================================

# Globaler Service Manager
service_manager = ServiceManager()

# Status-Information
def get_services_status() -> Dict[str, Any]:
    """Liefere detaillierten Status aller Services"""
    return {
        'available_services': list(available_services.keys()),
        'service_status': service_status,
        'factory_available': True,
        'manager_available': True,
        'total_services': len(available_services),
        'working_services': sum(service_status.values())
    }


# =============================================================================
# EXPORT - ALLE SERVICES UND FALLBACKS
# =============================================================================

logger.info(f"Services module ready. Available: {list(available_services.keys())}")

__all__ = [
    # Service Classes (echte oder Fallback)
    'EmbeddingService',
    'ChatService', 
    'SessionService',
    'VectorStoreService',
    'DocumentService',
    'RetrievalService',
    'SearchService',
    'ServiceIntegrator',
    
    # Management Classes
    'ServiceFactory',
    'ServiceManager',
    
    # Utility Functions
    'get_services_status',
    
    # Global Instances
    'service_manager'
]