#!/usr/bin/env python3
"""
Controllers Package - Request Orchestration Layer  
Industrielle RAG-Architektur - Phase 4 Migration

Sammelt alle Controller-Komponenten mit korrigierten Imports
und robuster Service-Integration für die Pipeline-Steuerung.

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

from typing import Dict, Any, Optional, List
import logging

# Core Imports
from core import get_logger, ServiceContainer

logger = get_logger(__name__)

# =============================================================================
# CONTROLLER IMPORTS MIT FALLBACK-HANDLING
# =============================================================================

# Verfügbare Controller tracken
AVAILABLE_CONTROLLERS = {}
CONTROLLER_IMPORT_ERRORS = {}

def safe_import_controller(controller_name: str, module_path: str, class_name: str):
    """
    Sichere Controller-Import-Funktion mit Fehlerbehandlung
    
    Args:
        controller_name: Name des Controllers für Tracking
        module_path: Pfad zum Controller-Modul
        class_name: Name der Controller-Klasse
    
    Returns:
        Controller-Klasse oder None bei Fehlern
    """
    try:
        module = __import__(module_path, fromlist=[class_name])
        controller_class = getattr(module, class_name)
        AVAILABLE_CONTROLLERS[controller_name] = controller_class
        logger.debug(f"✅ {controller_name} geladen")
        return controller_class
    except ImportError as e:
        CONTROLLER_IMPORT_ERRORS[controller_name] = str(e)
        logger.warning(f"⚠️ {controller_name} failed: {e}")
        AVAILABLE_CONTROLLERS[controller_name] = None
        return None
    except Exception as e:
        CONTROLLER_IMPORT_ERRORS[controller_name] = str(e)
        logger.error(f"❌ Unerwarteter Fehler beim Laden von {controller_name}: {e}")
        AVAILABLE_CONTROLLERS[controller_name] = None
        return None

print("Loading controllers module...")

# Core Controller (immer erforderlich)
PipelineController = safe_import_controller(
    "PipelineController",
    "controllers.pipeline_controller",
    "PipelineController"
)

SessionController = safe_import_controller(
    "SessionController", 
    "controllers.session_controller",
    "SessionController"
)

HealthController = safe_import_controller(
    "HealthController",
    "controllers.health_controller", 
    "HealthController"
)

# Status-Report
available_count = len([c for c in AVAILABLE_CONTROLLERS.values() if c is not None])
total_count = len(AVAILABLE_CONTROLLERS)

print(f"Controllers module ready. Available: {list(AVAILABLE_CONTROLLERS.keys())}")


# =============================================================================
# CONTROLLER FACTORY
# =============================================================================

class ControllerFactory:
    """
    Factory für Controller-Erstellung mit Dependency Injection
    """
    
    def __init__(self, service_container: ServiceContainer):
        self.container = service_container
        self.logger = get_logger(f"{__name__}.factory")
        self._controller_instances = {}
    
    def create_controller(self, controller_name: str, **kwargs) -> Optional[Any]:
        """
        Erstellt Controller-Instanz
        
        Args:
            controller_name: Name des Controllers
            **kwargs: Zusätzliche Parameter
        
        Returns:
            Controller-Instanz oder None
        """
        if controller_name not in AVAILABLE_CONTROLLERS:
            self.logger.error(f"Controller '{controller_name}' nicht bekannt")
            return None
        
        controller_class = AVAILABLE_CONTROLLERS[controller_name]
        if controller_class is None:
            self.logger.error(f"Controller '{controller_name}' nicht verfügbar")
            return None
        
        try:
            # Singleton-Pattern für Controller
            if controller_name in self._controller_instances:
                return self._controller_instances[controller_name]
            
            # Controller erstellen mit Service-Dependencies
            if controller_name == "PipelineController":
                # Service-Dependencies aus Container holen
                service_manager = self.container.get_service("service_manager")
                controller = controller_class(service_manager=service_manager, **kwargs)
                
            elif controller_name == "SessionController":
                # Session-spezifische Dependencies
                session_service = self.container.get_service("session_service")
                controller = controller_class(session_service=session_service, **kwargs)
                
            elif controller_name == "HealthController":
                # Health-Check-Dependencies
                service_manager = self.container.get_service("service_manager")
                controller = controller_class(service_manager=service_manager, **kwargs)
                
            else:
                # Generische Controller-Erstellung
                controller = controller_class(service_container=self.container, **kwargs)
            
            self._controller_instances[controller_name] = controller
            self.logger.info(f"Controller '{controller_name}' erfolgreich erstellt")
            return controller
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen von Controller '{controller_name}': {e}")
            return None
    
    def get_controller_status(self) -> Dict[str, Any]:
        """
        Gibt Status aller Controller zurück
        
        Returns:
            Dict mit Controller-Status
        """
        return {
            "available_controllers": {k: v is not None for k, v in AVAILABLE_CONTROLLERS.items()},
            "import_errors": CONTROLLER_IMPORT_ERRORS.copy(),
            "created_instances": list(self._controller_instances.keys()),
            "total_available": available_count,
            "total_controllers": total_count
        }


# =============================================================================
# CONTROLLER MANAGER
# =============================================================================

class ControllerManager:
    """
    Zentraler Manager für alle Request-Controller
    """
    
    def __init__(self, service_container: ServiceContainer):
        self.container = service_container
        self.factory = ControllerFactory(service_container)
        self.logger = get_logger(f"{__name__}.manager")
        
        # Core Controller
        self._pipeline_controller = None
        self._session_controller = None
        self._health_controller = None
        
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialisiert alle verfügbaren Controller
        
        Returns:
            bool: True wenn Core-Controller verfügbar
        """
        try:
            self.logger.info("Initialisiere Controller-Manager...")
            
            # Core Controller initialisieren
            self._pipeline_controller = self.factory.create_controller("PipelineController")
            self._session_controller = self.factory.create_controller("SessionController")
            self._health_controller = self.factory.create_controller("HealthController")
            
            # Prüfe ob mindestens Core-Controller verfügbar sind
            core_controllers_ok = all([
                self._pipeline_controller is not None,
                self._session_controller is not None,
                self._health_controller is not None
            ])
            
            if not core_controllers_ok:
                self.logger.warning("Nicht alle Core-Controller verfügbar")
                
            self._initialized = True
            self.logger.info("Controller-Manager initialisiert")
            
            return core_controllers_ok
            
        except Exception as e:
            self.logger.error(f"Fehler bei Controller-Manager-Initialisierung: {e}")
            return False
    
    # Getter-Methoden für Controller
    def get_pipeline_controller(self) -> Optional[Any]:
        """Gibt PipelineController zurück"""
        return self._pipeline_controller
    
    def get_session_controller(self) -> Optional[Any]:
        """Gibt SessionController zurück"""
        return self._session_controller
    
    def get_health_controller(self) -> Optional[Any]:
        """Gibt HealthController zurück"""
        return self._health_controller
    
    def get_all_controllers(self) -> Dict[str, Any]:
        """
        Gibt alle verfügbaren Controller-Instanzen zurück
        
        Returns:
            Dict mit Controller-Name -> Controller-Instanz
        """
        return {
            "pipeline_controller": self._pipeline_controller,
            "session_controller": self._session_controller,
            "health_controller": self._health_controller
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Gibt Health-Status aller Controller zurück
        
        Returns:
            Dict mit Health-Status
        """
        controllers = self.get_all_controllers()
        
        status = {
            "initialized": self._initialized,
            "controllers": {}
        }
        
        for name, controller in controllers.items():
            if controller is not None:
                # Prüfe ob Controller Health-Check-Methode hat
                if hasattr(controller, 'health_check'):
                    try:
                        status["controllers"][name] = controller.health_check()
                    except Exception as e:
                        status["controllers"][name] = {
                            "status": "error",
                            "error": str(e)
                        }
                else:
                    status["controllers"][name] = {
                        "status": "available",
                        "has_health_check": False
                    }
            else:
                status["controllers"][name] = {
                    "status": "unavailable",
                    "reason": CONTROLLER_IMPORT_ERRORS.get(name, "Unknown")
                }
        
        # Allgemeine Statistiken
        available_controllers = len([c for c in controllers.values() if c is not None])
        status["summary"] = {
            "total_controllers": len(controllers),
            "available_controllers": available_controllers,
            "unavailable_controllers": len(controllers) - available_controllers,
            "core_controllers_ok": all([
                self._pipeline_controller is not None,
                self._session_controller is not None,
                self._health_controller is not None
            ])
        }
        
        return status


# =============================================================================
# GLOBALE FACTORY FUNCTIONS
# =============================================================================

def create_controller_manager(service_container: ServiceContainer) -> ControllerManager:
    """
    Erstellt ControllerManager mit Service-Container
    
    Args:
        service_container: Konfigurierter ServiceContainer
    
    Returns:
        ControllerManager: Konfigurierter Controller-Manager
    """
    manager = ControllerManager(service_container)
    
    if not manager.initialize():
        logger.warning("Controller-Manager konnte nicht vollständig initialisiert werden")
    
    return manager


def get_controller_availability() -> Dict[str, bool]:
    """
    Gibt Verfügbarkeit aller Controller zurück
    
    Returns:
        Dict mit Controller-Name -> Verfügbar (bool)
    """
    return {name: controller is not None for name, controller in AVAILABLE_CONTROLLERS.items()}


def get_controller_import_errors() -> Dict[str, str]:
    """
    Gibt alle Controller-Import-Fehler zurück
    
    Returns:
        Dict mit Controller-Name -> Fehlermeldung
    """
    return CONTROLLER_IMPORT_ERRORS.copy()


# =============================================================================
# CONVENIENCE FUNCTIONS FÜR LEGACY-KOMPATIBILITÄT
# =============================================================================

def get_pipeline_controller():
    """Legacy-kompatible Funktion für PipelineController"""
    return AVAILABLE_CONTROLLERS.get("PipelineController")

def get_session_controller():
    """Legacy-kompatible Funktion für SessionController"""
    return AVAILABLE_CONTROLLERS.get("SessionController")

def get_health_controller():
    """Legacy-kompatible Funktion für HealthController"""
    return AVAILABLE_CONTROLLERS.get("HealthController")


# =============================================================================
# REQUEST/RESPONSE DATENSTRUKTUREN
# =============================================================================

# Import von standardisierten Request/Response-Strukturen
try:
    from .pipeline_controller import (
        PipelineRequest, PipelineResponse, PipelineStage
    )
    PIPELINE_TYPES_AVAILABLE = True
except ImportError:
    PIPELINE_TYPES_AVAILABLE = False
    # Fallback-Implementierungen
    class PipelineRequest:
        """Fallback PipelineRequest"""
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class PipelineResponse:
        """Fallback PipelineResponse"""
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class PipelineStage:
        """Fallback PipelineStage"""
        pass

try:
    from .session_controller import SessionType
    SESSION_TYPES_AVAILABLE = True
except ImportError:
    SESSION_TYPES_AVAILABLE = False
    # Fallback
    class SessionType:
        """Fallback SessionType"""
        pass


# =============================================================================
# CONVENIENCE FUNCTIONS FÜR PIPELINE-OPERATIONEN
# =============================================================================

def process_query_request(query: str, **kwargs):
    """
    Convenience-Funktion für Query-Processing
    
    Args:
        query: Benutzer-Anfrage
        **kwargs: Zusätzliche Parameter
    
    Returns:
        Pipeline-Response
    """
    pipeline_controller = get_pipeline_controller()
    if pipeline_controller and hasattr(pipeline_controller, 'process_query'):
        return pipeline_controller.process_query(query, **kwargs)
    else:
        raise RuntimeError("PipelineController nicht verfügbar oder nicht implementiert")


def process_indexing_request(file_path: str, **kwargs):
    """
    Convenience-Funktion für Document-Indexing
    
    Args:
        file_path: Pfad zur Dokument-Datei
        **kwargs: Zusätzliche Parameter
    
    Returns:
        Pipeline-Response
    """
    pipeline_controller = get_pipeline_controller()
    if pipeline_controller and hasattr(pipeline_controller, 'process_document'):
        return pipeline_controller.process_document(file_path, **kwargs)
    else:
        raise RuntimeError("PipelineController nicht verfügbar oder nicht implementiert")


def get_system_health_summary():
    """
    Convenience-Funktion für System-Health-Summary
    
    Returns:
        Health-Summary
    """
    health_controller = get_health_controller()
    if health_controller and hasattr(health_controller, 'get_system_health'):
        return health_controller.get_system_health()
    else:
        return {"status": "unavailable", "reason": "HealthController nicht verfügbar"}


def create_query_request(query: str, **kwargs) -> PipelineRequest:
    """
    Erstellt standardisierten Query-Request
    
    Args:
        query: Benutzer-Anfrage
        **kwargs: Zusätzliche Parameter
    
    Returns:
        PipelineRequest
    """
    return PipelineRequest(
        type="query",
        query=query,
        **kwargs
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

# Dynamische Exports basierend auf verfügbaren Controllern
__all__ = [
    # Manager und Factory
    'ControllerFactory', 'ControllerManager',
    
    # Factory Functions
    'create_controller_manager', 'get_controller_availability', 'get_controller_import_errors',
    
    # Legacy Functions
    'get_pipeline_controller', 'get_session_controller', 'get_health_controller',
    
    # Convenience Functions
    'process_query_request', 'process_indexing_request', 'get_system_health_summary',
    'create_query_request',
    
    # Data Types
    'PipelineRequest', 'PipelineResponse', 'PipelineStage', 'SessionType',
    
    # Status Constants
    'AVAILABLE_CONTROLLERS', 'CONTROLLER_IMPORT_ERRORS', 
    'PIPELINE_TYPES_AVAILABLE', 'SESSION_TYPES_AVAILABLE'
]

# Füge verfügbare Controller-Klassen zu Exports hinzu
for controller_name, controller_class in AVAILABLE_CONTROLLERS.items():
    if controller_class is not None:
        __all__.append(controller_name)
        globals()[controller_name] = controller_class


# =============================================================================
# LOGGING STATUS
# =============================================================================

if __name__ != "__main__":
    logger.info(f"Controllers Package geladen - {available_count}/{total_count} Controller verfügbar")
    if CONTROLLER_IMPORT_ERRORS:
        logger.warning(f"Controller-Import-Fehler: {list(CONTROLLER_IMPORT_ERRORS.keys())}")

        