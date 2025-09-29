#!/usr/bin/env python3
"""
Core Package - Import-Konflikte und fehlende Funktionen behoben
Industrielle RAG-Architektur - Service-orientierte Implementierung

KRITISCHE BUGFIXES:
- get_current_config -> get_config (Funktionsname korrigiert)
- ServiceContainer Import behoben
- Alle fehlenden Exception-Classes hinzugefügt
- Robuste Fallback-Mechanismen

Version: 4.0.0 - Service-orientierte Architektur
"""

# =============================================================================
# CORE IMPORTS - MIT BUGFIXES
# =============================================================================

try:
    # Logging System (zuerst laden)
    from .logger import (
        setup_logging,
        get_logger,
        log_performance,
        log_method_calls,
        emergency_log,
        RAGLogger,
        JSONFormatter,
        SafeFormatter
    )
    
    # Exceptions (vor Config laden)
    from .exceptions import (
        # Base Exception
        RAGException,
        ExceptionSeverity,
        ExceptionCategory,
        ErrorContext,
        
        # Configuration Exceptions
        ConfigurationException,
        ConfigurationError,
        ValidationError,
        
        # Service Exceptions
        ServiceError,
        ServiceException,            # SCHRITT 5: NEU HINZUFÜGEN
                
        # Pipeline Exceptions
        PipelineException,
        DocumentProcessingError,
        
        # Health Exceptions
        HealthCheckException,
        
        # Provider Exceptions
        LLMError,
        EmbeddingError,
        EmbeddingException,  # NEU HINZUGEFÜGT
        VectorStoreError,
        
        # Document Exceptions - NEU HINZUGEFÜGT
        DocumentException,
        RecordException,
        
        # Utilities
        create_error_context,
        handle_exception,
        exception_registry,
        
        # Interface/System Exceptions
        InterfaceException,
        RAGSystemException,          # <- Falls noch nicht da
        
        # Service Erweiterung
        ChatServiceError,          # NEU
        ResourceError,             # NEU
        SessionException          # NEU
    )
    
    # Configuration System (nach Exceptions)
    from .config import (
        RAGConfig,
        ApplicationConfig,
        LLMConfig,
        EmbeddingConfig,
        VectorStoreConfig,
        TextProcessingConfig,
        LoggingConfig,
        load_config,
        get_config,
        set_config,
        reset_config
    )
    
    # Container System (falls vorhanden)
    try:
        from .container import (
            ServiceContainer,  # KORRIGIERTER IMPORT
            ServiceRegistry,
            get_container,
            register_service,
            resolve_service
        )
        _CONTAINER_AVAILABLE = True
    except ImportError:
        # Fallback ServiceContainer Definition
        class ServiceContainer:
            """Minimale ServiceContainer Implementation als Fallback"""
            def __init__(self):
                self.services = {}
            
            def register(self, service_type, implementation):
                self.services[service_type] = implementation
            
            def resolve(self, service_type):
                return self.services.get(service_type)
        
        # Globale Container-Instanz
        _global_container = ServiceContainer()
        
        def get_container():
            return _global_container
        
        def register_service(service_type, implementation):
            _global_container.register(service_type, implementation)
        
        def resolve_service(service_type):
            return _global_container.resolve(service_type)
        
        class ServiceRegistry:
            """Minimale ServiceRegistry als Fallback"""
            def __init__(self):
                self.registry = {}
        
        _CONTAINER_AVAILABLE = False
    
    _CORE_INITIALIZED = True
    
except ImportError as e:
    # Fallback-Modus bei Import-Fehlern
    import logging
    print(f"WARNUNG: Core-Import-Fehler: {e}")
    
    # Minimale Fallback-Implementierungen
    def get_logger(name: str = "RAG"):
        return logging.getLogger(name)
    
    def setup_logging(**kwargs):
        logging.basicConfig(level=logging.INFO)
    
    def get_config():
        # Minimal-Konfiguration als Fallback
        from dataclasses import dataclass
        
        @dataclass
        class MinimalConfig:
            def __init__(self):
                self.llm = type('obj', (object,), {'providers': ['ollama'], 'model_name': 'llama3.1:8b'})()
                self.embeddings = type('obj', (object,), {'providers': ['ollama'], 'model_name': 'nomic-embed-text'})()
                self.vector_store = type('obj', (object,), {'providers': ['chroma'], 'collection_name': 'industrial_documents'})()
        
        return MinimalConfig()
    
    # Alias für Kompatibilität - BUGFIX
    get_current_config = get_config
    
    # Basis Exception
    class RAGException(Exception):
        pass
    
    # Minimale ServiceContainer
    class ServiceContainer:
        def __init__(self):
            self.services = {}
    
    _CORE_INITIALIZED = False


# =============================================================================
# KOMPATIBILITÄTS-ALIASE - BUGFIX
# =============================================================================

# Alias für get_current_config -> get_config (Import-Kompatibilität)
get_current_config = get_config

# Service Container global verfügbar machen
if not 'ServiceContainer' in globals():
    class ServiceContainer:
        """Fallback ServiceContainer für fehlende Implementierung"""
        def __init__(self):
            self.services = {}
        
        def register(self, service_type, implementation):
            self.services[service_type] = implementation
        
        def resolve(self, service_type):
            return self.services.get(service_type)


# =============================================================================
# CORE SYSTEM VALIDATION
# =============================================================================

def validate_core_system() -> bool:
    """
    Validiere dass alle Core-Komponenten korrekt geladen sind
    
    Returns:
        bool: True wenn alle Komponenten verfügbar sind
    """
    try:
        # Test Logger
        logger = get_logger("core_validation")
        logger.info("Logger Test erfolgreich")
        
        # Test Config
        config = get_config()
        if not config:
            return False
        
        # Test Exceptions
        test_exception = RAGException("Test Exception")
        if not isinstance(test_exception, Exception):
            return False
        
        return True
        
    except Exception as e:
        print(f"Core-System Validierung fehlgeschlagen: {e}")
        return False


def get_core_status() -> dict:
    """
    Liefere Status aller Core-Komponenten
    
    Returns:
        dict: Status-Informationen
    """
    status = {
        "initialized": _CORE_INITIALIZED,
        "logger_available": True,
        "config_available": True,
        "exceptions_available": True,
        "container_available": _CONTAINER_AVAILABLE,
        "validation_passed": False,
        "compatibility_aliases": True  # NEU
    }
    
    try:
        status["validation_passed"] = validate_core_system()
    except:
        status["validation_passed"] = False
    
    return status


# =============================================================================
# INITIALIZATION HELPER
# =============================================================================

def initialize_core_system(
    config_path: str = None,
    log_level: str = "INFO",
    force_reinit: bool = False
) -> bool:
    """
    Initialisiere das Core-System mit optionalen Parametern
    
    Args:
        config_path: Pfad zur Konfigurationsdatei
        log_level: Logging-Level
        force_reinit: Erzwinge Neuinitialisierung
        
    Returns:
        bool: True bei erfolgreicher Initialisierung
    """
    try:
        # Logging initialisieren
        setup_logging(level=log_level)
        logger = get_logger("core_init")
        
        # Konfiguration laden
        if config_path:
            config = load_config(config_path)
        else:
            config = get_config()
        
        # Validierung
        if validate_core_system():
            logger.info("Core-System erfolgreich initialisiert")
            return True
        else:
            logger.error("Core-System Validierung fehlgeschlagen")
            return False
            
    except Exception as e:
        print(f"Core-System Initialisierung fehlgeschlagen: {e}")
        return False


# =============================================================================
# EXPORT - ERWEITERTE LISTE
# =============================================================================

# Logging
__all__ = [
    # Logging
    "setup_logging",
    "get_logger", 
    "log_performance",
    "log_method_calls",
    "emergency_log",
    "RAGLogger",
    
    # Configuration
    "RAGConfig",
    "ApplicationConfig",
    "LLMConfig", 
    "EmbeddingConfig",
    "VectorStoreConfig",
    "TextProcessingConfig",
    "LoggingConfig",
    "load_config",
    "get_config",
    "get_current_config",  # KOMPATIBILITÄTS-ALIAS
    "set_config",
    "reset_config",
    
    # Exceptions - VOLLSTÄNDIGE LISTE
    "RAGException",
    "ExceptionSeverity",
    "ExceptionCategory", 
    "ErrorContext",
    "ConfigurationException",
    "ConfigurationError", 
    "ValidationError",
    "ServiceError",
    "ServiceException",          # SCHRITT 5: NEU HINZUFÜGEN
    "PipelineException",
    "DocumentProcessingError",
    "HealthCheckException",
    "LLMError",
    "EmbeddingError",
    "EmbeddingException",  # NEU HINZUGEFÜGT
    "VectorStoreError",
    "DocumentException",  # NEU HINZUGEFÜGT
    "RecordException",    # NEU HINZUGEFÜGT
    "create_error_context",
    "handle_exception",
    
    # Container
    "ServiceContainer",   # VERFÜGBAR GEMACHT
    "ServiceRegistry",
    "get_container",
    "register_service", 
    "resolve_service",
    
    # Core System
    "validate_core_system",
    "get_core_status",
    "initialize_core_system",
    
    # Config - mit Alias
    "get_config",
    "get_current_config",  # BUG-003 FIX
    
    # Exception - neue Classes
    "InterfaceException",    # BUG-001 FIX
    "RAGSystemException",    # BUG-002 FIX
    
    "ChatServiceError",        # NEU
    "ResourceError",           # NEU
    "SessionException"        # NEU
]

# Automatische Initialisierung beim Import
try:
    if _CORE_INITIALIZED:
        # Basis-Logging aktivieren
        setup_logging(level="INFO", console_output=True)
        
        # Status-Check
        status = get_core_status()
        if status["validation_passed"]:
            logger = get_logger("core")
            logger.info("Core-System beim Import erfolgreich validiert")
        else:
            print("WARNUNG: Core-System Validierung beim Import fehlgeschlagen")
            
except Exception as e:
    print(f"WARNUNG: Core-System Auto-Initialisierung fehlgeschlagen: {e}")


# =============================================================================
# FEHLENDE CONTAINER-FUNKTIONEN - INSTANT FIX
# =============================================================================

# Globale Container-Instanzen
if '_global_container' not in globals():
    _global_container = ServiceContainer()
    _global_registry = ServiceRegistry() if 'ServiceRegistry' in globals() else type('obj', (object,), {'registry': {}, 'register_service': lambda self, k, v: setattr(self, k, v), 'get_service': lambda self, k: getattr(self, k, None)})()

def get_service_container():
    """Gibt globalen Service-Container zurück - INSTANT FIX"""
    return _global_container

def reset_service_container():
    """Setzt Service-Container zurück"""
    global _global_container
    _global_container = ServiceContainer()
    return _global_container

def get_service_registry():
    """Gibt Service-Registry zurück"""
    return _global_registry

# get_container Alias falls noch nicht vorhanden
if 'get_container' not in globals():
    get_container = get_service_container

print("✅ Container-Funktionen instant fix angewendet")

# Package Metadata
__version__ = "4.0.0"
__author__ = "KI-Consultant für industrielle Automatisierung"
__description__ = "Core-System für service-orientierte RAG-Architektur"