#!/usr/bin/env python3
"""
Core Module Integration für RAG Chatbot Industrial

Zentrale Initialisierung und Integration aller Core-Komponenten:
- Konfigurationsverwaltung (config.py)
- Logging-System (logger.py) 
- Exception-Handling (exceptions.py)
- Dependency Injection (container.py)

Stellt einheitliche Schnittstelle für das gesamte RAG-System bereit.

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Core-Integration
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Core-Komponenten importieren
from .config import (
    RAGConfig, ConfigurationManager, ConfigurationFactory, ConfigurationValidator,
    get_config_manager, get_config, reset_config, generate_example_configs,
    # Enums
    LogLevel, LLMProvider, EmbeddingProvider, VectorStoreProvider
)

from .logger import (
    LoggingManager, StructuredJSONFormatter, AsyncFileHandler, ColoredConsoleHandler,
    LogContext, PerformanceMetrics,
    get_logging_manager, get_logger, setup_logging, shutdown_logging,
    log_performance, log_method_calls, log_context,
    log_system_info, log_config_info
)

from .exceptions import (
    # Base Exception
    RAGException, ErrorContext, ErrorSeverity, ErrorCategory,
    # Specific Exceptions
    ConfigurationError, InvalidConfigurationError, MissingConfigurationError,
    ValidationError, InvalidInputError, DocumentValidationError,
    ServiceError, DocumentProcessingError, EmbeddingError, VectorStoreError,
    LLMError, ChatServiceError,
    NetworkError, ExternalServiceError, APITimeoutError, APIRateLimitError,
    ResourceError, MemoryError, StorageError,
    # Utilities
    handle_exception, create_error_context, wrap_external_exception,
    ExceptionRegistry, get_exception_registry
)

from .container import (
    ServiceContainer, ServiceLifecycle, ServiceStatus, ServiceRegistration,
    DependencyInjectionError, ServiceNotRegisteredError, 
    CircularDependencyError, ServiceInitializationError,
    get_service_container, reset_service_container,
    inject, ServiceScope
)


# =============================================================================
# CORE-SYSTEM INITIALISIERUNG
# =============================================================================

class CoreSystem:
    """
    Zentrale Core-System-Initialisierung und -Verwaltung
    
    Koordiniert die Initialisierung aller Core-Komponenten und stellt
    einheitliche Lifecycle-Verwaltung bereit.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 environment: str = "development"):
        """
        Initialisiert Core-System
        
        Args:
            config_path (Optional[str]): Pfad zur Konfigurationsdatei
            environment (str): Umgebung (development, staging, production)
        """
        self.environment = environment
        self.config_path = config_path
        
        # Komponenten-Status
        self._config_manager: Optional[ConfigurationManager] = None
        self._logging_manager: Optional[LoggingManager] = None
        self._service_container: Optional[ServiceContainer] = None
        self._config: Optional[RAGConfig] = None
        
        # Initialisierungsstatus
        self._initialized = False
        self._initialization_error: Optional[Exception] = None
    
    def initialize(self) -> bool:
        """
        Initialisiert alle Core-Komponenten in der richtigen Reihenfolge
        
        Returns:
            bool: True wenn Initialisierung erfolgreich
        """
        if self._initialized:
            return True
        
        try:
            print("🔧 Initialisiere RAG Chatbot Core-System...")
            
            # 1. Konfiguration initialisieren
            self._initialize_configuration()
            print("✅ Konfiguration geladen")
            
            # 2. Logging initialisieren
            self._initialize_logging()
            print("✅ Logging-System aktiviert")
            
            # 3. Exception-Registry initialisieren
            self._initialize_exception_handling()
            print("✅ Exception-Handling bereit")
            
            # 4. Dependency Injection initialisieren
            self._initialize_dependency_injection()
            print("✅ Service-Container konfiguriert")
            
            # 5. System-Informationen loggen
            self._log_startup_information()
            
            self._initialized = True
            print("🎯 Core-System erfolgreich initialisiert!")
            
            return True
            
        except Exception as e:
            self._initialization_error = e
            print(f"❌ Core-System Initialisierung fehlgeschlagen: {e}")
            return False
    
    def _initialize_configuration(self) -> None:
        """Initialisiert Konfigurationsmanagement"""
        # Beispiel-Konfigurationen generieren falls nicht vorhanden
        config_dir = Path("./config")
        if not config_dir.exists() or not any(config_dir.glob("*.yaml")):
            print("📝 Generiere Beispiel-Konfigurationsdateien...")
            generate_example_configs()
        
        # Konfiguration laden
        self._config_manager = get_config_manager(
            config_path=self.config_path,
            environment=self.environment
        )
        
        self._config = self._config_manager.load_config()
        
        # Konfiguration validieren
        validator = ConfigurationValidator()
        validation_errors = validator.validate_comprehensive(self._config)
        
        if validation_errors:
            print("⚠️  Konfigurationsvalidierung-Warnings:")
            for error in validation_errors[:5]:  # Nur erste 5 Fehler anzeigen
                print(f"  - {error}")
            if len(validation_errors) > 5:
                print(f"  ... und {len(validation_errors) - 5} weitere")
    
    def _initialize_logging(self) -> None:
        """Initialisiert Logging-System"""
        setup_logging(self._config)
        self._logging_manager = get_logging_manager()
    
    def _initialize_exception_handling(self) -> None:
        """Initialisiert Exception-Handling"""
        # Exception-Registry ist automatisch verfügbar
        registry = get_exception_registry()
        
        # Test-Exception für Validierung
        try:
            raise RAGException(
                message="Core-System Test-Exception",
                error_code="CORE_TEST",
                severity=ErrorSeverity.LOW
            )
        except RAGException:
            pass  # Expected - nur für Registry-Test
    
    def _initialize_dependency_injection(self) -> None:
        """Initialisiert Dependency Injection Container"""
        self._service_container = get_service_container(self._config)
        
        # Core-Services im Container registrieren
        self._register_core_services()
    
    def _register_core_services(self) -> None:
        """Registriert Core-Services im DI-Container"""
        container = self._service_container
        
        # Konfiguration als Singleton
        container.register_instance(RAGConfig, self._config)
        container.register_instance(ConfigurationManager, self._config_manager)
        
        # Logging-Manager als Singleton
        container.register_instance(LoggingManager, self._logging_manager)
        
        # Exception-Registry als Singleton
        exception_registry = get_exception_registry()
        container.register_instance(ExceptionRegistry, exception_registry)
        
        # Container selbst registrieren (bereits gemacht, aber explizit für Klarheit)
        container.register_instance(ServiceContainer, container)
    
    def _log_startup_information(self) -> None:
        """Loggt System-Startup-Informationen"""
        logger = get_logger("core.system")
        
        # System-Informationen
        log_system_info(logger)
        
        # Konfigurationsinformationen
        log_config_info(logger, self._config)
        
        # Core-System Status
        logger.info(
            "RAG Chatbot Core-System erfolgreich initialisiert",
            extra={
                'extra_data': {
                    'event_type': 'core_system_startup',
                    'environment': self.environment,
                    'config_path': self.config_path,
                    'components_initialized': [
                        'configuration', 'logging', 'exceptions', 'dependency_injection'
                    ]
                }
            }
        )
    
    def shutdown(self) -> None:
        """Beendet Core-System ordnungsgemäß"""
        if not self._initialized:
            return
        
        logger = get_logger("core.system")
        logger.info("Beginne Core-System Shutdown...")
        
        try:
            # Service-Container beenden
            if self._service_container:
                self._service_container.dispose()
                reset_service_container()
            
            # Exception-Registry leeren
            registry = get_exception_registry()
            registry.clear_registry()
            
            # Logging-System beenden
            if self._logging_manager:
                shutdown_logging()
            
            # Konfiguration zurücksetzen
            reset_config()
            
            self._initialized = False
            print("✅ Core-System ordnungsgemäß beendet")
            
        except Exception as e:
            print(f"⚠️  Fehler beim Core-System Shutdown: {e}")
    
    def get_health_status(self) -> dict:
        """
        Holt Gesundheitsstatus des Core-Systems
        
        Returns:
            dict: Detaillierter Gesundheitsstatus
        """
        if not self._initialized:
            return {
                "status": "not_initialized",
                "initialization_error": str(self._initialization_error) if self._initialization_error else None
            }
        
        # Container-Statistiken
        container_stats = self._service_container.get_container_statistics() if self._service_container else {}
        
        # Exception-Statistiken
        exception_registry = get_exception_registry()
        exception_stats = exception_registry.get_exception_statistics()
        
        return {
            "status": "healthy",
            "environment": self.environment,
            "config_loaded": self._config is not None,
            "logging_active": self._logging_manager is not None,
            "container_active": self._service_container is not None,
            "container_stats": container_stats,
            "exception_stats": exception_stats,
            "config_summary": {
                "app_name": self._config.app.name,
                "version": self._config.app.version,
                "debug_mode": self._config.app.debug_mode,
                "llm_provider": self._config.llm.default_provider.value,
                "embedding_provider": self._config.embeddings.default_provider.value
            } if self._config else None
        }


# =============================================================================
# GLOBALE CORE-SYSTEM INSTANZ
# =============================================================================

_core_system: Optional[CoreSystem] = None


def initialize_core_system(config_path: Optional[str] = None,
                          environment: str = "development") -> bool:
    """
    Initialisiert globales Core-System
    
    Args:
        config_path (Optional[str]): Pfad zur Konfigurationsdatei
        environment (str): Umgebung (development, staging, production)
        
    Returns:
        bool: True wenn Initialisierung erfolgreich
    """
    global _core_system
    
    if _core_system is not None:
        print("ℹ️  Core-System bereits initialisiert")
        return True
    
    _core_system = CoreSystem(config_path, environment)
    return _core_system.initialize()


def shutdown_core_system() -> None:
    """Beendet globales Core-System"""
    global _core_system
    
    if _core_system:
        _core_system.shutdown()
        _core_system = None


def get_core_system() -> Optional[CoreSystem]:
    """
    Holt globale Core-System Instanz
    
    Returns:
        Optional[CoreSystem]: Core-System Instanz oder None
    """
    return _core_system


def is_core_system_initialized() -> bool:
    """
    Prüft ob Core-System initialisiert ist
    
    Returns:
        bool: True wenn initialisiert
    """
    return _core_system is not None and _core_system._initialized


# =============================================================================
# CONVENIENCE-FUNKTIONEN FÜR HÄUFIGE OPERATIONEN
# =============================================================================

def get_current_config() -> RAGConfig:
    """
    Convenience-Funktion für aktuelle Konfiguration
    
    Returns:
        RAGConfig: Aktuelle Konfiguration
    """
    return get_config()


def get_component_logger(component_name: str) -> 'logging.Logger':
    """
    Convenience-Funktion für Komponenten-Logger
    
    Args:
        component_name (str): Name der Komponente
        
    Returns:
        logging.Logger: Konfigurierter Logger
    """
    return get_logger(component_name, "core")


def create_service_context(component: str, 
                          operation: str,
                          **metadata) -> ErrorContext:
    """
    Convenience-Funktion für Service-Error-Context
    
    Args:
        component (str): Service-Komponente
        operation (str): Operation
        **metadata: Zusätzliche Metadaten
        
    Returns:
        ErrorContext: Erstellter Kontext
    """
    return create_error_context(
        component=component,
        operation=operation,
        **metadata
    )


# =============================================================================
# INTEGRATION-HILFSFUNKTIONEN
# =============================================================================

def validate_core_dependencies() -> bool:
    """
    Validiert dass alle erforderlichen Core-Abhängigkeiten verfügbar sind
    
    Returns:
        bool: True wenn alle Dependencies verfügbar
    """
    required_modules = [
        'yaml', 'logging', 'threading', 'pathlib', 'dataclasses', 'typing', 'enum'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Fehlende Python-Module: {', '.join(missing_modules)}")
        return False
    
    return True


def setup_development_environment() -> bool:
    """
    Richtet Development-Umgebung für Core-System ein
    
    Returns:
        bool: True wenn Setup erfolgreich
    """
    try:
        # Verzeichnisstruktur erstellen
        directories = [
            "./config",
            "./logs", 
            "./data",
            "./data/vectorstore",
            "./data/uploads",
            "./data/cache"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Beispiel-Konfigurationen generieren
        generate_example_configs()
        
        # Core-System mit Development-Config initialisieren
        return initialize_core_system(environment="development")
        
    except Exception as e:
        print(f"❌ Fehler beim Development-Setup: {e}")
        return False


# =============================================================================
# MODULE EXPORTS - ALLES AUS CORE-KOMPONENTEN
# =============================================================================

__all__ = [
    # Core-System
    'CoreSystem', 'initialize_core_system', 'shutdown_core_system',
    'get_core_system', 'is_core_system_initialized',
    
    # Konfiguration
    'RAGConfig', 'ConfigurationManager', 'ConfigurationFactory', 'ConfigurationValidator',
    'get_config_manager', 'get_config', 'reset_config', 'generate_example_configs',
    'LogLevel', 'LLMProvider', 'EmbeddingProvider', 'VectorStoreProvider',
    
    # Logging
    'LoggingManager', 'StructuredJSONFormatter', 'AsyncFileHandler', 'ColoredConsoleHandler',
    'LogContext', 'PerformanceMetrics',
    'get_logging_manager', 'get_logger', 'setup_logging', 'shutdown_logging',
    'log_performance', 'log_method_calls', 'log_context',
    'log_system_info', 'log_config_info',
    
    # Exceptions
    'RAGException', 'ErrorContext', 'ErrorSeverity', 'ErrorCategory',
    'ConfigurationError', 'InvalidConfigurationError', 'MissingConfigurationError',
    'ValidationError', 'InvalidInputError', 'DocumentValidationError',
    'ServiceError', 'DocumentProcessingError', 'EmbeddingError', 'VectorStoreError',
    'LLMError', 'ChatServiceError',
    'NetworkError', 'ExternalServiceError', 'APITimeoutError', 'APIRateLimitError',
    'ResourceError', 'MemoryError', 'StorageError',
    'handle_exception', 'create_error_context', 'wrap_external_exception',
    'ExceptionRegistry', 'get_exception_registry',
    
    # Dependency Injection
    'ServiceContainer', 'ServiceLifecycle', 'ServiceStatus', 'ServiceRegistration',
    'DependencyInjectionError', 'ServiceNotRegisteredError', 
    'CircularDependencyError', 'ServiceInitializationError',
    'get_service_container', 'reset_service_container',
    'inject', 'ServiceScope',
    
    # Convenience-Funktionen
    'get_current_config', 'get_component_logger', 'create_service_context',
    
    # Setup-Funktionen
    'validate_core_dependencies', 'setup_development_environment'
]


# =============================================================================
# AUTO-INITIALISIERUNG FÜR DEVELOPMENT
# =============================================================================

def _auto_initialize_for_development():
    """Automatische Initialisierung für Development-Umgebung"""
    if not is_core_system_initialized():
        # Prüfe ob wir in Development-Kontext sind
        if (os.environ.get("RAG_ENVIRONMENT", "development") == "development" or
            os.environ.get("PYTHON_ENV") == "development"):
            
            print("🔄 Auto-Initialisierung für Development-Umgebung...")
            setup_development_environment()


# Auto-Initialisierung nur wenn Modul direkt importiert wird
if __name__ != "__main__":
    # Abhängigkeiten validieren
    if validate_core_dependencies():
        _auto_initialize_for_development()
    else:
        print("⚠️  Core-Dependencies nicht vollständig - manuelle Initialisierung erforderlich")


if __name__ == "__main__":
    # Demo und Testing
    print("RAG Chatbot Core-System Demo")
    print("============================")
    
    # Dependencies prüfen
    if not validate_core_dependencies():
        print("❌ Dependencies nicht erfüllt")
        sys.exit(1)
    
    # Development-Environment setup
    if setup_development_environment():
        print("✅ Development-Environment erfolgreich eingerichtet")
    
    # Health-Check
    core_system = get_core_system()
    if core_system:
        health = core_system.get_health_status()
        print(f"🏥 System-Health: {health['status']}")
        
        if health.get('config_summary'):
            config_summary = health['config_summary']
            print(f"📋 Config: {config_summary['app_name']} v{config_summary['version']}")
            print(f"🔧 Provider: LLM={config_summary['llm_provider']}, Embeddings={config_summary['embedding_provider']}")
        
        if health.get('container_stats'):
            container_stats = health['container_stats']
            print(f"🏗️  Container: {container_stats['total_registrations']} Services registriert")
    
    # Demo der Integration
    try:
        # Logger holen
        logger = get_component_logger("demo")
        
        # Konfiguration holen
        config = get_current_config()
        
        # Service-Container holen
        container = get_service_container()
        
        # Beispiel-Service definieren und registrieren
        class DemoService:
            def __init__(self, config: RAGConfig):
                self.config = config
                self.logger = get_component_logger("demo_service")
            
            def get_info(self) -> str:
                self.logger.info("Demo-Service wurde aufgerufen")
                return f"Demo-Service läuft mit {self.config.app.name}"
        
        # Service registrieren
        container.register_singleton(DemoService)
        
        # Service auflösen
        demo_service = container.resolve(DemoService)
        result = demo_service.get_info()
        
        print(f"🎯 Demo-Service: {result}")
        
        # Exception-Test
        try:
            raise ServiceError("Demo-Fehler für Testing", "demo_service")
        except ServiceError as e:
            logger.info(f"Exception-Handling funktioniert: {e.error_code}")
        
        print("✅ Core-System Integration vollständig funktional!")
        
    except Exception as e:
        print(f"❌ Demo-Fehler: {e}")
    
    finally:
        # Cleanup
        shutdown_core_system()