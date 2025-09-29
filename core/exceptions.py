#!/usr/bin/env python3
"""
Core Exceptions - Vollständige Exception-Hierarchie
Industrielle RAG-Architektur - ALLE FEHLENDEN CLASSES ERGÄNZT

KRITISCHE BUGFIXES:
- EmbeddingException hinzugefügt
- DocumentRecord Exception ergänzt
- Alle Import-Fehler behoben
"""

import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field


# =============================================================================
# EXCEPTION SEVERITY UND KATEGORIEN
# =============================================================================

class ExceptionSeverity(str, Enum):
    """Schweregrad von Exceptions für Monitoring und Alerting"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExceptionCategory(str, Enum):
    """Kategorisierung von Exceptions für bessere Analyse"""
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    SERVICE = "service"
    PIPELINE = "pipeline"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_API = "external_api"
    STORAGE = "storage"
    NETWORK = "network"
    HEALTH = "health"
    EMBEDDING = "embedding"
    SYSTEM = "system" 


# =============================================================================
# NAMENSKONFLIKT BEHEBEN: ErrorSeverity vs ExceptionSeverity
# =============================================================================

# PROBLEM: Code importiert 'ErrorSeverity', aber es ist als 'ExceptionSeverity' definiert
# FEHLER: cannot import name 'ErrorSeverity' from 'core.exceptions'

# LÖSUNG 1: Aliase in core/exceptions.py hinzufügen

# =============================================================================
# ALIASE FÜR KOMPATIBILITÄT zu core/exceptions.py HINZUFÜGEN
# =============================================================================

# Nach den bestehenden Enum-Definitionen hinzufügen:

# Kompatibilitäts-Aliase für unterschiedliche Namenskonventionen
ErrorSeverity = ExceptionSeverity      # Alias für ExceptionSeverity
ErrorCategory = ExceptionCategory      # Alias für ExceptionCategory


# =============================================================================
# ERROR CONTEXT UND METADATA
# =============================================================================

@dataclass
class ErrorContext:
    """Strukturierter Fehler-Kontext für bessere Debugging-Informationen"""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    component: Optional[str] = None
    operation: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    
    def __post_init__(self):
        """Auto-capture Stack-Trace wenn nicht explizit gesetzt"""
        if self.stack_trace is None:
            self.stack_trace = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertierung zu Dictionary für Logging"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "metadata": self.metadata,
            "stack_trace": self.stack_trace
        }


# =============================================================================
# BASE EXCEPTION CLASS
# =============================================================================

class RAGException(Exception):
    """
    Basis-Exception für alle RAG-System Exceptions
    Erweitert Standard Exception um strukturierte Metadaten
    """
    
    def __init__(
        self,
        message: str,
        severity: ExceptionSeverity = ExceptionSeverity.MEDIUM,
        category: ExceptionCategory = ExceptionCategory.SERVICE,
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.error_code = error_code or self._generate_error_code()
        self.context = context or ErrorContext()
        self.cause = cause
        self.timestamp = datetime.now(timezone.utc)
    
    def _generate_error_code(self) -> str:
        """Generiert automatischen Error-Code basierend auf Exception-Typ"""
        class_name = self.__class__.__name__
        return f"{class_name.upper().replace('EXCEPTION', '')}_ERROR"
    
    def to_dict(self) -> Dict[str, Any]:
        """Strukturierte Darstellung für Logging und Monitoring"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "exception_type": self.__class__.__name__,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None
        }
    
    def __str__(self) -> str:
        """String-Representation mit Error-Code"""
        return f"[{self.error_code}] {self.message}"


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================

class ConfigurationException(RAGException):
    """Exception für Konfigurationsfehler"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ExceptionSeverity.HIGH,
            category=ExceptionCategory.CONFIGURATION,
            **kwargs
        )
        self.config_key = config_key
        if config_key and self.context:
            self.context.metadata["config_key"] = config_key


class ValidationError(RAGException):
    """Exception für Validierungsfehler"""
    
    def __init__(self, message: str, field_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ExceptionSeverity.MEDIUM,
            category=ExceptionCategory.VALIDATION,
            **kwargs
        )
        self.field_name = field_name
        if field_name and self.context:
            self.context.metadata["field_name"] = field_name

class ConfigurationError(ConfigurationException):
    """
    Alias für ConfigurationException - Service-Kompatibilität
    
    SCHRITT 1: Behebt ImportError für:
    - EmbeddingService 
    - RetrievalService
    - SearchService
    
    Diese Klasse ist ein Alias für die bereits existierende ConfigurationException,
    um Kompatibilität mit Service-Imports zu gewährleisten.
    """
    pass  # Erbt alle Funktionalität von ConfigurationException

# =============================================================================
# SERVICE EXCEPTIONS
# =============================================================================

class ServiceError(RAGException):
    """Exception für Service-Layer Fehler"""
    
    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ExceptionSeverity.HIGH,
            category=ExceptionCategory.SERVICE,
            **kwargs
        )
        self.service_name = service_name
        if service_name and self.context:
            self.context.metadata["service_name"] = service_name

class ServiceException(ServiceError):
    """
    Alias für ServiceError - Controller-Kompatibilität
    
    SCHRITT 5 BUGFIX: Behebt ImportError für:
    - PipelineController: cannot import name 'ServiceException' from 'core.exceptions'
    - HealthController: cannot import name 'ServiceException' from 'core.exceptions'
    
    Diese Klasse ist ein direkter Alias für ServiceError,
    um Kompatibilität mit Controller-Imports zu gewährleisten.
    Controller verwenden oft 'ServiceException' statt 'ServiceError'.
    """
    pass  # Erbt alle Funktionalität von ServiceError
    
# =============================================================================
# PIPELINE EXCEPTIONS
# =============================================================================

class PipelineException(RAGException):
    """Exception für Pipeline-Verarbeitungsfehler"""
    
    def __init__(self, message: str, stage: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ExceptionSeverity.HIGH,
            category=ExceptionCategory.PIPELINE,
            **kwargs
        )
        self.stage = stage
        if stage and self.context:
            self.context.metadata["pipeline_stage"] = stage


class DocumentProcessingError(PipelineException):
    """Spezialisierte Exception für Dokumenten-Verarbeitung"""
    
    def __init__(self, message: str, document_path: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            stage="document_processing",
            **kwargs
        )
        self.document_path = document_path
        if document_path and self.context:
            self.context.metadata["document_path"] = document_path


# =============================================================================
# HEALTH CHECK EXCEPTIONS
# =============================================================================

class HealthCheckException(RAGException):
    """Exception für Health-Check Fehler"""
    
    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ExceptionSeverity.MEDIUM,
            category=ExceptionCategory.HEALTH,
            **kwargs
        )
        self.component = component
        if component and self.context:
            self.context.metadata["health_component"] = component


# =============================================================================
# EMBEDDING EXCEPTIONS - FEHLENDE KLASSE HINZUGEFÜGT
# =============================================================================

class EmbeddingException(RAGException):
    """Exception für Embedding-spezifische Fehler - NEU HINZUGEFÜGT"""
    
    def __init__(self, message: str, provider: Optional[str] = None, model: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ExceptionSeverity.HIGH,
            category=ExceptionCategory.EMBEDDING,
            **kwargs
        )
        self.provider = provider
        self.model = model
        
        if self.context:
            if provider:
                self.context.metadata["embedding_provider"] = provider
            if model:
                self.context.metadata["embedding_model"] = model


class EmbeddingError(EmbeddingException):
    """Alias für EmbeddingException (Rückwärtskompatibilität)"""
    pass


# =============================================================================
# INTEGRATION UND EXTERNE API EXCEPTIONS
# =============================================================================

class LLMError(RAGException):
    """Exception für LLM-Provider Fehler"""
    
    def __init__(self, message: str, provider: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ExceptionSeverity.HIGH,
            category=ExceptionCategory.EXTERNAL_API,
            **kwargs
        )
        self.provider = provider
        if provider and self.context:
            self.context.metadata["llm_provider"] = provider


class VectorStoreError(RAGException):
    """Exception für Vector Store Fehler"""
    
    def __init__(self, message: str, store_type: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ExceptionSeverity.HIGH,
            category=ExceptionCategory.STORAGE,
            **kwargs
        )
        self.store_type = store_type
        if store_type and self.context:
            self.context.metadata["vector_store_type"] = store_type


# =============================================================================
# DOCUMENT UND RECORD EXCEPTIONS - NEU HINZUGEFÜGT
# =============================================================================

class DocumentException(RAGException):
    """Exception für Dokument-spezifische Fehler"""
    
    def __init__(self, message: str, document_id: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ExceptionSeverity.MEDIUM,
            category=ExceptionCategory.STORAGE,
            **kwargs
        )
        self.document_id = document_id
        if document_id and self.context:
            self.context.metadata["document_id"] = document_id


class RecordException(RAGException):
    """Exception für Record-spezifische Fehler - NEU HINZUGEFÜGT"""
    
    def __init__(self, message: str, record_id: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ExceptionSeverity.MEDIUM,
            category=ExceptionCategory.STORAGE,
            **kwargs
        )
        self.record_id = record_id
        if record_id and self.context:
            self.context.metadata["record_id"] = record_id


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_error_context(
    component: Optional[str] = None,
    operation: Optional[str] = None,
    **metadata
) -> ErrorContext:
    """
    Helper-Funktion zur Erstellung von Error-Context
    
    Args:
        component: Name der betroffenen Komponente
        operation: Name der fehlgeschlagenen Operation
        **metadata: Zusätzliche Metadaten
    
    Returns:
        ErrorContext: Konfigurierter Error-Context
    """
    context = ErrorContext(
        component=component,
        operation=operation,
        metadata=metadata
    )
    return context


def handle_exception(
    exc: Exception,
    component: str,
    operation: str,
    severity: ExceptionSeverity = ExceptionSeverity.MEDIUM,
    **metadata
) -> RAGException:
    """
    Konvertiert Standard-Exceptions zu strukturierten RAG-Exceptions
    
    Args:
        exc: Original Exception
        component: Betroffene Komponente
        operation: Fehlgeschlagene Operation
        severity: Schweregrad
        **metadata: Zusätzliche Metadaten
    
    Returns:
        RAGException: Strukturierte Exception
    """
    context = create_error_context(
        component=component,
        operation=operation,
        **metadata
    )
    
    # Prüfe ob bereits RAGException
    if isinstance(exc, RAGException):
        return exc
    
    # Konvertiere zu RAGException
    return RAGException(
        message=str(exc),
        severity=severity,
        context=context,
        cause=exc
    )


# =============================================================================
# EXCEPTION REGISTRY UND MONITORING
# =============================================================================

class ExceptionRegistry:
    """Zentrale Registry für Exception-Tracking und -Analyse"""
    
    def __init__(self):
        self.exceptions: List[RAGException] = []
        self._stats: Dict[str, int] = {}
    
    def register(self, exception: RAGException) -> None:
        """Registriert Exception für Monitoring"""
        self.exceptions.append(exception)
        
        # Update Statistics
        exc_type = exception.__class__.__name__
        self._stats[exc_type] = self._stats.get(exc_type, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Liefert Exception-Statistiken"""
        return {
            "total_exceptions": len(self.exceptions),
            "by_type": self._stats.copy(),
            "recent_exceptions": [
                exc.to_dict() 
                for exc in self.exceptions[-10:]
            ]
        }
    
    def clear(self) -> None:
        """Löscht Registry (für Tests)"""
        self.exceptions.clear()
        self._stats.clear()


# Globale Registry-Instanz
exception_registry = ExceptionRegistry()

class InterfaceException(RAGException):
    """Exception für Interface-spezifische Fehler - BUG-001 BEHOBEN"""
    
    def __init__(
        self, 
        message: str, 
        interface_name: Optional[str] = None, 
        component_name: Optional[str] = None,
        user_action: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message=message,
            severity=ExceptionSeverity.MEDIUM,
            category=ExceptionCategory.INTEGRATION,
            **kwargs
        )
        self.interface_name = interface_name
        self.component_name = component_name
        self.user_action = user_action
        
        if self.context:
            if interface_name:
                self.context.metadata["interface_name"] = interface_name
            if component_name:
                self.context.metadata["component_name"] = component_name
            if user_action:
                self.context.metadata["user_action"] = user_action

class RAGSystemException(RAGException):
    """Exception für systemweite Fehler - BUG-002 BEHOBEN"""
    
    def __init__(
        self, 
        message: str, 
        system_component: Optional[str] = None,
        failure_stage: Optional[str] = None,
        recovery_possible: bool = False,
        **kwargs
    ):
        super().__init__(
            message=message,
            severity=ExceptionSeverity.HIGH,
            category=ExceptionCategory.SERVICE,
            **kwargs
        )
        self.system_component = system_component
        self.failure_stage = failure_stage
        self.recovery_possible = recovery_possible
        
        if self.context:
            if system_component:
                self.context.metadata["system_component"] = system_component
            if failure_stage:
                self.context.metadata["failure_stage"] = failure_stage
            self.context.metadata["recovery_possible"] = recovery_possible

# =============================================================================
# EXCEPTION Service
# =============================================================================

class ChatServiceError(ServiceError):

    """Exception für ChatService-spezifische Fehler"""
    def __init__(self, message: str, chat_context: Optional[str] = None, **kwargs):
        super().__init__(message=message, service_name="ChatService", **kwargs)
        self.chat_context = chat_context
        if chat_context and self.context:
            self.context.metadata["chat_context"] = chat_context

class ResourceError(ServiceError):
    """Exception für Ressourcen-bezogene Fehler (Memory, Storage, etc.)"""
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(message=message, **kwargs)
        self.resource_type = resource_type
        if resource_type and self.context:
            self.context.metadata["resource_type"] = resource_type

class SessionException(ServiceError):
    """Exception für Session-Management-Fehler"""
    def __init__(self, message: str, session_id: Optional[str] = None, **kwargs):
        super().__init__(message=message, service_name="SessionService", **kwargs)
        self.session_id = session_id
        if session_id and self.context:
            self.context.metadata["session_id"] = session_id

# =============================================================================
# EXPORT - VOLLSTÄNDIGE LISTE
# =============================================================================

__all__ = [
    # Enums
    "ExceptionSeverity",
    "ExceptionCategory",
    
    # Enums - Aliase für Kompatibilität
    "ErrorSeverity",             # ALIAS für ExceptionSeverity
    "ErrorCategory",             # ALIAS für ExceptionCategory
    
    # Data Classes
    "ErrorContext",
    
    # Base Exception
    "RAGException",
    
    # Configuration Exceptions
    "ConfigurationException",
    "ConfigurationError",        # SCHRITT 1: NEU HINZUGEFÜGT
    "ValidationError",
    
    # Service Exceptions
    "ServiceError",
    "ServiceException",          # SCHRITT 5: NEU HINZUFÜGEN
   
    
    # Pipeline Exceptions
    "PipelineException",
    "DocumentProcessingError",
    
    # Health Exceptions
    "HealthCheckException",
    
    # Provider Exceptions
    "LLMError",
    "EmbeddingError",  # Alias
    "EmbeddingException",  # NEU HINZUGEFÜGT
    "VectorStoreError",
    
    # Document/Record Exceptions - NEU HINZUGEFÜGT
    "DocumentException",
    "RecordException",
    
    # Utilities
    "create_error_context",
    "handle_exception",
    "exception_registry",
    
    #Interface Exceptions - BUG-001 FIX
    "InterfaceException",
    
    # System Exceptions - BUG-002 FIX  
    "RAGSystemException",
    
    # Service Erweiterungen
    "ChatServiceError",          # NEU
    "ResourceError",             # NEU  
    "SessionException"           # NEU
]