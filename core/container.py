#!/usr/bin/env python3
"""
Core Dependency Injection Container für RAG Chatbot Industrial

Modernes Dependency Injection System mit automatischer Service-Resolution,
Lifecycle-Management und Type-Hint-basierter Konfiguration nach extended_coding_guidelines.

Features:
- Automatische Constructor-Injection basierend auf Type-Hints
- Singleton-, Transient- und Scoped-Lifecycles
- Circular-Dependency-Detection und -Auflösung
- Factory-Pattern für komplexe Service-Erstellung
- Interface-basierte Service-Registration
- Hot-Swapping für Testing und Development
- Performance-optimierte Service-Resolution

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Core-Komponente
"""

import inspect
import threading
from abc import ABC, abstractmethod
from typing import (
    Dict, Any, Optional, Type, TypeVar, Generic, Callable, 
    Union, List, Set, get_type_hints, get_origin, get_args
)
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import weakref

# Import der anderen Core-Komponenten
from .config import get_config, RAGConfig
from .logger import get_logger
from .exceptions import (
    RAGException, ErrorSeverity, ErrorCategory, create_error_context
)

# =============================================================================
# SERVICE LIFECYCLE ENUMS
# =============================================================================

class ServiceLifecycle(str, Enum):
    """
    Lifecycle-Modi für Service-Instanzen
    
    Attributes:
        SINGLETON: Eine globale Instanz für die gesamte Anwendung
        TRANSIENT: Neue Instanz bei jeder Anfrage
        SCOPED: Eine Instanz pro Request/Thread-Scope
        FACTORY: Verwendung einer Factory-Funktion
    """
    SINGLETON = "singleton"      # Eine globale Instanz
    TRANSIENT = "transient"      # Neue Instanz bei jeder Anfrage
    SCOPED = "scoped"           # Eine Instanz pro Scope (Request/Thread)
    FACTORY = "factory"          # Factory-Function für Erstellung


class ServiceStatus(str, Enum):
    """
    Status von registrierten Services
    
    Attributes:
        REGISTERED: Service ist registriert aber noch nicht initialisiert
        INITIALIZING: Service wird gerade initialisiert
        ACTIVE: Service ist initialisiert und verfügbar
        ERROR: Fehler bei Service-Initialisierung
        DISPOSED: Service wurde ordnungsgemäß beendet
    """
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    DISPOSED = "disposed"


# =============================================================================
# DEPENDENCY INJECTION EXCEPTIONS
# =============================================================================

class DependencyInjectionError(RAGException):
    """Basis-Exception für Dependency Injection Fehler"""
    
    def __init__(self, message: str, service_type: str = None, **kwargs):
        super().__init__(
            message=message,
            error_code="DI_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            retry_possible=False,
            **kwargs
        )
        
        if service_type and self.context:
            self.context.additional_metadata['service_type'] = service_type


class ServiceNotRegisteredError(DependencyInjectionError):
    """Service ist nicht im Container registriert"""
    
    def __init__(self, service_type: str, **kwargs):
        super().__init__(
            message=f"Service '{service_type}' ist nicht registriert",
            error_code="SERVICE_NOT_REGISTERED",
            service_type=service_type,
            **kwargs
        )


class CircularDependencyError(DependencyInjectionError):
    """Zirkuläre Abhängigkeit erkannt"""
    
    def __init__(self, dependency_chain: List[str], **kwargs):
        chain_str = " -> ".join(dependency_chain)
        super().__init__(
            message=f"Zirkuläre Abhängigkeit erkannt: {chain_str}",
            error_code="CIRCULAR_DEPENDENCY",
            **kwargs
        )
        
        if self.context:
            self.context.additional_metadata['dependency_chain'] = dependency_chain


class ServiceInitializationError(DependencyInjectionError):
    """Fehler bei Service-Initialisierung"""
    
    def __init__(self, service_type: str, original_error: Exception, **kwargs):
        super().__init__(
            message=f"Fehler bei Initialisierung von '{service_type}': {str(original_error)}",
            error_code="SERVICE_INIT_ERROR",
            service_type=service_type,
            original_exception=original_error,
            **kwargs
        )


# =============================================================================
# SERVICE REGISTRATION UND METADATA
# =============================================================================

T = TypeVar('T')


@dataclass
class ServiceRegistration:
    """
    Metadaten für registrierte Services
    
    Attributes:
        service_type (Type): Interface oder Abstract Base Class
        implementation_type (Optional[Type]): Konkrete Implementierung
        factory (Optional[Callable]): Factory-Funktion für Service-Erstellung
        lifecycle (ServiceLifecycle): Lifecycle-Management
        instance (Optional[Any]): Gespeicherte Singleton-Instanz
        status (ServiceStatus): Aktueller Service-Status
        dependencies (Set[Type]): Abhängigkeiten dieses Services
        metadata (Dict[str, Any]): Zusätzliche Service-Metadaten
    """
    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable] = None
    lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON
    instance: Optional[Any] = None
    status: ServiceStatus = ServiceStatus.REGISTERED
    dependencies: Set[Type] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validiert Service-Registration nach Initialisierung"""
        if not self.implementation_type and not self.factory:
            raise ValueError("Entweder implementation_type oder factory muss gesetzt sein")
        
        if self.implementation_type and self.factory:
            raise ValueError("implementation_type und factory können nicht beide gesetzt sein")


@dataclass
class ServiceScope:
    """
    Scope für Scoped-Services (Request/Thread-spezifisch)
    
    Attributes:
        scope_id (str): Eindeutige Scope-ID
        instances (Dict[Type, Any]): Service-Instanzen in diesem Scope
        metadata (Dict[str, Any]): Scope-Metadaten
    """
    scope_id: str
    instances: Dict[Type, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# HAUPTCONTAINER-IMPLEMENTIERUNG
# =============================================================================

class ServiceContainer:
    """
    Moderner Dependency Injection Container
    
    Implementiert automatische Service-Resolution, Lifecycle-Management
    und Type-Hint-basierte Dependency-Injection.
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialisiert Service-Container
        
        Args:
            config (RAGConfig): Container-Konfiguration
        """
        self.config = config or get_config()
        self.logger = get_logger("container", "core")
        
        # Service-Registry
        self._registrations: Dict[Type, ServiceRegistration] = {}
        self._building_stack: Set[Type] = set()  # Für Circular-Dependency-Detection
        
        # Thread-Safety
        self._lock = threading.RLock()
        
        # Scoped Services
        self._scopes: Dict[str, ServiceScope] = {}
        self._thread_local = threading.local()
        
        # Performance-Metriken
        self._resolution_stats: Dict[str, int] = {}
        
        # Container selbst registrieren
        self.register_instance(ServiceContainer, self)
        
        # Core-Services registrieren
        self._register_core_services()
    
    def _register_core_services(self) -> None:
        """Registriert Core-Services des RAG-Systems"""
        # Konfiguration als Singleton registrieren
        self.register_instance(RAGConfig, self.config)
        
        # Logger-Factory registrieren
        self.register_factory(
            service_type=object,  # Platzhalter für Logger-Interface
            factory=lambda: get_logger("injected"),
            lifecycle=ServiceLifecycle.TRANSIENT
        )
    
    def register_singleton(self, 
                          service_type: Type[T], 
                          implementation_type: Type[T] = None) -> 'ServiceContainer':
        """
        Registriert Service als Singleton
        
        Args:
            service_type (Type[T]): Service-Interface oder Abstract Type
            implementation_type (Type[T]): Konkrete Implementierung
            
        Returns:
            ServiceContainer: Selbst für Method-Chaining
        """
        return self.register(
            service_type=service_type,
            implementation_type=implementation_type,
            lifecycle=ServiceLifecycle.SINGLETON
        )
    
    def register_transient(self, 
                          service_type: Type[T], 
                          implementation_type: Type[T] = None) -> 'ServiceContainer':
        """
        Registriert Service als Transient (neue Instanz bei jeder Anfrage)
        
        Args:
            service_type (Type[T]): Service-Interface oder Abstract Type
            implementation_type (Type[T]): Konkrete Implementierung
            
        Returns:
            ServiceContainer: Selbst für Method-Chaining
        """
        return self.register(
            service_type=service_type,
            implementation_type=implementation_type,
            lifecycle=ServiceLifecycle.TRANSIENT
        )
    
    def register_scoped(self, 
                       service_type: Type[T], 
                       implementation_type: Type[T] = None) -> 'ServiceContainer':
        """
        Registriert Service als Scoped (eine Instanz pro Scope)
        
        Args:
            service_type (Type[T]): Service-Interface oder Abstract Type
            implementation_type (Type[T]): Konkrete Implementierung
            
        Returns:
            ServiceContainer: Selbst für Method-Chaining
        """
        return self.register(
            service_type=service_type,
            implementation_type=implementation_type,
            lifecycle=ServiceLifecycle.SCOPED
        )
    
    def register(self, 
                service_type: Type[T],
                implementation_type: Type[T] = None,
                lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON) -> 'ServiceContainer':
        """
        Registriert Service im Container
        
        Args:
            service_type (Type[T]): Service-Interface oder Abstract Type
            implementation_type (Type[T]): Konkrete Implementierung
            lifecycle (ServiceLifecycle): Lifecycle-Management
            
        Returns:
            ServiceContainer: Selbst für Method-Chaining
        """
        with self._lock:
            # Implementierung ermitteln falls nicht angegeben
            if implementation_type is None:
                implementation_type = service_type
            
            # Abhängigkeiten analysieren
            dependencies = self._analyze_dependencies(implementation_type)
            
            # Service registrieren - KORRIGIERT!
            registration = ServiceRegistration(
                service_type=service_type,
                implementation_type=implementation_type,  # KORREKT: implementation_type
                lifecycle=lifecycle,  # KORREKT: lifecycle Parameter
                dependencies=dependencies  # KORREKT: dependencies
            )
            
            self._registrations[service_type] = registration
            
            # SAFETY FIX: String-sichere Logging
            service_name = getattr(service_type, '__name__', str(service_type))
            impl_name = getattr(implementation_type, '__name__', str(implementation_type))
            
            self.logger.debug(
                f"Service registriert: {service_name} -> {impl_name}",
                extra={
                    'extra_data': {
                        'service_type': service_name,
                        'implementation_type': impl_name,
                        'lifecycle': lifecycle.value,
                        'dependency_count': len(dependencies)
                    }
                }
            )
            
            return self
            
    def register_instance(self, 
                         service_type: Type[T], 
                         instance: T) -> 'ServiceContainer':
        """
        Registriert bereits erstellte Service-Instanz
        
        Args:
            service_type (Type[T]): Service-Type
            instance (T): Service-Instanz
            
        Returns:
            ServiceContainer: Selbst für Method-Chaining
        """
        with self._lock:
            registration = ServiceRegistration(
                service_type=service_type,
                implementation_type=type(instance),  # FIX: implementation_type hinzufügen
                lifecycle=ServiceLifecycle.SINGLETON,
                instance=instance,
                status=ServiceStatus.ACTIVE
            )
            
            self._registrations[service_type] = registration
            
            self.logger.debug(f"Service-Instanz registriert: {service_type.__name__}")
            
            return self
    
    def register_factory(self, 
                        service_type: Type[T],
                        factory: Callable[[], T],
                        lifecycle: ServiceLifecycle = ServiceLifecycle.TRANSIENT) -> 'ServiceContainer':
        """
        Registriert Factory-Funktion für Service-Erstellung
        
        Args:
            service_type (Type[T]): Service-Type
            factory (Callable[[], T]): Factory-Funktion
            lifecycle (ServiceLifecycle): Lifecycle-Management
            
        Returns:
            ServiceContainer: Selbst für Method-Chaining
        """
        with self._lock:
            registration = ServiceRegistration(
                service_type=service_type,
                factory=factory,
                lifecycle=lifecycle
            )
            
            self._registrations[service_type] = registration
            
            self.logger.debug(f"Service-Factory registriert: {service_type.__name__}")
            
            return self
    
    def resolve(self, service_type: Type[T]) -> T:
        """
        Löst Service-Abhängigkeit auf und gibt Instanz zurück
        
        Args:
            service_type (Type[T]): Gewünschter Service-Type
            
        Returns:
            T: Service-Instanz
            
        Raises:
            ServiceNotRegisteredError: Service nicht registriert
            CircularDependencyError: Zirkuläre Abhängigkeit
            ServiceInitializationError: Initialisierungsfehler
        """
        with self._lock:
            # Performance-Tracking
            service_name = service_type.__name__
            self._resolution_stats[service_name] = self._resolution_stats.get(service_name, 0) + 1
            
            # Circular-Dependency-Check
            if service_type in self._building_stack:
                chain = list(self._building_stack) + [service_type]
                raise CircularDependencyError([t.__name__ for t in chain])
            
            # Service-Registration finden
            if service_type not in self._registrations:
                raise ServiceNotRegisteredError(service_type.__name__)
            
            registration = self._registrations[service_type]
            
            try:
                # Lifecycle-spezifische Resolution
                if registration.lifecycle == ServiceLifecycle.SINGLETON:
                    return self._resolve_singleton(registration)
                elif registration.lifecycle == ServiceLifecycle.TRANSIENT:
                    return self._resolve_transient(registration)
                elif registration.lifecycle == ServiceLifecycle.SCOPED:
                    return self._resolve_scoped(registration)
                elif registration.lifecycle == ServiceLifecycle.FACTORY:
                    return self._resolve_factory(registration)
                else:
                    raise ValueError(f"Unbekannter Lifecycle: {registration.lifecycle}")
                    
            except Exception as e:
                registration.status = ServiceStatus.ERROR
                
                if isinstance(e, DependencyInjectionError):
                    raise
                
                raise ServiceInitializationError(
                    service_type=service_type.__name__,
                    original_error=e
                )
    
    def _resolve_singleton(self, registration: ServiceRegistration) -> Any:
        """Löst Singleton-Service auf"""
        if registration.instance is not None:
            return registration.instance
        
        # Neue Instanz erstellen
        registration.status = ServiceStatus.INITIALIZING
        
        if registration.factory:
            instance = self._call_factory(registration.factory)
        else:
            instance = self._create_instance(registration.implementation_type)
        
        registration.instance = instance
        registration.status = ServiceStatus.ACTIVE
        
        return instance
    
    def _resolve_transient(self, registration: ServiceRegistration) -> Any:
        """Löst Transient-Service auf (neue Instanz)"""
        if registration.factory:
            return self._call_factory(registration.factory)
        else:
            return self._create_instance(registration.implementation_type)
    
    def _resolve_scoped(self, registration: ServiceRegistration) -> Any:
        """Löst Scoped-Service auf (eine Instanz pro Scope)"""
        current_scope = self._get_current_scope()
        
        if registration.service_type in current_scope.instances:
            return current_scope.instances[registration.service_type]
        
        # Neue Instanz für aktuellen Scope erstellen
        if registration.factory:
            instance = self._call_factory(registration.factory)
        else:
            instance = self._create_instance(registration.implementation_type)
        
        current_scope.instances[registration.service_type] = instance
        return instance
    
    def _resolve_factory(self, registration: ServiceRegistration) -> Any:
        """Löst Factory-Service auf"""
        return self._call_factory(registration.factory)
    
    def _create_instance(self, implementation_type: Type) -> Any:
        """
        Erstellt Service-Instanz mit automatischer Dependency-Injection
        
        Args:
            implementation_type (Type): Zu instanziierende Klasse
            
        Returns:
            Any: Erstellte Instanz
        """
        # Circular-Dependency-Tracking
        self._building_stack.add(implementation_type)
        
        try:
            # Constructor-Parameter analysieren
            constructor = implementation_type.__init__
            type_hints = get_type_hints(constructor)
            signature = inspect.signature(constructor)
            
            # Dependencies auflösen
            kwargs = {}
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                
                param_type = type_hints.get(param_name)
                if param_type:
                    # Abhängigkeit auflösen
                    dependency = self.resolve(param_type)
                    kwargs[param_name] = dependency
                elif param.default is not inspect.Parameter.empty:
                    # Default-Wert verwenden
                    kwargs[param_name] = param.default
                else:
                    # Optional Parameter ohne Type-Hint
                    self.logger.warning(
                        f"Parameter '{param_name}' in {implementation_type.__name__} "
                        f"hat keinen Type-Hint und keinen Default-Wert"
                    )
            
            # Instanz erstellen
            instance = implementation_type(**kwargs)
            
            self.logger.debug(
                f"Service-Instanz erstellt: {implementation_type.__name__}",
                extra={
                    'extra_data': {
                        'type': implementation_type.__name__,
                        'resolved_dependencies': len(kwargs)
                    }
                }
            )
            
            return instance
            
        finally:
            self._building_stack.discard(implementation_type)
    
    def _call_factory(self, factory: Callable) -> Any:
        """
        Ruft Factory-Funktion mit Dependency-Injection auf
        
        Args:
            factory (Callable): Factory-Funktion
            
        Returns:
            Any: Von Factory erstellte Instanz
        """
        # Factory-Parameter analysieren
        signature = inspect.signature(factory)
        type_hints = get_type_hints(factory)
        
        kwargs = {}
        for param_name, param in signature.parameters.items():
            param_type = type_hints.get(param_name)
            if param_type:
                dependency = self.resolve(param_type)
                kwargs[param_name] = dependency
            elif param.default is not inspect.Parameter.empty:
                kwargs[param_name] = param.default
        
        return factory(**kwargs)
    
    def _analyze_dependencies(self, implementation_type: Type) -> Set[Type]:
        """
        Analysiert Abhängigkeiten eines Service-Types
        
        Args:
            implementation_type (Type): Zu analysierende Klasse
            
        Returns:
            Set[Type]: Set der Abhängigkeitstypen
        """
        dependencies = set()
        
        try:
            constructor = implementation_type.__init__
            type_hints = get_type_hints(constructor)
            signature = inspect.signature(constructor)
            
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                
                param_type = type_hints.get(param_name)
                if param_type:
                    dependencies.add(param_type)
        
        except Exception:
            # Bei Fehlern in der Analyse leere Dependencies zurückgeben
            pass
        
        return dependencies
    
    def _get_current_scope(self) -> ServiceScope:
        """
        Holt aktuellen Service-Scope für Thread
        
        Returns:
            ServiceScope: Aktueller Scope
        """
        scope_id = getattr(self._thread_local, 'scope_id', None)
        
        if scope_id is None:
            # Neuen Scope für Thread erstellen
            scope_id = f"thread_{threading.current_thread().ident}"
            self._thread_local.scope_id = scope_id
            
            if scope_id not in self._scopes:
                self._scopes[scope_id] = ServiceScope(scope_id=scope_id)
        
        return self._scopes[scope_id]
    
    def create_scope(self, scope_id: str = None) -> 'ServiceScope':
        """
        Erstellt neuen Service-Scope
        
        Args:
            scope_id (str): Scope-ID (auto-generiert falls None)
            
        Returns:
            ServiceScope: Neuer Scope
        """
        if scope_id is None:
            scope_id = f"scope_{len(self._scopes)}"
        
        scope = ServiceScope(scope_id=scope_id)
        self._scopes[scope_id] = scope
        
        return scope
    
    def dispose_scope(self, scope_id: str) -> None:
        """
        Beendet und entfernt Service-Scope
        
        Args:
            scope_id (str): Zu beendender Scope
        """
        if scope_id in self._scopes:
            scope = self._scopes[scope_id]
            
            # Disposable Services ordnungsgemäß beenden
            for instance in scope.instances.values():
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception as e:
                        self.logger.warning(f"Fehler beim Beenden von Service-Instanz: {e}")
            
            del self._scopes[scope_id]
    
    def is_registered(self, service_type: Type) -> bool:
        """
        Prüft ob Service registriert ist
        
        Args:
            service_type (Type): Zu prüfender Service-Type
            
        Returns:
            bool: True wenn registriert
        """
        return service_type in self._registrations
    
    def get_registration_info(self, service_type: Type) -> Optional[Dict[str, Any]]:
        """
        Holt Registrierungs-Informationen für Service
        
        Args:
            service_type (Type): Service-Type
            
        Returns:
            Optional[Dict[str, Any]]: Registrierungs-Infos oder None
        """
        if service_type not in self._registrations:
            return None
        
        registration = self._registrations[service_type]
        
        return {
            'service_type': service_type.__name__,
            'implementation_type': registration.implementation_type.__name__ if registration.implementation_type else None,
            'lifecycle': registration.lifecycle.value,
            'status': registration.status.value,
            'has_factory': registration.factory is not None,
            'has_instance': registration.instance is not None,
            'dependency_count': len(registration.dependencies)
        }
    
    def get_container_statistics(self) -> Dict[str, Any]:
        """
        Holt Container-Statistiken für Monitoring
        
        Returns:
            Dict[str, Any]: Container-Statistiken
        """
        return {
            'total_registrations': len(self._registrations),
            'active_scopes': len(self._scopes),
            'resolution_stats': self._resolution_stats,
            'lifecycle_distribution': {
                lifecycle.value: sum(
                    1 for reg in self._registrations.values()
                    if reg.lifecycle == lifecycle
                )
                for lifecycle in ServiceLifecycle
            },
            'status_distribution': {
                status.value: sum(
                    1 for reg in self._registrations.values()
                    if reg.status == status
                )
                for status in ServiceStatus
            }
        }
    
    def get_service(self, service_type: Type[T]) -> Optional[T]:
        """
        Alias für resolve() - Legacy-Kompatibilität
        
        Args:
            service_type (Type[T]): Gewünschter Service-Type
            
        Returns:
            Optional[T]: Service-Instanz oder None bei Fehlern
        """
        try:
            return self.resolve(service_type)
        except (ServiceNotRegisteredError, ServiceInitializationError, CircularDependencyError) as e:
            self.logger.warning(f"Service {service_type.__name__} konnte nicht aufgelöst werden: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unerwarteter Fehler bei Service-Resolution für {service_type.__name__}: {e}")
            return None
    
    def dispose(self) -> None:
        """Beendet Container und alle verwalteten Services"""
        with self._lock:
            # Alle Scopes beenden
            for scope_id in list(self._scopes.keys()):
                self.dispose_scope(scope_id)
            
            # Singleton-Instanzen beenden
            for registration in self._registrations.values():
                if registration.instance and hasattr(registration.instance, 'dispose'):
                    try:
                        registration.instance.dispose()
                        registration.status = ServiceStatus.DISPOSED
                    except Exception as e:
                        self.logger.warning(f"Fehler beim Beenden von Service: {e}")
            
            # Container-State zurücksetzen
            self._registrations.clear()
            self._building_stack.clear()
            self._scopes.clear()
            self._resolution_stats.clear()
            
            self.logger.info("Service-Container ordnungsgemäß beendet")


# =============================================================================
# DECORATOR FÜR AUTOMATISCHE DEPENDENCY INJECTION
# =============================================================================

def inject(*dependency_types: Type):
    """
    Decorator für automatische Dependency-Injection in Funktionen
    
    Args:
        *dependency_types: Types der zu injizierenden Dependencies
        
    Returns:
        Decorator: Function mit Dependency-Injection
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, container: ServiceContainer = None, **kwargs):
            if container is None:
                container = get_service_container()
            
            # Dependencies auflösen
            for dep_type in dependency_types:
                if dep_type.__name__.lower() not in kwargs:
                    dependency = container.resolve(dep_type)
                    kwargs[dep_type.__name__.lower()] = dependency
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# =============================================================================
# GLOBALER CONTAINER
# =============================================================================

# Singleton-Pattern für globalen Container
_global_container: Optional[ServiceContainer] = None
_container_lock = threading.RLock()


def get_service_container(config: RAGConfig = None) -> ServiceContainer:
    """
    Holt globale ServiceContainer-Instanz (Singleton)
    
    Args:
        config (RAGConfig): Container-Konfiguration
        
    Returns:
        ServiceContainer: Globale Container-Instanz
    """
    global _global_container
    
    with _container_lock:
        if _global_container is None:
            _global_container = ServiceContainer(config)
        
        return _global_container


def reset_service_container() -> None:
    """Setzt globalen Service-Container zurück (für Tests)"""
    global _global_container
    
    with _container_lock:
        if _global_container:
            _global_container.dispose()
            _global_container = None


# =============================================================================
# CONTEXT-MANAGER FÜR SERVICE-SCOPES
# =============================================================================

class ServiceScope:
    """Context-Manager für Service-Scopes"""
    
    def __init__(self, container: ServiceContainer = None, scope_id: str = None):
        """
        Initialisiert Service-Scope Context-Manager
        
        Args:
            container (ServiceContainer): Container-Instanz
            scope_id (str): Scope-ID (auto-generiert falls None)
        """
        self.container = container or get_service_container()
        self.scope_id = scope_id
        self._original_scope_id = None
    
    def __enter__(self) -> 'ServiceScope':
        """Aktiviert Service-Scope"""
        # Aktuellen Scope speichern
        self._original_scope_id = getattr(
            self.container._thread_local, 'scope_id', None
        )
        
        # Neuen Scope erstellen falls keine ID angegeben
        if self.scope_id is None:
            scope = self.container.create_scope()
            self.scope_id = scope.scope_id
        
        # Scope für aktuellen Thread setzen
        self.container._thread_local.scope_id = self.scope_id
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Deaktiviert Service-Scope"""
        # Ursprünglichen Scope wiederherstellen
        if self._original_scope_id:
            self.container._thread_local.scope_id = self._original_scope_id
        else:
            if hasattr(self.container._thread_local, 'scope_id'):
                delattr(self.container._thread_local, 'scope_id')
        
        # Scope beenden
        self.container.dispose_scope(self.scope_id)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'ServiceLifecycle', 'ServiceStatus',
    
    # Datenstrukturen
    'ServiceRegistration', 'ServiceScope',
    
    # Exceptions
    'DependencyInjectionError', 'ServiceNotRegisteredError',
    'CircularDependencyError', 'ServiceInitializationError',
    
    # Container-Klassen
    'ServiceContainer',
    
    # Decorators
    'inject',
    
    # Globale Funktionen
    'get_service_container', 'reset_service_container',
    
    # Context-Manager
    'ServiceScope'
]


if __name__ == "__main__":
    # Beispiel-Nutzung und Testing
    print("RAG Chatbot Dependency Injection Container")
    print("==========================================")
    
    # Test-Services definieren
    class ITestService(ABC):
        @abstractmethod
        def get_message(self) -> str:
            pass
    
    class TestService(ITestService):
        def __init__(self, config: RAGConfig):
            self.config = config
        
        def get_message(self) -> str:
            return f"Service funktioniert mit Config: {self.config.app.name}"
    
    class ConsumerService:
        def __init__(self, test_service: ITestService):
            self.test_service = test_service
        
        def do_work(self) -> str:
            return f"Consumer: {self.test_service.get_message()}"
    
    # Container testen
    container = get_service_container()
    
    # Services registrieren
    container.register_singleton(ITestService, TestService)
    container.register_transient(ConsumerService)
    
    # Services auflösen
    test_service = container.resolve(ITestService)
    consumer = container.resolve(ConsumerService)
    
    print(f"Test-Service: {test_service.get_message()}")
    print(f"Consumer: {consumer.do_work()}")
    
    # Container-Statistiken anzeigen
    stats = container.get_container_statistics()
    print(f"Container-Statistiken: {stats}")
    
    # Scope-Test
    with ServiceScope() as scope:
        scoped_consumer = container.resolve(ConsumerService)
        print(f"Scoped Consumer: {scoped_consumer.do_work()}")
    
    # Decorator-Test
    @inject(ITestService)
    def test_function(message: str, itestservice: ITestService):
        return f"{message}: {itestservice.get_message()}"
    
    result = test_function("Decorator-Test", container=container)
    print(f"Decorator-Test: {result}")
    
    print("✅ Dependency Injection Container erfolgreich getestet")
    
    # Cleanup
    reset_service_container()
    
    
    # =============================================================================
# CONTAINER-REPARATUR - AM ENDE VON core/container.py ANHÄNGEN
# =============================================================================

# Diese Datei einfach am Ende von core/container.py anhängen

import threading
from typing import Dict, Any, Optional, Type, List

# =============================================================================
# FEHLENDE KLASSEN FÜR IMPORT-KOMPATIBILITÄT
# =============================================================================

class DependencyInjector:
    """
    Dependency Injector - Wrapper um ServiceContainer
    Stellt die erwartete API für Service-Registration bereit
    """
    
    def __init__(self, container: Optional['ServiceContainer'] = None):
        """
        Initialisiert DependencyInjector
        
        Args:
            container: Optional ServiceContainer, falls None wird globaler verwendet
        """
        self.container = container or get_service_container()
        self.logger = get_logger(f"{__name__}.injector")
        
    def register_singleton(self, service_type: Type, implementation: Optional[Type] = None) -> 'DependencyInjector':
        """
        Registriert Service als Singleton
        
        Args:
            service_type: Interface oder Service-Type
            implementation: Konkrete Implementierung (optional)
            
        Returns:
            DependencyInjector: Für Method-Chaining
        """
        try:
            if hasattr(self.container, 'register_singleton'):
                self.container.register_singleton(service_type, implementation)
            else:
                # Fallback für einfache Container
                if not hasattr(self.container, '_singletons'):
                    self.container._singletons = {}
                self.container._singletons[service_type] = implementation or service_type
                
            self.logger.debug(f"Singleton registriert: {service_type.__name__}")
            
        except Exception as e:
            self.logger.warning(f"Singleton-Registrierung fehlgeschlagen: {e}")
            
        return self
    
    def register_transient(self, service_type: Type, implementation: Optional[Type] = None) -> 'DependencyInjector':
        """
        Registriert Service als Transient (neue Instanz bei jeder Anfrage)
        
        Args:
            service_type: Interface oder Service-Type
            implementation: Konkrete Implementierung (optional)
            
        Returns:
            DependencyInjector: Für Method-Chaining
        """
        try:
            if hasattr(self.container, 'register_transient'):
                self.container.register_transient(service_type, implementation)
            else:
                # Fallback für einfache Container
                if not hasattr(self.container, '_transients'):
                    self.container._transients = {}
                self.container._transients[service_type] = implementation or service_type
                
            self.logger.debug(f"Transient registriert: {service_type.__name__}")
            
        except Exception as e:
            self.logger.warning(f"Transient-Registrierung fehlgeschlagen: {e}")
            
        return self
    
    def register_instance(self, service_type: Type, instance: Any) -> 'DependencyInjector':
        """
        Registriert konkrete Service-Instanz
        
        Args:
            service_type: Service-Type
            instance: Konkrete Instanz
            
        Returns:
            DependencyInjector: Für Method-Chaining
        """
        try:
            if hasattr(self.container, 'register_instance'):
                self.container.register_instance(service_type, instance)
            else:
                # Fallback für einfache Container
                if not hasattr(self.container, '_instances'):
                    self.container._instances = {}
                self.container._instances[service_type] = instance
                
            self.logger.debug(f"Instanz registriert: {service_type.__name__}")
            
        except Exception as e:
            self.logger.warning(f"Instanz-Registrierung fehlgeschlagen: {e}")
            
        return self
    
    def resolve(self, service_type: Type) -> Any:
        """
        Löst Service-Abhängigkeit auf
        
        Args:
            service_type: Gewünschter Service-Type
            
        Returns:
            Service-Instanz
        """
        try:
            if hasattr(self.container, 'resolve'):
                return self.container.resolve(service_type)
            else:
                # Fallback-Resolution
                # Prüfe Instanzen zuerst
                if hasattr(self.container, '_instances') and service_type in self.container._instances:
                    return self.container._instances[service_type]
                
                # Dann Singletons
                if hasattr(self.container, '_singletons') and service_type in self.container._singletons:
                    impl_type = self.container._singletons[service_type]
                    if not hasattr(self.container, '_singleton_instances'):
                        self.container._singleton_instances = {}
                    if service_type not in self.container._singleton_instances:
                        self.container._singleton_instances[service_type] = impl_type()
                    return self.container._singleton_instances[service_type]
                
                # Dann Transients
                if hasattr(self.container, '_transients') and service_type in self.container._transients:
                    impl_type = self.container._transients[service_type]
                    return impl_type()
                
                # Last Resort: Direkte Instanziierung
                return service_type()
                
        except Exception as e:
            self.logger.error(f"Service-Resolution fehlgeschlagen für {service_type.__name__}: {e}")
            raise
    
    def dispose(self):
        """Beendet DependencyInjector und gibt Ressourcen frei"""
        if hasattr(self.container, 'dispose'):
            self.container.dispose()


class ServiceRegistry:
    """
    Service Registry - Name-basierte Service-Verwaltung
    Ermöglicht String-basierte Service-Registrierung und -Auflösung
    """
    
    def __init__(self, container: Optional['ServiceContainer'] = None):
        """
        Initialisiert ServiceRegistry
        
        Args:
            container: Optional ServiceContainer für Type-basierte Services
        """
        self.container = container or get_service_container()
        self.logger = get_logger(f"{__name__}.registry")
        self._services: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._dependencies: Dict[str, List[str]] = {}
        self._lock = threading.RLock()
    
    def register_service(
        self,
        name: str,
        service_instance: Any,
        dependencies: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registriert Service mit Namen
        
        Args:
            name: Service-Name
            service_instance: Service-Instanz
            dependencies: Service-Abhängigkeiten (optional)
            metadata: Zusätzliche Metadaten (optional)
        """
        with self._lock:
            try:
                self._services[name] = service_instance
                
                # Metadaten speichern
                self._metadata[name] = metadata or {}
                self._metadata[name].update({
                    'registration_time': time.time(),
                    'service_type': type(service_instance).__name__
                })
                
                # Dependencies verarbeiten
                if dependencies:
                    if hasattr(dependencies, 'depends_on'):
                        # ServiceDependency-Objekt
                        self._dependencies[name] = dependencies.depends_on
                    elif isinstance(dependencies, (list, tuple)):
                        # Liste von Abhängigkeiten
                        self._dependencies[name] = list(dependencies)
                    else:
                        # Einzelne Abhängigkeit
                        self._dependencies[name] = [str(dependencies)]
                
                # Auch im Container registrieren falls möglich
                if hasattr(service_instance, '__class__'):
                    try:
                        if hasattr(self.container, 'register_instance'):
                            self.container.register_instance(service_instance.__class__, service_instance)
                    except Exception:
                        pass  # Soft failure
                
                self.logger.info(f"Service '{name}' registriert")
                
            except Exception as e:
                self.logger.error(f"Service-Registrierung fehlgeschlagen für '{name}': {e}")
                raise
    
    def get_service(self, name: str) -> Optional[Any]:
        """
        Holt Service nach Namen
        
        Args:
            name: Service-Name
            
        Returns:
            Service-Instanz oder None
        """
        with self._lock:
            service = self._services.get(name)
            if service is None:
                self.logger.warning(f"Service '{name}' nicht gefunden")
            return service
    
    def list_services(self) -> List[str]:
        """
        Listet alle registrierten Service-Namen
        
        Returns:
            Liste der Service-Namen
        """
        with self._lock:
            return list(self._services.keys())
    
    def get_service_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Holt Metadaten für Service
        
        Args:
            name: Service-Name
            
        Returns:
            Metadaten-Dictionary oder None
        """
        with self._lock:
            return self._metadata.get(name)
    
    def get_service_dependencies(self, name: str) -> List[str]:
        """
        Holt Abhängigkeiten für Service
        
        Args:
            name: Service-Name
            
        Returns:
            Liste der Abhängigkeiten
        """
        with self._lock:
            return self._dependencies.get(name, [])
    
    def remove_service(self, name: str) -> bool:
        """
        Entfernt Service aus Registry
        
        Args:
            name: Service-Name
            
        Returns:
            True wenn entfernt, False wenn nicht gefunden
        """
        with self._lock:
            if name in self._services:
                # Service ordnungsgemäß beenden falls möglich
                service = self._services[name]
                if hasattr(service, 'dispose'):
                    try:
                        service.dispose()
                    except Exception as e:
                        self.logger.warning(f"Fehler beim Beenden von Service '{name}': {e}")
                
                # Aus allen Registries entfernen
                del self._services[name]
                self._metadata.pop(name, None)
                self._dependencies.pop(name, None)
                
                self.logger.info(f"Service '{name}' entfernt")
                return True
            return False
    
    def clear_registry(self):
        """Leert komplette Registry"""
        with self._lock:
            # Alle Services ordnungsgemäß beenden
            for name, service in self._services.items():
                if hasattr(service, 'dispose'):
                    try:
                        service.dispose()
                    except Exception as e:
                        self.logger.warning(f"Fehler beim Beenden von Service '{name}': {e}")
            
            self._services.clear()
            self._metadata.clear()
            self._dependencies.clear()
            
            self.logger.info("Service-Registry geleert")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Holt Registry-Statistiken
        
        Returns:
            Statistiken-Dictionary
        """
        with self._lock:
            return {
                'total_services': len(self._services),
                'service_names': list(self._services.keys()),
                'services_with_dependencies': len(self._dependencies),
                'total_dependencies': sum(len(deps) for deps in self._dependencies.values())
            }


# =============================================================================
# GLOBALE REGISTRY-INSTANZ
# =============================================================================

_global_registry: Optional[ServiceRegistry] = None
_registry_lock = threading.RLock()

def get_service_registry() -> ServiceRegistry:
    """
    Holt globale ServiceRegistry-Instanz (Singleton)
    
    Returns:
        ServiceRegistry: Globale Registry-Instanz
    """
    global _global_registry
    
    with _registry_lock:
        if _global_registry is None:
            _global_registry = ServiceRegistry()
        return _global_registry

def reset_service_registry():
    """Setzt globale Registry zurück"""
    global _global_registry
    with _registry_lock:
        if _global_registry is not None:
            _global_registry.clear_registry()
        _global_registry = None


# =============================================================================
# FACTORY FUNCTIONS FÜR KOMPATIBILITÄT
# =============================================================================

def create_container_with_defaults() -> ServiceContainer:
    """
    Erstellt ServiceContainer mit Standard-Konfiguration
    
    Returns:
        ServiceContainer: Konfigurierter Container
    """
    try:
        container = get_service_container()
        
        # Standard-Services registrieren falls Container erweitert werden soll
        # (Wird später implementiert)
        
        return container
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Container-Erstellung fehlgeschlagen: {e}")
        # Fallback: Minimaler Container
        return ServiceContainer()


# =============================================================================
# __all__ EXPORT ERGÄNZUNG
# =============================================================================

# Diese Zeilen zur bestehenden __all__ Liste hinzufügen:
# (Falls __all__ am Ende der Datei noch nicht existiert, komplett hinzufügen)

try:
    # Zur bestehenden __all__ hinzufügen
    __all__.extend([
        'DependencyInjector',
        'ServiceRegistry', 
        'get_service_registry',
        'reset_service_registry',
        'create_container_with_defaults',
        'get_service_container', 
        'reset_service_container', 
        'get_service_registry'
    ])
except NameError:
    # Falls __all__ nicht existiert, komplett definieren
    __all__ = [
        'ServiceContainer',
        'ServiceLifecycle', 
        'ServiceStatus',
        'DependencyInjector',
        'ServiceRegistry',
        'get_service_container',
        'reset_service_container',
        'register_service',
        'get_service',
        'get_service_registry',
        'reset_service_registry',
        'create_container_with_defaults'
        'get_service_container', 
        'reset_service_container', 
        'ServiceRegistry', 
        'get_service_registry'
    ]

# =============================================================================
# IMPORT-KOMPATIBILITÄT SICHERSTELLEN
# =============================================================================

# Zusätzliche Imports falls noch nicht vorhanden
import time

# Logger-Import sicherstellen
try:
    from .logger import get_logger
except ImportError:
    def get_logger(name):
        import logging
        return logging.getLogger(name)

print("✅ Container-Reparatur abgeschlossen - DependencyInjector und ServiceRegistry verfügbar")

# =============================================================================
# FEHLENDE GLOBALE SERVICE-FUNKTIONEN
# =============================================================================

def get_container():
    """Alias für get_service_container - Import-Kompatibilität"""
    return get_service_container()

def register_service(name: str, service_instance, lifecycle=None):
    """Globale register_service Funktion"""
    try:
        registry = get_service_registry()
        registry.register_service(name, service_instance)
        print(f"✅ Service '{name}' registriert")
    except Exception as e:
        print(f"⚠️ Service-Registration failed für '{name}': {e}")

def get_service(name: str):
    """Globale get_service Funktion"""
    try:
        registry = get_service_registry()
        return registry.get_service(name)
    except Exception:
        return None

print("✅ Container-Service-Funktionen erweitert")

_global_service_container = None

def get_service_container():
    global _global_service_container
    if _global_service_container is None:
        _global_service_container = ServiceContainer()
    return _global_service_container

def reset_service_container():
    global _global_service_container
    _global_service_container = ServiceContainer()
    return _global_service_container

_global_service_registry = None

def get_service_registry():
    global _global_service_registry  
    if _global_service_registry is None:
        _global_service_registry = ServiceRegistry()
    return _global_service_registry
    
    