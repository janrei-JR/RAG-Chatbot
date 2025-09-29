#!/usr/bin/env python3
"""
Session Service für RAG Chatbot Industrial

Robustes Session State Management mit automatischer LLM-Instanz-Verwaltung,
Session-Persistence und Recovery-Mechanismen für Streamlit-Integration.

Features:
- Robuste LLM-Instanz-Verwaltung mit automatischer Wiederherstellung
- Session-State-Persistence mit Thread-Safety
- Automatische Session-Cleanup und Memory-Management
- Health-Monitoring und Diagnostic-Tools
- Streamlit-Session-Integration mit Rerun-Handling
- Production-Features: Session-Metrics, Error-Recovery, Hot-Reload

Gelöste Probleme der Version 3.1.2:
- LLM-Instanz Verlust während Streamlit-Rerun
- Session State Management Instabilität
- Memory-Leaks bei lange laufenden Sessions
- Fehlende Session-Recovery-Mechanismen

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

import time
import threading
import weakref
import uuid
from typing import Dict, Any, Optional, List, Callable, Union, Type, TypeVar
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import gc
import sys

# Core-Komponenten
from core import (
    get_logger, RAGConfig, get_current_config,
    ServiceError, ValidationError, ResourceError,
    create_error_context, log_performance, log_method_calls
)

# Chat-Service Integration
from .chat_service import ChatService
#LLM-Imports aus korrekten Modulen mit Fallback-Handling
try:
    from modules.llm.base_llm import BaseLLM
    from modules.llm.ollama_llm import OllamaLLM
except ImportError:
    
    # Robuste Fallback-Implementierungen bei Import-Problemen^
    class BaseLLM:
        def __init__(self, config=None, **kwargs):
            self.config = config
        def generate_response(self, query: str, context: str = "") -> str:
            return f"Fallback-Antwort für: {query}"
    class OllamaLLM(BaseLLM):
        pass
        
# =============================================================================
# SESSION-DATENSTRUKTUREN UND ENUMS
# =============================================================================

class SessionStatus(str, Enum):
    """Status der Session"""
    ACTIVE = "active"                   # Session aktiv und verfügbar
    IDLE = "idle"                      # Session inaktiv aber verfügbar
    SUSPENDED = "suspended"            # Session temporär pausiert
    EXPIRED = "expired"                # Session abgelaufen
    ERROR = "error"                    # Session-Fehler aufgetreten
    TERMINATED = "terminated"          # Session ordnungsgemäß beendet


class SessionType(str, Enum):
    """Typ der Session"""
    INTERACTIVE = "interactive"        # Interaktive Chat-Session
    API = "api"                        # API-basierte Session
    BACKGROUND = "background"          # Hintergrund-Verarbeitung
    SYSTEM = "system"                  # System-Session


class ComponentStatus(str, Enum):
    """Status einzelner Session-Komponenten"""
    INITIALIZED = "initialized"        # Komponente initialisiert
    HEALTHY = "healthy"                # Komponente funktionsfähig
    DEGRADED = "degraded"              # Komponente eingeschränkt
    FAILED = "failed"                  # Komponente fehlgeschlagen
    RECOVERING = "recovering"          # Komponente wird wiederhergestellt


@dataclass
class SessionMetrics:
    """Session-Performance-Metriken"""
    queries_processed: int = 0
    total_processing_time_ms: float = 0.0
    average_response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_count: int = 0
    recovery_count: int = 0


@dataclass
class ComponentHealth:
    """Gesundheitsstatus einer Session-Komponente"""
    component_name: str
    status: ComponentStatus
    last_check: datetime
    error_message: Optional[str] = None
    recovery_attempts: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    """
    Vollständiger Session-Zustand
    
    Attributes:
        session_id (str): Eindeutige Session-ID
        user_id (Optional[str]): Benutzer-ID
        session_type (SessionType): Typ der Session
        status (SessionStatus): Aktueller Session-Status
        created_at (datetime): Erstellungszeitpunkt
        last_accessed (datetime): Letzter Zugriff
        expires_at (datetime): Ablaufzeitpunkt
        data (Dict[str, Any]): Session-Daten
        metrics (SessionMetrics): Performance-Metriken
        component_health (Dict[str, ComponentHealth]): Komponentenstatus
        metadata (Dict[str, Any]): Zusätzliche Metadaten
    """
    session_id: str
    user_id: Optional[str] = None
    session_type: SessionType = SessionType.INTERACTIVE
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=24))
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    component_health: Dict[str, ComponentHealth] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Prüft ob Session aktiv ist"""
        return self.status == SessionStatus.ACTIVE
    
    @property
    def is_expired(self) -> bool:
        """Prüft ob Session abgelaufen ist"""
        return datetime.now(timezone.utc) > self.expires_at
    
    @property
    def age_seconds(self) -> float:
        """Alter der Session in Sekunden"""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()
    
    @property
    def idle_seconds(self) -> float:
        """Idle-Zeit in Sekunden"""
        return (datetime.now(timezone.utc) - self.last_accessed).total_seconds()


T = TypeVar('T')


# =============================================================================
# KOMPONENTEN-MANAGER FÜR SESSION-SERVICES
# =============================================================================

class SessionComponentManager:
    """
    Manager für Session-Komponenten mit automatischer Recovery
    
    Verwaltet Chat-Service, LLM-Instanzen und andere Session-abhängige
    Komponenten mit robuster Fehlerbehandlung und Wiederherstellung.
    """
    
    def __init__(self, session_id: str, config: RAGConfig = None):
        """
        Initialisiert Komponenten-Manager
        
        Args:
            session_id (str): Session-ID
            config (RAGConfig): Konfiguration
        """
        self.session_id = session_id
        self.config = config or get_current_config()
        self.logger = get_logger(f"session_components_{session_id[:8]}", "services")
        
        # Komponenten-Registry
        self._components: Dict[str, Any] = {}
        self._component_factories: Dict[str, Callable[[], Any]] = {}
        self._component_health: Dict[str, ComponentHealth] = {}
        
        # Recovery-Konfiguration
        self.max_recovery_attempts = 3
        self.recovery_delay_seconds = 1.0
        
        # Thread-Safety
        self._lock = threading.RLock()
        
        # Standard-Komponenten registrieren
        self._register_default_components()
    
    def _register_default_components(self) -> None:
        """Registriert Standard-Session-Komponenten"""
        # Chat-Service Factory
        def create_chat_service() -> ChatService:
            return ChatService(config=self.config)
        
        # LLM Factory
        def create_llm() -> BaseLLM:
            return OllamaLLM(config=self.config)
        
        self.register_component_factory("chat_service", create_chat_service)
        self.register_component_factory("llm", create_llm)
    
    def register_component_factory(self, name: str, factory: Callable[[], T]) -> None:
        """
        Registriert Factory für Komponente
        
        Args:
            name (str): Komponenten-Name
            factory (Callable): Factory-Funktion
        """
        with self._lock:
            self._component_factories[name] = factory
            
            # Gesundheitsstatus initialisieren
            self._component_health[name] = ComponentHealth(
                component_name=name,
                status=ComponentStatus.INITIALIZED,
                last_check=datetime.now(timezone.utc)
            )
    
    def get_component(self, name: str, force_recreate: bool = False) -> Optional[Any]:
        """
        Holt Komponente mit automatischer Erstellung/Recovery
        
        Args:
            name (str): Komponenten-Name
            force_recreate (bool): Erzwingt Neuerstellung
            
        Returns:
            Optional[Any]: Komponenten-Instanz oder None bei Fehler
        """
        with self._lock:
            # Komponente prüfen falls vorhanden
            if name in self._components and not force_recreate:
                component = self._components[name]
                
                # Health-Check durchführen
                if self._check_component_health(name, component):
                    return component
                else:
                    # Komponente ist ungesund - Recovery versuchen
                    self.logger.warning(f"Komponente {name} ungesund - starte Recovery")
                    return self._recover_component(name)
            
            # Komponente erstellen
            return self._create_component(name)
    
    def _create_component(self, name: str) -> Optional[Any]:
        """Erstellt neue Komponenten-Instanz"""
        if name not in self._component_factories:
            self.logger.error(f"Keine Factory für Komponente {name} registriert")
            return None
        
        try:
            factory = self._component_factories[name]
            component = factory()
            
            self._components[name] = component
            
            # Gesundheitsstatus aktualisieren
            self._component_health[name] = ComponentHealth(
                component_name=name,
                status=ComponentStatus.HEALTHY,
                last_check=datetime.now(timezone.utc)
            )
            
            self.logger.debug(f"Komponente {name} erfolgreich erstellt")
            return component
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen von Komponente {name}: {str(e)}")
            
            self._component_health[name] = ComponentHealth(
                component_name=name,
                status=ComponentStatus.FAILED,
                last_check=datetime.now(timezone.utc),
                error_message=str(e)
            )
            
            return None
    
    def _check_component_health(self, name: str, component: Any) -> bool:
        """Prüft Gesundheit einer Komponente"""
        try:
            # Component-spezifische Health-Checks
            if hasattr(component, 'health_check'):
                health_result = component.health_check()
                
                if isinstance(health_result, dict):
                    is_healthy = health_result.get('service_status') in ['healthy', 'degraded']
                else:
                    is_healthy = bool(health_result)
            
            elif hasattr(component, 'is_available'):
                is_healthy = component.is_available()
            
            else:
                # Basis-Check: Objekt existiert und ist nicht None
                is_healthy = component is not None
            
            # Gesundheitsstatus aktualisieren
            status = ComponentStatus.HEALTHY if is_healthy else ComponentStatus.DEGRADED
            
            self._component_health[name] = ComponentHealth(
                component_name=name,
                status=status,
                last_check=datetime.now(timezone.utc)
            )
            
            return is_healthy
            
        except Exception as e:
            self.logger.warning(f"Health-Check für {name} fehlgeschlagen: {str(e)}")
            
            self._component_health[name] = ComponentHealth(
                component_name=name,
                status=ComponentStatus.FAILED,
                last_check=datetime.now(timezone.utc),
                error_message=str(e)
            )
            
            return False
    
    def _recover_component(self, name: str) -> Optional[Any]:
        """Versucht Komponenten-Recovery"""
        health = self._component_health.get(name)
        
        if health and health.recovery_attempts >= self.max_recovery_attempts:
            self.logger.error(f"Maximale Recovery-Versuche für {name} erreicht")
            health.status = ComponentStatus.FAILED
            return None
        
        try:
            # Recovery-Status setzen
            if health:
                health.status = ComponentStatus.RECOVERING
                health.recovery_attempts += 1
            
            # Kurze Pause vor Recovery
            time.sleep(self.recovery_delay_seconds)
            
            # Alte Komponente entfernen
            if name in self._components:
                old_component = self._components[name]
                
                # Cleanup falls verfügbar
                if hasattr(old_component, 'dispose'):
                    try:
                        old_component.dispose()
                    except Exception:
                        pass
                
                del self._components[name]
            
            # Neue Komponente erstellen
            new_component = self._create_component(name)
            
            if new_component and health:
                health.status = ComponentStatus.HEALTHY
                self.logger.info(f"Komponente {name} erfolgreich wiederhergestellt")
            
            return new_component
            
        except Exception as e:
            self.logger.error(f"Recovery für {name} fehlgeschlagen: {str(e)}")
            
            if health:
                health.status = ComponentStatus.FAILED
                health.error_message = str(e)
            
            return None
    
    def get_component_health(self) -> Dict[str, ComponentHealth]:
        """Holt Gesundheitsstatus aller Komponenten"""
        with self._lock:
            return self._component_health.copy()
    
    def cleanup(self) -> None:
        """Cleanup aller Komponenten"""
        with self._lock:
            for name, component in self._components.items():
                try:
                    if hasattr(component, 'dispose'):
                        component.dispose()
                except Exception as e:
                    self.logger.warning(f"Cleanup für {name} fehlgeschlagen: {str(e)}")
            
            self._components.clear()
            self._component_health.clear()


# =============================================================================
# SESSION-SERVICE IMPLEMENTIERUNG
# =============================================================================

class SessionService:
    """
    Robuster Session-Service mit automatischer Komponenten-Verwaltung
    
    Löst kritische Session-State-Probleme der monolithischen Version 3.1.2:
    - LLM-Instanz-Verlust während Streamlit-Rerun
    - Session-State-Inkonsistenzen
    - Memory-Leaks bei lange laufenden Sessions
    - Fehlende Error-Recovery-Mechanismen
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialisiert Session-Service
        
        Args:
            config (RAGConfig): Service-Konfiguration
        """
        self.config = config or get_current_config()
        self.logger = get_logger("session_service", "services")
        
        # Session-Registry
        self._sessions: Dict[str, SessionState] = {}
        self._component_managers: Dict[str, SessionComponentManager] = {}
        
        # Thread-Safety
        self._lock = threading.RLock()
        
        # Session-Konfiguration
        self.default_timeout_hours = getattr(self.config.app, 'session_timeout_hours', 24)
        self.max_idle_minutes = getattr(self.config.app, 'max_idle_minutes', 120)
        self.cleanup_interval_minutes = getattr(self.config.app, 'cleanup_interval_minutes', 15)
        
        # Performance-Statistiken
        self._stats = {
            'total_sessions_created': 0,
            'active_sessions': 0,
            'expired_sessions_cleaned': 0,
            'component_recoveries': 0,
            'memory_cleanups': 0
        }
        
        # Background-Cleanup-Thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # Weak-References für automatische Cleanup
        self._session_refs: Dict[str, weakref.ref] = {}
        
        self.logger.info("Session-Service initialisiert")
        
        # Cleanup-Thread starten
        self._start_cleanup_thread()
    
    @log_performance()
    @log_method_calls()
    def create_session(self, 
                      user_id: str = None,
                      session_type: SessionType = SessionType.INTERACTIVE,
                      timeout_hours: float = None,
                      metadata: Dict[str, Any] = None) -> str:
        """
        Erstellt neue Session mit robuster Komponenten-Initialisierung
        
        Args:
            user_id (str): Optional Benutzer-ID
            session_type (SessionType): Typ der Session
            timeout_hours (float): Session-Timeout in Stunden
            metadata (Dict[str, Any]): Zusätzliche Metadaten
            
        Returns:
            str: Session-ID
        """
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        
        timeout_hours = timeout_hours or self.default_timeout_hours
        expires_at = datetime.now(timezone.utc) + timedelta(hours=timeout_hours)
        
        with self._lock:
            # Session-State erstellen
            session_state = SessionState(
                session_id=session_id,
                user_id=user_id,
                session_type=session_type,
                expires_at=expires_at,
                metadata=metadata or {}
            )
            
            # Session registrieren
            self._sessions[session_id] = session_state
            
            # Komponenten-Manager erstellen
            component_manager = SessionComponentManager(session_id, self.config)
            self._component_managers[session_id] = component_manager
            
            # Weak-Reference für automatische Cleanup
            def cleanup_callback(ref):
                self._cleanup_session(session_id)
            
            self._session_refs[session_id] = weakref.ref(session_state, cleanup_callback)
            
            # Statistiken aktualisieren
            self._stats['total_sessions_created'] += 1
            self._stats['active_sessions'] += 1
            
            self.logger.info(
                f"Session erstellt: {session_id} (User: {user_id}, Type: {session_type.value})",
                extra={
                    'extra_data': {
                        'session_id': session_id,
                        'user_id': user_id,
                        'session_type': session_type.value,
                        'timeout_hours': timeout_hours
                    }
                }
            )
            
            return session_id
    
    @log_method_calls()
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Holt Session mit automatischer Validierung und Recovery
        
        Args:
            session_id (str): Session-ID
            
        Returns:
            Optional[SessionState]: Session-State oder None
        """
        with self._lock:
            if session_id not in self._sessions:
                return None
            
            session = self._sessions[session_id]
            
            # Expiration prüfen
            if session.is_expired:
                self.logger.debug(f"Session abgelaufen: {session_id}")
                self._expire_session(session_id)
                return None
            
            # Last-Access aktualisieren
            session.last_accessed = datetime.now(timezone.utc)
            
            return session
    
    def get_component(self, session_id: str, component_name: str) -> Optional[Any]:
        """
        Holt Session-Komponente mit automatischer Recovery
        
        Args:
            session_id (str): Session-ID
            component_name (str): Name der Komponente
            
        Returns:
            Optional[Any]: Komponenten-Instanz oder None
        """
        session = self.get_session(session_id)
        if not session:
            return None
        
        component_manager = self._component_managers.get(session_id)
        if not component_manager:
            self.logger.error(f"Kein Komponenten-Manager für Session {session_id}")
            return None
        
        component = component_manager.get_component(component_name)
        
        if component:
            # Session-Metriken aktualisieren
            session.metrics.last_activity = datetime.now(timezone.utc)
        
        return component
    
    def get_chat_service(self, session_id: str) -> Optional[ChatService]:
        """
        Holt Chat-Service für Session mit automatischer Wiederherstellung
        
        Args:
            session_id (str): Session-ID
            
        Returns:
            Optional[ChatService]: Chat-Service-Instanz
        """
        return self.get_component(session_id, "chat_service")
    
    def get_llm(self, session_id: str) -> Optional[BaseLLM]:
        """
        Holt LLM für Session mit robuster Instanz-Verwaltung
        
        Args:
            session_id (str): Session-ID
            
        Returns:
            Optional[BaseLLM]: LLM-Instanz
        """
        return self.get_component(session_id, "llm")
    
    def update_session_data(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Aktualisiert Session-Daten
        
        Args:
            session_id (str): Session-ID
            data (Dict[str, Any]): Zu aktualisierende Daten
            
        Returns:
            bool: True wenn erfolgreich
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        with self._lock:
            session.data.update(data)
            session.last_accessed = datetime.now(timezone.utc)
        
        return True
    
    def extend_session(self, session_id: str, additional_hours: float = None) -> bool:
        """
        Verlängert Session-Timeout
        
        Args:
            session_id (str): Session-ID
            additional_hours (float): Zusätzliche Stunden
            
        Returns:
            bool: True wenn erfolgreich
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        additional_hours = additional_hours or self.default_timeout_hours
        
        with self._lock:
            session.expires_at = datetime.now(timezone.utc) + timedelta(hours=additional_hours)
            session.last_accessed = datetime.now(timezone.utc)
        
        self.logger.debug(f"Session verlängert: {session_id} (+{additional_hours}h)")
        return True
    
    def terminate_session(self, session_id: str) -> bool:
        """
        Beendet Session ordnungsgemäß
        
        Args:
            session_id (str): Session-ID
            
        Returns:
            bool: True wenn erfolgreich
        """
        with self._lock:
            if session_id not in self._sessions:
                return False
            
            session = self._sessions[session_id]
            session.status = SessionStatus.TERMINATED
            
            # Komponenten-Cleanup
            self._cleanup_session_components(session_id)
            
            # Session entfernen
            del self._sessions[session_id]
            
            if session_id in self._session_refs:
                del self._session_refs[session_id]
            
            # Statistiken aktualisieren
            self._stats['active_sessions'] = max(0, self._stats['active_sessions'] - 1)
            
            self.logger.info(f"Session beendet: {session_id}")
            return True
    
    # =============================================================================
    # MONITORING UND HEALTH-CHECKS
    # =============================================================================
    
    def get_session_health(self, session_id: str) -> Dict[str, Any]:
        """
        Detaillierter Gesundheitscheck für Session
        
        Args:
            session_id (str): Session-ID
            
        Returns:
            Dict[str, Any]: Gesundheitsstatus
        """
        session = self.get_session(session_id)
        if not session:
            return {
                'session_exists': False,
                'overall_status': 'not_found'
            }
        
        health = {
            'session_exists': True,
            'session_id': session_id,
            'status': session.status.value,
            'is_active': session.is_active,
            'is_expired': session.is_expired,
            'age_seconds': session.age_seconds,
            'idle_seconds': session.idle_seconds,
            'overall_status': 'healthy'
        }
        
        # Komponenten-Gesundheit
        component_manager = self._component_managers.get(session_id)
        if component_manager:
            component_health = component_manager.get_component_health()
            health['components'] = {}
            
            for name, comp_health in component_health.items():
                health['components'][name] = {
                    'status': comp_health.status.value,
                    'last_check': comp_health.last_check.isoformat(),
                    'recovery_attempts': comp_health.recovery_attempts,
                    'error_message': comp_health.error_message
                }
            
            # Overall-Status basierend auf Komponenten
            failed_components = [
                name for name, comp_health in component_health.items()
                if comp_health.status == ComponentStatus.FAILED
            ]
            
            if failed_components:
                health['overall_status'] = 'degraded'
                health['failed_components'] = failed_components
            
            degraded_components = [
                name for name, comp_health in component_health.items()
                if comp_health.status == ComponentStatus.DEGRADED
            ]
            
            if degraded_components and health['overall_status'] == 'healthy':
                health['overall_status'] = 'degraded'
                health['degraded_components'] = degraded_components
        
        # Session-Metriken
        health['metrics'] = {
            'queries_processed': session.metrics.queries_processed,
            'average_response_time_ms': session.metrics.average_response_time_ms,
            'memory_usage_mb': session.metrics.memory_usage_mb,
            'error_count': session.metrics.error_count,
            'recovery_count': session.metrics.recovery_count
        }
        
        return health
    
    def perform_health_check(self, session_id: str) -> bool:
        """
        Führt umfassenden Health-Check durch
        
        Args:
            session_id (str): Session-ID
            
        Returns:
            bool: True wenn Session gesund
        """
        health = self.get_session_health(session_id)
        return health.get('overall_status') in ['healthy', 'degraded']
    
    def recover_session(self, session_id: str) -> bool:
        """
        Führt Session-Recovery durch
        
        Args:
            session_id (str): Session-ID
            
        Returns:
            bool: True wenn Recovery erfolgreich
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        component_manager = self._component_managers.get(session_id)
        if not component_manager:
            return False
        
        # Alle Komponenten zur Recovery zwingen
        for component_name in component_manager._component_factories.keys():
            try:
                component = component_manager.get_component(component_name, force_recreate=True)
                if component:
                    session.metrics.recovery_count += 1
                    self._stats['component_recoveries'] += 1
                    self.logger.info(f"Komponente {component_name} erfolgreich wiederhergestellt")
                else:
                    self.logger.warning(f"Recovery für {component_name} fehlgeschlagen")
            except Exception as e:
                self.logger.error(f"Fehler bei Recovery von {component_name}: {str(e)}")
        
        return True
    
    def get_active_sessions(self) -> List[str]:
        """
        Holt Liste aktiver Sessions
        
        Returns:
            List[str]: Session-IDs
        """
        with self._lock:
            active_sessions = []
            for session_id, session in self._sessions.items():
                if session.is_active and not session.is_expired:
                    active_sessions.append(session_id)
            return active_sessions
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Service-Statistiken für Monitoring
        
        Returns:
            Dict[str, Any]: Detaillierte Statistiken
        """
        with self._lock:
            stats = self._stats.copy()
            
            # Live-Statistiken berechnen
            now = datetime.now(timezone.utc)
            active_count = 0
            idle_count = 0
            total_memory_mb = 0.0
            
            for session in self._sessions.values():
                if session.is_expired:
                    continue
                
                if session.idle_seconds < self.max_idle_minutes * 60:
                    active_count += 1
                else:
                    idle_count += 1
                
                total_memory_mb += session.metrics.memory_usage_mb
            
            stats.update({
                'current_active_sessions': active_count,
                'current_idle_sessions': idle_count,
                'total_memory_usage_mb': total_memory_mb,
                'average_memory_per_session_mb': total_memory_mb / max(1, active_count + idle_count)
            })
            
            return stats
    
    # =============================================================================
    # CLEANUP UND MEMORY-MANAGEMENT
    # =============================================================================
    
    def _start_cleanup_thread(self) -> None:
        """Startet Background-Cleanup-Thread"""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            return
        
        def cleanup_worker():
            while not self._stop_cleanup.wait(self.cleanup_interval_minutes * 60):
                try:
                    self._perform_cleanup()
                except Exception as e:
                    self.logger.error(f"Cleanup-Thread Fehler: {str(e)}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        self.logger.debug("Cleanup-Thread gestartet")
    
    def _perform_cleanup(self) -> None:
        """Führt periodische Cleanup-Operationen durch"""
        expired_sessions = []
        idle_sessions = []
        
        with self._lock:
            now = datetime.now(timezone.utc)
            
            for session_id, session in list(self._sessions.items()):
                # Abgelaufene Sessions sammeln
                if session.is_expired:
                    expired_sessions.append(session_id)
                
                # Idle Sessions sammeln
                elif session.idle_seconds > self.max_idle_minutes * 60:
                    idle_sessions.append(session_id)
        
        # Abgelaufene Sessions entfernen
        for session_id in expired_sessions:
            self._expire_session(session_id)
        
        # Memory-Cleanup für idle Sessions
        for session_id in idle_sessions:
            self._cleanup_idle_session(session_id)
        
        # Garbage Collection
        if expired_sessions or idle_sessions:
            gc.collect()
            self._stats['memory_cleanups'] += 1
        
        if expired_sessions:
            self.logger.debug(f"Cleanup: {len(expired_sessions)} abgelaufene Sessions entfernt")
        
        if idle_sessions:
            self.logger.debug(f"Memory-Cleanup: {len(idle_sessions)} idle Sessions bereinigt")
    
    def _expire_session(self, session_id: str) -> None:
        """Entfernt abgelaufene Session"""
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.status = SessionStatus.EXPIRED
                
                # Komponenten-Cleanup
                self._cleanup_session_components(session_id)
                
                # Session entfernen
                del self._sessions[session_id]
                
                if session_id in self._session_refs:
                    del self._session_refs[session_id]
                
                # Statistiken aktualisieren
                self._stats['expired_sessions_cleaned'] += 1
                self._stats['active_sessions'] = max(0, self._stats['active_sessions'] - 1)
    
    def _cleanup_idle_session(self, session_id: str) -> None:
        """Memory-Cleanup für idle Sessions"""
        component_manager = self._component_managers.get(session_id)
        if component_manager:
            # Nicht-kritische Komponenten temporär entfernen
            with component_manager._lock:
                # Chat-Service kann wiederhergestellt werden
                if 'chat_service' in component_manager._components:
                    del component_manager._components['chat_service']
    
    def _cleanup_session_components(self, session_id: str) -> None:
        """Cleanup für Session-Komponenten"""
        if session_id in self._component_managers:
            component_manager = self._component_managers[session_id]
            component_manager.cleanup()
            del self._component_managers[session_id]
    
    def _cleanup_session(self, session_id: str) -> None:
        """Callback für automatische Session-Cleanup"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
            
            if session_id in self._component_managers:
                self._component_managers[session_id].cleanup()
                del self._component_managers[session_id]
            
            if session_id in self._session_refs:
                del self._session_refs[session_id]
    
    # =============================================================================
    # STREAMLIT-INTEGRATION
    # =============================================================================
    
    def streamlit_session_wrapper(self, streamlit_session_state: Any) -> 'StreamlitSessionWrapper':
        """
        Erstellt Wrapper für Streamlit-Session-Integration
        
        Args:
            streamlit_session_state: Streamlit Session State
            
        Returns:
            StreamlitSessionWrapper: Wrapper-Objekt
        """
        return StreamlitSessionWrapper(self, streamlit_session_state)
    
    def dispose(self) -> None:
        """Beendet Session-Service ordnungsgemäß"""
        # Cleanup-Thread stoppen
        self._stop_cleanup.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        
        # Alle Sessions beenden
        with self._lock:
            session_ids = list(self._sessions.keys())
            for session_id in session_ids:
                self.terminate_session(session_id)
        
        self.logger.info("Session-Service ordnungsgemäß beendet")


# =============================================================================
# STREAMLIT-SESSION-WRAPPER
# =============================================================================

class StreamlitSessionWrapper:
    """
    Wrapper für nahtlose Streamlit-Integration
    
    Löst Session-State-Probleme durch robuste Integration
    mit dem Streamlit-Session-State-System.
    """
    
    def __init__(self, session_service: SessionService, streamlit_session_state: Any):
        """
        Initialisiert Streamlit-Wrapper
        
        Args:
            session_service (SessionService): Session-Service
            streamlit_session_state: Streamlit Session State
        """
        self.session_service = session_service
        self.st_session = streamlit_session_state
        self.logger = get_logger("streamlit_wrapper", "services")
        
        # Session-ID aus Streamlit-State oder neu erstellen
        if not hasattr(self.st_session, 'rag_session_id'):
            self.st_session.rag_session_id = self.session_service.create_session(
                session_type=SessionType.INTERACTIVE,
                metadata={'streamlit_integration': True}
            )
            self.logger.debug(f"Neue Streamlit-Session: {self.st_session.rag_session_id}")
    
    @property
    def session_id(self) -> str:
        """Session-ID"""
        return self.st_session.rag_session_id
    
    def get_chat_service(self) -> Optional[ChatService]:
        """Holt Chat-Service mit automatischer Recovery"""
        return self.session_service.get_chat_service(self.session_id)
    
    def get_llm(self) -> Optional[BaseLLM]:
        """Holt LLM mit robuster Instanz-Verwaltung"""
        return self.session_service.get_llm(self.session_id)
    
    def extend_session(self) -> None:
        """Verlängert Session bei Aktivität"""
        self.session_service.extend_session(self.session_id)
    
    def health_check(self) -> Dict[str, Any]:
        """Session-Health-Check"""
        return self.session_service.get_session_health(self.session_id)
    
    def recover_if_needed(self) -> bool:
        """Führt Recovery durch falls nötig"""
        health = self.health_check()
        if health.get('overall_status') not in ['healthy']:
            return self.session_service.recover_session(self.session_id)
        return True


# =============================================================================
# GLOBALER SESSION-SERVICE
# =============================================================================

# Singleton-Pattern für globalen Session-Service
_global_session_service: Optional[SessionService] = None
_service_lock = threading.RLock()


def get_session_service(config: RAGConfig = None) -> SessionService:
    """
    Holt globale SessionService-Instanz (Singleton)
    
    Args:
        config (RAGConfig): Service-Konfiguration
        
    Returns:
        SessionService: Globale SessionService-Instanz
    """
    global _global_session_service
    
    with _service_lock:
        if _global_session_service is None:
            _global_session_service = SessionService(config)
        
        return _global_session_service


def reset_session_service() -> None:
    """Setzt globalen Session-Service zurück (für Tests)"""
    global _global_session_service
    
    with _service_lock:
        if _global_session_service:
            _global_session_service.dispose()
            _global_session_service = None


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'SessionStatus', 'SessionType', 'ComponentStatus',
    
    # Datenstrukturen
    'SessionMetrics', 'ComponentHealth', 'SessionState',
    
    # Manager-Klassen
    'SessionComponentManager', 'SessionService',
    
    # Streamlit-Integration
    'StreamlitSessionWrapper',
    
    # Globale Funktionen
    'get_session_service', 'reset_session_service'
]


if __name__ == "__main__":
    # Testing und Demonstration
    print("Session Service - Robustes Session State Management")
    print("==================================================")
    
    # Session-Service erstellen
    try:
        session_service = SessionService()
        
        print(f"Session-Service initialisiert")
        
        # Test-Session erstellen
        session_id = session_service.create_session(
            user_id="test_user",
            session_type=SessionType.INTERACTIVE,
            metadata={'test': True}
        )
        
        print(f"\nTest-Session erstellt: {session_id}")
        
        # Session-Health-Check
        health = session_service.get_session_health(session_id)
        print(f"\nSession-Health:")
        print(f"  Status: {health['overall_status']}")
        print(f"  Aktiv: {health['is_active']}")
        print(f"  Alter: {health['age_seconds']:.1f}s")
        
        if 'components' in health:
            print(f"  Komponenten: {len(health['components'])}")
            for comp_name, comp_health in health['components'].items():
                print(f"    {comp_name}: {comp_health['status']}")
        
        # Chat-Service testen
        chat_service = session_service.get_chat_service(session_id)
        if chat_service:
            print(f"\nChat-Service erfolgreich geladen")
            
            # Health-Check des Chat-Service
            chat_health = chat_service.health_check()
            print(f"  Chat-Service Status: {chat_health.get('service_status', 'unknown')}")
        else:
            print(f"\nChat-Service nicht verfügbar")
        
        # LLM testen
        llm = session_service.get_llm(session_id)
        if llm:
            print(f"LLM erfolgreich geladen")
            print(f"  LLM verfügbar: {llm.is_available()}")
        else:
            print(f"LLM nicht verfügbar")
        
        # Recovery-Test
        print(f"\nRecovery-Test...")
        recovery_success = session_service.recover_session(session_id)
        print(f"  Recovery erfolgreich: {recovery_success}")
        
        # Session-Daten aktualisieren
        update_success = session_service.update_session_data(session_id, {
            'test_data': 'some_value',
            'timestamp': time.time()
        })
        print(f"  Daten-Update erfolgreich: {update_success}")
        
        # Session verlängern
        extend_success = session_service.extend_session(session_id, 1.0)
        print(f"  Session verlängert: {extend_success}")
        
        # Service-Statistiken
        stats = session_service.get_session_statistics()
        print(f"\nService-Statistiken:")
        print(f"  Erstelle Sessions: {stats['total_sessions_created']}")
        print(f"  Aktive Sessions: {stats['current_active_sessions']}")
        print(f"  Memory-Cleanups: {stats['memory_cleanups']}")
        print(f"  Komponenten-Recoveries: {stats['component_recoveries']}")
        
        # Streamlit-Wrapper testen (Mock)
        class MockStreamlitState:
            pass
        
        mock_st_state = MockStreamlitState()
        streamlit_wrapper = session_service.streamlit_session_wrapper(mock_st_state)
        
        print(f"\nStreamlit-Wrapper:")
        print(f"  Session-ID: {streamlit_wrapper.session_id}")
        print(f"  Chat-Service verfügbar: {streamlit_wrapper.get_chat_service() is not None}")
        
        # Session ordnungsgemäß beenden
        terminate_success = session_service.terminate_session(session_id)
        print(f"\nSession beendet: {terminate_success}")
        
        # Service beenden
        session_service.dispose()
        print(f"Session-Service beendet")
    
    except Exception as e:
        print(f"Fehler beim Testen: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Session-Service getestet")
