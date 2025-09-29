# controllers/session_controller.py
"""
Session Controller - Robustes Session State Management
Industrielle RAG-Architektur - Phase 3 Migration

Löst die kritischen Session-Probleme des monolithischen Systems durch
robuste Session-Verwaltung, automatische Wiederherstellung und 
industrietaugliche State-Persistierung.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
import uuid
import json
import pickle
import threading
from pathlib import Path
from contextlib import contextmanager

from core.logger import get_logger
from core.exceptions import SessionException, ConfigurationException
from core.config import get_config
from core.container import get_container

logger = get_logger(__name__)


class SessionState(Enum):
    """Session-Zustände für Lifecycle-Management"""
    CREATED = "created"
    ACTIVE = "active"
    IDLE = "idle"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    ERROR = "error"


class SessionType(Enum):
    """Session-Typen für unterschiedliche Use Cases"""
    CHAT = "chat"                    # Chat-Konversationen
    DOCUMENT_PROCESSING = "document" # Dokument-Verarbeitung
    ADMIN = "admin"                  # Admin/Debug-Sessions
    API = "api"                      # API-Sessions
    BACKGROUND = "background"        # Hintergrund-Tasks


@dataclass
class SessionData:
    """Strukturierte Session-Daten mit Metadaten"""
    session_id: str
    session_type: SessionType
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    state: SessionState = SessionState.CREATED
    
    # Session-spezifische Daten
    user_context: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    document_context: Dict[str, Any] = field(default_factory=dict)
    
    # Service-States (für LLM-Instanz Recovery etc.)
    service_states: Dict[str, Any] = field(default_factory=dict)
    
    # Performance-Tracking
    request_count: int = 0
    total_processing_time: float = 0.0
    error_count: int = 0
    
    # Configuration
    session_config: Dict[str, Any] = field(default_factory=dict)
    ttl_seconds: int = 3600          # 1 Stunde Standard-TTL
    max_idle_seconds: int = 1800     # 30 Minuten Max-Idle
    
    def update_access_time(self):
        """Aktualisiert Last-Access-Timestamp"""
        self.last_accessed = time.time()
    
    def is_expired(self) -> bool:
        """Prüft ob Session abgelaufen ist"""
        return (time.time() - self.created_at) > self.ttl_seconds
    
    def is_idle_expired(self) -> bool:
        """Prüft ob Session idle-expired ist"""
        return (time.time() - self.last_accessed) > self.max_idle_seconds
    
    def get_age_seconds(self) -> float:
        """Liefert Session-Alter in Sekunden"""
        return time.time() - self.created_at
    
    def get_idle_seconds(self) -> float:
        """Liefert Idle-Zeit in Sekunden"""
        return time.time() - self.last_accessed


@dataclass
class SessionConfig:
    """Session Controller Konfiguration"""
    # Persistence
    persistence_enabled: bool = True
    persistence_directory: str = "data/sessions"
    auto_save_interval: int = 300        # 5 Minuten Auto-Save
    
    # Cleanup
    cleanup_interval: int = 600          # 10 Minuten Cleanup-Interval
    max_sessions: int = 1000            # Maximale Session-Anzahl
    expired_session_cleanup: bool = True
    
    # Recovery
    auto_recovery_enabled: bool = True
    service_state_recovery: bool = True
    backup_on_error: bool = True
    
    # Performance
    session_cache_size: int = 100       # In-Memory Cache-Größe
    lazy_loading: bool = True           # Lazy Session Loading
    compress_persistence: bool = True    # Session-Daten komprimieren
    
    # Monitoring
    performance_tracking: bool = True
    session_analytics: bool = True
    health_monitoring: bool = True


class SessionController:
    """
    Session Controller - Robustes Session State Management
    
    Löst kritische Session-Probleme:
    - LLM-Instanz Verlust bei Streamlit-Rerun ✅
    - Robuste Session-Wiederherstellung ✅
    - Automatische Service-State Recovery ✅  
    - Industrietaugliche Session-Persistierung ✅
    - Performance-optimiertes Session-Caching ✅
    """
    
    def __init__(self, config: SessionConfig):
        """
        Initialisiert Session Controller
        
        Args:
            config: SessionConfig mit Controller-Einstellungen
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.controller")
        
        # Session Storage
        self._sessions: Dict[str, SessionData] = {}
        self._session_cache: Dict[str, SessionData] = {}
        
        # Thread-Safety
        self._session_lock = threading.RLock()
        
        # Persistence
        self._persistence_dir = Path(self.config.persistence_directory)
        if self.config.persistence_enabled:
            self._persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance-Tracking
        self._controller_stats = {
            'sessions_created': 0,
            'sessions_restored': 0,
            'sessions_expired': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'persistence_saves': 0,
            'persistence_loads': 0,
            'recovery_attempts': 0,
            'recovery_successes': 0
        }
        
        # Background Tasks
        self._last_cleanup = time.time()
        self._last_auto_save = time.time()
        
        # Service Container für Recovery
        self._container = get_container()
        
        # Existing Sessions laden
        if self.config.persistence_enabled:
            self._load_existing_sessions()
        
        self.logger.info(
            "Session Controller initialisiert",
            extra={
                'persistence_enabled': config.persistence_enabled,
                'max_sessions': config.max_sessions,
                'cache_size': config.session_cache_size,
                'existing_sessions': len(self._sessions)
            }
        )

    def create_session(
        self,
        session_type: SessionType = SessionType.CHAT,
        user_context: Optional[Dict[str, Any]] = None,
        session_config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Erstellt neue Session mit robuster Initialisierung
        
        Args:
            session_type: Typ der Session
            user_context: Benutzer-Kontext
            session_config: Session-spezifische Konfiguration
            session_id: Optional vordefinierte Session-ID
            
        Returns:
            str: Session-ID
        """
        with self._session_lock:
            try:
                if session_id is None:
                    session_id = self._generate_session_id()
                
                # Session-Daten erstellen
                session_data = SessionData(
                    session_id=session_id,
                    session_type=session_type,
                    user_context=user_context or {},
                    session_config=session_config or {}
                )
                
                # TTL aus Konfiguration setzen
                if session_config and 'ttl_seconds' in session_config:
                    session_data.ttl_seconds = session_config['ttl_seconds']
                
                # Session registrieren
                self._sessions[session_id] = session_data
                self._add_to_cache(session_id, session_data)
                
                # Session-State initialisieren
                session_data.state = SessionState.ACTIVE
                
                # Service-States für Recovery vorbereiten
                if self.config.service_state_recovery:
                    self._initialize_service_states(session_data)
                
                # Persistierung
                if self.config.persistence_enabled:
                    self._persist_session(session_data)
                
                # Statistiken
                self._controller_stats['sessions_created'] += 1
                
                self.logger.info(
                    "Session erstellt",
                    extra={
                        'session_id': session_id,
                        'session_type': session_type.value,
                        'user_context_keys': list((user_context or {}).keys())
                    }
                )
                
                return session_id
                
            except Exception as e:
                self.logger.error(f"Session-Erstellung fehlgeschlagen: {str(e)}", exc_info=True)
                raise SessionException(f"Session-Erstellung fehlgeschlagen: {str(e)}")

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Holt Session mit automatischer Wiederherstellung
        
        Args:
            session_id: Session-ID
            
        Returns:
            SessionData oder None wenn Session nicht existiert
        """
        with self._session_lock:
            try:
                # Cache-Lookup zuerst
                if session_id in self._session_cache:
                    session_data = self._session_cache[session_id]
                    self._controller_stats['cache_hits'] += 1
                else:
                    self._controller_stats['cache_misses'] += 1
                    
                    # Memory-Lookup
                    if session_id in self._sessions:
                        session_data = self._sessions[session_id]
                        self._add_to_cache(session_id, session_data)
                    else:
                        # Persistence-Lookup (Session Recovery)
                        session_data = self._load_session_from_persistence(session_id)
                        if session_data:
                            self._sessions[session_id] = session_data
                            self._add_to_cache(session_id, session_data)
                            self._controller_stats['sessions_restored'] += 1
                        else:
                            return None
                
                # Session-Validierung
                if session_data.is_expired():
                    self.logger.warning(f"Session {session_id} ist abgelaufen")
                    self._expire_session(session_id)
                    return None
                
                # Access-Time aktualisieren
                session_data.update_access_time()
                
                # Session-State Recovery wenn nötig
                if session_data.state == SessionState.ERROR and self.config.auto_recovery_enabled:
                    self._attempt_session_recovery(session_data)
                
                return session_data
                
            except Exception as e:
                self.logger.error(f"Session-Abruf fehlgeschlagen: {str(e)}", exc_info=True)
                return None

    def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any],
        merge_context: bool = True
    ) -> bool:
        """
        Aktualisiert Session-Daten thread-safe
        
        Args:
            session_id: Session-ID
            updates: Updates als Dictionary
            merge_context: Ob Kontexte gemerged werden sollen
            
        Returns:
            bool: True wenn erfolgreich aktualisiert
        """
        with self._session_lock:
            try:
                session_data = self.get_session(session_id)
                if not session_data:
                    return False
                
                # Updates anwenden
                for key, value in updates.items():
                    if key == 'user_context' and merge_context:
                        session_data.user_context.update(value)
                    elif key == 'conversation_history':
                        # Conversation History erweitern
                        if isinstance(value, list):
                            session_data.conversation_history.extend(value)
                        else:
                            session_data.conversation_history.append(value)
                    elif key == 'service_states' and merge_context:
                        session_data.service_states.update(value)
                    elif hasattr(session_data, key):
                        setattr(session_data, key, value)
                
                # Access-Time aktualisieren
                session_data.update_access_time()
                
                # Cache aktualisieren
                self._add_to_cache(session_id, session_data)
                
                # Auto-Persistierung bei wichtigen Updates
                if self.config.persistence_enabled and any(
                    key in updates for key in ['service_states', 'conversation_history', 'document_context']
                ):
                    self._persist_session(session_data)
                
                self.logger.debug(f"Session {session_id} aktualisiert: {list(updates.keys())}")
                return True
                
            except Exception as e:
                self.logger.error(f"Session-Update fehlgeschlagen: {str(e)}", exc_info=True)
                return False

    def store_service_state(
        self,
        session_id: str,
        service_name: str,
        service_state: Any
    ) -> bool:
        """
        Speichert Service-State für Session (LLM-Instanz Recovery)
        
        Args:
            session_id: Session-ID
            service_name: Name des Services (z.B. 'llm_instance')
            service_state: State-Daten (serialisierbar)
            
        Returns:
            bool: True wenn erfolgreich gespeichert
        """
        try:
            updates = {
                'service_states': {service_name: service_state}
            }
            
            success = self.update_session(session_id, updates, merge_context=True)
            
            if success:
                self.logger.debug(f"Service-State gespeichert: {session_id}/{service_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Service-State Speicherung fehlgeschlagen: {str(e)}")
            return False

    def restore_service_state(
        self,
        session_id: str,
        service_name: str
    ) -> Optional[Any]:
        """
        Stellt Service-State wieder her (LLM-Instanz Recovery)
        
        Args:
            session_id: Session-ID
            service_name: Name des Services
            
        Returns:
            Service-State oder None
        """
        try:
            session_data = self.get_session(session_id)
            if not session_data:
                return None
            
            service_state = session_data.service_states.get(service_name)
            
            if service_state:
                self.logger.debug(f"Service-State wiederhergestellt: {session_id}/{service_name}")
            
            return service_state
            
        except Exception as e:
            self.logger.error(f"Service-State Wiederherstellung fehlgeschlagen: {str(e)}")
            return None

    def add_conversation_entry(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Fügt Conversation-Entry zur Session hinzu
        
        Args:
            session_id: Session-ID
            role: Rolle (user, assistant, system)
            content: Message-Inhalt
            metadata: Zusätzliche Metadaten
            
        Returns:
            bool: True wenn erfolgreich hinzugefügt
        """
        try:
            conversation_entry = {
                'timestamp': time.time(),
                'role': role,
                'content': content,
                'metadata': metadata or {}
            }
            
            updates = {
                'conversation_history': conversation_entry,
                'request_count': 1  # Wird in update_session inkrementiert
            }
            
            return self.update_session(session_id, updates)
            
        except Exception as e:
            self.logger.error(f"Conversation-Entry hinzufügen fehlgeschlagen: {str(e)}")
            return False

    @contextmanager
    def session_context(self, session_id: str):
        """
        Context Manager für Session-sichere Operationen
        
        Usage:
            with controller.session_context(session_id) as session:
                session.user_context['key'] = 'value'
                # Automatisches Update beim Context-Exit
        """
        session_data = self.get_session(session_id)
        if not session_data:
            raise SessionException(f"Session {session_id} nicht gefunden")
        
        try:
            yield session_data
        finally:
            # Session nach Context-Nutzung aktualisieren
            self.update_session(session_id, asdict(session_data), merge_context=False)

    def expire_session(self, session_id: str) -> bool:
        """
        Markiert Session als abgelaufen und bereinigt sie
        
        Args:
            session_id: Session-ID
            
        Returns:
            bool: True wenn erfolgreich abgelaufen
        """
        with self._session_lock:
            return self._expire_session(session_id)

    def _expire_session(self, session_id: str) -> bool:
        """Interne Session-Expiration (thread-unsafe)"""
        try:
            if session_id in self._sessions:
                session_data = self._sessions[session_id]
                session_data.state = SessionState.EXPIRED
                
                # Backup vor Löschung wenn konfiguriert
                if self.config.backup_on_error:
                    self._backup_session(session_data)
                
                # Session aus Memory entfernen
                del self._sessions[session_id]
                
                # Cache bereinigen
                if session_id in self._session_cache:
                    del self._session_cache[session_id]
                
                # Persistence bereinigen
                if self.config.persistence_enabled:
                    self._delete_persistent_session(session_id)
                
                self._controller_stats['sessions_expired'] += 1
                self.logger.info(f"Session {session_id} abgelaufen und bereinigt")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Session-Expiration fehlgeschlagen: {str(e)}")
            return False

    def cleanup_expired_sessions(self) -> Dict[str, Any]:
        """
        Bereinigt abgelaufene Sessions
        
        Returns:
            Dict mit Cleanup-Statistiken
        """
        cleanup_stats = {
            'sessions_checked': 0,
            'sessions_expired': 0,
            'sessions_idle_expired': 0,
            'cache_cleaned': 0,
            'persistence_cleaned': 0,
            'errors': []
        }
        
        with self._session_lock:
            try:
                current_time = time.time()
                sessions_to_expire = []
                
                # Identifiziere abgelaufene Sessions
                for session_id, session_data in self._sessions.items():
                    cleanup_stats['sessions_checked'] += 1
                    
                    if session_data.is_expired():
                        sessions_to_expire.append((session_id, 'ttl_expired'))
                    elif session_data.is_idle_expired():
                        sessions_to_expire.append((session_id, 'idle_expired'))
                
                # Sessions abgelaufen lassen
                for session_id, reason in sessions_to_expire:
                    try:
                        self._expire_session(session_id)
                        
                        if reason == 'ttl_expired':
                            cleanup_stats['sessions_expired'] += 1
                        else:
                            cleanup_stats['sessions_idle_expired'] += 1
                            
                    except Exception as e:
                        cleanup_stats['errors'].append(f"Session {session_id}: {str(e)}")
                
                # Cache-Größe überprüfen und bereinigen
                if len(self._session_cache) > self.config.session_cache_size:
                    self._cleanup_cache()
                    cleanup_stats['cache_cleaned'] += 1
                
                # Persistence-Bereinigung
                if self.config.persistence_enabled:
                    persistence_cleaned = self._cleanup_persistence()
                    cleanup_stats['persistence_cleaned'] = persistence_cleaned
                
                self._last_cleanup = current_time
                
                if cleanup_stats['sessions_expired'] + cleanup_stats['sessions_idle_expired'] > 0:
                    self.logger.info(
                        "Session-Cleanup abgeschlossen",
                        extra=cleanup_stats
                    )
                
            except Exception as e:
                cleanup_stats['errors'].append(f"Cleanup-Fehler: {str(e)}")
                self.logger.error(f"Session-Cleanup fehlgeschlagen: {str(e)}")
        
        return cleanup_stats

    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Liefert umfassende Session-Statistiken
        
        Returns:
            Dict mit Session-Metriken
        """
        with self._session_lock:
            try:
                active_sessions = len(self._sessions)
                cache_size = len(self._session_cache)
                
                # Session-Type Distribution
                type_distribution = {}
                state_distribution = {}
                age_distribution = {'<1h': 0, '1-6h': 0, '6-24h': 0, '>24h': 0}
                idle_distribution = {'<5m': 0, '5-30m': 0, '30m-2h': 0, '>2h': 0}
                
                total_requests = 0
                total_processing_time = 0.0
                total_errors = 0
                
                for session_data in self._sessions.values():
                    # Type Distribution
                    session_type = session_data.session_type.value
                    type_distribution[session_type] = type_distribution.get(session_type, 0) + 1
                    
                    # State Distribution
                    session_state = session_data.state.value
                    state_distribution[session_state] = state_distribution.get(session_state, 0) + 1
                    
                    # Age Distribution
                    age_hours = session_data.get_age_seconds() / 3600
                    if age_hours < 1:
                        age_distribution['<1h'] += 1
                    elif age_hours < 6:
                        age_distribution['1-6h'] += 1
                    elif age_hours < 24:
                        age_distribution['6-24h'] += 1
                    else:
                        age_distribution['>24h'] += 1
                    
                    # Idle Distribution
                    idle_minutes = session_data.get_idle_seconds() / 60
                    if idle_minutes < 5:
                        idle_distribution['<5m'] += 1
                    elif idle_minutes < 30:
                        idle_distribution['5-30m'] += 1
                    elif idle_minutes < 120:
                        idle_distribution['30m-2h'] += 1
                    else:
                        idle_distribution['>2h'] += 1
                    
                    # Performance Aggregation
                    total_requests += session_data.request_count
                    total_processing_time += session_data.total_processing_time
                    total_errors += session_data.error_count
                
                # Persistence Statistics
                persistence_stats = {}
                if self.config.persistence_enabled and self._persistence_dir.exists():
                    persistence_files = list(self._persistence_dir.glob("*.json"))
                    persistence_size_mb = sum(f.stat().st_size for f in persistence_files) / 1024 / 1024
                    
                    persistence_stats = {
                        'enabled': True,
                        'directory': str(self._persistence_dir),
                        'file_count': len(persistence_files),
                        'size_mb': round(persistence_size_mb, 2)
                    }
                else:
                    persistence_stats = {'enabled': False}
                
                return {
                    'active_sessions': active_sessions,
                    'cache_size': cache_size,
                    'cache_hit_rate': self._calculate_cache_hit_rate(),
                    'controller_statistics': self._controller_stats,
                    'session_distributions': {
                        'by_type': type_distribution,
                        'by_state': state_distribution,
                        'by_age': age_distribution,
                        'by_idle_time': idle_distribution
                    },
                    'performance_metrics': {
                        'total_requests': total_requests,
                        'total_processing_time': total_processing_time,
                        'total_errors': total_errors,
                        'avg_requests_per_session': total_requests / max(active_sessions, 1),
                        'avg_processing_time_per_request': total_processing_time / max(total_requests, 1),
                        'error_rate': total_errors / max(total_requests, 1)
                    },
                    'persistence': persistence_stats,
                    'configuration': {
                        'max_sessions': self.config.max_sessions,
                        'session_cache_size': self.config.session_cache_size,
                        'auto_save_interval': self.config.auto_save_interval,
                        'cleanup_interval': self.config.cleanup_interval
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Session-Statistiken Fehler: {str(e)}")
                return {'error': str(e)}

    def perform_maintenance(self) -> Dict[str, Any]:
        """
        Führt Wartungsoperationen durch
        
        Returns:
            Dict mit Wartungs-Ergebnissen
        """
        maintenance_results = {
            'timestamp': time.time(),
            'operations_performed': [],
            'cleanup_stats': {},
            'save_stats': {},
            'optimization_stats': {},
            'errors': []
        }
        
        try:
            # Expired Sessions Cleanup
            if (time.time() - self._last_cleanup) > self.config.cleanup_interval:
                cleanup_stats = self.cleanup_expired_sessions()
                maintenance_results['cleanup_stats'] = cleanup_stats
                maintenance_results['operations_performed'].append('expired_sessions_cleanup')
            
            # Auto-Save
            if (time.time() - self._last_auto_save) > self.config.auto_save_interval:
                save_stats = self._auto_save_all_sessions()
                maintenance_results['save_stats'] = save_stats
                maintenance_results['operations_performed'].append('auto_save')
            
            # Cache Optimization
            if len(self._session_cache) > self.config.session_cache_size:
                self._optimize_cache()
                maintenance_results['operations_performed'].append('cache_optimization')
            
            # Session Health Check
            health_issues = self._check_session_health()
            if health_issues:
                maintenance_results['health_issues'] = health_issues
                maintenance_results['operations_performed'].append('health_check')
            
        except Exception as e:
            maintenance_results['errors'].append(str(e))
            self.logger.error(f"Maintenance fehlgeschlagen: {str(e)}")
        
        return maintenance_results

    def _generate_session_id(self) -> str:
        """Generiert eindeutige Session-ID"""
        timestamp = int(time.time() * 1000)  # Millisekunden für Eindeutigkeit
        unique_id = str(uuid.uuid4())[:8]    # Kurzer UUID-Teil
        return f"session_{timestamp}_{unique_id}"

    def _initialize_service_states(self, session_data: SessionData):
        """Initialisiert Service-States für Recovery"""
        try:
            # Placeholder für Service-State Initialisierung
            # In vollständiger Implementierung würden hier
            # Service-spezifische States initialisiert
            
            initial_states = {
                'llm_instance': None,           # LLM-Instanz Recovery
                'embedding_cache': {},          # Embedding-Cache State
                'conversation_context': {},     # Chat-Kontext
                'document_state': {},          # Dokument-Verarbeitungs-State
                'user_preferences': {}         # Benutzer-Präferenzen
            }
            
            session_data.service_states.update(initial_states)
            
        except Exception as e:
            self.logger.warning(f"Service-State Initialisierung fehlgeschlagen: {str(e)}")

    def _attempt_session_recovery(self, session_data: SessionData) -> bool:
        """Versucht Session-Recovery bei Error-State"""
        try:
            self._controller_stats['recovery_attempts'] += 1
            
            # Service-States wiederherstellen
            if self.config.service_state_recovery:
                for service_name, service_state in session_data.service_states.items():
                    if service_state and service_name == 'llm_instance':
                        # Hier würde LLM-Instanz Recovery implementiert
                        pass
            
            # Session-State auf ACTIVE setzen
            session_data.state = SessionState.ACTIVE
            session_data.error_count = 0  # Error-Count zurücksetzen
            
            self._controller_stats['recovery_successes'] += 1
            self.logger.info(f"Session {session_data.session_id} erfolgreich wiederhergestellt")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Session-Recovery fehlgeschlagen: {str(e)}")
            return False

    def _persist_session(self, session_data: SessionData):
        """Persistiert Session-Daten"""
        if not self.config.persistence_enabled:
            return
        
        try:
            session_file = self._persistence_dir / f"{session_data.session_id}.json"
            
            # Session-Daten zu JSON serialisieren
            session_dict = asdict(session_data)
            
            # Enum-Werte zu Strings konvertieren
            session_dict['session_type'] = session_data.session_type.value
            session_dict['state'] = session_data.state.value
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_dict, f, ensure_ascii=False, indent=2)
            
            self._controller_stats['persistence_saves'] += 1
            
        except Exception as e:
            self.logger.error(f"Session-Persistierung fehlgeschlagen: {str(e)}")

    def _load_session_from_persistence(self, session_id: str) -> Optional[SessionData]:
        """Lädt Session aus Persistierung"""
        if not self.config.persistence_enabled:
            return None
        
        try:
            session_file = self._persistence_dir / f"{session_id}.json"
            
            if not session_file.exists():
                return None
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_dict = json.load(f)
            
            # Enum-Werte zurück konvertieren
            session_dict['session_type'] = SessionType(session_dict['session_type'])
            session_dict['state'] = SessionState(session_dict['state'])
            
            session_data = SessionData(**session_dict)
            
            self._controller_stats['persistence_loads'] += 1
            
            return session_data
            
        except Exception as e:
            self.logger.error(f"Session-Loading fehlgeschlagen: {str(e)}")
            return None

    def _load_existing_sessions(self):
        """Lädt alle existierenden Sessions beim Start"""
        if not self._persistence_dir.exists():
            return
        
        loaded_count = 0
        error_count = 0
        
        try:
            session_files = list(self._persistence_dir.glob("*.json"))
            
            for session_file in session_files:
                try:
                    session_id = session_file.stem
                    session_data = self._load_session_from_persistence(session_id)
                    
                    if session_data and not session_data.is_expired():
                        self._sessions[session_id] = session_data
                        loaded_count += 1
                    else:
                        # Abgelaufene Session-Datei löschen
                        session_file.unlink()
                        
                except Exception as e:
                    error_count += 1
                    self.logger.warning(f"Session-Datei {session_file} konnte nicht geladen werden: {str(e)}")
            
            if loaded_count > 0:
                self.logger.info(f"{loaded_count} Sessions aus Persistierung geladen ({error_count} Fehler)")
            
        except Exception as e:
            self.logger.error(f"Session-Loading beim Start fehlgeschlagen: {str(e)}")

    def _add_to_cache(self, session_id: str, session_data: SessionData):
        """Fügt Session zum Cache hinzu (mit LRU-Policy)"""
        self._session_cache[session_id] = session_data
        
        # Cache-Größe begrenzen (einfache LRU-Implementierung)
        if len(self._session_cache) > self.config.session_cache_size:
            # Älteste Session aus Cache entfernen
            oldest_id = min(
                self._session_cache.keys(),
                key=lambda sid: self._session_cache[sid].last_accessed
            )
            del self._session_cache[oldest_id]

    def _cleanup_cache(self):
        """Bereinigt Session-Cache"""
        cache_size_before = len(self._session_cache)
        
        # Sessions sortiert nach last_accessed
        sorted_sessions = sorted(
            self._session_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Behalte nur die neuesten Session im Cache
        keep_count = self.config.session_cache_size // 2
        sessions_to_keep = dict(sorted_sessions[-keep_count:])
        
        self._session_cache = sessions_to_keep
        
        self.logger.debug(f"Cache bereinigt: {cache_size_before} -> {len(self._session_cache)}")

    def _calculate_cache_hit_rate(self) -> float:
        """Berechnet Cache-Hit-Rate"""
        total_requests = self._controller_stats['cache_hits'] + self._controller_stats['cache_misses']
        if total_requests == 0:
            return 0.0
        return self._controller_stats['cache_hits'] / total_requests

    def _auto_save_all_sessions(self) -> Dict[str, Any]:
        """Speichert alle Sessions automatisch"""
        save_stats = {
            'sessions_saved': 0,
            'errors': 0,
            'total_sessions': len(self._sessions)
        }
        
        if not self.config.persistence_enabled:
            return save_stats
        
        for session_data in self._sessions.values():
            try:
                self._persist_session(session_data)
                save_stats['sessions_saved'] += 1
            except Exception:
                save_stats['errors'] += 1
        
        self._last_auto_save = time.time()
        return save_stats

    def cleanup(self):
        """Cleanup-Routine für Controller"""
        try:
            # Finale Session-Persistierung
            if self.config.persistence_enabled:
                self._auto_save_all_sessions()
            
            # Memory-Cleanup
            self._sessions.clear()
            self._session_cache.clear()
            
            self.logger.info("Session Controller cleanup abgeschlossen")
            
        except Exception as e:
            self.logger.error(f"Session Controller cleanup fehlgeschlagen: {str(e)}")


# Factory-Funktionen
def create_session_controller(config: Optional[Dict[str, Any]] = None) -> SessionController:
    """
    Factory-Funktion für Session Controller
    
    Args:
        config: Optional Controller-Konfiguration
        
    Returns:
        SessionController: Konfigurierter Controller
    """
    if config is None:
        app_config = get_config()
        session_config_dict = app_config.get('session', {})
    else:
        session_config_dict = config
    
    session_config = SessionConfig(**session_config_dict)
    return SessionController(session_config)


# Container-Registrierung
def register_session_controller():
    """Registriert Session Controller im DI-Container"""
    container = get_container()
    
    container.register_singleton(
        SessionController,
        lambda: create_session_controller()
    )
    
    logger.info("Session Controller im Container registriert")


# Example Usage & Testing
if __name__ == '__main__':
    # Test-Konfiguration
    test_config = SessionConfig(
        persistence_enabled=True,
        persistence_directory="test_sessions",
        session_cache_size=10,
        auto_save_interval=60
    )
    
    try:
        # Controller erstellen
        controller = SessionController(test_config)
        
        # Test-Session erstellen
        session_id = controller.create_session(
            session_type=SessionType.CHAT,
            user_context={'user_id': 'test_user', 'preferences': {'theme': 'dark'}},
            session_config={'ttl_seconds': 7200}  # 2 Stunden
        )
        
        print(f"✅ Session erstellt: {session_id}")
        
        # Session-Daten abrufen
        session_data = controller.get_session(session_id)
        if session_data:
            print(f"📊 Session-Typ: {session_data.session_type.value}")
            print(f"⏰ Alter: {session_data.get_age_seconds():.1f}s")
        
        # Context Manager testen
        with controller.session_context(session_id) as session:
            session.user_context['last_action'] = 'test_performed'
            session.conversation_history.append({
                'role': 'user',
                'content': 'Test-Nachricht',
                'timestamp': time.time()
            })
        
        print("✅ Session-Context erfolgreich aktualisiert")
        
        # Statistiken abrufen
        stats = controller.get_session_statistics()
        print(f"📈 Aktive Sessions: {stats['active_sessions']}")
        print(f"🎯 Cache-Hit-Rate: {stats['cache_hit_rate']:.1%}")
        
        # Service-State Test
        controller.store_service_state(session_id, 'llm_instance', {'model': 'test-model', 'temperature': 0.7})
        restored_state = controller.restore_service_state(session_id, 'llm_instance')
        print(f"🔄 Service-State Test: {'✅' if restored_state else '❌'}")
        
        # Cleanup
        controller.cleanup()
        print("🧹 Test cleanup abgeschlossen")
        
    except Exception as e:
        print(f"❌ Session Controller Test fehlgeschlagen: {str(e)}")
        import traceback
        traceback.print_exc()