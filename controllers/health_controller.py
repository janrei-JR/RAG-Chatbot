# controllers/health_controller.py
"""
Health Controller - System-Gesundheitsüberwachung
Industrielle RAG-Architektur - Phase 3 Migration

Umfassende Gesundheitsüberwachung aller System-Komponenten mit
industrietauglichen Monitoring-Features, Alert-Management und
automatischer Problemerkennung für kritische Infrastrukturen.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import psutil
import platform
from pathlib import Path

from core.logger import get_logger
from core.exceptions import HealthCheckException, ServiceException
from core.config import get_config
from core.container import get_container

logger = get_logger(__name__)


class HealthStatus(Enum):
    """System-Gesundheitsstatus"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert-Schweregrade"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Einzelne Gesundheitsmetrik"""
    name: str
    value: Any
    status: HealthStatus
    threshold: Optional[float] = None
    unit: str = ""
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def is_healthy(self) -> bool:
        """Prüft ob Metrik gesund ist"""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


@dataclass
class HealthAlert:
    """System-Gesundheitsalert"""
    alert_id: str
    component: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_at: Optional[float] = None
    
    def resolve(self):
        """Markiert Alert als gelöst"""
        self.resolved = True
        self.resolved_at = time.time()


@dataclass
class ComponentHealth:
    """Gesundheitsstatus einer Komponente"""
    component_name: str
    status: HealthStatus
    metrics: List[HealthMetric] = field(default_factory=list)
    alerts: List[HealthAlert] = field(default_factory=list)
    last_check: float = field(default_factory=time.time)
    check_duration: float = 0.0
    error_message: Optional[str] = None
    
    def add_metric(self, metric: HealthMetric):
        """Fügt Metrik hinzu"""
        self.metrics.append(metric)
    
    def add_alert(self, alert: HealthAlert):
        """Fügt Alert hinzu"""
        self.alerts.append(alert)
    
    def get_active_alerts(self) -> List[HealthAlert]:
        """Liefert aktive (ungelöste) Alerts"""
        return [alert for alert in self.alerts if not alert.resolved]


@dataclass
class SystemHealth:
    """Gesamtsystem-Gesundheitsstatus"""
    overall_status: HealthStatus
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    system_metrics: List[HealthMetric] = field(default_factory=list)
    active_alerts: List[HealthAlert] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    check_duration: float = 0.0


@dataclass
class HealthConfig:
    """Health Controller Konfiguration"""
    # Check-Intervalle
    health_check_interval: int = 60        # Sekunden zwischen Health-Checks
    detailed_check_interval: int = 300     # Sekunden zwischen detaillierten Checks
    metric_retention_hours: int = 24       # Stunden für Metrik-Aufbewahrung
    
    # Thresholds
    cpu_threshold_warning: float = 70.0    # CPU-Warnung bei >70%
    cpu_threshold_critical: float = 90.0   # CPU-Kritisch bei >90%
    memory_threshold_warning: float = 80.0 # RAM-Warnung bei >80%
    memory_threshold_critical: float = 95.0 # RAM-Kritisch bei >95%
    disk_threshold_warning: float = 85.0   # Disk-Warnung bei >85%
    disk_threshold_critical: float = 95.0  # Disk-Kritisch bei >95%
    
    # Service-spezifische Thresholds
    response_time_warning: float = 5.0     # Response-Zeit Warnung bei >5s
    response_time_critical: float = 15.0   # Response-Zeit Kritisch bei >15s
    error_rate_warning: float = 0.05       # Error-Rate Warnung bei >5%
    error_rate_critical: float = 0.20      # Error-Rate Kritisch bei >20%
    
    # Alert-Management
    alert_retention_hours: int = 168       # 7 Tage Alert-Aufbewahrung
    alert_cooldown_minutes: int = 5        # Mindestabstand zwischen gleichen Alerts
    auto_resolve_alerts: bool = True       # Automatisches Alert-Resolving
    
    # Monitoring
    enable_system_monitoring: bool = True
    enable_service_monitoring: bool = True
    enable_performance_monitoring: bool = True
    detailed_logging: bool = True


class HealthController:
    """
    Health Controller - Umfassende System-Gesundheitsüberwachung
    
    Features:
    - Multi-Layer Health-Checks (System, Services, Performance)
    - Intelligentes Alert-Management mit Cooldowns
    - Trend-Analyse und Predictive-Monitoring
    - Automatische Problemerkennung und -Kategorisierung
    - Industrietaugliche Metriken und Dashboards
    """
    
    def __init__(self, config: HealthConfig):
        """
        Initialisiert Health Controller
        
        Args:
            config: HealthConfig mit Monitoring-Einstellungen
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.controller")
        
        # System State
        self._last_health_check = 0.0
        self._last_detailed_check = 0.0
        self._health_history: List[SystemHealth] = []
        self._metric_history: Dict[str, List[HealthMetric]] = {}
        self._active_alerts: Dict[str, HealthAlert] = {}
        self._alert_cooldowns: Dict[str, float] = {}
        
        # Thread-Safety
        self._health_lock = threading.RLock()
        
        # Service Container für Component-Checks
        self._container = get_container()
        
        # Registered Health Check Functions
        self._health_check_functions: Dict[str, Callable[[], ComponentHealth]] = {}
        
        # System Info Cache
        self._system_info = self._collect_system_info()
        
        self.logger.info(
            "Health Controller initialisiert",
            extra={
                'check_interval': config.health_check_interval,
                'system_monitoring': config.enable_system_monitoring,
                'service_monitoring': config.enable_service_monitoring,
                'platform': platform.system()
            }
        )

    def register_health_check(self, component_name: str, check_function: Callable[[], ComponentHealth]):
        """
        Registriert Health-Check-Funktion für Komponente
        
        Args:
            component_name: Name der Komponente
            check_function: Funktion die ComponentHealth zurückgibt
        """
        self._health_check_functions[component_name] = check_function
        self.logger.info(f"Health-Check registriert: {component_name}")

    def perform_health_check(self, detailed: bool = False) -> SystemHealth:
        """
        Führt umfassenden System-Health-Check durch
        
        Args:
            detailed: Ob detaillierte Checks durchgeführt werden sollen
            
        Returns:
            SystemHealth: Umfassender Gesundheitsstatus
        """
        start_time = time.time()
        
        with self._health_lock:
            try:
                system_health = SystemHealth(overall_status=HealthStatus.UNKNOWN)
                
                # System-Level Checks
                if self.config.enable_system_monitoring:
                    system_components = self._check_system_health(detailed)
                    system_health.components.update(system_components)
                
                # Service-Level Checks
                if self.config.enable_service_monitoring:
                    service_components = self._check_service_health(detailed)
                    system_health.components.update(service_components)
                
                # Performance Checks
                if self.config.enable_performance_monitoring:
                    performance_components = self._check_performance_health(detailed)
                    system_health.components.update(performance_components)
                
                # Registered Component Checks
                registered_components = self._check_registered_components()
                system_health.components.update(registered_components)
                
                # Overall Status bestimmen
                system_health.overall_status = self._determine_overall_status(system_health.components)
                
                # Active Alerts sammeln
                system_health.active_alerts = self._collect_active_alerts(system_health.components)
                
                # System-Level Metriken
                system_health.system_metrics = self._collect_system_metrics()
                
                # Timing
                system_health.check_duration = time.time() - start_time
                
                # Alert-Processing
                self._process_alerts(system_health)
                
                # History-Update
                self._update_health_history(system_health)
                
                # Timestamps aktualisieren
                if detailed:
                    self._last_detailed_check = time.time()
                else:
                    self._last_health_check = time.time()
                
                self.logger.info(
                    "Health-Check abgeschlossen",
                    extra={
                        'overall_status': system_health.overall_status.value,
                        'components_checked': len(system_health.components),
                        'active_alerts': len(system_health.active_alerts),
                        'check_duration': system_health.check_duration,
                        'detailed': detailed
                    }
                )
                
                return system_health
                
            except Exception as e:
                self.logger.error(f"Health-Check fehlgeschlagen: {str(e)}", exc_info=True)
                
                # Fallback Health-Status bei Fehlern
                return SystemHealth(
                    overall_status=HealthStatus.CRITICAL,
                    check_duration=time.time() - start_time,
                    active_alerts=[HealthAlert(
                        alert_id=f"health_check_error_{int(time.time())}",
                        component="health_controller",
                        severity=AlertSeverity.CRITICAL,
                        message=f"Health-Check Fehler: {str(e)}"
                    )]
                )

    def _check_system_health(self, detailed: bool = False) -> Dict[str, ComponentHealth]:
        """Überprüft System-Ressourcen Health"""
        components = {}
        
        try:
            # CPU Health
            cpu_component = self._check_cpu_health()
            components['cpu'] = cpu_component
            
            # Memory Health
            memory_component = self._check_memory_health()
            components['memory'] = memory_component
            
            # Disk Health
            disk_component = self._check_disk_health()
            components['disk'] = disk_component
            
            if detailed:
                # Network Health (nur bei detaillierten Checks)
                network_component = self._check_network_health()
                components['network'] = network_component
                
                # Process Health
                process_component = self._check_process_health()
                components['processes'] = process_component
            
        except Exception as e:
            self.logger.error(f"System Health-Check fehlgeschlagen: {str(e)}")
            components['system_error'] = ComponentHealth(
                component_name="system_error",
                status=HealthStatus.CRITICAL,
                error_message=str(e)
            )
        
        return components

    def _check_cpu_health(self) -> ComponentHealth:
        """Überprüft CPU-Gesundheit"""
        try:
            # CPU-Auslastung über kurzen Zeitraum messen
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Status bestimmen
            if cpu_percent >= self.config.cpu_threshold_critical:
                status = HealthStatus.CRITICAL
            elif cpu_percent >= self.config.cpu_threshold_warning:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            component = ComponentHealth(
                component_name="cpu",
                status=status
            )
            
            # Metriken hinzufügen
            component.add_metric(HealthMetric(
                name="cpu_usage_percent",
                value=cpu_percent,
                status=status,
                threshold=self.config.cpu_threshold_warning,
                unit="%",
                description="CPU-Auslastung"
            ))
            
            component.add_metric(HealthMetric(
                name="cpu_count",
                value=cpu_count,
                status=HealthStatus.HEALTHY,
                unit="cores",
                description="Anzahl CPU-Kerne"
            ))
            
            if cpu_freq:
                component.add_metric(HealthMetric(
                    name="cpu_frequency",
                    value=cpu_freq.current,
                    status=HealthStatus.HEALTHY,
                    unit="MHz",
                    description="Aktuelle CPU-Frequenz"
                ))
            
            # Alert bei hoher CPU-Last
            if status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                alert = HealthAlert(
                    alert_id=f"cpu_high_{int(time.time())}",
                    component="cpu",
                    severity=AlertSeverity.CRITICAL if status == HealthStatus.CRITICAL else AlertSeverity.WARNING,
                    message=f"Hohe CPU-Auslastung: {cpu_percent:.1f}%",
                    details={'cpu_percent': cpu_percent, 'threshold': self.config.cpu_threshold_warning}
                )
                component.add_alert(alert)
            
            return component
            
        except Exception as e:
            return ComponentHealth(
                component_name="cpu",
                status=HealthStatus.UNKNOWN,
                error_message=str(e)
            )

    def _check_memory_health(self) -> ComponentHealth:
        """Überprüft Memory-Gesundheit"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_percent = memory.percent
            
            # Status bestimmen
            if memory_percent >= self.config.memory_threshold_critical:
                status = HealthStatus.CRITICAL
            elif memory_percent >= self.config.memory_threshold_warning:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            component = ComponentHealth(
                component_name="memory",
                status=status
            )
            
            # Memory Metriken
            component.add_metric(HealthMetric(
                name="memory_usage_percent",
                value=memory_percent,
                status=status,
                threshold=self.config.memory_threshold_warning,
                unit="%",
                description="RAM-Auslastung"
            ))
            
            component.add_metric(HealthMetric(
                name="memory_total_gb",
                value=round(memory.total / (1024**3), 2),
                status=HealthStatus.HEALTHY,
                unit="GB",
                description="Gesamter RAM"
            ))
            
            component.add_metric(HealthMetric(
                name="memory_available_gb",
                value=round(memory.available / (1024**3), 2),
                status=HealthStatus.HEALTHY,
                unit="GB",
                description="Verfügbarer RAM"
            ))
            
            # Swap-Metriken
            if swap.total > 0:
                component.add_metric(HealthMetric(
                    name="swap_usage_percent",
                    value=swap.percent,
                    status=HealthStatus.HEALTHY if swap.percent < 50 else HealthStatus.DEGRADED,
                    threshold=50.0,
                    unit="%",
                    description="Swap-Auslastung"
                ))
            
            # Alert bei hoher Memory-Nutzung
            if status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                alert = HealthAlert(
                    alert_id=f"memory_high_{int(time.time())}",
                    component="memory",
                    severity=AlertSeverity.CRITICAL if status == HealthStatus.CRITICAL else AlertSeverity.WARNING,
                    message=f"Hohe Memory-Auslastung: {memory_percent:.1f}%",
                    details={'memory_percent': memory_percent, 'available_gb': round(memory.available / (1024**3), 2)}
                )
                component.add_alert(alert)
            
            return component
            
        except Exception as e:
            return ComponentHealth(
                component_name="memory",
                status=HealthStatus.UNKNOWN,
                error_message=str(e)
            )

    def _check_disk_health(self) -> ComponentHealth:
        """Überprüft Disk-Gesundheit"""
        try:
            # Hauptpartition prüfen
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Status bestimmen
            if disk_percent >= self.config.disk_threshold_critical:
                status = HealthStatus.CRITICAL
            elif disk_percent >= self.config.disk_threshold_warning:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            component = ComponentHealth(
                component_name="disk",
                status=status
            )
            
            # Disk Metriken
            component.add_metric(HealthMetric(
                name="disk_usage_percent",
                value=round(disk_percent, 2),
                status=status,
                threshold=self.config.disk_threshold_warning,
                unit="%",
                description="Festplatten-Auslastung"
            ))
            
            component.add_metric(HealthMetric(
                name="disk_total_gb",
                value=round(disk_usage.total / (1024**3), 2),
                status=HealthStatus.HEALTHY,
                unit="GB",
                description="Gesamter Speicherplatz"
            ))
            
            component.add_metric(HealthMetric(
                name="disk_free_gb",
                value=round(disk_usage.free / (1024**3), 2),
                status=HealthStatus.HEALTHY,
                unit="GB",
                description="Freier Speicherplatz"
            ))
            
            # Disk I/O Statistiken
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    component.add_metric(HealthMetric(
                        name="disk_read_mb",
                        value=round(disk_io.read_bytes / (1024**2), 2),
                        status=HealthStatus.HEALTHY,
                        unit="MB",
                        description="Gelesene Daten seit Start"
                    ))
                    
                    component.add_metric(HealthMetric(
                        name="disk_write_mb",
                        value=round(disk_io.write_bytes / (1024**2), 2),
                        status=HealthStatus.HEALTHY,
                        unit="MB",
                        description="Geschriebene Daten seit Start"
                    ))
            except:
                pass  # Disk I/O nicht verfügbar
            
            # Alert bei wenig Speicherplatz
            if status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                alert = HealthAlert(
                    alert_id=f"disk_full_{int(time.time())}",
                    component="disk",
                    severity=AlertSeverity.CRITICAL if status == HealthStatus.CRITICAL else AlertSeverity.WARNING,
                    message=f"Wenig Speicherplatz: {disk_percent:.1f}% belegt",
                    details={'disk_percent': disk_percent, 'free_gb': round(disk_usage.free / (1024**3), 2)}
                )
                component.add_alert(alert)
            
            return component
            
        except Exception as e:
            return ComponentHealth(
                component_name="disk",
                status=HealthStatus.UNKNOWN,
                error_message=str(e)
            )

    def _check_service_health(self, detailed: bool = False) -> Dict[str, ComponentHealth]:
        """Überprüft Service-Gesundheit"""
        components = {}
        
        try:
            # Container-Services prüfen
            services_to_check = [
                ('document_service', 'DocumentService'),
                ('embedding_service', 'EmbeddingService'),
                ('search_service', 'SearchService'),
                ('chat_service', 'ChatService'),
                ('pipeline_controller', 'RAGPipelineController'),
                ('session_controller', 'SessionController')
            ]
            
            for service_name, service_class_name in services_to_check:
                try:
                    component = self._check_individual_service_health(service_name, service_class_name, detailed)
                    components[service_name] = component
                except Exception as e:
                    components[service_name] = ComponentHealth(
                        component_name=service_name,
                        status=HealthStatus.UNKNOWN,
                        error_message=str(e)
                    )
        
        except Exception as e:
            self.logger.error(f"Service Health-Check fehlgeschlagen: {str(e)}")
            components['service_error'] = ComponentHealth(
                component_name="service_error",
                status=HealthStatus.CRITICAL,
                error_message=str(e)
            )
        
        return components

    def _check_individual_service_health(self, service_name: str, service_class_name: str, detailed: bool) -> ComponentHealth:
        """Überprüft einzelnen Service"""
        try:
            # Service aus Container holen
            service = self._container.get_service_by_name(service_class_name)
            
            if not service:
                return ComponentHealth(
                    component_name=service_name,
                    status=HealthStatus.CRITICAL,
                    error_message="Service nicht im Container gefunden"
                )
            
            component = ComponentHealth(
                component_name=service_name,
                status=HealthStatus.HEALTHY
            )
            
            # Service-spezifische Health-Checks
            if hasattr(service, 'get_service_health'):
                service_health = service.get_service_health()
                
                # Status aus Service-Health ableiten
                service_status = service_health.get('service_status', 'unknown')
                if service_status == 'healthy':
                    component.status = HealthStatus.HEALTHY
                elif service_status == 'degraded':
                    component.status = HealthStatus.DEGRADED
                elif service_status == 'unhealthy':
                    component.status = HealthStatus.UNHEALTHY
                else:
                    component.status = HealthStatus.UNKNOWN
                
                # Service-Metriken extrahieren
                if 'performance_metrics' in service_health:
                    perf_metrics = service_health['performance_metrics']
                    
                    # Response Time Metrik
                    if 'avg_processing_time' in perf_metrics:
                        avg_time = perf_metrics['avg_processing_time']
                        time_status = HealthStatus.HEALTHY
                        
                        if avg_time >= self.config.response_time_critical:
                            time_status = HealthStatus.CRITICAL
                        elif avg_time >= self.config.response_time_warning:
                            time_status = HealthStatus.DEGRADED
                        
                        component.add_metric(HealthMetric(
                            name=f"{service_name}_response_time",
                            value=round(avg_time, 3),
                            status=time_status,
                            threshold=self.config.response_time_warning,
                            unit="seconds",
                            description=f"Durchschnittliche Response-Zeit"
                        ))
                    
                    # Error Rate Metrik
                    if 'error_rate' in perf_metrics:
                        error_rate = perf_metrics['error_rate']
                        error_status = HealthStatus.HEALTHY
                        
                        if error_rate >= self.config.error_rate_critical:
                            error_status = HealthStatus.CRITICAL
                        elif error_rate >= self.config.error_rate_warning:
                            error_status = HealthStatus.DEGRADED
                        
                        component.add_metric(HealthMetric(
                            name=f"{service_name}_error_rate",
                            value=round(error_rate, 4),
                            status=error_status,
                            threshold=self.config.error_rate_warning,
                            unit="ratio",
                            description=f"Fehlerrate"
                        ))
                
                # Alerts aus Service-Health
                if 'alerts' in service_health:
                    for alert_data in service_health['alerts']:
                        alert = HealthAlert(
                            alert_id=f"{service_name}_{alert_data.get('type', 'unknown')}_{int(time.time())}",
                            component=service_name,
                            severity=AlertSeverity(alert_data.get('severity', 'warning')),
                            message=alert_data.get('message', 'Service Alert'),
                            details=alert_data
                        )
                        component.add_alert(alert)
            
            elif hasattr(service, 'health_check'):
                # Einfacher Health-Check
                is_healthy = service.health_check()
                component.status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
                
                component.add_metric(HealthMetric(
                    name=f"{service_name}_available",
                    value=is_healthy,
                    status=component.status,
                    description=f"Service Verfügbarkeit"
                ))
            
            else:
                # Service ohne Health-Check - als verfügbar betrachten
                component.add_metric(HealthMetric(
                    name=f"{service_name}_loaded",
                    value=True,
                    status=HealthStatus.HEALTHY,
                    description=f"Service geladen"
                ))
            
            return component
            
        except Exception as e:
            return ComponentHealth(
                component_name=service_name,
                status=HealthStatus.CRITICAL,
                error_message=str(e)
            )

    def _determine_overall_status(self, components: Dict[str, ComponentHealth]) -> HealthStatus:
        """Bestimmt Overall-Status basierend auf Component-Status"""
        if not components:
            return HealthStatus.UNKNOWN
        
        statuses = [comp.status for comp in components.values()]
        
        # Critical wenn irgendeine Komponente critical ist
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        
        # Unhealthy wenn mehrere Komponenten unhealthy sind
        unhealthy_count = statuses.count(HealthStatus.UNHEALTHY)
        if unhealthy_count >= 2:
            return HealthStatus.UNHEALTHY
        elif unhealthy_count == 1:
            return HealthStatus.DEGRADED
        
        # Degraded wenn irgendeine Komponente degraded ist
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # Healthy wenn alle Komponenten healthy sind
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        
        # Default: Degraded bei gemischten Status
        return HealthStatus.DEGRADED

    def get_health_summary(self) -> Dict[str, Any]:
        """
        Liefert zusammengefasste Gesundheitsinformationen
        
        Returns:
            Dict mit Health-Summary
        """
        try:
            # Aktuellen Health-Check durchführen
            current_health = self.perform_health_check()
            
            # Summary erstellen
            summary = {
                'overall_status': current_health.overall_status.value,
                'timestamp': current_health.timestamp,
                'check_duration': current_health.check_duration,
                'components': {
                    name: {
                        'status': comp.status.value,
                        'metrics_count': len(comp.metrics),
                        'active_alerts': len(comp.get_active_alerts()),
                        'last_check': comp.last_check,
                        'error': comp.error_message
                    }
                    for name, comp in current_health.components.items()
                },
                'alerts': {
                    'total': len(current_health.active_alerts),
                    'by_severity': self._count_alerts_by_severity(current_health.active_alerts),
                    'recent': [
                        {
                            'component': alert.component,
                            'severity': alert.severity.value,
                            'message': alert.message,
                            'age_minutes': (time.time() - alert.timestamp) / 60
                        }
                        for alert in current_health.active_alerts[-5:]  # Letzten 5 Alerts
                    ]
                },
                'system_info': self._system_info,
                'controller_stats': {
                    'health_checks_performed': len(self._health_history),
                    'last_check_time': self._last_health_check,
                    'last_detailed_check': self._last_detailed_check,
                    'registered_checks': len(self._health_check_functions)
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Health-Summary Fehler: {str(e)}")
            return {
                'overall_status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

    def cleanup(self):
        """Cleanup-Routine für Health Controller"""
        try:
            # History bereinigen
            self._cleanup_old_data()
            
            # Memory cleanup
            self._health_history.clear()
            self._metric_history.clear()
            self._active_alerts.clear()
            
            self.logger.info("Health Controller cleanup abgeschlossen")
            
        except Exception as e:
            self.logger.error(f"Health Controller cleanup fehlgeschlagen: {str(e)}")


# Factory-Funktionen
def create_health_controller(config: Optional[Dict[str, Any]] = None) -> HealthController:
    """Factory-Funktion für Health Controller"""
    if config is None:
        app_config = get_config()
        health_config_dict = app_config.get('health', {})
    else:
        health_config_dict = config
    
    health_config = HealthConfig(**health_config_dict)
    return HealthController(health_config)


# Container-Registrierung
def register_health_controller():
    """Registriert Health Controller im DI-Container"""
    container = get_container()
    
    container.register_singleton(
        HealthController,
        lambda: create_health_controller()
    )
    
    logger.info("Health Controller im Container registriert")