#!/usr/bin/env python3
"""
Service Integration - Minimale Version für App-Start
"""
import time
from typing import Dict, Any, List, Optional
from core import get_logger

class ServiceIntegrator:
    """Minimaler ServiceIntegrator für sofortigen App-Start"""
    
    def __init__(self, config=None):
        self.logger = get_logger(__name__)
        self.config = config
        self.services = {}
        self._start_time = time.time()
        self.logger.info("ServiceIntegrator initialisiert")
    
    def register_service(self, name: str, service_instance: Any) -> None:
        """Registriert einen Service"""
        self.services[name] = service_instance
        self.logger.info(f"Service '{name}' registriert")
    
    def get_service(self, name: str) -> Optional[Any]:
        """Holt Service-Instanz"""
        return self.services.get(name)
    
    def health_check(self) -> Dict[str, Any]:
        """Health-Check"""
        return {
            'status': 'healthy',
            'services': len(self.services),
            'uptime': time.time() - self._start_time
        }

# Export
__all__ = ['ServiceIntegrator']

# =============================================================================
# FACTORY FUNCTIONS - KRITISCHER HOTFIX
# =============================================================================

_service_integrator_instance = None

def get_service_integrator() -> ServiceIntegrator:
    """
    Factory-Funktion für ServiceIntegrator-Singleton
    
    Returns:
        ServiceIntegrator: Globale Integrator-Instanz
    """
    global _service_integrator_instance
    
    if _service_integrator_instance is None:
        try:
            from core import get_logger, get_config
            logger = get_logger("service_integration.factory")
            logger.info("Erstelle ServiceIntegrator-Instanz...")
            
            _service_integrator_instance = ServiceIntegrator()
            logger.info("✅ ServiceIntegrator-Instanz erstellt")
            
        except Exception as e:
            # Fallback bei Initialisierung-Fehlern
            print(f"⚠️ ServiceIntegrator-Initialisierung fehlgeschlagen: {e}")
            
            class ServiceIntegratorFallback:
                def __init__(self):
                    pass
                def get_service_status(self):
                    return {"status": "fallback"}
                def validate_service_integration(self):
                    return True
            
            _service_integrator_instance = ServiceIntegratorFallback()
    
    return _service_integrator_instance


def validate_service_integration() -> bool:
    """
    Validiert Service-Integration
    
    Returns:
        bool: True wenn Integration OK
    """
    try:
        integrator = get_service_integrator()
        return hasattr(integrator, 'get_service_status')
    except Exception:
        return False


# ServiceStatus und ServiceEventType für Kompatibilität
class ServiceStatus:
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceEventType:
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
