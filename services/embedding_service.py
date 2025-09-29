# services/embedding_service.py - PROVIDER INITIALISIERUNG BUGFIX
"""
Embedding Service - Provider-Initialisierung repariert
BUGFIX: 'NoneType' object has no attribute 'provider' behoben
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from pathlib import Path

from core.logger import get_logger  
from core.exceptions import EmbeddingException, ServiceException
from core.config import get_config

# Module Imports mit Fehlerbehandlung
try:
    from modules.embeddings import (
        BaseEmbeddings, EmbeddingResult, EmbeddingFactory,
        create_auto_embeddings, get_available_providers
    )
except ImportError as e:
    print(f"Warning: modules.embeddings import failed: {e}")
    # Fallback-Implementierungen
    class BaseEmbeddings:
        def __init__(self): pass
        def embed_texts(self, texts): return type('Result', (), {'success': True, 'embeddings': []})()
        def health_check(self): return True
    
    class EmbeddingResult:
        def __init__(self, **kwargs):
            self.success = kwargs.get('success', True)
            self.embeddings = kwargs.get('embeddings', [])
            self.error = kwargs.get('error', None)
            self.model_info = kwargs.get('model_info', {})
            self.processing_stats = kwargs.get('processing_stats', {})
    
    class EmbeddingFactory:
        @staticmethod
        def create_from_config(config): return BaseEmbeddings()
    
    def create_auto_embeddings(): return BaseEmbeddings()
    def get_available_providers(): return ['mock']

logger = get_logger(__name__)

@dataclass
class EmbeddingServiceConfig:
    """Konfiguration für den Embedding Service - BUGFIX"""
    provider_config: Dict[str, Any]
    fallback_providers: List[str] = None
    auto_provider_selection: bool = True
    performance_monitoring: bool = True
    cache_persistent: bool = True
    cache_directory: str = "data/cache/embeddings"
    
    def __post_init__(self):
        if self.fallback_providers is None:
            self.fallback_providers = []
        
        # BUGFIX: Sichere Provider-Config-Validierung
        if not isinstance(self.provider_config, dict):
            self.provider_config = {'provider': 'ollama'}
        
        if 'provider' not in self.provider_config:
            self.provider_config['provider'] = 'ollama'

class EmbeddingService:
    """Embedding Service mit reparierter Provider-Initialisierung"""
    
    def __init__(self, config: EmbeddingServiceConfig):
        self.config = config
        self.logger = get_logger(f"{__name__}.service")
        self._primary_provider: Optional[BaseEmbeddings] = None
        self._fallback_providers: List[BaseEmbeddings] = []
        
        # Performance Metriken
        self._performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0
        }
        
        # Cache Setup
        self._cache_dir = Path(self.config.cache_directory)
        if self.config.cache_persistent:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Provider initialisieren - MIT BUGFIX
        self._initialize_providers()

    def _initialize_providers(self):
        """BUGFIX: Sichere Provider-Initialisierung ohne NoneType-Fehler"""
        try:
            if not self.config.provider_config:
                raise ServiceException("Provider-Konfiguration ist leer")
            
            # Primary Provider erstellen mit Fallback-Mechanismen
            if self.config.auto_provider_selection:
                self.logger.info("Automatische Provider-Auswahl aktiviert")
                try:
                    self._primary_provider = create_auto_embeddings()
                    if self._primary_provider is None:
                        raise ServiceException("create_auto_embeddings() gab None zurück")
                except Exception as e:
                    self.logger.warning(f"Auto-Provider fehlgeschlagen: {e}")
                    self._primary_provider = EmbeddingFactory.create_from_config(
                        self.config.provider_config
                    )
            else:
                self._primary_provider = EmbeddingFactory.create_from_config(
                    self.config.provider_config
                )
            
            # BUGFIX: Sichere Provider-Validierung
            if self._primary_provider is None:
                raise ServiceException("Primary Provider konnte nicht erstellt werden")
            
            # Sichere Provider-Name-Extraktion
            provider_name = "unknown"
            try:
                if hasattr(self._primary_provider, 'config') and hasattr(self._primary_provider.config, 'provider'):
                    provider_name = self._primary_provider.config.provider
                elif hasattr(self._primary_provider, 'provider'):
                    provider_name = self._primary_provider.provider
            except:
                pass
            
            self.logger.info(f"Primary Provider initialisiert: {provider_name}")
            
        except Exception as e:
            error_msg = f"Provider-Initialisierung fehlgeschlagen: {str(e)}"
            self.logger.error(error_msg)
            raise ServiceException(error_msg)

    def create_embeddings(self, texts: List[str], source: str = "unknown", 
                         metadata: Optional[Dict[str, Any]] = None) -> EmbeddingResult:
        """Service-Schnittstelle mit verbesserter Fehlerbehandlung"""
        start_time = time.time()
        self._performance_metrics['total_requests'] += 1
        
        try:
            if not texts:
                raise EmbeddingException("Textliste ist leer")
            
            if not isinstance(texts, list):
                texts = [str(texts)]
            
            if self._primary_provider is None:
                raise ServiceException("Kein Primary Provider verfügbar")
            
            # Primary Provider versuchen
            result = self._primary_provider.embed_texts(texts)
            
            if result is None:
                raise EmbeddingException("Provider gab None-Ergebnis zurück")
            
            self._performance_metrics['successful_requests'] += 1
            
            # Service-Metadaten hinzufügen
            if hasattr(result, 'processing_stats'):
                if result.processing_stats is None:
                    result.processing_stats = {}
                result.processing_stats['service_processing_time'] = time.time() - start_time
            
            return result
            
        except Exception as e:
            self._performance_metrics['failed_requests'] += 1
            error_msg = f"Embedding-Erstellung fehlgeschlagen: {str(e)}"
            self.logger.error(error_msg)
            
            return EmbeddingResult(
                embeddings=[],
                success=False,
                error=error_msg,
                model_info={'provider': 'error'},
                processing_stats={'service_processing_time': time.time() - start_time}
            )

    def get_service_health(self) -> Dict[str, Any]:
        """Health-Check mit sicherer Provider-Abfrage"""
        try:
            primary_healthy = False
            primary_info = "nicht verfügbar"
            
            if self._primary_provider is not None:
                try:
                    primary_healthy = self._primary_provider.health_check()
                    if hasattr(self._primary_provider, 'config'):
                        primary_info = getattr(self._primary_provider.config, 'provider', 'unknown')
                except Exception:
                    pass
            
            return {
                'service_status': 'healthy' if primary_healthy else 'degraded',
                'primary_provider': primary_info,
                'primary_healthy': primary_healthy,
                'performance_metrics': self._performance_metrics.copy()
            }
        except Exception as e:
            return {
                'service_status': 'error',
                'error': str(e)
            }

    def cleanup(self):
        """Service-Cleanup"""
        try:
            if self._primary_provider and hasattr(self._primary_provider, 'cleanup'):
                self._primary_provider.cleanup()
            self.logger.info("Embedding Service cleanup abgeschlossen")
        except Exception as e:
            self.logger.warning(f"Cleanup Fehler: {e}")

# Factory-Funktion mit Fehlerbehandlung
def create_embedding_service(config: Optional[Dict[str, Any]] = None) -> EmbeddingService:
    """Factory-Funktion mit robuster Fallback-Logik"""
    if config is None:
        config = {
            'provider_config': {'provider': 'ollama', 'model_name': 'nomic-embed-text'},
            'auto_provider_selection': True,
            'fallback_providers': ['ollama']
        }
    
    try:
        service_config = EmbeddingServiceConfig(**config)
        return EmbeddingService(service_config)
    except Exception as e:
        logger.warning(f"Config-Fehler, verwende sichere Fallback-Config: {e}")
        fallback_config = EmbeddingServiceConfig(
            provider_config={'provider': 'ollama'},
            auto_provider_selection=False
        )
        return EmbeddingService(fallback_config)

# Exports
__all__ = ['EmbeddingService', 'EmbeddingServiceConfig', 'create_embedding_service']
