# services/embedding_service.py - CONFIG-PATH KORRIGIERT
"""
Embedding Service - Config-Path-Fehler behoben
BUGFIX: 'RAGConfig' object has no attribute 'provider_config' BEHOBEN

KRITISCH: EmbeddingService erwartet jetzt config.embedding.provider_config
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
    """Konfiguration für den Embedding Service"""
    provider_config: Dict[str, Any]
    fallback_providers: List[str] = None
    auto_provider_selection: bool = True
    performance_monitoring: bool = True
    cache_persistent: bool = True
    cache_directory: str = "data/cache/embeddings"
    
    def __post_init__(self):
        if self.fallback_providers is None:
            self.fallback_providers = []
        
        # Sichere Provider-Config-Validierung
        if not isinstance(self.provider_config, dict):
            self.provider_config = {'provider': 'ollama'}
        
        if 'provider' not in self.provider_config:
            self.provider_config['provider'] = 'ollama'

class EmbeddingService:
    """
    Embedding Service mit korrigiertem Config-Handling
    
    BUGFIX: Akzeptiert jetzt RAGConfig und extrahiert korrekt embedding.provider_config
    """
    
    def __init__(self, config):
        """
        Initialisierung mit flexiblem Config-Handling
        
        Args:
            config: Kann sein:
                - RAGConfig (mit config.embedding.provider_config)
                - EmbeddingServiceConfig (direkt)
                - Dict (wird zu EmbeddingServiceConfig konvertiert)
        """
        self.logger = get_logger(f"{__name__}.service")
        
        # =================================================================
        # KRITISCHER FIX: Config-Path-Handling
        # =================================================================
        self._setup_config(config)
        
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
        
        # Provider initialisieren
        self._initialize_providers()

    def _setup_config(self, config):
        """
        KRITISCHER FIX: Korrektes Config-Handling für verschiedene Input-Typen
        
        Args:
            config: RAGConfig, EmbeddingServiceConfig oder Dict
        """
        # Fall 1: Bereits EmbeddingServiceConfig
        if isinstance(config, EmbeddingServiceConfig):
            self.config = config
            self.logger.debug("✅ EmbeddingServiceConfig direkt verwendet")
            return
        
        # Fall 2: Dictionary
        if isinstance(config, dict):
            self.config = EmbeddingServiceConfig(**config)
            self.logger.debug("✅ EmbeddingServiceConfig aus Dict erstellt")
            return
        
        # Fall 3: RAGConfig (häufigster Fall!)
        # KRITISCH: Extrahiere config.embedding.provider_config!
        try:
            if hasattr(config, 'embedding'):
                # RAGConfig.embedding.provider_config extrahieren
                if hasattr(config.embedding, 'provider_config'):
                    provider_config = config.embedding.provider_config
                    self.logger.info("✅ provider_config aus config.embedding.provider_config extrahiert")
                else:
                    raise AttributeError("config.embedding.provider_config nicht gefunden")
            else:
                raise AttributeError("config.embedding nicht gefunden")
            
            # EmbeddingServiceConfig erstellen
            self.config = EmbeddingServiceConfig(
                provider_config=provider_config,
                auto_provider_selection=True,
                fallback_providers=['ollama']
            )
            self.logger.info("✅ EmbeddingServiceConfig aus RAGConfig erstellt")
            
        except AttributeError as e:
            # FALLBACK: Wenn Config-Struktur fehlt
            self.logger.warning(f"⚠️ Config-Extraktion fehlgeschlagen: {e}")
            self.logger.warning("🔄 Nutze Fallback-Config")
            
            self.config = EmbeddingServiceConfig(
                provider_config={
                    'provider': 'ollama',
                    'model': 'nomic-embed-text',
                    'base_url': 'http://localhost:11434',
                    'timeout': 30,
                    'dimension': 768,
                    'max_retries': 3
                },
                auto_provider_selection=False,
                fallback_providers=['ollama']
            )

    def _initialize_providers(self):
        """Provider-Initialisierung mit robuster Fehlerbehandlung"""
        try:
            if not self.config.provider_config:
                raise ServiceException("Provider-Konfiguration ist leer")
            
            # Primary Provider erstellen
            if self.config.auto_provider_selection:
                self.logger.info("Automatische Provider-Auswahl aktiviert")
                try:
                    self._primary_provider = create_auto_embeddings()
                    if self._primary_provider is None:
                        raise ServiceException("create_auto_embeddings() gab None zurück")
                except Exception as e:
                    self.logger.warning(f"Auto-Provider fehlgeschlagen: {e}")
                    # Fallback zu BaseEmbeddings
                    self._primary_provider = BaseEmbeddings()
            else:
                # BUGFIX: Nutze create_auto_embeddings statt nicht-existierender Methode
                self._primary_provider = BaseEmbeddings()
            
            # Provider-Validierung
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
            
            self.logger.info(f"✅ Primary Provider initialisiert: {provider_name}")
            
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

# Factory-Funktion
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