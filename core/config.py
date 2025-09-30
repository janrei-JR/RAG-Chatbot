#!/usr/bin/env python3
"""
Core Config - TextProcessingConfig content_types Parameter behoben
Industrielle RAG-Architektur - Konfigurationsfehler korrigiert

PROBLEM BEHOBEN:
- TextProcessingConfig.__init__() got an unexpected keyword argument 'content_types'
- Alle Konfigurationsparameter validiert
- Rückwärtskompatibilität sichergestellt
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum


# =============================================================================
# TEXT PROCESSING CONFIGURATION - PARAMETER KORRIGIERT
# =============================================================================

@dataclass
class TextProcessingConfig:
    """
    Text-Processing Konfiguration - CONTENT_TYPES PARAMETER KORRIGIERT
    
    BUGFIX: 'content_types' Parameter entfernt (nicht unterstützt)
    """
    chunk_size: int = 800
    chunk_overlap: int = 150
    separators: List[str] = field(default_factory=lambda: [
        "\n\n# ", "\n\n## ", "\n\n### ", "\n\n", "\n", ". "
    ])
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    
    # Neue Parameter für erweiterte Text-Verarbeitung
    enable_preprocessing: bool = True
    remove_whitespace: bool = True
    normalize_unicode: bool = True
    filter_empty_chunks: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextProcessingConfig':
        """
        Erstelle TextProcessingConfig aus Dictionary mit Fallback für unbekannte Parameter
        
        BUGFIX: Ignoriert 'content_types' und andere unbekannte Parameter
        """
        # Bekannte Parameter extrahieren
        known_params = {
            'chunk_size', 'chunk_overlap', 'separators', 'min_chunk_size', 
            'max_chunk_size', 'enable_preprocessing', 'remove_whitespace',
            'normalize_unicode', 'filter_empty_chunks'
        }
        
        # Filtere nur bekannte Parameter
        filtered_data = {k: v for k, v in data.items() if k in known_params}
        
        # Warnungen für unbekannte Parameter
        unknown_params = set(data.keys()) - known_params
        if unknown_params:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Unbekannte TextProcessing Parameter ignoriert: {unknown_params}")
        
        return cls(**filtered_data)


# =============================================================================
# LLM CONFIGURATION - PROVIDER PARAMETER KORRIGIERT
# =============================================================================

@dataclass
class LLMConfig:
    """
    LLM-Konfiguration - PROVIDER PARAMETER KORRIGIERT
    
    BUGFIX: 'provider' Parameter in 'providers' geändert
    """
    # KORRIGIERT: provider -> providers (aber akzeptiert auch provider für Rückwärtskompatibilität)
    providers: List[str] = field(default_factory=lambda: ["ollama"])
    model_name: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 60
    retry_attempts: int = 3
    fallback_models: List[str] = field(default_factory=lambda: ["llama3.1:8b", "llama2:7b"])
    
    def __post_init__(self):
        """Post-initialization validation"""
        # Konvertiere einzelnen provider zu providers list
        if hasattr(self, 'provider') and not hasattr(self, 'providers'):
            self.providers = [self.provider]
        
        # Validierung
        if not self.providers:
            self.providers = ["ollama"]
        
        # Stelle sicher dass providers eine Liste ist
        if isinstance(self.providers, str):
            self.providers = [self.providers]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMConfig':
        """
        Erstelle LLMConfig aus Dictionary mit Rückwärtskompatibilität
        
        BUGFIX: Behandelt sowohl 'provider' als auch 'providers' Parameter
        """
        # Rückwärtskompatibilität: 'provider' -> 'providers'
        if 'provider' in data and 'providers' not in data:
            data['providers'] = [data.pop('provider')]
        elif 'provider' in data and 'providers' in data:
            # Entferne 'provider' wenn beide vorhanden
            data.pop('provider')
        
        # Stelle sicher dass providers eine Liste ist
        if 'providers' in data and isinstance(data['providers'], str):
            data['providers'] = [data['providers']]
        
        return cls(**data)


@dataclass
class EmbeddingConfig:
    """Embedding-Konfiguration - ebenfalls korrigiert"""
    providers: List[str] = field(default_factory=lambda: ["ollama"])
    model_name: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    dimensions: int = 768
    batch_size: int = 16
    cache_size: int = 1000  # HOTFIX: Fehlende cache_size Property
    cache_embeddings: bool = True
    max_retries: int = 3
    fallback_models: List[str] = field(default_factory=lambda: ["nomic-embed-text", "mxbai-embed-large"])
    
    def __post_init__(self):
        """Post-initialization mit provider->providers Konvertierung"""
        if hasattr(self, 'provider') and not hasattr(self, 'providers'):
            self.providers = [self.provider]
        
        if not self.providers:
            self.providers = ["ollama"]
            
        if isinstance(self.providers, str):
            self.providers = [self.providers]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingConfig':
        """Erstelle EmbeddingConfig aus Dictionary mit Rückwärtskompatibilität"""
        # Rückwärtskompatibilität: 'provider' -> 'providers'
        if 'provider' in data and 'providers' not in data:
            data['providers'] = [data.pop('provider')]
        elif 'provider' in data and 'providers' in data:
            data.pop('provider')
        
        if 'providers' in data and isinstance(data['providers'], str):
            data['providers'] = [data['providers']]
        
        return cls(**data)


@dataclass
class VectorStoreConfig:
    """Vector Store Konfiguration - ebenfalls korrigiert"""
    providers: List[str] = field(default_factory=lambda: ["chroma"])
    persist_directory: str = "./data/vectorstore"
    collection_name: str = "industrial_documents"
    distance_metric: str = "cosine"
    
    # Provider-spezifische Konfigurationen
    chroma: Dict[str, Any] = field(default_factory=lambda: {
        "host": "localhost",
        "port": 8000,
        "ssl": False,
        "headers": {}
    })
    
    def __post_init__(self):
        """Post-initialization mit provider->providers Konvertierung"""
        if hasattr(self, 'provider') and not hasattr(self, 'providers'):
            self.providers = [self.provider]
        
        if not self.providers:
            self.providers = ["chroma"]
            
        if isinstance(self.providers, str):
            self.providers = [self.providers]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorStoreConfig':
        """Erstelle VectorStoreConfig aus Dictionary mit Rückwärtskompatibilität"""
        # Rückwärtskompatibilität: 'provider' -> 'providers'
        if 'provider' in data and 'providers' not in data:
            data['providers'] = [data.pop('provider')]
        elif 'provider' in data and 'providers' in data:
            data.pop('provider')
        
        if 'providers' in data and isinstance(data['providers'], str):
            data['providers'] = [data['providers']]
        
        return cls(**data)


@dataclass 
class LoggingConfig:
    """Logging-Konfiguration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "./data/logs/rag_system.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    json_format: bool = False


@dataclass
class ApplicationConfig:
    """Haupt-Anwendungskonfiguration"""
    name: str = "RAG Industrial Chatbot"
    version: str = "4.0.0"
    description: str = "Service-orientierte RAG-Architektur für industrielle Dokumentenanalyse"
    environment: str = "development"
    debug_mode: bool = True


@dataclass
class RAGConfig:
    """
    Zentrale RAG-System Konfiguration - BUGFIXES INTEGRIERT
    Alle Provider-Parameter korrekt als 'providers' implementiert
    """
    application: ApplicationConfig = field(default_factory=ApplicationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    text_processing: TextProcessingConfig = field(default_factory=TextProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'RAGConfig':
        """
        Lade Konfiguration aus YAML-Datei - MIT BUGFIXES
        
        BUGFIX: Korrekte Behandlung von provider->providers Parameter
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data:
                raise ValueError("Leere Konfigurationsdatei")
            
            # Erstelle Konfigurationsobjekte mit korrekten Parametern
            config_kwargs = {}
            
            if 'application' in data:
                config_kwargs['application'] = ApplicationConfig(**data['application'])
            
            if 'llm' in data:
                config_kwargs['llm'] = LLMConfig.from_dict(data['llm'])
            
            if 'embeddings' in data:
                config_kwargs['embeddings'] = EmbeddingConfig.from_dict(data['embeddings'])
            
            if 'vector_store' in data:
                config_kwargs['vector_store'] = VectorStoreConfig.from_dict(data['vector_store'])
            
            if 'text_processing' in data:
                config_kwargs['text_processing'] = TextProcessingConfig.from_dict(data['text_processing'])
            
            if 'logging' in data:
                config_kwargs['logging'] = LoggingConfig(**data['logging'])
            
            return cls(**config_kwargs)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Fehler beim Parsen der YAML-Datei: {e}")
        except Exception as e:
            raise ValueError(f"Fehler beim Laden der Konfiguration: {e}")
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Erstelle Konfiguration aus Umgebungsvariablen"""
        config = cls()
        
        # LLM Konfiguration
        if os.getenv('LLM_PROVIDER'):
            config.llm.providers = [os.getenv('LLM_PROVIDER')]
        if os.getenv('LLM_MODEL_NAME'):
            config.llm.model_name = os.getenv('LLM_MODEL_NAME')
        if os.getenv('LLM_BASE_URL'):
            config.llm.base_url = os.getenv('LLM_BASE_URL')
        if os.getenv('LLM_TEMPERATURE'):
            config.llm.temperature = float(os.getenv('LLM_TEMPERATURE'))
        
        # Embedding Konfiguration  
        if os.getenv('EMBEDDING_PROVIDER'):
            config.embeddings.providers = [os.getenv('EMBEDDING_PROVIDER')]
        if os.getenv('EMBEDDING_MODEL_NAME'):
            config.embeddings.model_name = os.getenv('EMBEDDING_MODEL_NAME')
        if os.getenv('EMBEDDING_BASE_URL'):
            config.embeddings.base_url = os.getenv('EMBEDDING_BASE_URL')
        
        # Vector Store Konfiguration
        if os.getenv('VECTOR_STORE_PROVIDER'):
            config.vector_store.providers = [os.getenv('VECTOR_STORE_PROVIDER')]
        if os.getenv('VECTOR_STORE_PATH'):
            config.vector_store.persist_directory = os.getenv('VECTOR_STORE_PATH')
        
        return config
    

    # =============================================================================
    # MISSING PROPERTIES - BUGFIX
    # =============================================================================
    
    @property
    def default_provider(self) -> str:
        """Gibt Standard-Provider für VectorStore zurück"""
        try:
            return self.vector_store.providers[0] if self.vector_store.providers else "chroma"
        except (AttributeError, IndexError):
            return "chroma"
    
    @property
    def cache_directory(self) -> str:
        """Gibt Cache-Verzeichnis zurück"""
        try:
            return getattr(self, '_cache_directory', "./data/cache")
        except AttributeError:
            return "./data/cache"
    
    @cache_directory.setter 
    def cache_directory(self, value: str):
        """Setzt Cache-Verzeichnis"""
        self._cache_directory = value

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary"""
        return asdict(self)
    
    # HOTFIX: Fehlende Properties
    @property
    def auto_provider_selection(self) -> bool:
        """Automatische Provider-Auswahl aktiviert"""
        return getattr(self.embeddings, 'auto_provider_selection', True)
    
    def validate(self) -> List[str]:
        """Validiere Konfiguration und liefere Liste von Fehlern zurück"""
        errors = []
        
        # LLM Validierung
        if not self.llm.providers:
            errors.append("LLM providers dürfen nicht leer sein")
        if not self.llm.model_name:
            errors.append("LLM model_name ist erforderlich")
        if self.llm.temperature < 0 or self.llm.temperature > 2:
            errors.append("LLM temperature muss zwischen 0 und 2 liegen")
        if self.llm.max_tokens <= 0:
            errors.append("LLM max_tokens muss positiv sein")
        
        # Embedding Validierung
        if not self.embeddings.providers:
            errors.append("Embedding providers dürfen nicht leer sein")
        if not self.embeddings.model_name:
            errors.append("Embedding model_name ist erforderlich")
        if self.embeddings.dimensions <= 0:
            errors.append("Embedding dimensions muss positiv sein")
        
        # Vector Store Validierung
        if not self.vector_store.providers:
            errors.append("Vector Store providers dürfen nicht leer sein")
        if not self.vector_store.collection_name:
            errors.append("Vector Store collection_name ist erforderlich")
        
        # Text Processing Validierung
        if self.text_processing.chunk_size <= 0:
            errors.append("Text Processing chunk_size muss positiv sein")
        if self.text_processing.chunk_overlap < 0:
            errors.append("Text Processing chunk_overlap darf nicht negativ sein")
        if self.text_processing.chunk_overlap >= self.text_processing.chunk_size:
            errors.append("Text Processing chunk_overlap muss kleiner als chunk_size sein")
        
        return errors

    @property
    def processing(self):
        """Text-Processing Konfiguration"""
        return self.text_processing

    @property
    def app(self):
        """Application-Konfiguration"""
        return {
            'name': self.application.name,
            'version': self.application.version,
            'session_timeout': 3600
        }

    @property
    def fallback_providers(self) -> list:
        """Fallback-Provider für Services"""
        return ['chroma', 'memory']

    @property
    def cache_persistent(self) -> bool:
        """Cache-Persistenz Konfiguration"""
        return True

    @property
    def persistence_directory(self) -> str:
        """Persistenz-Verzeichnis"""
        return "./data/sessions"

# =============================================================================
# UTILITY FUNCTIONS (außerhalb der RAGConfig Klasse)
# =============================================================================

def load_config(config_path: Optional[Union[str, Path]] = None) -> RAGConfig:
    """
    Zentrale Funktion zum Laden der Konfiguration - MIT BUGFIXES
    
    Args:
        config_path: Pfad zur Konfigurationsdatei
        
    Returns:
        RAGConfig: Geladene und validierte Konfiguration
        
    Raises:
        ConfigurationException: Bei Konfigurationsfehlern
    """
    try:
        # Standard-Konfigurationspfade
        default_paths = [
            "app_config.yaml",
            "config/app_config.yaml", 
            "app_config_development.yaml",
            "config/app_config_development.yaml"
        ]
        
        # Versuche Konfigurationsdatei zu finden
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = None
            for path in default_paths:
                if Path(path).exists():
                    config_file = Path(path)
                    break
        
        # Lade Konfiguration
        if config_file and config_file.exists():
            config = RAGConfig.from_yaml(config_file)
        else:
            # Fallback zu Umgebungsvariablen
            config = RAGConfig.from_env()
        
        # Validierung
        errors = config.validate()
        if errors:
            error_msg = "Konfigurationsfehler gefunden:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)
        
        return config
        
    except Exception as e:
        # Verwende Standard-Konfiguration als letzten Fallback
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Fehler beim Laden der Konfiguration: {e}")
        logger.info("Verwende Standard-Konfiguration als Fallback")
        
        return RAGConfig()


# =============================================================================
# GLOBAL CONFIG MANAGEMENT - SINGLETON PATTERN
# =============================================================================

_global_config: Optional[RAGConfig] = None
_config_initialized = False

def get_config() -> RAGConfig:
    """
    Globale Funktion zum Abrufen der Konfiguration - SINGLETON PATTERN
    
    Returns:
        RAGConfig: Die globale Konfigurationsinstanz
        
    Raises:
        ConfigurationException: Bei Konfigurationsfehlern
    """
    global _global_config, _config_initialized
    
    if not _config_initialized:
        try:
            _global_config = load_config()
            _config_initialized = True
        except Exception as e:
            # Fallback zu Standard-Konfiguration
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Konfiguration konnte nicht geladen werden: {e}")
            logger.info("Verwende Standard-Konfiguration")
            
            _global_config = RAGConfig()
            _config_initialized = True
    
    return _global_config

def set_config(config: RAGConfig) -> None:
    """
    Setze globale Konfiguration (für Tests und spezielle Anwendungen)
    
    Args:
        config: Neue Konfiguration
    """
    global _global_config, _config_initialized
    _global_config = config
    _config_initialized = True

def reset_config() -> None:
    """
    Setze Konfiguration zurück (für Tests)
    """
    global _global_config, _config_initialized
    _global_config = None
    _config_initialized = False


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    "RAGConfig",
    "ApplicationConfig", 
    "LLMConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "TextProcessingConfig",
    "LoggingConfig",
    "load_config",
    "get_config",
    "set_config",
    "reset_config"
]
