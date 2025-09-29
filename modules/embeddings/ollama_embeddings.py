#!/usr/bin/env python3
"""
Ollama Embeddings für RAG Chatbot Industrial

Lokale Embedding-Erzeugung mit Ollama-Server für Cloud-unabhängige industrielle
RAG-Systeme. Optimiert für On-Premise-Deployments ohne externe API-Abhängigkeiten.

Features:
- Lokale Ollama-Modell-Integration (nomic-embed-text, mxbai-embed-large, etc.)
- Robuste Verbindungsverwaltung mit Retry-Mechanismen
- Automatische Modell-Download und -Management
- Performance-optimierte Batch-Verarbeitung
- Health-Checks und Monitoring für Produktionsumgebungen

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

import json
import time
import requests
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import urllib.parse

# Base Embeddings
from .base_embeddings import (
    BaseEmbeddings, EmbeddingProvider, EmbeddingStatus,
    EmbeddingResult, BatchEmbeddingResult
)

# Core-Komponenten
from core import (
    get_logger, RAGConfig, get_current_config,
    DocumentProcessingError, ValidationError,
    create_error_context, log_performance
)


# =============================================================================
# OLLAMA-SPEZIFISCHE KONFIGURATION UND DATENSTRUKTUREN
# =============================================================================

@dataclass
class OllamaModelInfo:
    """
    Informationen über verfügbare Ollama-Modelle
    
    Attributes:
        name (str): Modell-Name
        size (str): Modell-Größe (z.B. "435MB")
        digest (str): Modell-Hash
        modified_at (str): Letzte Änderung
        dimension (int): Embedding-Dimension (falls bekannt)
        context_length (int): Maximale Kontext-Länge
    """
    name: str
    size: str
    digest: str
    modified_at: str
    dimension: int = 0
    context_length: int = 2048


class OllamaEmbeddings(BaseEmbeddings):
    """
    Ollama-basierte Embedding-Implementierung für lokale KI-Systeme
    
    Unterstützt verschiedene Ollama-Embedding-Modelle:
    - nomic-embed-text (768 Dimensionen)
    - mxbai-embed-large (1024 Dimensionen) 
    - snowflake-arctic-embed (1024 Dimensionen)
    - jina/jina-embeddings-v2-base-en (768 Dimensionen)
    """
    
    # Standard-Embedding-Modelle mit bekannten Dimensionen
    SUPPORTED_MODELS = {
        'nomic-embed-text': {'dimension': 768, 'context_length': 2048},
        'mxbai-embed-large': {'dimension': 1024, 'context_length': 512},
        'snowflake-arctic-embed': {'dimension': 1024, 'context_length': 512},
        'jina/jina-embeddings-v2-base-en': {'dimension': 768, 'context_length': 8192},
        'sentence-transformers/all-MiniLM-L6-v2': {'dimension': 384, 'context_length': 256}
    }
    
    def __init__(self, 
                 config: RAGConfig = None,
                 model_name: str = None,
                 base_url: str = None):
        """
        Initialisiert Ollama Embeddings
        
        Args:
            config (RAGConfig): Konfiguration
            model_name (str): Ollama-Modell (default: nomic-embed-text)
            base_url (str): Ollama-Server URL (default: http://localhost:11434)
        """
        # Base-Klasse initialisieren
        super().__init__(config)
        
        # Provider-Eigenschaften setzen
        self.provider = EmbeddingProvider.OLLAMA
        
        # Ollama-spezifische Konfiguration
        self.base_url = base_url or getattr(self.config.embeddings, 'ollama_base_url', 'http://localhost:11434')
        self.model_name = model_name or getattr(self.config.embeddings, 'ollama_model', 'nomic-embed-text')
        
        # Modell-Eigenschaften aus bekannten Modellen laden
        if self.model_name in self.SUPPORTED_MODELS:
            model_info = self.SUPPORTED_MODELS[self.model_name]
            self.dimension = model_info['dimension']
            self.max_tokens = model_info['context_length']
        else:
            # Fallback-Werte für unbekannte Modelle
            self.dimension = 768
            self.max_tokens = 2048
        
        # HTTP-Session für Connection-Pooling
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'RAG-Industrial-Embeddings/4.0.0'
        })
        
        # Timeouts
        self.connect_timeout = getattr(self.config.embeddings, 'connect_timeout', 10.0)
        self.read_timeout = getattr(self.config.embeddings, 'read_timeout', 60.0)
        
        # Retry-Konfiguration
        self.max_retries = getattr(self.config.embeddings, 'max_retries', 3)
        self.retry_delay = getattr(self.config.embeddings, 'retry_delay', 1.0)
        
        # Ollama-spezifische Statistiken
        self._ollama_stats = {
            'server_requests': 0,
            'server_errors': 0,
            'model_pulls': 0,
            'connection_failures': 0,
            'retry_attempts': 0
        }
        
        # Initialisierung und Modell-Check
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialisiert Modell und lädt es bei Bedarf herunter"""
        try:
            # Server-Verbindung prüfen
            if not self._validate_connection():
                self.logger.warning(f"Ollama-Server nicht erreichbar: {self.base_url}")
                return
            
            # Verfügbare Modelle abrufen
            available_models = self._get_available_models()
            model_names = [model.name for model in available_models]
            
            if self.model_name not in model_names:
                self.logger.info(f"Modell {self.model_name} nicht verfügbar. Lade herunter...")
                
                if self._pull_model(self.model_name):
                    self.logger.info(f"Modell {self.model_name} erfolgreich geladen")
                else:
                    self.logger.error(f"Fehler beim Laden von Modell {self.model_name}")
            else:
                # Modell-Dimensionen aktualisieren falls verfügbar
                self._update_model_info(available_models)
                self.logger.info(f"Ollama-Modell verfügbar: {self.model_name} (Dim: {self.dimension})")
                
        except Exception as e:
            self.logger.error(f"Fehler bei Modell-Initialisierung: {str(e)}")
    
    # =============================================================================
    # ABSTRACT METHODS IMPLEMENTATION
    # =============================================================================
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Erstellt Embedding mit Ollama-API
        
        Args:
            text (str): Zu verarbeitender Text
            
        Returns:
            List[float]: Embedding-Vektor
        """
        url = f"{self.base_url}/api/embeddings"
        
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        
        # Retry-Mechanismus
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self._ollama_stats['server_requests'] += 1
                
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=(self.connect_timeout, self.read_timeout)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get('embedding', [])
                    
                    if not embedding:
                        raise DocumentProcessingError("Leeres Embedding von Ollama erhalten")
                    
                    # Dimension validieren und ggf. aktualisieren
                    if self.dimension != len(embedding):
                        self.logger.debug(f"Dimension aktualisiert: {self.dimension} -> {len(embedding)}")
                        self.dimension = len(embedding)
                    
                    return embedding
                
                elif response.status_code == 404:
                    # Modell nicht gefunden - versuche Download
                    if attempt == 0 and self._pull_model(self.model_name):
                        continue  # Retry mit geladenem Modell
                    else:
                        raise DocumentProcessingError(f"Ollama-Modell nicht verfügbar: {self.model_name}")
                
                else:
                    error_msg = f"Ollama-API Fehler: {response.status_code} - {response.text}"
                    raise DocumentProcessingError(error_msg)
                    
            except requests.exceptions.RequestException as e:
                last_exception = e
                self._ollama_stats['connection_failures'] += 1
                
                if attempt < self.max_retries:
                    self._ollama_stats['retry_attempts'] += 1
                    self.logger.warning(f"Ollama-Request fehlgeschlagen (Versuch {attempt + 1}): {str(e)}")
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    self._ollama_stats['server_errors'] += 1
                    break
            
            except Exception as e:
                last_exception = e
                self._ollama_stats['server_errors'] += 1
                break
        
        # Alle Versuche fehlgeschlagen
        raise DocumentProcessingError(
            f"Ollama-Embedding nach {self.max_retries + 1} Versuchen fehlgeschlagen: {str(last_exception)}"
        )
    
    def _validate_connection(self) -> bool:
        """
        Validiert Verbindung zum Ollama-Server
        
        Returns:
            bool: True wenn Server erreichbar
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(
                url, 
                timeout=(self.connect_timeout, 5.0)  # Kurzer Timeout für Health-Check
            )
            return response.status_code == 200
            
        except Exception as e:
            self.logger.debug(f"Ollama-Server Verbindungstest fehlgeschlagen: {str(e)}")
            return False
    
    # =============================================================================
    # OLLAMA-SPEZIFISCHE METHODEN
    # =============================================================================
    
    def _get_available_models(self) -> List[OllamaModelInfo]:
        """
        Holt Liste verfügbarer Modelle vom Ollama-Server
        
        Returns:
            List[OllamaModelInfo]: Verfügbare Modelle
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, timeout=(self.connect_timeout, 10.0))
            
            if response.status_code == 200:
                data = response.json()
                models = []
                
                for model_data in data.get('models', []):
                    model_info = OllamaModelInfo(
                        name=model_data.get('name', ''),
                        size=model_data.get('size', '0'),
                        digest=model_data.get('digest', ''),
                        modified_at=model_data.get('modified_at', '')
                    )
                    
                    # Bekannte Dimensionen hinzufügen
                    if model_info.name in self.SUPPORTED_MODELS:
                        known_info = self.SUPPORTED_MODELS[model_info.name]
                        model_info.dimension = known_info['dimension']
                        model_info.context_length = known_info['context_length']
                    
                    models.append(model_info)
                
                return models
            
            else:
                raise DocumentProcessingError(f"Fehler beim Abrufen der Modelle: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen verfügbarer Modelle: {str(e)}")
            return []
    
    def _pull_model(self, model_name: str) -> bool:
        """
        Lädt Modell vom Ollama-Server herunter
        
        Args:
            model_name (str): Name des Modells
            
        Returns:
            bool: True wenn erfolgreich geladen
        """
        try:
            self.logger.info(f"Lade Ollama-Modell herunter: {model_name}")
            
            url = f"{self.base_url}/api/pull"
            payload = {"name": model_name}
            
            # Längerer Timeout für Modell-Download
            response = self.session.post(
                url,
                json=payload,
                timeout=(self.connect_timeout, 600.0),  # 10 Minuten für Download
                stream=True
            )
            
            if response.status_code == 200:
                # Progress-Tracking für große Downloads
                for line in response.iter_lines():
                    if line:
                        try:
                            progress_data = json.loads(line.decode('utf-8'))
                            status = progress_data.get('status', '')
                            
                            if 'pulling' in status.lower():
                                self.logger.debug(f"Modell-Download: {status}")
                            elif 'success' in status.lower():
                                self._ollama_stats['model_pulls'] += 1
                                return True
                            elif 'error' in status.lower():
                                error_msg = progress_data.get('error', 'Unbekannter Fehler')
                                self.logger.error(f"Modell-Download Fehler: {error_msg}")
                                return False
                                
                        except json.JSONDecodeError:
                            continue
                
                return True  # Erfolgreich falls keine Fehler
            
            else:
                self.logger.error(f"Modell-Download fehlgeschlagen: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Fehler beim Modell-Download: {str(e)}")
            return False
    
    def _update_model_info(self, available_models: List[OllamaModelInfo]) -> None:
        """Aktualisiert Modell-Informationen basierend auf verfügbaren Modellen"""
        for model in available_models:
            if model.name == self.model_name and model.dimension > 0:
                if self.dimension != model.dimension:
                    self.logger.info(f"Modell-Dimension aktualisiert: {self.dimension} -> {model.dimension}")
                    self.dimension = model.dimension
                
                if model.context_length > 0:
                    self.max_tokens = model.context_length
                break
    
    # =============================================================================
    # PERFORMANCE UND MONITORING
    # =============================================================================
    
    @log_performance()
    def embed_batch_optimized(self, texts: List[str]) -> BatchEmbeddingResult:
        """
        Ollama-optimierte Batch-Verarbeitung mit Connection-Pooling
        
        Args:
            texts (List[str]): Zu verarbeitende Texte
            
        Returns:
            BatchEmbeddingResult: Batch-Ergebnis
        """
        if not texts:
            return BatchEmbeddingResult(
                results=[],
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                cached_requests=0,
                total_processing_time_ms=0.0
            )
        
        # Server-Status vor Batch-Verarbeitung prüfen
        if not self._validate_connection():
            raise DocumentProcessingError("Ollama-Server nicht erreichbar für Batch-Verarbeitung")
        
        # Standard-Batch-Verarbeitung mit optimierter Session
        return self.embed_batch(texts)
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Holt Server-Informationen vom Ollama-Server
        
        Returns:
            Dict[str, Any]: Server-Informationen
        """
        try:
            url = f"{self.base_url}/api/version"
            response = self.session.get(url, timeout=(self.connect_timeout, 5.0))
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Umfassender Health-Check für Produktionsumgebungen
        
        Returns:
            Dict[str, Any]: Health-Status
        """
        health_status = {
            'server_reachable': False,
            'model_available': False,
            'embedding_functional': False,
            'server_info': {},
            'available_models': [],
            'performance_ok': False,
            'overall_status': 'unhealthy'
        }
        
        try:
            # 1. Server-Erreichbarkeit
            health_status['server_reachable'] = self._validate_connection()
            
            if health_status['server_reachable']:
                # 2. Server-Informationen
                health_status['server_info'] = self.get_server_info()
                
                # 3. Verfügbare Modelle
                available_models = self._get_available_models()
                health_status['available_models'] = [m.name for m in available_models]
                
                # 4. Modell verfügbar
                health_status['model_available'] = self.model_name in health_status['available_models']
                
                # 5. Embedding-Funktionalität testen
                if health_status['model_available']:
                    try:
                        test_result = self.embed_text("Health check test", "health_check")
                        health_status['embedding_functional'] = test_result.is_valid
                        
                        # 6. Performance prüfen
                        health_status['performance_ok'] = test_result.processing_time_ms < 5000  # < 5s
                        
                    except Exception:
                        health_status['embedding_functional'] = False
                        health_status['performance_ok'] = False
            
            # Gesamt-Status bestimmen
            if (health_status['server_reachable'] and 
                health_status['model_available'] and 
                health_status['embedding_functional']):
                health_status['overall_status'] = 'healthy'
            elif health_status['server_reachable']:
                health_status['overall_status'] = 'degraded'
            else:
                health_status['overall_status'] = 'unhealthy'
                
        except Exception as e:
            health_status['error'] = str(e)
        
        return health_status
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Erweiterte Performance-Statistiken mit Ollama-spezifischen Metriken
        
        Returns:
            Dict[str, Any]: Detaillierte Performance-Daten
        """
        # Basis-Statistiken von Parent-Klasse
        stats = super().get_performance_statistics()
        
        # Ollama-spezifische Statistiken hinzufügen
        stats['ollama'] = self._ollama_stats.copy()
        
        # Server-Informationen
        stats['server'] = {
            'base_url': self.base_url,
            'connection_timeout': self.connect_timeout,
            'read_timeout': self.read_timeout,
            'max_retries': self.max_retries
        }
        
        # Fehlerrate berechnen
        total_requests = self._ollama_stats['server_requests']
        if total_requests > 0:
            stats['ollama']['error_rate'] = (
                self._ollama_stats['server_errors'] / total_requests
            )
            stats['ollama']['retry_rate'] = (
                self._ollama_stats['retry_attempts'] / total_requests
            )
        
        return stats
    
    def __del__(self):
        """Cleanup beim Objektabbau"""
        if hasattr(self, 'session'):
            self.session.close()


# =============================================================================
# OLLAMA EMBEDDINGS FACTORY
# =============================================================================

class OllamaEmbeddingsFactory:
    """Factory für verschiedene Ollama-Embedding-Konfigurationen"""
    
    @staticmethod
    def create_nomic_embeddings(config: RAGConfig = None, 
                               base_url: str = None) -> OllamaEmbeddings:
        """
        Erstellt Ollama Embeddings mit nomic-embed-text Modell
        
        Args:
            config (RAGConfig): Konfiguration
            base_url (str): Ollama-Server URL
            
        Returns:
            OllamaEmbeddings: Nomic-Embeddings
        """
        return OllamaEmbeddings(
            config=config,
            model_name='nomic-embed-text',
            base_url=base_url
        )
    
    @staticmethod
    def create_mxbai_embeddings(config: RAGConfig = None,
                               base_url: str = None) -> OllamaEmbeddings:
        """
        Erstellt Ollama Embeddings mit mxbai-embed-large Modell
        
        Args:
            config (RAGConfig): Konfiguration  
            base_url (str): Ollama-Server URL
            
        Returns:
            OllamaEmbeddings: MXBAI-Embeddings
        """
        return OllamaEmbeddings(
            config=config,
            model_name='mxbai-embed-large',
            base_url=base_url
        )
    
    @staticmethod
    def create_multilingual_embeddings(config: RAGConfig = None,
                                     base_url: str = None) -> OllamaEmbeddings:
        """
        Erstellt Ollama Embeddings mit mehrsprachigem Modell
        
        Args:
            config (RAGConfig): Konfiguration
            base_url (str): Ollama-Server URL
            
        Returns:
            OllamaEmbeddings: Mehrsprachige Embeddings
        """
        return OllamaEmbeddings(
            config=config,
            model_name='jina/jina-embeddings-v2-base-en',
            base_url=base_url
        )
    
    @staticmethod
    def create_lightweight_embeddings(config: RAGConfig = None,
                                    base_url: str = None) -> OllamaEmbeddings:
        """
        Erstellt leichtgewichtige Embeddings für Performance
        
        Args:
            config (RAGConfig): Konfiguration
            base_url (str): Ollama-Server URL
            
        Returns:
            OllamaEmbeddings: Lightweight Embeddings
        """
        return OllamaEmbeddings(
            config=config,
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            base_url=base_url
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Hauptklasse
    'OllamaEmbeddings',
    
    # Datenstrukturen
    'OllamaModelInfo',
    
    # Factory
    'OllamaEmbeddingsFactory'
]


if __name__ == "__main__":
    # Testing und Demonstration
    print("Ollama Embeddings - Lokale Embedding-Erzeugung")
    print("==============================================")
    
    # Ollama-Embeddings erstellen (erfordert laufenden Ollama-Server)
    try:
        embeddings = OllamaEmbeddings(
            model_name='nomic-embed-text',
            base_url='http://localhost:11434'
        )
        
        print(f"Ollama-Embeddings initialisiert: {embeddings}")
        
        # Health-Check durchführen
        health_status = embeddings.health_check()
        print(f"\nHealth-Check:")
        print(f"  Server erreichbar: {health_status['server_reachable']}")
        print(f"  Modell verfügbar: {health_status['model_available']}")
        print(f"  Embedding funktional: {health_status['embedding_functional']}")
        print(f"  Gesamt-Status: {health_status['overall_status']}")
        
        # Nur weitere Tests wenn Server gesund
        if health_status['overall_status'] in ['healthy', 'degraded']:
            
            # Einzelnes Embedding testen
            test_text = "Dies ist ein Test für lokale Ollama-Embeddings in industrieller Umgebung."
            result = embeddings.embed_text(test_text)
            
            print(f"\nEinzel-Embedding:")
            print(f"  Status: {result.status}")
            print(f"  Dimension: {result.dimension}")
            print(f"  Verarbeitungszeit: {result.processing_time_ms:.1f}ms")
            print(f"  Gültig: {result.is_valid}")
            
            # Cache-Test
            cached_result = embeddings.embed_text(test_text)  # Sollte aus Cache kommen
            print(f"  Cached Status: {cached_result.status}")
            
            # Batch-Test
            test_texts = [
                "Sicherheitshinweis: Vor Wartungsarbeiten Spannung freischalten.",
                "Technische Daten: 400V AC, 50Hz, 10A Nennstrom.",
                "Installation: Gerät in IP65-Schaltschrank montieren.",
                "Troubleshooting: Bei Alarm A001 Sensor prüfen."
            ]
            
            batch_result = embeddings.embed_batch(test_texts)
            print(f"\nBatch-Embedding:")
            print(f"  Gesamt Requests: {batch_result.total_requests}")
            print(f"  Erfolgreich: {batch_result.successful_requests}")
            print(f"  Erfolgsrate: {batch_result.success_rate:.1%}")
            print(f"  Durchschnittliche Zeit: {batch_result.average_processing_time_ms:.1f}ms")
            
            # Verfügbare Modelle anzeigen
            available_models = embeddings._get_available_models()
            if available_models:
                print(f"\nVerfügbare Modelle:")
                for model in available_models[:5]:  # Erste 5 Modelle
                    print(f"  {model.name} (Größe: {model.size})")
            
            # Performance-Statistiken
            stats = embeddings.get_performance_statistics()
            print(f"\nPerformance-Statistiken:")
            print(f"  Gesamt Requests: {stats['total_requests']}")
            print(f"  Cache Hit-Rate: {stats['cache_hit_rate']:.1%}")
            print(f"  Server-Requests: {stats['ollama']['server_requests']}")
            print(f"  Server-Fehler: {stats['ollama']['server_errors']}")
            
            if 'error_rate' in stats['ollama']:
                print(f"  Fehlerrate: {stats['ollama']['error_rate']:.1%}")
        
        else:
            print("\nOllama-Server nicht verfügbar für weitere Tests")
            print("Stellen Sie sicher, dass Ollama läuft: ollama serve")
            print("Und das gewünschte Modell verfügbar ist: ollama pull nomic-embed-text")
    
    except Exception as e:
        print(f"Fehler beim Testen: {str(e)}")
        print("\nHinweis: Für Ollama-Embeddings muss ein Ollama-Server laufen.")
        print("Installation: https://ollama.ai")
        print("Start: ollama serve")
        print("Modell laden: ollama pull nomic-embed-text")
    
    # Factory-Tests
    print(f"\n--- Factory-Tests ---")
    try:
        # Verschiedene Embedding-Varianten testen
        factories = [
            ('Nomic', OllamaEmbeddingsFactory.create_nomic_embeddings),
            ('MXBAI', OllamaEmbeddingsFactory.create_mxbai_embeddings),
            ('Lightweight', OllamaEmbeddingsFactory.create_lightweight_embeddings)
        ]
        
        for name, factory_method in factories:
            try:
                factory_embeddings = factory_method()
                print(f"  {name}: {factory_embeddings.model_name} (Dim: {factory_embeddings.dimension})")
            except Exception as e:
                print(f"  {name}: Fehler - {str(e)}")
    
    except Exception as e:
        print(f"Factory-Test Fehler: {str(e)}")
    
    print("\n✅ Ollama-Embeddings getestet")