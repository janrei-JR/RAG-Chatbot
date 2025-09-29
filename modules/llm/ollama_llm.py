#!/usr/bin/env python3
"""
Ollama LLM - Local Language Model Integration
Industrielle RAG-Architektur - Module Layer

Ollama-spezifische LLM-Implementierung für lokale, cloud-unabhängige
Sprachmodelle in industriellen Umgebungen. Unterstützt verschiedene
Open-Source-Modelle und bietet vollständige Kontrolle über Datenverarbeitung.

Features:
- Lokale LLM-Ausführung ohne Cloud-Abhängigkeiten
- Unterstützung für Llama2, Mistral, CodeLlama, Vicuna und weitere
- Streaming-Support für real-time responses
- Model-Management: Download, Update, Removal
- Performance-Monitoring und Health-Checks
- Industrielle Prompt-Templates für technische Dokumentation

Autor: KI-Consultant für industrielle Automatisierung  
Version: 4.0.0 - Service-orientierte Architektur
"""

import json
import time
import requests
from typing import Dict, Any, List, Optional, Iterator, Union
from dataclasses import dataclass, field

# Core-Komponenten
from core import get_logger, ValidationError, create_error_context

# Base LLM-Komponenten
from .base_llm import (
    BaseLLM,
    LLMConfig,
    LLMMessage, 
    LLMResponse,
    LLMStreamChunk,
    LLMProvider,
    ResponseFormat,
    StreamingMode
)


# =============================================================================
# OLLAMA LLM KONFIGURATION
# =============================================================================

@dataclass
class OllamaLLMConfig(LLMConfig):
    """Erweiterte Konfiguration für Ollama LLM"""
    
    # Ollama-spezifische Parameter
    base_url: str = "http://localhost:11434"  # Ollama Server URL
    keep_alive: str = "5m"                    # Model keep-alive Zeit
    num_predict: int = -1                     # Max prediction tokens (-1 = unbegrenzt)
    num_ctx: int = 2048                       # Context window size
    num_batch: int = 512                      # Batch size für processing
    num_gqa: int = 1                          # Group query attention
    num_gpu: int = -1                         # GPU layers (-1 = auto)
    main_gpu: int = 0                         # Main GPU index
    low_vram: bool = False                    # Low VRAM mode
    f16_kv: bool = True                       # Half precision key/value
    logits_all: bool = False                  # Return logits für all tokens
    vocab_only: bool = False                  # Only load vocabulary
    use_mmap: bool = True                     # Use memory mapping
    use_mlock: bool = False                   # Use memory locking
    embedding_only: bool = False              # Only use for embeddings
    num_thread: int = -1                      # Number of threads (-1 = auto)
    
    # Sampling Parameter (zusätzlich zu base config)
    repeat_last_n: int = 64                   # Look back n tokens for repeat penalty
    repeat_penalty: float = 1.1               # Penalty für wiederholte tokens
    seed: int = -1                            # Random seed (-1 = random)
    stop: List[str] = field(default_factory=list)  # Stop sequences
    tfs_z: float = 1.0                        # Tail free sampling
    num_predict_max: int = 4096               # Maximum tokens to predict
    typical_p: float = 1.0                    # Typical sampling
    mirostat: int = 0                         # Mirostat sampling (0=disabled, 1=Mirostat, 2=Mirostat 2.0)
    mirostat_eta: float = 0.1                 # Mirostat learning rate
    mirostat_tau: float = 5.0                 # Mirostat target entropy
    penalize_newline: bool = True             # Penalize newlines
    
    # Modell-Management
    auto_pull_model: bool = True              # Auto-download model falls nicht verfügbar
    model_tag: Optional[str] = None           # Spezifischer Model-Tag/Version
    
    def __post_init__(self):
        """Validierung der Ollama-Konfiguration"""
        super().__post_init__()
        
        # Provider setzen
        self.provider = LLMProvider.OLLAMA
        
        # Ollama-spezifische Validierung
        if self.num_ctx < 256:
            self.num_ctx = 2048
            
        if self.num_batch < 1:
            self.num_batch = 512
            
        if not (1.0 <= self.repeat_penalty <= 2.0):
            self.repeat_penalty = 1.1
            
        # URL normalisieren
        if not self.base_url.startswith('http'):
            self.base_url = f"http://{self.base_url}"
        
        if self.base_url.endswith('/'):
            self.base_url = self.base_url[:-1]


# =============================================================================
# OLLAMA LLM IMPLEMENTIERUNG
# =============================================================================

class OllamaLLM(BaseLLM):
    """
    Ollama LLM für lokale Sprachmodelle
    
    Implementiert vollständige Ollama-Integration für industrielle
    RAG-Anwendungen mit Fokus auf:
    - Lokale Ausführung ohne externe Abhängigkeiten
    - Unterstützung verschiedener Open-Source-Modelle
    - Performance-Optimierung für Produktionsumgebungen
    - Robuste Fehlerbehandlung und Health-Monitoring
    """
    
    def __init__(self, config: OllamaLLMConfig):
        """
        Initialisiert Ollama LLM
        
        Args:
            config: Ollama-spezifische Konfiguration
        """
        super().__init__(config)
        self.ollama_config = config
        
        # HTTP Session für Connection-Pooling
        self.session = requests.Session()
        self.session.timeout = self.config.request_timeout
        
        # Verfügbare Modelle Cache
        self._available_models: Optional[List[Dict[str, Any]]] = None
        self._models_last_updated: Optional[time.time] = None
        
        # Performance-Metriken
        self._model_load_time = 0.0
        self._context_switches = 0
        self._gpu_memory_usage = 0
        
        # Initialisierung
        self._initialize_ollama()
        
        self.logger.info(f"Ollama LLM initialisiert: {config.model} @ {config.base_url}")

    def _initialize_ollama(self):
        """Initialisiert Ollama-Verbindung und Model-Verfügbarkeit"""
        try:
            # Server-Verfügbarkeit prüfen
            health_url = f"{self.ollama_config.base_url}/api/tags"
            response = self.session.get(health_url, timeout=5.0)
            response.raise_for_status()
            
            # Modell-Verfügbarkeit prüfen
            if self.ollama_config.auto_pull_model:
                self._ensure_model_available()
            
            self.logger.info("Ollama-Initialisierung erfolgreich")
            
        except Exception as e:
            self.logger.error(f"Ollama-Initialisierung fehlgeschlagen: {e}")
            if self.ollama_config.auto_pull_model:
                self.logger.info("Versuche Model-Download...")
                self._pull_model()

    def _generate_impl(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """
        Implementiert Ollama-Text-Generation
        
        Args:
            messages: Conversation-Messages
            **kwargs: Zusätzliche Ollama-Parameter
            
        Returns:
            LLMResponse: Generierte Response
        """
        start_time = time.time()
        
        try:
            # Request-Payload erstellen
            payload = self._build_generate_payload(messages, **kwargs)
            
            # API-Request
            generate_url = f"{self.ollama_config.base_url}/api/generate"
            response = self.session.post(generate_url, json=payload, timeout=self.config.request_timeout)
            response.raise_for_status()
            
            # Response verarbeiten
            response_data = response.json()
            
            # LLM-Response erstellen
            llm_response = self._parse_generate_response(response_data, start_time)
            
            # Context-Management
            if response_data.get('context'):
                self._handle_context_update(response_data['context'])
            
            return llm_response
            
        except requests.RequestException as e:
            raise ValidationError(f"Ollama API-Request fehlgeschlagen: {str(e)}") from e
        except Exception as e:
            raise ValidationError(f"Ollama-Generation fehlgeschlagen: {str(e)}") from e

    def _stream_impl(self, messages: List[LLMMessage], **kwargs) -> Iterator[LLMStreamChunk]:
        """
        Implementiert Ollama-Streaming-Generation
        
        Args:
            messages: Conversation-Messages
            **kwargs: Zusätzliche Ollama-Parameter
            
        Yields:
            LLMStreamChunk: Response-Chunks
        """
        try:
            # Request-Payload mit Streaming
            payload = self._build_generate_payload(messages, stream=True, **kwargs)
            
            # Streaming-Request
            generate_url = f"{self.ollama_config.base_url}/api/generate"
            response = self.session.post(
                generate_url, 
                json=payload, 
                timeout=self.config.request_timeout,
                stream=True
            )
            response.raise_for_status()
            
            # Stream-Chunks verarbeiten
            chunk_index = 0
            accumulated_content = ""
            
            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue
                
                try:
                    chunk_data = json.loads(line)
                    
                    # Response-Content extrahieren
                    if 'response' in chunk_data:
                        chunk_content = chunk_data['response']
                        accumulated_content += chunk_content
                        
                        # Stream-Chunk erstellen
                        is_complete = chunk_data.get('done', False)
                        
                        yield LLMStreamChunk(
                            content=chunk_content,
                            chunk_index=chunk_index,
                            is_complete=is_complete,
                            metadata={
                                'model': chunk_data.get('model'),
                                'total_duration': chunk_data.get('total_duration'),
                                'load_duration': chunk_data.get('load_duration'),
                                'sample_count': chunk_data.get('sample_count'),
                                'sample_duration': chunk_data.get('sample_duration'),
                                'prompt_eval_count': chunk_data.get('prompt_eval_count'),
                                'prompt_eval_duration': chunk_data.get('prompt_eval_duration'),
                                'eval_count': chunk_data.get('eval_count'),
                                'eval_duration': chunk_data.get('eval_duration')
                            }
                        )
                        
                        chunk_index += 1
                        
                        if is_complete:
                            break
                            
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Ungültiger JSON in Stream-Chunk: {line}")
                    continue
                    
        except requests.RequestException as e:
            # Error-Chunk senden
            yield LLMStreamChunk(
                content=f"Streaming-Error: {str(e)}",
                chunk_index=-1,
                is_complete=True,
                metadata={'error': True}
            )
        except Exception as e:
            yield LLMStreamChunk(
                content=f"Unerwarteter Streaming-Error: {str(e)}",
                chunk_index=-1,
                is_complete=True,
                metadata={'error': True}
            )

    def _build_generate_payload(self, messages: List[LLMMessage], stream: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Erstellt Ollama API-Request-Payload
        
        Args:
            messages: Conversation-Messages
            stream: Streaming aktivieren
            **kwargs: Zusätzliche Parameter
            
        Returns:
            Dict[str, Any]: Request-Payload
        """
        # Prompt aus Messages erstellen
        prompt = self._messages_to_prompt(messages)
        
        # Basis-Payload
        payload = {
            "model": self.ollama_config.model,
            "prompt": prompt,
            "stream": stream,
            "options": self._build_options(**kwargs)
        }
        
        # Keep-alive
        if self.ollama_config.keep_alive:
            payload["keep_alive"] = self.ollama_config.keep_alive
        
        # Format
        if self.config.response_format == ResponseFormat.JSON:
            payload["format"] = "json"
        
        # Stop-Sequences
        if self.ollama_config.stop:
            payload["options"]["stop"] = self.ollama_config.stop
        
        return payload

    def _build_options(self, **kwargs) -> Dict[str, Any]:
        """Erstellt Ollama Options-Dictionary"""
        options = {
            # Generation-Parameter
            "temperature": kwargs.get('temperature', self.config.temperature),
            "top_p": kwargs.get('top_p', self.config.top_p),
            "top_k": kwargs.get('top_k', self.config.top_k),
            
            # Ollama-spezifische Parameter
            "num_predict": kwargs.get('num_predict', self.ollama_config.num_predict),
            "num_ctx": kwargs.get('num_ctx', self.ollama_config.num_ctx),
            "repeat_last_n": kwargs.get('repeat_last_n', self.ollama_config.repeat_last_n),
            "repeat_penalty": kwargs.get('repeat_penalty', self.ollama_config.repeat_penalty),
            "seed": kwargs.get('seed', self.ollama_config.seed),
            "tfs_z": kwargs.get('tfs_z', self.ollama_config.tfs_z),
            "typical_p": kwargs.get('typical_p', self.ollama_config.typical_p),
            "mirostat": kwargs.get('mirostat', self.ollama_config.mirostat),
            "mirostat_eta": kwargs.get('mirostat_eta', self.ollama_config.mirostat_eta),
            "mirostat_tau": kwargs.get('mirostat_tau', self.ollama_config.mirostat_tau),
            "penalize_newline": kwargs.get('penalize_newline', self.ollama_config.penalize_newline),
            
            # Performance-Parameter
            "num_batch": kwargs.get('num_batch', self.ollama_config.num_batch),
            "num_gqa": kwargs.get('num_gqa', self.ollama_config.num_gqa),
            "num_gpu": kwargs.get('num_gpu', self.ollama_config.num_gpu),
            "main_gpu": kwargs.get('main_gpu', self.ollama_config.main_gpu),
            "low_vram": kwargs.get('low_vram', self.ollama_config.low_vram),
            "f16_kv": kwargs.get('f16_kv', self.ollama_config.f16_kv),
            "use_mmap": kwargs.get('use_mmap', self.ollama_config.use_mmap),
            "use_mlock": kwargs.get('use_mlock', self.ollama_config.use_mlock),
            "num_thread": kwargs.get('num_thread', self.ollama_config.num_thread)
        }
        
        # None-Werte entfernen
        return {k: v for k, v in options.items() if v is not None}

    def _messages_to_prompt(self, messages: List[LLMMessage]) -> str:
        """
        Konvertiert Messages zu Ollama-Prompt-Format
        
        Args:
            messages: Message-Liste
            
        Returns:
            str: Formatierter Prompt
        """
        prompt_parts = []
        
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
        
        # Abschließenden Assistant-Prompt hinzufügen
        if not prompt_parts[-1].startswith("User:"):
            prompt_parts.append("Assistant:")
        else:
            prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)

    def _parse_generate_response(self, response_data: Dict[str, Any], start_time: float) -> LLMResponse:
        """
        Parst Ollama API-Response zu LLMResponse
        
        Args:
            response_data: Ollama API-Response
            start_time: Start-Zeit für Processing-Time
            
        Returns:
            LLMResponse: Strukturierte Response
        """
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Token-Usage berechnen
        prompt_eval_count = response_data.get('prompt_eval_count', 0)
        eval_count = response_data.get('eval_count', 0)
        total_tokens = prompt_eval_count + eval_count
        
        # Usage-Dictionary
        usage = {
            'prompt_tokens': prompt_eval_count,
            'completion_tokens': eval_count,
            'total_tokens': total_tokens,
            'prompt_eval_duration': response_data.get('prompt_eval_duration', 0),
            'eval_duration': response_data.get('eval_duration', 0)
        }
        
        # Metadata sammeln
        metadata = {
            'model_info': response_data.get('model'),
            'total_duration': response_data.get('total_duration'),
            'load_duration': response_data.get('load_duration'),
            'sample_count': response_data.get('sample_count'),
            'sample_duration': response_data.get('sample_duration'),
            'context_length': len(response_data.get('context', [])),
            'ollama_version': self._get_ollama_version()
        }
        
        # Model-Load-Time tracken
        if response_data.get('load_duration'):
            self._model_load_time = response_data['load_duration'] / 1000000  # ns zu ms
        
        return LLMResponse(
            content=response_data.get('response', ''),
            model=self.config.model,
            provider=self.config.provider.value,
            usage=usage,
            metadata=metadata,
            processing_time_ms=processing_time_ms,
            finish_reason='stop' if response_data.get('done') else 'length'
        )

    def _handle_context_update(self, context: List[int]):
        """Behandelt Context-Updates für Session-Management"""
        # Context kann für folgende Requests wiederverwendet werden
        # Implementierung abhängig von Anwendungsfall
        self._context_switches += 1
        self.logger.debug(f"Context aktualisiert: {len(context)} Tokens")

    def _validate_connection(self) -> bool:
        """
        Validiert Ollama-Verbindung
        
        Returns:
            bool: True wenn Verbindung OK
        """
        try:
            health_url = f"{self.ollama_config.base_url}/api/tags"
            response = self.session.get(health_url, timeout=5.0)
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.warning(f"Ollama-Verbindung fehlgeschlagen: {e}")
            return False

    def _ensure_model_available(self) -> bool:
        """
        Stellt sicher dass Modell verfügbar ist
        
        Returns:
            bool: True wenn Modell verfügbar
        """
        try:
            available_models = self.get_available_models()
            model_names = [model['name'] for model in available_models]
            
            if self.config.model in model_names:
                return True
            
            # Modell mit Tag suchen
            if self.ollama_config.model_tag:
                full_model_name = f"{self.config.model}:{self.ollama_config.model_tag}"
                if full_model_name in model_names:
                    return True
            
            # Auto-Pull falls aktiviert
            if self.ollama_config.auto_pull_model:
                self.logger.info(f"Modell '{self.config.model}' nicht gefunden. Starte Download...")
                return self._pull_model()
            
            return False
            
        except Exception as e:
            self.logger.error(f"Model-Verfügbarkeits-Check fehlgeschlagen: {e}")
            return False

    def _pull_model(self) -> bool:
        """
        Lädt Modell von Ollama herunter
        
        Returns:
            bool: True wenn Download erfolgreich
        """
        try:
            model_name = self.config.model
            if self.ollama_config.model_tag:
                model_name = f"{self.config.model}:{self.ollama_config.model_tag}"
            
            pull_url = f"{self.ollama_config.base_url}/api/pull"
            payload = {"name": model_name}
            
            self.logger.info(f"Starte Model-Download: {model_name}")
            
            response = self.session.post(
                pull_url, 
                json=payload, 
                timeout=300,  # 5 Minuten Timeout für Download
                stream=True
            )
            response.raise_for_status()
            
            # Download-Progress verfolgen
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():
                    try:
                        progress_data = json.loads(line)
                        if 'status' in progress_data:
                            self.logger.info(f"Download-Status: {progress_data['status']}")
                            
                        if progress_data.get('status') == 'success':
                            self.logger.info(f"Model-Download abgeschlossen: {model_name}")
                            return True
                            
                    except json.JSONDecodeError:
                        continue
            
            return False
            
        except Exception as e:
            self.logger.error(f"Model-Download fehlgeschlagen: {e}")
            return False

    def get_available_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Holt verfügbare Modelle von Ollama
        
        Args:
            force_refresh: Cache ignorieren und neu laden
            
        Returns:
            List[Dict[str, Any]]: Verfügbare Modelle
        """
        # Cache-Check
        if (not force_refresh and 
            self._available_models and 
            self._models_last_updated and 
            time.time() - self._models_last_updated < 300):  # 5 Minuten Cache
            return self._available_models
        
        try:
            tags_url = f"{self.ollama_config.base_url}/api/tags"
            response = self.session.get(tags_url, timeout=10.0)
            response.raise_for_status()
            
            data = response.json()
            models = data.get('models', [])
            
            # Cache aktualisieren
            self._available_models = models
            self._models_last_updated = time.time()
            
            self.logger.debug(f"Verfügbare Modelle aktualisiert: {len(models)} Modelle")
            return models
            
        except Exception as e:
            self.logger.error(f"Modell-Liste-Abruf fehlgeschlagen: {e}")
            return self._available_models or []

    def delete_model(self, model_name: str) -> bool:
        """
        Löscht Modell von Ollama
        
        Args:
            model_name: Name des zu löschenden Modells
            
        Returns:
            bool: True wenn erfolgreich gelöscht
        """
        try:
            delete_url = f"{self.ollama_config.base_url}/api/delete"
            payload = {"name": model_name}
            
            response = self.session.delete(delete_url, json=payload, timeout=30.0)
            response.raise_for_status()
            
            self.logger.info(f"Modell gelöscht: {model_name}")
            
            # Cache invalidieren
            self._available_models = None
            return True
            
        except Exception as e:
            self.logger.error(f"Modell-Löschung fehlgeschlagen: {model_name}, Fehler: {e}")
            return False

    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Holt detaillierte Modell-Informationen
        
        Args:
            model_name: Modell-Name (default: aktuelles Modell)
            
        Returns:
            Dict[str, Any]: Modell-Informationen
        """
        target_model = model_name or self.config.model
        
        try:
            show_url = f"{self.ollama_config.base_url}/api/show"
            payload = {"name": target_model}
            
            response = self.session.post(show_url, json=payload, timeout=10.0)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Modell-Info-Abruf fehlgeschlagen: {target_model}, Fehler: {e}")
            return {}

    def _get_ollama_version(self) -> str:
        """Holt Ollama-Version"""
        try:
            version_url = f"{self.ollama_config.base_url}/api/version"
            response = self.session.get(version_url, timeout=5.0)
            response.raise_for_status()
            
            data = response.json()
            return data.get('version', 'unknown')
            
        except Exception:
            return 'unknown'

    def _custom_health_check(self) -> Optional[Dict[str, Any]]:
        """
        Ollama-spezifische Health-Checks
        
        Returns:
            Dict[str, Any]: Health-Status der Ollama-Komponenten
        """
        health_data = {
            'base_url': self.ollama_config.base_url,
            'model': self.config.model,
            'ollama_version': self._get_ollama_version()
        }
        
        # Modell-Verfügbarkeit prüfen
        try:
            available_models = self.get_available_models()
            model_names = [model['name'] for model in available_models]
            
            health_data.update({
                'model_available': self.config.model in model_names,
                'available_models_count': len(available_models),
                'available_models': model_names[:10]  # Top 10 für Übersicht
            })
            
        except Exception as e:
            health_data['model_check_error'] = str(e)
        
        # Modell-Details
        try:
            model_info = self.get_model_info()
            if model_info:
                health_data.update({
                    'model_size': model_info.get('size'),
                    'model_format': model_info.get('details', {}).get('format'),
                    'model_family': model_info.get('details', {}).get('family'),
                    'parameter_size': model_info.get('details', {}).get('parameter_size')
                })
        except Exception as e:
            health_data['model_info_error'] = str(e)
        
        # Performance-Metriken
        health_data.update({
            'model_load_time_ms': self._model_load_time,
            'context_switches': self._context_switches,
            'gpu_memory_usage': self._gpu_memory_usage
        })
        
        # Server-Ressourcen (falls verfügbar)
        try:
            # Einfacher Memory-Check über API
            ps_url = f"{self.ollama_config.base_url}/api/ps"
            response = self.session.get(ps_url, timeout=5.0)
            if response.status_code == 200:
                ps_data = response.json()
                health_data['running_models'] = len(ps_data.get('models', []))
        except Exception:
            pass  # Nicht kritisch falls nicht verfügbar
        
        return health_data

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Erweiterte Performance-Statistiken für Ollama LLM
        
        Returns:
            Dict[str, Any]: Detaillierte Performance-Metriken
        """
        base_stats = super().get_performance_stats()
        
        ollama_stats = {
            'base_url': self.ollama_config.base_url,
            'model_load_time_ms': self._model_load_time,
            'context_switches': self._context_switches,
            'gpu_memory_usage': self._gpu_memory_usage,
            'ollama_version': self._get_ollama_version()
        }
        
        # Model-specific stats
        if self._total_requests > 0:
            ollama_stats['avg_context_switches_per_request'] = self._context_switches / self._total_requests
        
        # Configuration stats
        ollama_stats.update({
            'num_ctx': self.ollama_config.num_ctx,
            'num_gpu': self.ollama_config.num_gpu,
            'low_vram_mode': self.ollama_config.low_vram,
            'keep_alive': self.ollama_config.keep_alive
        })
        
        # Stats kombinieren
        base_stats.update(ollama_stats)
        return base_stats

    def update_ollama_config(self, **kwargs):
        """
        Aktualisiert Ollama-spezifische Konfiguration
        
        Args:
            **kwargs: Zu aktualisierende Ollama-Parameter
        """
        for key, value in kwargs.items():
            if hasattr(self.ollama_config, key):
                old_value = getattr(self.ollama_config, key)
                setattr(self.ollama_config, key, value)
                self.logger.info(f"Ollama-Config aktualisiert: {key} = {old_value} -> {value}")
            else:
                self.logger.warning(f"Unbekannter Ollama-Config-Parameter: {key}")

    def switch_model(self, model_name: str, auto_pull: bool = None) -> bool:
        """
        Wechselt zu anderem Modell
        
        Args:
            model_name: Neuer Modell-Name
            auto_pull: Auto-Download aktivieren (default: Config-Wert)
            
        Returns:
            bool: True wenn Wechsel erfolgreich
        """
        if auto_pull is None:
            auto_pull = self.ollama_config.auto_pull_model
        
        old_model = self.config.model
        
        try:
            # Modell-Name aktualisieren
            self.config.model = model_name
            self.ollama_config.model = model_name
            
            # Modell-Verfügbarkeit prüfen
            if not self._ensure_model_available():
                # Rollback bei Fehler
                self.config.model = old_model
                self.ollama_config.model = old_model
                return False
            
            self.logger.info(f"Modell gewechselt: {old_model} -> {model_name}")
            return True
            
        except Exception as e:
            # Rollback bei Fehler
            self.config.model = old_model
            self.ollama_config.model = old_model
            self.logger.error(f"Modell-Wechsel fehlgeschlagen: {e}")
            return False

    def __del__(self):
        """Cleanup beim Zerstören der Instanz"""
        if hasattr(self, 'session'):
            self.session.close()


# =============================================================================
# OLLAMA PROMPT TEMPLATES
# =============================================================================

class OllamaPromptTemplates:
    """
    Sammlung industrieller Prompt-Templates für Ollama LLMs
    """
    
    @staticmethod
    def industrial_system_prompt() -> str:
        """System-Prompt für industrielle Anwendungen"""
        return """Du bist ein technischer Assistent für industrielle Automatisierung und Fertigung. 
Du hilfst bei Fragen zu:
- SPS/PLC-Programmierung (Siemens, Allen-Bradley, Schneider, etc.)
- HMI/SCADA-Systemen 
- Antriebstechnik und Motorsteuerungen
- Sensorik und Aktorik
- Industrienetzwerke (Profibus, Profinet, Modbus, etc.)
- Safety-Systemen und funktionaler Sicherheit
- Wartung und Instandhaltung

Antworte präzise, technisch korrekt und praxisorientiert. Verwende deutsche Fachbegriffe und gib konkrete Lösungsansätze."""

    @staticmethod
    def troubleshooting_prompt() -> str:
        """Prompt für Troubleshooting-Szenarien"""
        return """Du bist ein Experte für industrielle Fehlerdiagnose. 
Analysiere systematisch:
1. Symptombeschreibung
2. Mögliche Ursachen 
3. Diagnoseschritte
4. Lösungsansätze
5. Präventionsmaßnahmen

Strukturiere deine Antwort logisch und praxisnah."""

    @staticmethod
    def safety_prompt() -> str:
        """Prompt für Safety-relevante Themen"""
        return """Du bist ein Experte für funktionale Sicherheit in der Industrie.
Berücksichtige immer:
- Relevante Normen (IEC 61508, IEC 62061, ISO 13849)
- Safety-Kategorien und SIL-Level
- Risikobeurteilung
- Fail-Safe-Prinzipien
- CE-Konformität

WICHTIG: Bei sicherheitskritischen Themen weise auf die Notwendigkeit einer fachkundigen Überprüfung hin."""

    @staticmethod
    def programming_prompt() -> str:
        """Prompt für SPS-Programmierung"""
        return """Du bist ein SPS-Programmierexperte mit Erfahrung in:
- Ladder Logic (LAD)
- Function Block Diagram (FBD) 
- Structured Text (ST)
- Sequential Function Chart (SFC)
- Statement List (AWL/STL)

Erkläre Code-Beispiele ausführlich und weise auf Best Practices hin."""


# =============================================================================
# FACTORY UND UTILITY FUNCTIONS
# =============================================================================

def create_ollama_llm(config: Dict[str, Any]) -> OllamaLLM:
    """
    Factory-Funktion für Ollama LLM-Erstellung
    
    Args:
        config: Konfiguration als Dictionary
        
    Returns:
        OllamaLLM: Konfigurierte Ollama-LLM Instanz
    """
    ollama_config = OllamaLLMConfig(
        name=config.get('name', 'ollama_llm'),
        model=config['model'],  # Required
        base_url=config.get('base_url', 'http://localhost:11434'),
        **{k: v for k, v in config.items() if k not in ['name', 'model', 'base_url']}
    )
    
    return OllamaLLM(ollama_config)


def get_recommended_ollama_models() -> List[Dict[str, str]]:
    """
    Holt empfohlene Ollama-Modelle für industrielle Anwendungen
    
    Returns:
        List[Dict[str, str]]: Empfohlene Modelle mit Beschreibungen
    """
    return [
        {
            'name': 'llama2:13b',
            'description': 'Llama 2 13B - Ausgewogen zwischen Performance und Qualität',
            'use_case': 'Allgemeine technische Beratung und Dokumentation'
        },
        {
            'name': 'codellama:13b',
            'description': 'Code Llama 13B - Spezialisiert auf Code-Generation',
            'use_case': 'SPS-Programmierung und Automatisierungscode'
        },
        {
            'name': 'mistral:7b',
            'description': 'Mistral 7B - Schnell und effizient',
            'use_case': 'Schnelle Anfragen und Chat-Anwendungen'
        },
        {
            'name': 'llama2:70b',
            'description': 'Llama 2 70B - Höchste Qualität (benötigt viel RAM)',
            'use_case': 'Komplexe technische Analysen und Troubleshooting'
        },
        {
            'name': 'vicuna:13b',
            'description': 'Vicuna 13B - Gute Conversational AI',
            'use_case': 'Interaktive Beratung und Schulungsunterstützung'
        }
    ]


# Registrierung im LLM-Registry (wird von __init__.py aufgerufen)
def register_ollama_llm():
    """Registriert Ollama LLM im globalen Registry"""
    from .base_llm import LLMRegistry
    
    LLMRegistry.register('ollama', OllamaLLM)