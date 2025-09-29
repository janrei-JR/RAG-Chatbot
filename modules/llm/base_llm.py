"""
Base LLM - Abstract Base Class für alle LLM-Implementierungen

CRITICAL SYNTAX FIX - Komplette Neuerstellung der defekten Datei

Diese Datei ist der Basis-Baustein für alle LLM-Provider und definiert:
- Standardisierte LLM-Schnittstelle
- Message-basierte Conversation-Handling
- Token-Usage und Cost-Tracking
- Health-Checks und Performance-Monitoring
"""

import time
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Iterator, Union
from dataclasses import dataclass, field
from enum import Enum
from core.logger import get_logger
from core.exceptions import ValidationError, ConfigurationException

# =============================================================================
# LLM ENUMS UND DATENSTRUKTUREN
# =============================================================================

class LLMProvider(Enum):
    """Unterstützte LLM-Provider"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

class MessageRole(Enum):
    """Message-Rollen für Conversation"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

@dataclass
class LLMMessage:
    """Single Message in einer Conversation"""
    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    token_count: Optional[int] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class LLMResponse:
    """Response von LLM-Generation"""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    conversation_id: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def __post_init__(self):
        if not self.usage:
            self.usage = {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }

@dataclass
class LLMStreamChunk:
    """Chunk von Streaming-Response"""
    content: str
    chunk_index: int
    is_complete: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class LLMConfig:
    """Konfiguration für LLM-Instanz"""
    name: str
    provider: LLMProvider
    model: str
    
    # Generation-Parameter
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # System-Prompts
    system_prompt_template: Optional[str] = None
    
    # Context-Management
    context_window: int = 4096
    enable_context_management: bool = True
    
    # Performance
    request_timeout: int = 60
    max_retries: int = 3
    
    # Provider-spezifische Config
    provider_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validierung der LLM-Konfiguration"""
        if not (0.0 <= self.temperature <= 2.0):
            raise ValidationError("temperature muss zwischen 0.0 und 2.0 liegen")
        
        if self.max_tokens < 1:
            self.max_tokens = 2048
        
        if not (0.0 <= self.top_p <= 1.0):
            self.top_p = 0.9

# =============================================================================
# ABSTRACT BASE LLM
# =============================================================================

class BaseLLM(ABC):
    """
    Abstract Base Class für alle LLM-Implementierungen
    
    Definiert standardisierte Schnittstelle für:
    - Text-Generation mit konfigurierbaren Parametern
    - Streaming-Support für real-time responses
    - Message-basierte Conversation-Handling
    - Token-Usage und Cost-Tracking
    - Health-Checks und Performance-Monitoring
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialisiert Base LLM
        
        Args:
            config: LLM-Konfiguration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Conversation-Management
        self._conversation_history: List[LLMMessage] = []
        
        # Performance-Metriken
        self._total_requests = 0
        self._successful_requests = 0
        self._total_processing_time = 0.0
        self._token_usage = {'prompt': 0, 'completion': 0, 'total': 0}
        
        self.logger.info(f"LLM initialisiert: {config.name} ({config.provider.value})")
    
    @property
    def name(self) -> str:
        """Name des LLMs"""
        return self.config.name
    
    @property
    def provider(self) -> LLMProvider:
        """Provider des LLMs"""
        return self.config.provider
    
    @property
    def model(self) -> str:
        """Model-Name"""
        return self.config.model
    
    # =========================================================================
    # ABSTRACT METHODS - Müssen von Subklassen implementiert werden
    # =========================================================================
    
    @abstractmethod
    def _generate_impl(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """
        Implementiert tatsächliche Text-Generation
        
        Args:
            messages: Conversation-Messages
            **kwargs: Provider-spezifische Parameter
            
        Returns:
            LLMResponse: Generierte Response
        """
        pass
    
    @abstractmethod
    def _stream_impl(self, messages: List[LLMMessage], **kwargs) -> Iterator[LLMStreamChunk]:
        """
        Implementiert Streaming-Text-Generation
        
        Args:
            messages: Conversation-Messages
            **kwargs: Provider-spezifische Parameter
            
        Yields:
            LLMStreamChunk: Response-Chunks
        """
        pass
    
    @abstractmethod
    def _health_check_impl(self) -> Dict[str, Any]:
        """
        Implementiert Provider-spezifischen Health-Check
        
        Returns:
            Dict[str, Any]: Health-Status
        """
        pass
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generiert Text-Response für Prompt
        
        Args:
            prompt: Input-Prompt
            **kwargs: Generation-Parameter (überschreibt Config)
            
        Returns:
            LLMResponse: Generierte Antwort
        """
        start_time = time.time()
        
        try:
            # Message-Liste erstellen
            messages = self._build_messages(prompt, **kwargs)
            
            # Context-Management
            if self.config.enable_context_management:
                messages = self._manage_context(messages)
            
            # Generation ausführen
            response = self._generate_impl(messages, **kwargs)
            
            # Conversation-History aktualisieren
            if self.config.enable_context_management:
                self._update_conversation_history(messages, response)
            
            # Metriken aktualisieren
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(response, True, processing_time)
            
            return response
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Text-Generation fehlgeschlagen: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Error-Metriken
            self._update_metrics(None, success=False, processing_time_ms=processing_time_ms)
            
            raise ValidationError(error_msg) from e
    
    def stream(self, prompt: str, **kwargs) -> Iterator[LLMStreamChunk]:
        """
        Streamt Text-Response für Prompt
        
        Args:
            prompt: Input-Prompt
            **kwargs: Generation-Parameter
            
        Yields:
            LLMStreamChunk: Response-Chunks
        """
        try:
            # Message-Liste erstellen
            messages = self._build_messages(prompt, **kwargs)
            
            # Context-Management
            if self.config.enable_context_management:
                messages = self._manage_context(messages)
            
            # Streaming-Generation
            for chunk in self._stream_impl(messages, **kwargs):
                yield chunk
                
        except Exception as e:
            error_msg = f"Streaming-Generation fehlgeschlagen: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ValidationError(error_msg) from e
    
    def chat(self, 
             prompt: str, 
             system_prompt: Optional[str] = None,
             conversation_history: Optional[List[LLMMessage]] = None,
             **kwargs) -> LLMResponse:
        """
        Chat-Interface mit erweiterten Optionen
        
        Args:
            prompt: User-Prompt
            system_prompt: Optional System-Prompt
            conversation_history: Optional explizite History
            **kwargs: Generation-Parameter
            
        Returns:
            LLMResponse: Chat-Response
        """
        start_time = time.time()
        
        try:
            # Message-Liste mit Chat-Context erstellen
            messages = self._build_messages(
                prompt, 
                system_prompt=system_prompt,
                conversation_history=conversation_history
            )
            
            # Context-Management
            if self.config.enable_context_management:
                messages = self._manage_context(messages)
            
            # Chat-Generation
            response = self._generate_impl(messages, **kwargs)
            
            # Conversation-History aktualisieren
            if self.config.enable_context_management:
                self._update_conversation_history(messages, response)
            
            # Metriken aktualisieren
            processing_time = (time.time() - start_time) * 1000
            self._update_metrics(response, True, processing_time)
            
            return response
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Chat-Generation fehlgeschlagen: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Error-Metriken
            self._update_metrics(None, success=False, processing_time_ms=processing_time_ms)
            
            raise ValidationError(error_msg) from e
    
    def health_check(self) -> Dict[str, Any]:
        """
        Führt Health-Check aus
        
        Returns:
            Dict[str, Any]: Health-Status
        """
        try:
            # Basis-Health-Informationen
            health_data = {
                'name': self.name,
                'provider': self.provider.value,
                'model': self.model,
                'status': 'unknown',
                'performance_metrics': self._get_performance_metrics(),
                'config': {
                    'temperature': self.config.temperature,
                    'max_tokens': self.config.max_tokens,
                    'context_window': self.config.context_window
                }
            }
            
            # Provider-spezifischen Health-Check ausführen
            provider_health = self._health_check_impl()
            health_data.update(provider_health)
            
            # Status bestimmen
            if provider_health.get('available', False):
                health_data['status'] = 'healthy'
            else:
                health_data['status'] = 'unhealthy'
            
            return health_data
            
        except Exception as e:
            self.logger.error(f"Health-Check fehlgeschlagen: {e}")
            return {
                'name': self.name,
                'status': 'error',
                'error': str(e)
            }
    
    # =========================================================================
    # INTERNAL HELPER METHODS
    # =========================================================================
    
    def _build_messages(self, 
                       prompt: str, 
                       system_prompt: Optional[str] = None,
                       conversation_history: Optional[List[LLMMessage]] = None) -> List[LLMMessage]:
        """
        Baut Message-Liste für LLM-Request
        
        Args:
            prompt: User-Prompt
            system_prompt: Optional System-Prompt
            conversation_history: Optional History
            
        Returns:
            List[LLMMessage]: Message-Liste
        """
        messages = []
        
        # System-Prompt hinzufügen
        final_system_prompt = system_prompt or self.config.system_prompt_template
        if final_system_prompt:
            messages.append(LLMMessage(
                role="system",
                content=final_system_prompt,
                metadata={'source': 'config'}
            ))
        
        # Conversation-History hinzufügen
        if conversation_history:
            messages.extend(conversation_history)
        elif self.config.enable_context_management and self._conversation_history:
            # Gespeicherte History verwenden
            messages.extend(self._conversation_history[-10:])  # Letzte 10 Messages
        
        # User-Prompt hinzufügen
        messages.append(LLMMessage(
            role="user",
            content=prompt,
            metadata={'source': 'current_request'}
        ))
        
        return messages
    
    def _manage_context(self, messages: List[LLMMessage]) -> List[LLMMessage]:
        """
        Context-Management für Token-Limits
        
        Args:
            messages: Original Messages
            
        Returns:
            List[LLMMessage]: Gefilterte Messages innerhalb Context-Limit
        """
        # Einfache Implementierung: Token-Schätzung basierend auf Zeichen
        total_length = sum(len(msg.content) for msg in messages)
        estimated_tokens = total_length // 4  # Grobe Schätzung: 4 Zeichen = 1 Token
        
        # Wenn unter Limit, alle Messages zurückgeben
        if estimated_tokens <= self.config.context_window:
            return messages
        
        # Sonst: System-Prompt behalten, Rest kürzen
        system_messages = [msg for msg in messages if msg.role == "system"]
        other_messages = [msg for msg in messages if msg.role != "system"]
        
        # Neueste Messages bevorzugen
        reduced_messages = system_messages
        current_length = sum(len(msg.content) for msg in system_messages) // 4
        
        for msg in reversed(other_messages):
            msg_tokens = len(msg.content) // 4
            if current_length + msg_tokens <= self.config.context_window:
                reduced_messages.insert(-len(system_messages), msg)
                current_length += msg_tokens
            else:
                break
        
        self.logger.warning(f"Context gekürzt: {len(messages)} -> {len(reduced_messages)} Messages")
        return reduced_messages
    
    def _update_conversation_history(self, messages: List[LLMMessage], response: LLMResponse):
        """
        Aktualisiert Conversation-History
        
        Args:
            messages: Request-Messages
            response: LLM-Response
        """
        # User-Message zur History hinzufügen
        user_messages = [msg for msg in messages if msg.role == "user"]
        if user_messages:
            self._conversation_history.append(user_messages[-1])
        
        # Assistant-Response hinzufügen
        assistant_message = LLMMessage(
            role="assistant",
            content=response.content,
            metadata={
                'model': response.model,
                'processing_time_ms': response.processing_time_ms,
                'usage': response.usage
            }
        )
        self._conversation_history.append(assistant_message)
        
        # History-Limit einhalten (max 20 Messages)
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]
    
    def _update_metrics(self, response: Optional[LLMResponse], success: bool, processing_time_ms: float):
        """
        Aktualisiert Performance-Metriken
        
        Args:
            response: LLM-Response (None bei Fehler)
            success: Ob Request erfolgreich war
            processing_time_ms: Verarbeitungszeit in Millisekunden
        """
        self._total_requests += 1
        self._total_processing_time += processing_time_ms
        
        if success:
            self._successful_requests += 1
            
            if response and response.usage:
                self._token_usage['prompt'] += response.usage.get('prompt_tokens', 0)
                self._token_usage['completion'] += response.usage.get('completion_tokens', 0)
                self._token_usage['total'] += response.usage.get('total_tokens', 0)
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """
        Holt aktuelle Performance-Metriken
        
        Returns:
            Dict[str, Any]: Performance-Daten
        """
        success_rate = self._successful_requests / max(self._total_requests, 1)
        avg_processing_time = self._total_processing_time / max(self._total_requests, 1)
        
        return {
            'total_requests': self._total_requests,
            'successful_requests': self._successful_requests,
            'success_rate': round(success_rate, 3),
            'average_processing_time_ms': round(avg_processing_time, 2),
            'total_tokens_used': self._token_usage['total'],
            'conversation_length': len(self._conversation_history)
        }
    
    def clear_conversation_history(self):
        """Leert Conversation-History"""
        self._conversation_history.clear()
        self.logger.info("Conversation-History geleert")
    
    def get_conversation_history(self) -> List[LLMMessage]:
        """
        Holt aktuelle Conversation-History
        
        Returns:
            List[LLMMessage]: Conversation-History
        """
        return self._conversation_history.copy()

# =============================================================================
# LLM REGISTRY FÜR DYNAMIC LOADING
# =============================================================================

class LLMRegistry:
    """Registry für verfügbare LLM-Implementierungen"""
    
    _llms: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, llm_class: type):
        """
        Registriert neue LLM-Implementierung
        
        Args:
            name: Eindeutiger Name des LLMs
            llm_class: LLM-Klasse (muss BaseLLM erweitern)
        """
        if not issubclass(llm_class, BaseLLM):
            raise ValidationError(f"LLM-Klasse muss BaseLLM erweitern")
        
        cls._llms[name] = llm_class
        logger = get_logger(__name__)
        logger.info(f"LLM '{name}' registriert: {llm_class.__name__}")
    
    @classmethod
    def get_available_llms(cls) -> List[str]:
        """
        Holt Liste verfügbarer LLM-Namen
        
        Returns:
            List[str]: Verfügbare LLM-Namen
        """
        return list(cls._llms.keys())
    
    @classmethod
    def create_llm(cls, name: str, config: LLMConfig, **kwargs) -> BaseLLM:
        """
        Erstellt LLM-Instanz nach Name
        
        Args:
            name: Name des zu erstellenden LLMs
            config: LLM-Konfiguration
            **kwargs: Zusätzliche Parameter für LLM-Konstruktor
            
        Returns:
            BaseLLM: LLM-Instanz
        """
        if name not in cls._llms:
            available = ', '.join(cls._llms.keys())
            raise ValidationError(f"Unbekanntes LLM '{name}'. Verfügbar: {available}")
        
        llm_class = cls._llms[name]
        return llm_class(config, **kwargs)
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Prüft ob LLM registriert ist"""
        return name in cls._llms

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_llm_config(name: str,
                     provider: Union[str, LLMProvider],
                     model: str,
                     **kwargs) -> LLMConfig:
    """
    Convenience-Funktion für LLM-Config-Erstellung
    
    Args:
        name: Name des LLMs
        provider: Provider (Enum oder String)
        model: Model-Name
        **kwargs: Zusätzliche Config-Parameter
        
    Returns:
        LLMConfig: Konfigurierte LLMConfig-Instanz
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider)
    
    return LLMConfig(
        name=name,
        provider=provider,
        model=model,
        **kwargs
    )

def validate_llm_response(response: LLMResponse) -> bool:
    """
    Validiert LLM-Response auf Korrektheit
    
    Args:
        response: Zu validierende LLMResponse
        
    Returns:
        bool: True wenn valid
    """
    try:
        # Basis-Validierung
        if not isinstance(response, LLMResponse):
            return False
        
        if not response.content or not isinstance(response.content, str):
            return False
        
        if not response.model or not isinstance(response.model, str):
            return False
        
        if response.processing_time_ms < 0:
            return False
        
        # Token-Usage Validierung
        if response.usage:
            for key, value in response.usage.items():
                if not isinstance(value, int) or value < 0:
                    return False
        
        return True
        
    except Exception:
        return False

def estimate_tokens(text: str) -> int:
    """
    Schätzt Token-Anzahl für Text
    
    Args:
        text: Zu schätzender Text
        
    Returns:
        int: Geschätzte Token-Anzahl
    """
    # Einfache Schätzung: ~4 Zeichen = 1 Token
    return max(1, len(text) // 4)
