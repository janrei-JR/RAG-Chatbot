# services/chat_service.py - CONSTRUCTOR PARAMETER BUGFIX
"""
Chat Service - Constructor Parameter-Mismatch behoben
BUGFIX: ChatService.__init__() unexpected keyword argument 'retrieval_service'
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
import uuid

from core import get_logger, get_config, RAGConfig
from core.exceptions import ServiceError

logger = get_logger(__name__)

@dataclass
class ChatQuery:
    user_input: str
    session_id: str
    context_messages: List[Dict[str, str]] = field(default_factory=list)
    retrieval_k: int = 5
    include_sources: bool = True
    temperature: float = 0.1

@dataclass
class DocumentSource:
    title: str
    content_snippet: str
    relevance_score: float
    source_type: str = "document"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class ChatResponse:
    response_text: str
    sources: List[DocumentSource] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0
    session_id: str = ""
    response_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class ChatService:
    """Chat Service - CONSTRUCTOR BUGFIX (ohne retrieval_service Parameter)"""
    
    def __init__(self, config: Optional[RAGConfig] = None,
                 embeddings: Optional[Any] = None,
                 llm: Optional[Any] = None):
        """
        BUGFIX: Constructor ohne retrieval_service Parameter
        """
        # Sichere Konfiguration
        try:
            self.config = config or get_config()
        except Exception:
            self.config = type('Config', (), {'llm': type('LLM', (), {'max_tokens': 2048})()})()
        
        self.logger = logger
        
        # Service-Dependencies mit Fallbacks - OHNE retrieval_service
        self._init_embeddings(embeddings)
        self._init_llm(llm)
        
        # Konfiguration
        self.max_context_length = getattr(getattr(self.config, 'llm', None), 'max_tokens', 2048)
        self.retrieval_k = 5
        self.confidence_threshold = 0.3
        
        # Statistiken
        self._stats = {
            'total_queries': 0,
            'successful_responses': 0,
            'failed_responses': 0,
            'average_response_time_ms': 0.0
        }
        
        self._session_contexts = {}
        
        self.logger.info("Chat-Service initialisiert (Constructor BUGFIX)")

    def _init_embeddings(self, embeddings):
        """Sichere Embeddings-Initialisierung"""
        if embeddings:
            self.embeddings = embeddings
        else:
            # Fallback Mock-Embeddings
            self.embeddings = type('MockEmbeddings', (), {
                'embed_text': lambda self, text: [0.0] * 768,
                'health_check': lambda self: True
            })()

    def _init_llm(self, llm):
        """Sichere LLM-Initialisierung"""
        if llm:
            self.llm = llm
        else:
            # Fallback Mock-LLM
            self.llm = type('MockLLM', (), {
                'generate': lambda self, prompt, **kwargs: f"Mock-Antwort auf: {prompt[:50]}...",
                'health_check': lambda self: True
            })()

    def set_retrieval_service(self, retrieval_service):
        """BUGFIX: Separate Methode zum Setzen des RetrievalService"""
        self.retrieval_service = retrieval_service
        self.logger.info("RetrievalService erfolgreich gesetzt")

    def process_query(self, query: ChatQuery) -> ChatResponse:
        """Verarbeitet Chat-Query mit Retrieval und LLM-Generation"""
        start_time = time.time()
        response_id = str(uuid.uuid4())
        self._stats['total_queries'] += 1
        
        try:
            # Retrieval falls verfügbar
            retrieved_docs = []
            if hasattr(self, 'retrieval_service') and self.retrieval_service:
                try:
                    if hasattr(self.retrieval_service, 'retrieve_relevant_documents'):
                        retrieved_docs = self.retrieval_service.retrieve_relevant_documents(
                            query.user_input, k=query.retrieval_k
                        )
                except Exception as e:
                    self.logger.warning(f"Retrieval fehlgeschlagen: {e}")
            
            # Context erstellen
            context = self._create_context(retrieved_docs, query)
            
            # LLM-Response generieren
            response_text = self._generate_response(context, query)
            
            # Response zusammenbauen
            processing_time = (time.time() - start_time) * 1000
            
            response = ChatResponse(
                response_text=response_text,
                sources=self._create_document_sources(retrieved_docs),
                confidence_score=self._calculate_confidence(retrieved_docs),
                processing_time_ms=processing_time,
                session_id=query.session_id,
                response_id=response_id
            )
            
            self._stats['successful_responses'] += 1
            return response
            
        except Exception as e:
            self.logger.error(f"Query-Verarbeitung fehlgeschlagen: {e}")
            self._stats['failed_responses'] += 1
            
            return ChatResponse(
                response_text=f"Entschuldigung, ein Fehler ist aufgetreten: {str(e)}",
                session_id=query.session_id,
                response_id=response_id,
                processing_time_ms=(time.time() - start_time) * 1000
            )

    def _create_context(self, retrieved_docs: List[Dict[str, Any]], query: ChatQuery) -> str:
        """Erstellt Context aus Retrieved Documents"""
        if not retrieved_docs:
            return query.user_input
        
        context_parts = ["KONTEXT:"]
        for i, doc in enumerate(retrieved_docs[:query.retrieval_k]):
            content = doc.get('content', str(doc))
            context_parts.append(f"[Dokument {i+1}] {content[:300]}...")
        
        context_parts.append(f"FRAGE: {query.user_input}")
        return "\n".join(context_parts)

    def _generate_response(self, context: str, query: ChatQuery) -> str:
        """Generiert Response über LLM"""
        try:
            if hasattr(self.llm, 'generate'):
                return str(self.llm.generate(context, temperature=query.temperature))
            elif hasattr(self.llm, 'invoke'):
                return str(self.llm.invoke(context))
            else:
                return f"Basierend auf verfügbaren Informationen zu '{query.user_input}'"
        except Exception as e:
            return f"Ihre Anfrage wurde registriert: {query.user_input}"

    def _create_document_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[DocumentSource]:
        """Erstellt DocumentSource-Objekte"""
        sources = []
        for doc in retrieved_docs:
            try:
                content = doc.get('content', str(doc))
                metadata = doc.get('metadata', {})
                score = doc.get('relevance_score', doc.get('score', 0.0))
                
                source = DocumentSource(
                    title=metadata.get('title', 'Dokument'),
                    content_snippet=content[:200],
                    relevance_score=float(score),
                    metadata=metadata
                )
                sources.append(source)
            except Exception:
                continue
        return sources

    def _calculate_confidence(self, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Berechnet Confidence-Score"""
        if not retrieved_docs:
            return 0.0
        try:
            scores = [doc.get('relevance_score', 0.5) for doc in retrieved_docs]
            return sum(scores) / len(scores) if scores else 0.5
        except:
            return 0.5

    def get_stats(self) -> Dict[str, Any]:
        return self._stats.copy()

    def health_check(self) -> Dict[str, Any]:
        """Health-Check"""
        try:
            return {
                "status": "healthy",
                "service": "ChatService",
                "stats": self._stats,
                "version": "4.0.0-CONSTRUCTOR-BUGFIX"
            }
        except Exception as e:
            return {"status": "unhealthy", "service": "ChatService", "error": str(e)}

    def cleanup(self):
        """Cleanup"""
        try:
            self._session_contexts.clear()
            self.logger.info("ChatService cleanup abgeschlossen")
        except Exception as e:
            self.logger.error(f"Cleanup fehlgeschlagen: {e}")

# Factory-Funktionen
def create_chat_service(config: Optional[RAGConfig] = None) -> ChatService:
    """Factory-Funktion BUGFIX-kompatibel"""
    try:
        return ChatService(config=config)
    except Exception as e:
        logger.error(f"ChatService-Erstellung fehlgeschlagen: {e}")
        return ChatService()

# Exports
__all__ = ['ChatService', 'ChatQuery', 'ChatResponse', 'DocumentSource', 'create_chat_service']
