#!/usr/bin/env python3
"""
Retrieval Service - Business Logic Layer
Industrielle RAG-Architektur - Phase 4 Migration

Orchestriert Such-Algorithmen und Retrieval-Strategien mit intelligenter
Query-Analyse, Multi-Strategy-Suche und Production-Features für
optimale Dokumenten-Relevanz in industriellen RAG-Anwendungen.

Features:
- Multi-Strategy Retrieval: Semantic, Keyword, Hybrid und MMR-Suche
- Intelligente Query-Klassifizierung und Strategy-Selection
- Result-Fusion und Re-Ranking für optimale Relevanz
- Performance-Monitoring und Caching für Production-Deployment
- Industrial-Domain spezifische Such-Optimierungen
- Fallback-Mechanismen und Error-Recovery

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

import time
import re
import hashlib
import threading
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from abc import ABC, abstractmethod

# Core-Komponenten
from core import (
    get_logger, get_config, RAGConfig,
    ServiceError, ValidationError, create_error_context,
    log_performance, log_method_calls
)

# Service-Integrationen
from services.vector_store_service import VectorStoreService
from services.embedding_service import EmbeddingService


# =============================================================================
# RETRIEVAL SERVICE DATENSTRUKTUREN UND ENUMS
# =============================================================================

class QueryType(str, Enum):
    """Typ der Benutzer-Anfrage für optimierte Such-Strategien"""
    FACTUAL = "factual"                    # Fakten-basierte Fragen
    PROCEDURAL = "procedural"              # Schritt-für-Schritt Anleitungen
    TROUBLESHOOTING = "troubleshooting"    # Problemlösung und Diagnose
    SPECIFICATION = "specification"        # Technische Spezifikationen
    COMPARATIVE = "comparative"            # Vergleichende Anfragen
    CONTEXTUAL = "contextual"              # Kontextabhängige Fragen
    EXPLORATORY = "exploratory"            # Explorative Recherche


class RetrievalStrategy(str, Enum):
    """Verfügbare Retrieval-Strategien"""
    SEMANTIC = "semantic"                  # Rein semantische Vektorsuche
    KEYWORD = "keyword"                    # Keyword-basierte BM25-Suche
    HYBRID = "hybrid"                      # Kombination Semantic + Keyword
    MMR = "mmr"                           # Maximum Marginal Relevance
    ADAPTIVE = "adaptive"                  # Intelligente Strategy-Auswahl
    MULTI_STRATEGY = "multi_strategy"      # Alle Strategien parallel


class RetrievalMode(str, Enum):
    """Retrieval-Modi für verschiedene Anwendungsfälle"""
    PRECISION = "precision"                # Hohe Präzision, weniger Ergebnisse
    RECALL = "recall"                      # Hohe Vollständigkeit, mehr Ergebnisse
    BALANCED = "balanced"                  # Ausgewogen zwischen Precision/Recall
    SPEED = "speed"                        # Optimiert für schnelle Antworten
    QUALITY = "quality"                    # Optimiert für beste Relevanz


@dataclass
class QueryAnalysis:
    """Analyse-Ergebnis einer Benutzer-Anfrage"""
    query: str
    query_type: QueryType
    domain_keywords: List[str] = field(default_factory=list)
    technical_terms: List[str] = field(default_factory=list)
    intent_confidence: float = 0.0
    suggested_strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE
    expected_result_count: int = 5
    requires_sequential_info: bool = False
    language: str = "de"
    
    @property
    def is_technical_query(self) -> bool:
        """Prüft ob Query technisch ist"""
        return len(self.technical_terms) > 0 or self.query_type in [
            QueryType.SPECIFICATION, QueryType.TROUBLESHOOTING
        ]


@dataclass
class RetrievedDocument:
    """Einzelnes Retrieval-Ergebnis mit Metadaten"""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    mmr_score: float = 0.0
    retrieval_strategy: str = ""
    source_collection: str = ""
    document_id: Optional[str] = None
    
    @property
    def combined_score(self) -> float:
        """Kombinierter Score aus allen Strategien"""
        return max(self.relevance_score, self.semantic_score, self.keyword_score, self.mmr_score)


@dataclass
class RetrievalRequest:
    """Request-Parameter für Retrieval-Operation"""
    query: str
    collection_name: str
    strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE
    mode: RetrievalMode = RetrievalMode.BALANCED
    k: int = 5
    fetch_k: int = 20
    score_threshold: float = 0.0
    
    # Hybrid-Parameter
    hybrid_alpha: float = 0.5  # 0.0 = nur keyword, 1.0 = nur semantic
    
    # MMR-Parameter
    mmr_lambda: float = 0.5  # Diversity vs. Relevance
    
    # Post-Processing
    rerank: bool = True
    deduplicate: bool = True
    
    # Query-Analyse (optional, wird automatisch erstellt)
    query_analysis: Optional[QueryAnalysis] = None
    
    # Erweiterte Parameter
    filters: Dict[str, Any] = field(default_factory=dict)
    boost_keywords: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-Initialisierung mit Validierung"""
        if self.k <= 0:
            raise ValidationError("k muss größer als 0 sein")
        if self.fetch_k < self.k:
            self.fetch_k = max(self.k * 2, 20)
        if not (0.0 <= self.hybrid_alpha <= 1.0):
            raise ValidationError("hybrid_alpha muss zwischen 0.0 und 1.0 liegen")


@dataclass
class RetrievalResult:
    """Ergebnis einer Retrieval-Operation"""
    success: bool
    documents: List[RetrievedDocument]
    query: str
    strategy_used: RetrievalStrategy
    total_found: int
    processing_time_ms: float
    
    # Query-Analyse
    query_analysis: Optional[QueryAnalysis] = None
    
    # Performance-Metriken
    vector_search_time_ms: float = 0.0
    keyword_search_time_ms: float = 0.0
    fusion_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    
    # Debugging-Info
    debug_info: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    
    @property
    def has_results(self) -> bool:
        """Prüft ob Ergebnisse vorhanden sind"""
        return len(self.documents) > 0
    
    @property
    def avg_relevance_score(self) -> float:
        """Durchschnittlicher Relevance-Score"""
        if not self.documents:
            return 0.0
        return sum(doc.relevance_score for doc in self.documents) / len(self.documents)


# =============================================================================
# QUERY ANALYZER
# =============================================================================

class QueryAnalyzer:
    """
    Intelligente Query-Analyse für optimierte Retrieval-Strategien
    
    Analysiert Benutzer-Anfragen und schlägt passende Retrieval-Strategien vor.
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.query_analyzer")
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialisiert Pattern für Query-Klassifikation"""
        self.query_patterns = {
            QueryType.FACTUAL: [
                r'\b(?:was ist|was sind|definiere|erkläre|bedeutung)\b',
                r'\b(?:wie viel|wie viele|wann|wo|wer)\b'
            ],
            QueryType.PROCEDURAL: [
                r'\b(?:wie|schritt|anleitung|prozess|verfahren)\b',
                r'\b(?:reihenfolge|nacheinander|zuerst|dann|anschließend)\b'
            ],
            QueryType.TROUBLESHOOTING: [
                r'\b(?:problem|fehler|störung|defekt|funktioniert nicht)\b',
                r'\b(?:troubleshooting|diagnose|beheben|reparieren|lösung)\b'
            ],
            QueryType.SPECIFICATION: [
                r'\b(?:spezifikation|technische daten|parameter|eigenschaften)\b',
                r'\b(?:norm|standard|anforderung|richtlinie)\b'
            ],
            QueryType.COMPARATIVE: [
                r'\b(?:vergleich|unterschied|besser|schlechter|vs)\b',
                r'\b(?:alternative|option|auswahl|empfehlung)\b'
            ]
        }
        
        self.technical_terms = [
            'motor', 'sensor', 'aktor', 'steuerung', 'regelung', 'plc', 'scada',
            'pneumatik', 'hydraulik', 'antrieb', 'frequenzumrichter', 'servo',
            'encoder', 'resolver', 'bus', 'protokoll', 'ethernet', 'profinet'
        ]
        
        self.domain_keywords = [
            'industrie', 'automation', 'fertigung', 'produktion', 'anlage',
            'maschine', 'roboter', 'förderband', 'handling', 'montage'
        ]
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analysiert Query und gibt Analyse-Ergebnis zurück
        
        Args:
            query: Benutzer-Query
            
        Returns:
            QueryAnalysis: Analyse-Ergebnis
        """
        query_lower = query.lower()
        
        # Query-Typ bestimmen
        query_type = self._classify_query_type(query_lower)
        
        # Keywords extrahieren
        domain_keywords = [kw for kw in self.domain_keywords if kw in query_lower]
        technical_terms = [term for term in self.technical_terms if term in query_lower]
        
        # Strategie vorschlagen
        suggested_strategy = self._suggest_strategy(query_type, len(technical_terms))
        
        # Result-Count schätzen
        expected_count = self._estimate_result_count(query_type)
        
        # Weitere Eigenschaften
        requires_sequential = self._requires_sequential_info(query_lower)
        
        return QueryAnalysis(
            query=query,
            query_type=query_type,
            domain_keywords=domain_keywords,
            technical_terms=technical_terms,
            intent_confidence=0.8,  # Placeholder
            suggested_strategy=suggested_strategy,
            expected_result_count=expected_count,
            requires_sequential_info=requires_sequential,
            language="de"
        )
    
    def _classify_query_type(self, query_lower: str) -> QueryType:
        """Klassifiziert Query-Typ basierend auf Patterns"""
        scores = {}
        
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            scores[query_type] = score
        
        # Besten Match finden
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] > 0:
                return best_type
        
        return QueryType.FACTUAL  # Default
    
    def _suggest_strategy(self, query_type: QueryType, technical_term_count: int) -> RetrievalStrategy:
        """Schlägt Retrieval-Strategie basierend auf Query-Typ vor"""
        if query_type == QueryType.FACTUAL:
            return RetrievalStrategy.SEMANTIC
        elif query_type == QueryType.PROCEDURAL:
            return RetrievalStrategy.MMR  # Für diverse, schrittweise Info
        elif query_type == QueryType.TROUBLESHOOTING:
            return RetrievalStrategy.HYBRID  # Keyword + Semantic
        elif query_type == QueryType.SPECIFICATION:
            return RetrievalStrategy.KEYWORD  # Exakte technische Terms
        elif technical_term_count > 2:
            return RetrievalStrategy.HYBRID
        else:
            return RetrievalStrategy.SEMANTIC
    
    def _estimate_result_count(self, query_type: QueryType) -> int:
        """Schätzt optimale Anzahl Ergebnisse"""
        if query_type == QueryType.FACTUAL:
            return 3  # Wenige, präzise Antworten
        elif query_type == QueryType.PROCEDURAL:
            return 5  # Schritt-für-Schritt, nicht zu viele
        elif query_type == QueryType.TROUBLESHOOTING:
            return 7  # Verschiedene Lösungsansätze
        elif query_type == QueryType.EXPLORATORY:
            return 10  # Breite Abdeckung
        else:
            return 5  # Standard
    
    def _requires_sequential_info(self, query_lower: str) -> bool:
        """Prüft ob Query sequentielle Information benötigt"""
        sequential_indicators = [
            'schritt', 'reihenfolge', 'nacheinander', 'zuerst', 'dann', 'anschließend'
        ]
        return any(indicator in query_lower for indicator in sequential_indicators)


# =============================================================================
# RETRIEVAL SERVICE IMPLEMENTIERUNG
# =============================================================================

class RetrievalService:
    """
    Retrieval Service - Such-Orchestrierung und Strategy-Management
    
    Zentrale Schnittstelle für intelligente Dokumenten-Suche mit:
    - Multi-Strategy Retrieval (Semantic, Keyword, Hybrid, MMR)
    - Intelligente Query-Analyse und Strategy-Selection
    - Result-Fusion und Re-Ranking für optimale Relevanz
    - Performance-Monitoring und Caching
    - Industrial-Domain spezifische Optimierungen
    """
    
    def __init__(self, 
                 vector_store_service: VectorStoreService,
                 embedding_service: Optional[EmbeddingService] = None,
                 config: RAGConfig = None):
        """
        Initialisiert Retrieval Service
        
        Args:
            vector_store_service: Vector Store Service für Dokumenten-Zugriff
            embedding_service: Embedding Service für Query-Vektorisierung
            config: RAG-System-Konfiguration
        """
        self.logger = get_logger(__name__)
        self.vector_store_service = vector_store_service
        self.embedding_service = embedding_service
        self.config = config or get_config()
        
        # Query-Analyzer
        self.query_analyzer = QueryAnalyzer()
        
        # Retrieval-Strategien
        self._strategies = {}
        self._initialize_strategies()
        
        # Performance-Monitoring
        self._performance_stats = {
            'total_queries': 0,
            'total_time_ms': 0.0,
            'strategy_usage': {},
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Einfacher Query-Cache
        self._query_cache = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = 1000
        
        self.logger.info(f"Retrieval Service initialisiert mit {len(self._strategies)} Strategien")
    
    def _initialize_strategies(self):
        """Initialisiert verfügbare Retrieval-Strategien"""
        try:
            # Registriere verfügbare Strategien
            self._strategies = {
                RetrievalStrategy.SEMANTIC: self._semantic_search,
                RetrievalStrategy.KEYWORD: self._keyword_search,
                RetrievalStrategy.HYBRID: self._hybrid_search,
                RetrievalStrategy.MMR: self._mmr_search,
                RetrievalStrategy.ADAPTIVE: self._adaptive_search,
                RetrievalStrategy.MULTI_STRATEGY: self._multi_strategy_search
            }
            
            self.logger.debug("Retrieval-Strategien initialisiert")
            
        except Exception as e:
            self.logger.error(f"Strategie-Initialisierung fehlgeschlagen: {str(e)}")
    
    # =============================================================================
    # PUBLIC API
    # =============================================================================
    
    @log_method_calls
    @log_performance
    def retrieve_documents(self, request: RetrievalRequest) -> RetrievalResult:
        """
        Hauptmethode für Dokumenten-Retrieval
        
        Args:
            request: Retrieval-Request mit Parametern
            
        Returns:
            RetrievalResult: Retrieval-Ergebnisse
        """
        start_time = time.time()
        
        try:
            # Query-Analyse durchführen
            if not request.query_analysis:
                request.query_analysis = self.query_analyzer.analyze_query(request.query)
            
            # Cache-Check
            cache_key = self._generate_cache_key(request)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self._performance_stats['cache_hits'] += 1
                self.logger.debug(f"Cache-Hit für Query: {request.query[:50]}...")
                return cached_result
            
            self._performance_stats['cache_misses'] += 1
            
            # Strategie-Auswahl
            if request.strategy == RetrievalStrategy.ADAPTIVE:
                strategy = request.query_analysis.suggested_strategy
            else:
                strategy = request.strategy
            
            self.logger.debug(f"Verwende Retrieval-Strategie: {strategy} für Query: {request.query[:50]}...")
            
            # Retrieval ausführen
            if strategy in self._strategies:
                documents = self._strategies[strategy](request)
            else:
                self.logger.warning(f"Unbekannte Strategie {strategy}, fallback auf Hybrid")
                documents = self._hybrid_search(request)
            
            # Post-Processing
            if request.rerank and len(documents) > 1:
                rerank_start = time.time()
                documents = self._rerank_documents(documents, request)
                rerank_time = (time.time() - rerank_start) * 1000
            else:
                rerank_time = 0.0
            
            # Ergebnis zusammenstellen
            processing_time_ms = (time.time() - start_time) * 1000
            
            result = RetrievalResult(
                success=True,
                documents=documents[:request.k],
                query=request.query,
                strategy_used=strategy,
                total_found=len(documents),
                processing_time_ms=processing_time_ms,
                query_analysis=request.query_analysis,
                rerank_time_ms=rerank_time
            )
            
            # Cache-Update
            self._cache_result(cache_key, result)
            
            # Performance-Stats aktualisieren
            self._update_performance_stats(strategy, processing_time_ms)
            
            return result
            
        except Exception as e:
            error_context = create_error_context(
                component="services.retrieval_service",
                operation="retrieve_documents",
                query=request.query
            )
            
            self.logger.error(f"Retrieval fehlgeschlagen: {str(e)}")
            
            return RetrievalResult(
                success=False,
                documents=[],
                query=request.query,
                strategy_used=request.strategy,
                total_found=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def search(self, 
               query: str,
               collection_name: str = "default",
               k: int = 5,
               strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE) -> RetrievalResult:
        """
        Vereinfachte Search-API für schnelle Nutzung
        
        Args:
            query: Such-Query
            collection_name: Name der Collection
            k: Anzahl gewünschter Ergebnisse
            strategy: Retrieval-Strategie
            
        Returns:
            RetrievalResult: Such-Ergebnisse
        """
        request = RetrievalRequest(
            query=query,
            collection_name=collection_name,
            k=k,
            strategy=strategy
        )
        
        return self.retrieve_documents(request)
    
    # =============================================================================
    # RETRIEVAL STRATEGIES
    # =============================================================================
    
    def _semantic_search(self, request: RetrievalRequest) -> List[RetrievedDocument]:
        """Semantische Vektorsuche"""
        try:
            # Vector Store Service für semantische Suche nutzen
            vector_results = self.vector_store_service.similarity_search(
                collection_name=request.collection_name,
                query=request.query,
                k=request.fetch_k,
                score_threshold=request.score_threshold,
                filters=request.filters
            )
            
            documents = []
            for result in vector_results:
                doc = RetrievedDocument(
                    content=result.get('content', ''),
                    metadata=result.get('metadata', {}),
                    relevance_score=result.get('score', 0.0),
                    semantic_score=result.get('score', 0.0),
                    retrieval_strategy=RetrievalStrategy.SEMANTIC.value,
                    source_collection=request.collection_name,
                    document_id=result.get('id')
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Semantic Search Fehler: {str(e)}")
            return []
    
    def _keyword_search(self, request: RetrievalRequest) -> List[RetrievedDocument]:
        """Keyword-basierte BM25-Suche (Fallback auf Vector Store)"""
        try:
            # Wenn Vector Store BM25 unterstützt, nutze das
            if hasattr(self.vector_store_service, 'keyword_search'):
                keyword_results = self.vector_store_service.keyword_search(
                    collection_name=request.collection_name,
                    query=request.query,
                    k=request.fetch_k,
                    score_threshold=request.score_threshold
                )
            else:
                # Fallback auf Semantic Search mit Keyword-Boosting
                self.logger.debug("Keyword Search: Fallback auf Semantic Search mit Keyword-Boosting")
                keyword_results = self.vector_store_service.similarity_search(
                    collection_name=request.collection_name,
                    query=request.query,
                    k=request.fetch_k,
                    score_threshold=request.score_threshold,
                    filters=request.filters
                )
            
            documents = []
            query_tokens = set(request.query.lower().split())
            
            for result in keyword_results:
                content = result.get('content', '')
                content_tokens = set(content.lower().split())
                keyword_overlap = len(query_tokens.intersection(content_tokens))
                keyword_score = keyword_overlap / len(query_tokens) if query_tokens else 0.0
                
                doc = RetrievedDocument(
                    content=content,
                    metadata=result.get('metadata', {}),
                    relevance_score=result.get('score', 0.0),
                    keyword_score=keyword_score,
                    retrieval_strategy=RetrievalStrategy.KEYWORD.value,
                    source_collection=request.collection_name,
                    document_id=result.get('id')
                )
                
                # Adjustiere Score basierend auf Keyword-Overlap
                doc.relevance_score = (doc.relevance_score + keyword_score) / 2
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Keyword Search Fehler: {str(e)}")
            return []
    
    def _hybrid_search(self, request: RetrievalRequest) -> List[RetrievedDocument]:
        """Hybrid-Suche: Kombination Semantic + Keyword"""
        try:
            # Beide Strategien parallel ausführen
            semantic_start = time.time()
            semantic_docs = self._semantic_search(request)
            semantic_time = (time.time() - semantic_start) * 1000
            
            keyword_start = time.time()
            keyword_docs = self._keyword_search(request)
            keyword_time = (time.time() - keyword_start) * 1000
            
            # Ergebnisse fusionieren
            fusion_start = time.time()
            fused_docs = self._fuse_results(
                semantic_docs, keyword_docs, 
                alpha=request.hybrid_alpha
            )
            fusion_time = (time.time() - fusion_start) * 1000
            
            # Strategy-Tag setzen
            for doc in fused_docs:
                doc.retrieval_strategy = RetrievalStrategy.HYBRID.value
            
            self.logger.debug(f"Hybrid Search: Semantic={len(semantic_docs)}, Keyword={len(keyword_docs)}, Fused={len(fused_docs)}")
            
            return fused_docs
            
        except Exception as e:
            self.logger.error(f"Hybrid Search Fehler: {str(e)}")
            return self._semantic_search(request)  # Fallback
    
    def _mmr_search(self, request: RetrievalRequest) -> List[RetrievedDocument]:
        """Maximum Marginal Relevance Suche"""
        try:
            # Nutze Vector Store MMR wenn verfügbar
            if hasattr(self.vector_store_service, 'max_marginal_relevance_search'):
                mmr_results = self.vector_store_service.max_marginal_relevance_search(
                    collection_name=request.collection_name,
                    query=request.query,
                    k=request.k,
                    fetch_k=request.fetch_k,
                    lambda_mult=request.mmr_lambda
                )
            else:
                # Fallback auf Semantic mit simulierter Diversifizierung
                self.logger.debug("MMR Search: Fallback auf Semantic mit Diversifizierung")
                semantic_docs = self._semantic_search(request)
                mmr_results = self._simulate_mmr(semantic_docs, request.k, request.mmr_lambda)
            
            documents = []
            for i, result in enumerate(mmr_results):
                doc = RetrievedDocument(
                    content=result.get('content', ''),
                    metadata=result.get('metadata', {}),
                    relevance_score=result.get('score', 0.0),
                    mmr_score=result.get('score', 0.0),
                    retrieval_strategy=RetrievalStrategy.MMR.value,
                    source_collection=request.collection_name,
                    document_id=result.get('id')
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"MMR Search Fehler: {str(e)}")
            return self._semantic_search(request)  # Fallback
    
    def _adaptive_search(self, request: RetrievalRequest) -> List[RetrievedDocument]:
        """Adaptive Suche basierend auf Query-Analyse"""
        try:
            # Query-Analyse nutzen um beste Strategie zu wählen
            if not request.query_analysis:
                request.query_analysis = self.query_analyzer.analyze_query(request.query)
            
            suggested_strategy = request.query_analysis.suggested_strategy
            
            # Rekursiver Aufruf mit vorgeschlagener Strategie
            adaptive_request = RetrievalRequest(
                query=request.query,
                collection_name=request.collection_name,
                strategy=suggested_strategy,  # Wichtig: Nicht ADAPTIVE um Endlosschleife zu vermeiden
                mode=request.mode,
                k=request.query_analysis.expected_result_count,
                fetch_k=request.fetch_k,
                score_threshold=request.score_threshold,
                hybrid_alpha=request.hybrid_alpha,
                mmr_lambda=request.mmr_lambda,
                rerank=request.rerank,
                deduplicate=request.deduplicate,
                query_analysis=request.query_analysis,
                filters=request.filters,
                boost_keywords=request.boost_keywords
            )
            
            return self._strategies[suggested_strategy](adaptive_request)
            
        except Exception as e:
            self.logger.error(f"Adaptive Search Fehler: {str(e)}")
            return self._hybrid_search(request)  # Fallback
    
    def _multi_strategy_search(self, request: RetrievalRequest) -> List[RetrievedDocument]:
        """Multi-Strategy Suche: Alle Strategien parallel"""
        try:
            all_strategies = [
                RetrievalStrategy.SEMANTIC,
                RetrievalStrategy.KEYWORD, 
                RetrievalStrategy.HYBRID,
                RetrievalStrategy.MMR
            ]
            
            all_documents = []
            
            for strategy in all_strategies:
                try:
                    strategy_request = RetrievalRequest(
                        query=request.query,
                        collection_name=request.collection_name,
                        strategy=strategy,
                        k=request.k,
                        fetch_k=request.fetch_k,
                        score_threshold=request.score_threshold,
                        hybrid_alpha=request.hybrid_alpha,
                        mmr_lambda=request.mmr_lambda,
                        rerank=False,  # Disable reranking for individual strategies
                        deduplicate=False,
                        query_analysis=request.query_analysis,
                        filters=request.filters,
                        boost_keywords=request.boost_keywords
                    )
                    
                    strategy_docs = self._strategies[strategy](strategy_request)
                    all_documents.extend(strategy_docs)
                    
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy} failed: {str(e)}")
            
            # Duplikate entfernen und nach Relevanz sortieren
            unique_docs = self._deduplicate_documents(all_documents)
            unique_docs.sort(key=lambda x: x.combined_score, reverse=True)
            
            # Strategy-Tag setzen
            for doc in unique_docs:
                doc.retrieval_strategy = RetrievalStrategy.MULTI_STRATEGY.value
            
            return unique_docs
            
        except Exception as e:
            self.logger.error(f"Multi-Strategy Search Fehler: {str(e)}")
            return self._hybrid_search(request)  # Fallback
    
    # =============================================================================
    # RESULT PROCESSING
    # =============================================================================
    
    def _fuse_results(self, 
                     semantic_docs: List[RetrievedDocument],
                     keyword_docs: List[RetrievedDocument],
                     alpha: float = 0.5) -> List[RetrievedDocument]:
        """
        Fusioniert Ergebnisse verschiedener Such-Strategien
        
        Args:
            semantic_docs: Semantische Such-Ergebnisse
            keyword_docs: Keyword-basierte Such-Ergebnisse  
            alpha: Gewichtung (0.0 = nur keyword, 1.0 = nur semantic)
            
        Returns:
            List[RetrievedDocument]: Fusionierte Ergebnisse
        """
        try:
            # Dokumente nach Content-Hash indizieren für Deduplication
            semantic_by_hash = {}
            keyword_by_hash = {}
            
            for doc in semantic_docs:
                content_hash = hashlib.md5(doc.content.encode()).hexdigest()
                semantic_by_hash[content_hash] = doc
            
            for doc in keyword_docs:
                content_hash = hashlib.md5(doc.content.encode()).hexdigest()
                keyword_by_hash[content_hash] = doc
            
            fused_docs = []
            all_hashes = set(semantic_by_hash.keys()) | set(keyword_by_hash.keys())
            
            for content_hash in all_hashes:
                semantic_doc = semantic_by_hash.get(content_hash)
                keyword_doc = keyword_by_hash.get(content_hash)
                
                if semantic_doc and keyword_doc:
                    # Beide Strategien haben das Dokument gefunden
                    fused_score = (alpha * semantic_doc.semantic_score + 
                                 (1 - alpha) * keyword_doc.keyword_score)
                    
                    fused_doc = RetrievedDocument(
                        content=semantic_doc.content,
                        metadata=semantic_doc.metadata,
                        relevance_score=fused_score,
                        semantic_score=semantic_doc.semantic_score,
                        keyword_score=keyword_doc.keyword_score,
                        retrieval_strategy=RetrievalStrategy.HYBRID.value,
                        source_collection=semantic_doc.source_collection,
                        document_id=semantic_doc.document_id
                    )
                    fused_docs.append(fused_doc)
                    
                elif semantic_doc:
                    # Nur semantische Suche hat Dokument gefunden
                    semantic_doc.relevance_score = semantic_doc.semantic_score * alpha
                    fused_docs.append(semantic_doc)
                    
                elif keyword_doc:
                    # Nur Keyword-Suche hat Dokument gefunden
                    keyword_doc.relevance_score = keyword_doc.keyword_score * (1 - alpha)
                    fused_docs.append(keyword_doc)
            
            # Nach fusioniertem Score sortieren
            fused_docs.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return fused_docs
            
        except Exception as e:
            self.logger.error(f"Result-Fusion fehlgeschlagen: {str(e)}")
            return semantic_docs + keyword_docs  # Fallback: Einfache Konkatenation
    
    def _rerank_documents(self, 
                         documents: List[RetrievedDocument],
                         request: RetrievalRequest) -> List[RetrievedDocument]:
        """
        Re-Ranking der Dokumente basierend auf erweiterten Kriterien
        
        Args:
            documents: Liste der zu rerankenden Dokumente
            request: Original-Request für Kontext
            
        Returns:
            List[RetrievedDocument]: Re-ranked Dokumente
        """
        try:
            if not documents:
                return documents
            
            query_lower = request.query.lower()
            query_tokens = set(query_lower.split())
            
            # Boost-Keywords berücksichtigen
            boost_tokens = set(kw.lower() for kw in request.boost_keywords)
            
            for doc in documents:
                content_lower = doc.content.lower()
                content_tokens = set(content_lower.split())
                
                # Base Score
                rerank_score = doc.relevance_score
                
                # Keyword-Overlap Boost
                keyword_overlap = len(query_tokens.intersection(content_tokens))
                if query_tokens:
                    overlap_ratio = keyword_overlap / len(query_tokens)
                    rerank_score += overlap_ratio * 0.2
                
                # Boost-Keywords
                boost_overlap = len(boost_tokens.intersection(content_tokens))
                if boost_tokens:
                    boost_ratio = boost_overlap / len(boost_tokens)
                    rerank_score += boost_ratio * 0.3
                
                # Query-Position Boost (Query-Terms am Anfang höher gewichten)
                first_100_chars = content_lower[:100]
                early_matches = sum(1 for token in query_tokens if token in first_100_chars)
                if query_tokens:
                    early_ratio = early_matches / len(query_tokens)
                    rerank_score += early_ratio * 0.1
                
                # Content-Length Normalisierung (zu kurze oder zu lange Texte abwerten)
                content_len = len(doc.content)
                if 50 <= content_len <= 1000:
                    rerank_score += 0.05  # Sweet spot
                elif content_len < 20:
                    rerank_score -= 0.1   # Zu kurz
                elif content_len > 3000:
                    rerank_score -= 0.05  # Zu lang
                
                doc.relevance_score = rerank_score
            
            # Nach Re-Ranking Score sortieren
            documents.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Re-Ranking fehlgeschlagen: {str(e)}")
            return documents  # Fallback: Unveränderte Reihenfolge
    
    def _deduplicate_documents(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        Entfernt Duplikate basierend auf Content-Ähnlichkeit
        
        Args:
            documents: Liste der zu deduplizierenden Dokumente
            
        Returns:
            List[RetrievedDocument]: Deduplizierte Dokumente
        """
        try:
            if not documents:
                return documents
            
            seen_hashes = set()
            unique_docs = []
            
            for doc in documents:
                # Content-Hash für Deduplication
                content_hash = hashlib.md5(doc.content.encode()).hexdigest()
                
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_docs.append(doc)
            
            self.logger.debug(f"Deduplication: {len(documents)} -> {len(unique_docs)} Dokumente")
            
            return unique_docs
            
        except Exception as e:
            self.logger.error(f"Deduplication fehlgeschlagen: {str(e)}")
            return documents  # Fallback
    
    def _simulate_mmr(self, documents: List[RetrievedDocument], k: int, lambda_mult: float) -> List[Dict[str, Any]]:
        """
        Simuliert MMR-Algorithmus für Diversifizierung
        
        Args:
            documents: Eingangsdokumente
            k: Anzahl gewünschter Ergebnisse
            lambda_mult: Gewichtung zwischen Relevanz und Diversität
            
        Returns:
            List[Dict]: MMR-diversifizierte Ergebnisse
        """
        try:
            if not documents or k <= 0:
                return []
            
            # Konvertiere zu Dict-Format
            candidates = []
            for doc in documents:
                candidates.append({
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'score': doc.relevance_score,
                    'id': doc.document_id
                })
            
            if len(candidates) <= k:
                return candidates
            
            selected = []
            remaining = candidates.copy()
            
            # Erstes Dokument: Höchste Relevanz
            if remaining:
                first_doc = max(remaining, key=lambda x: x['score'])
                selected.append(first_doc)
                remaining.remove(first_doc)
            
            # Restliche Dokumente: MMR-Kriterium
            while len(selected) < k and remaining:
                best_mmr_score = -1
                best_doc = None
                
                for candidate in remaining:
                    # Relevanz-Score
                    relevance = candidate['score']
                    
                    # Maximale Ähnlichkeit zu bereits ausgewählten Dokumenten
                    max_sim = 0.0
                    for sel_doc in selected:
                        # Einfache Token-basierte Ähnlichkeit
                        cand_tokens = set(candidate['content'].lower().split())
                        sel_tokens = set(sel_doc['content'].lower().split())
                        
                        if cand_tokens and sel_tokens:
                            intersection = len(cand_tokens.intersection(sel_tokens))
                            union = len(cand_tokens.union(sel_tokens))
                            similarity = intersection / union if union > 0 else 0.0
                            max_sim = max(max_sim, similarity)
                    
                    # MMR Score berechnen
                    mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                    
                    if mmr_score > best_mmr_score:
                        best_mmr_score = mmr_score
                        best_doc = candidate
                
                if best_doc:
                    selected.append(best_doc)
                    remaining.remove(best_doc)
                else:
                    break
            
            return selected
            
        except Exception as e:
            self.logger.error(f"MMR-Simulation fehlgeschlagen: {str(e)}")
            return [{'content': doc.content, 'metadata': doc.metadata, 'score': doc.relevance_score, 'id': doc.document_id} 
                   for doc in documents[:k]]
    
    # =============================================================================
    # CACHING UND PERFORMANCE
    # =============================================================================
    
    def _generate_cache_key(self, request: RetrievalRequest) -> str:
        """Generiert Cache-Key für Request"""
        key_data = {
            'query': request.query,
            'collection': request.collection_name,
            'strategy': request.strategy.value,
            'k': request.k,
            'score_threshold': request.score_threshold,
            'hybrid_alpha': request.hybrid_alpha,
            'mmr_lambda': request.mmr_lambda
        }
        
        key_string = str(sorted(key_data.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[RetrievalResult]:
        """Holt Ergebnis aus Cache"""
        with self._cache_lock:
            return self._query_cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, result: RetrievalResult) -> None:
        """Speichert Ergebnis im Cache"""
        with self._cache_lock:
            # Cache-Size begrenzen
            if len(self._query_cache) >= self._max_cache_size:
                # Älteste Einträge entfernen (FIFO)
                oldest_keys = list(self._query_cache.keys())[:10]
                for key in oldest_keys:
                    del self._query_cache[key]
            
            self._query_cache[cache_key] = result
    
    def _update_performance_stats(self, strategy: RetrievalStrategy, time_ms: float) -> None:
        """Aktualisiert Performance-Statistiken"""
        self._performance_stats['total_queries'] += 1
        self._performance_stats['total_time_ms'] += time_ms
        
        strategy_key = strategy.value
        if strategy_key not in self._performance_stats['strategy_usage']:
            self._performance_stats['strategy_usage'][strategy_key] = {
                'count': 0,
                'total_time_ms': 0.0,
                'avg_time_ms': 0.0
            }
        
        strategy_stats = self._performance_stats['strategy_usage'][strategy_key]
        strategy_stats['count'] += 1
        strategy_stats['total_time_ms'] += time_ms
        strategy_stats['avg_time_ms'] = strategy_stats['total_time_ms'] / strategy_stats['count']
    
    # =============================================================================
    # SERVICE MANAGEMENT
    # =============================================================================
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Holt Service-Statistiken"""
        stats = self._performance_stats.copy()
        
        # Zusätzliche Metriken berechnen
        if stats['total_queries'] > 0:
            stats['avg_time_ms'] = stats['total_time_ms'] / stats['total_queries']
        else:
            stats['avg_time_ms'] = 0.0
        
        stats['cache_size'] = len(self._query_cache)
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0.0
        )
        
        return stats
    
    def get_service_health(self) -> Dict[str, Any]:
        """Holt Service-Health-Status"""
        try:
            health_data = {
                'status': 'healthy',
                'vector_store_available': bool(self.vector_store_service),
                'embedding_service_available': bool(self.embedding_service),
                'strategies_available': len(self._strategies),
                'query_analyzer_available': bool(self.query_analyzer),
                'cache_size': len(self._query_cache),
                'performance_stats': self.get_service_stats(),
                'last_check': datetime.now(timezone.utc).isoformat()
            }
            
            # Vector Store Health prüfen
            try:
                vector_health = self.vector_store_service.get_health()
                health_data['vector_store_health'] = vector_health
                if vector_health.get('status') != 'healthy':
                    health_data['status'] = 'degraded'
            except:
                health_data['vector_store_health'] = {'status': 'error'}
                health_data['status'] = 'degraded'
            
            return health_data
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now(timezone.utc).isoformat()
            }
    
    def clear_cache(self) -> None:
        """Leert den Query-Cache"""
        with self._cache_lock:
            self._query_cache.clear()
        self.logger.info("Query-Cache geleert")
    
    def reset_stats(self) -> None:
        """Setzt Performance-Statistiken zurück"""
        self._performance_stats = {
            'total_queries': 0,
            'total_time_ms': 0.0,
            'strategy_usage': {},
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.logger.info("Performance-Statistiken zurückgesetzt")


# =============================================================================
# FACTORY FUNCTIONS UND SINGLETON
# =============================================================================

_retrieval_service_instance: Optional[RetrievalService] = None
_service_lock = threading.Lock()

def get_retrieval_service(
    vector_store_service: VectorStoreService = None,
    embedding_service: EmbeddingService = None,
    config: RAGConfig = None
) -> RetrievalService:
    """
    Holt Retrieval Service Singleton-Instanz
    
    Args:
        vector_store_service: Vector Store Service (bei erster Erstellung)
        embedding_service: Embedding Service (optional)
        config: RAG-Konfiguration (optional)
        
    Returns:
        RetrievalService: Service-Instanz
    """
    global _retrieval_service_instance
    
    if _retrieval_service_instance is None:
        with _service_lock:
            if _retrieval_service_instance is None:
                if not vector_store_service:
                    raise ValueError("vector_store_service ist erforderlich für erste Erstellung")
                
                _retrieval_service_instance = RetrievalService(
                    vector_store_service=vector_store_service,
                    embedding_service=embedding_service,
                    config=config
                )
    
    return _retrieval_service_instance


def create_retrieval_service(
    vector_store_service: VectorStoreService,
    embedding_service: EmbeddingService = None,
    config: RAGConfig = None
) -> RetrievalService:
    """
    Erstellt neue Retrieval Service Instanz (für Testing/Multi-Instance)
    
    Args:
        vector_store_service: Vector Store Service
        embedding_service: Embedding Service (optional)
        config: RAG-Konfiguration (optional)
        
    Returns:
        RetrievalService: Neue Service-Instanz
    """
    return RetrievalService(
        vector_store_service=vector_store_service,
        embedding_service=embedding_service,
        config=config
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Haupt-Service-Klasse
    'RetrievalService',
    
    # Datenstrukturen
    'QueryType', 'RetrievalStrategy', 'RetrievalMode',
    'QueryAnalysis', 'RetrievedDocument', 'RetrievalRequest', 'RetrievalResult',
    
    # Query-Analyzer
    'QueryAnalyzer',
    
    # Factory-Funktionen
    'get_retrieval_service', 'create_retrieval_service'
]
