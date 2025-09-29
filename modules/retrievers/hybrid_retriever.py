#!/usr/bin/env python3
"""
Hybrid Retriever - Combined Vector and Keyword Search
Industrielle RAG-Architektur - Module Layer

Kombiniert semantische Vector-Suche mit Keyword-basierter Suche für
optimale Retrieval-Ergebnisse in industriellen RAG-Anwendungen.

Features:
- Hybrid-Algorithmus mit konfigurierbarer Gewichtung (Reciprocal Rank Fusion)
- Parallele Vector- und Keyword-Suche für Performance-Optimierung
- Intelligente Score-Normalisierung und Fusion-Strategien  
- Adaptive Retrieval-Modi für verschiedene Anwendungsfälle
- Production-Features: Fallback-Mechanismen, Health-Monitoring, Caching

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

import asyncio
import math
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

# Core-Komponenten  
from core import get_logger, ValidationError, create_error_context
from core.config import get_config

# Base Retriever-Komponenten
from .base_retriever import (
    BaseRetriever, 
    RetrieverConfig, 
    RetrievalQuery, 
    RetrievalResult, 
    RetrievalMode,
    Document
)


# =============================================================================
# HYBRID RETRIEVER KONFIGURATION
# =============================================================================

class FusionStrategy(str, Enum):
    """Fusion-Strategien für Hybrid-Retrieval"""
    RRF = "reciprocal_rank_fusion"     # Reciprocal Rank Fusion (Standard)
    WEIGHTED_SUM = "weighted_sum"      # Gewichtete Summe der Scores
    MAX_SCORE = "max_score"            # Maximum der Scores
    MIN_SCORE = "min_score"            # Minimum der Scores 
    AVERAGE = "average"                # Durchschnitt der Scores
    BAYESIAN = "bayesian"              # Bayesian Score Fusion


@dataclass
class HybridRetrieverConfig(RetrieverConfig):
    """Erweiterte Konfiguration für Hybrid Retriever"""
    
    # Fusion-Parameter
    fusion_strategy: FusionStrategy = FusionStrategy.RRF
    vector_weight: float = 0.7         # Gewichtung Vector-Suche (0.0-1.0)
    keyword_weight: float = 0.3        # Gewichtung Keyword-Suche (0.0-1.0)
    rrf_k: int = 60                   # RRF-Parameter k (Standard: 60)
    
    # Performance-Parameter
    parallel_search: bool = True       # Parallele Suche aktivieren
    max_workers: int = 2              # Thread-Pool-Größe  
    timeout_seconds: float = 30.0     # Timeout für Such-Operationen
    
    # Retrieval-Parameter pro Modalität
    vector_k_multiplier: float = 1.5  # Vector k = query.k * multiplier
    keyword_k_multiplier: float = 1.5 # Keyword k = query.k * multiplier
    min_score_threshold: float = 0.0  # Minimaler Score für Ergebnisse
    
    # Fallback-Verhalten
    fallback_to_vector: bool = True    # Vector-only bei Keyword-Fehler
    fallback_to_keyword: bool = True   # Keyword-only bei Vector-Fehler  
    require_both_sources: bool = False # Beide Quellen erforderlich
    
    # Quality-Filter
    remove_duplicates: bool = True     # Duplikate entfernen
    duplicate_threshold: float = 0.95  # Ähnlichkeitsschwelle für Duplikate
    max_content_overlap: float = 0.8   # Max. Content-Überschneidung

    def __post_init__(self):
        """Validierung der Hybrid-Konfiguration"""
        super().__post_init__()
        
        # Gewichtungs-Validierung
        if not (0.0 <= self.vector_weight <= 1.0):
            raise ValidationError("vector_weight muss zwischen 0.0 und 1.0 liegen")
        
        if not (0.0 <= self.keyword_weight <= 1.0):
            raise ValidationError("keyword_weight muss zwischen 0.0 und 1.0 liegen")
        
        # Normalisierung der Gewichte
        total_weight = self.vector_weight + self.keyword_weight
        if total_weight > 0:
            self.vector_weight = self.vector_weight / total_weight
            self.keyword_weight = self.keyword_weight / total_weight
        
        # Parameter-Validierung
        if self.rrf_k < 1:
            raise ValidationError("rrf_k muss >= 1 sein")
            
        if self.max_workers < 1:
            self.max_workers = 1


# =============================================================================
# HYBRID RETRIEVER IMPLEMENTIERUNG  
# =============================================================================

class HybridRetriever(BaseRetriever):
    """
    Hybrid Retriever kombiniert Vector- und Keyword-basierte Suche
    
    Verwendet verschiedene Fusion-Strategien um die Vorteile beider
    Suchansätze zu kombinieren:
    - Vector-Suche: Semantische Ähnlichkeit, Kontext-Verständnis
    - Keyword-Suche: Exakte Begriffe, Fachterminologie, Präzision
    
    Features:
    - Reciprocal Rank Fusion (RRF) als Standard-Algorithmus
    - Parallele Ausführung für Performance-Optimierung  
    - Intelligente Duplikat-Erkennung und -Entfernung
    - Adaptive Modi basierend auf Query-Eigenschaften
    - Robuste Fallback-Mechanismen
    """
    
    def __init__(self, 
                 config: HybridRetrieverConfig,
                 vector_retriever = None,
                 keyword_retriever = None):
        """
        Initialisiert Hybrid Retriever
        
        Args:
            config: Hybrid Retriever-Konfiguration
            vector_retriever: Vector-basierter Retriever (Injection)
            keyword_retriever: Keyword-basierter Retriever (Injection)
        """
        super().__init__(config)
        self.hybrid_config = config
        
        # Retriever-Komponenten
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        
        # Thread Pool für parallele Suche
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers) if config.parallel_search else None
        
        # Performance-Metriken  
        self._vector_queries = 0
        self._keyword_queries = 0
        self._fusion_operations = 0
        self._fallback_count = 0
        self._duplicate_removals = 0
        
        self.logger.info(f"Hybrid Retriever initialisiert: {config.name}")
        self.logger.info(f"Fusion: {config.fusion_strategy.value}, Gewichte: V={config.vector_weight:.2f}, K={config.keyword_weight:.2f}")

    def _retrieve_impl(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Implementiert Hybrid-Retrieval-Logik
        
        Args:
            query: Retrieval-Query mit Parametern
            
        Returns:
            RetrievalResult: Fusionierte Ergebnisse aus beiden Retrieval-Modi
        """
        try:
            # Query-Anpassung für Sub-Retriever
            vector_k = max(1, int(query.k * self.hybrid_config.vector_k_multiplier))
            keyword_k = max(1, int(query.k * self.hybrid_config.keyword_k_multiplier))
            
            # Parallel oder sequenziell ausführen
            if self.hybrid_config.parallel_search and self.executor:
                vector_results, keyword_results = self._parallel_search(query, vector_k, keyword_k)
            else:
                vector_results, keyword_results = self._sequential_search(query, vector_k, keyword_k)
            
            # Fusion-Strategie anwenden
            fused_results = self._fuse_results(
                vector_results=vector_results,
                keyword_results=keyword_results, 
                original_query=query
            )
            
            # Post-Processing
            final_results = self._post_process_results(fused_results, query)
            
            # Metriken aktualisieren  
            self._fusion_operations += 1
            
            return RetrievalResult(
                documents=final_results[:query.k],  # Top-k Limitierung
                query=query,
                total_found=len(final_results),
                processing_time_ms=0.0,  # Wird von BaseRetriever gesetzt
                metadata={
                    'fusion_strategy': self.hybrid_config.fusion_strategy.value,
                    'vector_results': len(vector_results) if vector_results else 0,
                    'keyword_results': len(keyword_results) if keyword_results else 0,
                    'fusion_count': len(fused_results),
                    'duplicates_removed': self._duplicate_removals
                }
            )
            
        except Exception as e:
            error_context = create_error_context(
                operation="hybrid_retrieve", 
                query=query.text,
                config=self.hybrid_config.name
            )
            self.logger.error(f"Hybrid Retrieval Fehler: {e}", extra=error_context)
            raise

    def _parallel_search(self, 
                        query: RetrievalQuery, 
                        vector_k: int, 
                        keyword_k: int) -> Tuple[Optional[List], Optional[List]]:
        """
        Führt parallele Suche in Vector- und Keyword-Retriever aus
        
        Args:
            query: Original Query
            vector_k: k für Vector-Suche  
            keyword_k: k für Keyword-Suche
            
        Returns:
            Tuple[vector_results, keyword_results]: Ergebnisse beider Retriever
        """
        vector_results = None
        keyword_results = None
        
        # Futures für parallele Ausführung
        futures = {}
        
        # Vector-Suche starten
        if self.vector_retriever:
            vector_query = RetrievalQuery(
                text=query.text,
                k=vector_k,
                filters=query.filters,
                mode=query.mode,
                score_threshold=query.score_threshold,
                metadata=query.metadata
            )
            futures['vector'] = self.executor.submit(self._safe_vector_search, vector_query)
        
        # Keyword-Suche starten  
        if self.keyword_retriever:
            keyword_query = RetrievalQuery(
                text=query.text,
                k=keyword_k,
                filters=query.filters,
                mode=query.mode,
                score_threshold=query.score_threshold,
                metadata=query.metadata
            )
            futures['keyword'] = self.executor.submit(self._safe_keyword_search, keyword_query)
        
        # Ergebnisse sammeln
        completed_futures = as_completed(futures.values(), timeout=self.hybrid_config.timeout_seconds)
        
        try:
            for future in completed_futures:
                result = future.result()
                if result['type'] == 'vector':
                    vector_results = result['documents']
                    self._vector_queries += 1
                elif result['type'] == 'keyword':  
                    keyword_results = result['documents']
                    self._keyword_queries += 1
                    
        except Exception as e:
            self.logger.warning(f"Parallel search error: {e}")
            
        return vector_results, keyword_results

    def _sequential_search(self, 
                          query: RetrievalQuery,
                          vector_k: int, 
                          keyword_k: int) -> Tuple[Optional[List], Optional[List]]:
        """
        Führt sequenzielle Suche aus (Fallback wenn kein Threading)
        
        Args:
            query: Original Query
            vector_k: k für Vector-Suche
            keyword_k: k für Keyword-Suche
            
        Returns:
            Tuple[vector_results, keyword_results]: Ergebnisse beider Retriever
        """
        vector_results = None
        keyword_results = None
        
        # Vector-Suche
        if self.vector_retriever:
            try:
                vector_query = RetrievalQuery(
                    text=query.text, k=vector_k, filters=query.filters,
                    mode=query.mode, score_threshold=query.score_threshold
                )
                result = self.vector_retriever.retrieve(vector_query)
                vector_results = result.documents
                self._vector_queries += 1
            except Exception as e:
                self.logger.warning(f"Vector search failed: {e}")
        
        # Keyword-Suche
        if self.keyword_retriever:
            try:
                keyword_query = RetrievalQuery(
                    text=query.text, k=keyword_k, filters=query.filters,
                    mode=query.mode, score_threshold=query.score_threshold
                )
                result = self.keyword_retriever.retrieve(keyword_query)
                keyword_results = result.documents  
                self._keyword_queries += 1
            except Exception as e:
                self.logger.warning(f"Keyword search failed: {e}")
                
        return vector_results, keyword_results

    def _safe_vector_search(self, query: RetrievalQuery) -> Dict[str, Any]:
        """Thread-sichere Vector-Suche mit Error-Handling"""
        try:
            if not self.vector_retriever:
                return {'type': 'vector', 'documents': [], 'error': 'No vector retriever'}
                
            result = self.vector_retriever.retrieve(query)
            return {'type': 'vector', 'documents': result.documents}
        except Exception as e:
            self.logger.warning(f"Vector retriever error: {e}")
            return {'type': 'vector', 'documents': [], 'error': str(e)}

    def _safe_keyword_search(self, query: RetrievalQuery) -> Dict[str, Any]:
        """Thread-sichere Keyword-Suche mit Error-Handling"""
        try:
            if not self.keyword_retriever:
                return {'type': 'keyword', 'documents': [], 'error': 'No keyword retriever'}
                
            result = self.keyword_retriever.retrieve(query) 
            return {'type': 'keyword', 'documents': result.documents}
        except Exception as e:
            self.logger.warning(f"Keyword retriever error: {e}")
            return {'type': 'keyword', 'documents': [], 'error': str(e)}

    def _fuse_results(self, 
                     vector_results: Optional[List[Tuple[Document, float]]], 
                     keyword_results: Optional[List[Tuple[Document, float]]],
                     original_query: RetrievalQuery) -> List[Tuple[Document, float]]:
        """
        Fusioniert Ergebnisse beider Retriever mit gewählter Strategie
        
        Args:
            vector_results: Vector-Retrieval Ergebnisse
            keyword_results: Keyword-Retrieval Ergebnisse  
            original_query: Original Query für Kontext
            
        Returns:
            List[Tuple[Document, float]]: Fusionierte und sortierte Ergebnisse
        """
        # Fallback-Behandlung
        if not vector_results and not keyword_results:
            self.logger.warning("Keine Ergebnisse von beiden Retrievern")
            return []
        
        if not vector_results and self.hybrid_config.fallback_to_keyword:
            self.logger.info("Fallback zu Keyword-only Retrieval")
            self._fallback_count += 1
            return keyword_results or []
        
        if not keyword_results and self.hybrid_config.fallback_to_vector:
            self.logger.info("Fallback zu Vector-only Retrieval") 
            self._fallback_count += 1
            return vector_results or []
        
        if self.hybrid_config.require_both_sources and (not vector_results or not keyword_results):
            self.logger.warning("Beide Quellen erforderlich, aber nicht verfügbar")
            return []
        
        # Fusion-Strategie anwenden
        if self.hybrid_config.fusion_strategy == FusionStrategy.RRF:
            return self._reciprocal_rank_fusion(vector_results or [], keyword_results or [])
        elif self.hybrid_config.fusion_strategy == FusionStrategy.WEIGHTED_SUM:
            return self._weighted_sum_fusion(vector_results or [], keyword_results or [])
        elif self.hybrid_config.fusion_strategy == FusionStrategy.MAX_SCORE:
            return self._max_score_fusion(vector_results or [], keyword_results or [])
        elif self.hybrid_config.fusion_strategy == FusionStrategy.AVERAGE:
            return self._average_fusion(vector_results or [], keyword_results or [])
        else:
            # Default: RRF
            return self._reciprocal_rank_fusion(vector_results or [], keyword_results or [])

    def _reciprocal_rank_fusion(self, 
                               vector_results: List[Tuple[Document, float]], 
                               keyword_results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Reciprocal Rank Fusion - State-of-the-Art Hybrid Algorithm
        
        RRF Score = Σ(1 / (rank + k)) für jedes Vorkommen des Dokuments
        
        Args:
            vector_results: Vector-Ergebnisse mit Scores
            keyword_results: Keyword-Ergebnisse mit Scores
            
        Returns:
            List[Tuple[Document, float]]: Fusionierte Ergebnisse nach RRF-Score sortiert
        """
        rrf_scores: Dict[str, float] = {}
        document_map: Dict[str, Document] = {}
        k = self.hybrid_config.rrf_k
        
        # Vector-Results verarbeiten (Gewichtung anwenden)
        for rank, (doc, score) in enumerate(vector_results):
            doc_key = self._get_document_key(doc)
            rrf_score = self.hybrid_config.vector_weight / (rank + 1 + k)
            
            if doc_key in rrf_scores:
                rrf_scores[doc_key] += rrf_score
            else:
                rrf_scores[doc_key] = rrf_score
                document_map[doc_key] = doc
        
        # Keyword-Results verarbeiten (Gewichtung anwenden)  
        for rank, (doc, score) in enumerate(keyword_results):
            doc_key = self._get_document_key(doc)
            rrf_score = self.hybrid_config.keyword_weight / (rank + 1 + k)
            
            if doc_key in rrf_scores:
                rrf_scores[doc_key] += rrf_score
            else:
                rrf_scores[doc_key] = rrf_score
                document_map[doc_key] = doc
        
        # Nach RRF-Score sortieren (absteigend)
        sorted_results = [
            (document_map[doc_key], score)
            for doc_key, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return sorted_results

    def _weighted_sum_fusion(self, 
                            vector_results: List[Tuple[Document, float]], 
                            keyword_results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Gewichtete Summe der normalisierten Scores
        
        Final Score = vector_weight * norm_vector_score + keyword_weight * norm_keyword_score
        """
        # Score-Normalisierung 
        vector_scores = self._normalize_scores([score for _, score in vector_results])
        keyword_scores = self._normalize_scores([score for _, score in keyword_results])
        
        combined_scores: Dict[str, float] = {}
        document_map: Dict[str, Document] = {}
        
        # Vector-Scores hinzufügen
        for i, (doc, original_score) in enumerate(vector_results):
            doc_key = self._get_document_key(doc)
            normalized_score = vector_scores[i] if i < len(vector_scores) else 0.0
            weighted_score = self.hybrid_config.vector_weight * normalized_score
            
            combined_scores[doc_key] = weighted_score
            document_map[doc_key] = doc
        
        # Keyword-Scores hinzufügen
        for i, (doc, original_score) in enumerate(keyword_results):
            doc_key = self._get_document_key(doc)
            normalized_score = keyword_scores[i] if i < len(keyword_scores) else 0.0
            weighted_score = self.hybrid_config.keyword_weight * normalized_score
            
            if doc_key in combined_scores:
                combined_scores[doc_key] += weighted_score
            else:
                combined_scores[doc_key] = weighted_score
                document_map[doc_key] = doc
        
        # Sortieren nach kombinierten Scores
        sorted_results = [
            (document_map[doc_key], score)
            for doc_key, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return sorted_results

    def _max_score_fusion(self, 
                         vector_results: List[Tuple[Document, float]], 
                         keyword_results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Maximum-Score Fusion - nimmt den höchsten Score pro Dokument"""
        max_scores: Dict[str, float] = {}
        document_map: Dict[str, Document] = {}
        
        # Vector-Scores sammeln
        for doc, score in vector_results:
            doc_key = self._get_document_key(doc)
            max_scores[doc_key] = score
            document_map[doc_key] = doc
        
        # Keyword-Scores vergleichen  
        for doc, score in keyword_results:
            doc_key = self._get_document_key(doc)
            if doc_key in max_scores:
                max_scores[doc_key] = max(max_scores[doc_key], score)
            else:
                max_scores[doc_key] = score
                document_map[doc_key] = doc
        
        # Sortieren nach Max-Score
        sorted_results = [
            (document_map[doc_key], score)
            for doc_key, score in sorted(max_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return sorted_results

    def _average_fusion(self, 
                       vector_results: List[Tuple[Document, float]], 
                       keyword_results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Durchschnitts-Fusion - mittelt Scores bei überlappenden Dokumenten"""
        score_data: Dict[str, List[float]] = {}
        document_map: Dict[str, Document] = {}
        
        # Alle Scores sammeln
        for doc, score in vector_results + keyword_results:
            doc_key = self._get_document_key(doc)
            if doc_key in score_data:
                score_data[doc_key].append(score)
            else:
                score_data[doc_key] = [score]
                document_map[doc_key] = doc
        
        # Durchschnitt berechnen
        avg_scores = {
            doc_key: sum(scores) / len(scores)
            for doc_key, scores in score_data.items()
        }
        
        # Sortieren nach Durchschnitts-Score
        sorted_results = [
            (document_map[doc_key], score)
            for doc_key, score in sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return sorted_results

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Min-Max-Normalisierung der Scores auf [0, 1]
        
        Args:
            scores: Liste der Original-Scores
            
        Returns:
            List[float]: Normalisierte Scores
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)  # Alle gleich -> alle 1.0
        
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def _get_document_key(self, doc: Document) -> str:
        """
        Erstellt eindeutigen Key für Dokument-Identifikation
        
        Args:
            doc: Dokument
            
        Returns:
            str: Eindeutiger Document-Key
        """
        # Priorität: doc_id > source > content_hash
        if doc.doc_id:
            return f"id:{doc.doc_id}"
        elif doc.source:
            return f"src:{doc.source}"
        else:
            # Content-Hash als Fallback
            content_hash = hash(doc.content[:200])  # Erste 200 Zeichen
            return f"hash:{content_hash}"

    def _post_process_results(self, 
                             results: List[Tuple[Document, float]], 
                             query: RetrievalQuery) -> List[Tuple[Document, float]]:
        """
        Post-Processing der Fusion-Ergebnisse
        
        Args:
            results: Fusionierte Ergebnisse
            query: Original Query für Kontext
            
        Returns:
            List[Tuple[Document, float]]: Gefilterte und optimierte Ergebnisse
        """
        processed_results = results
        
        # Score-Filtering  
        if self.hybrid_config.min_score_threshold > 0.0:
            processed_results = [
                (doc, score) for doc, score in processed_results
                if score >= self.hybrid_config.min_score_threshold
            ]
        
        # Duplikat-Entfernung
        if self.hybrid_config.remove_duplicates:
            processed_results = self._remove_duplicates(processed_results)
        
        return processed_results

    def _remove_duplicates(self, 
                          results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Entfernt ähnliche/doppelte Dokumente basierend auf Content-Similarity
        
        Args:
            results: Ergebnisliste mit potentiellen Duplikaten
            
        Returns:
            List[Tuple[Document, float]]: Bereinigte Ergebnisse ohne Duplikate
        """
        if not results or len(results) <= 1:
            return results
        
        unique_results = []
        seen_contents = []
        
        for doc, score in results:
            is_duplicate = False
            
            # Vergleich mit bereits gesehenen Inhalten
            for seen_content in seen_contents:
                similarity = self._calculate_content_similarity(doc.content, seen_content)
                
                if similarity >= self.hybrid_config.duplicate_threshold:
                    is_duplicate = True
                    self._duplicate_removals += 1
                    break
            
            if not is_duplicate:
                unique_results.append((doc, score))
                seen_contents.append(doc.content)
        
        return unique_results

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Berechnet Ähnlichkeit zwischen zwei Content-Strings
        
        Verwendet Jaccard-Similarity auf Word-Level für Performance
        
        Args:
            content1: Erster Content-String
            content2: Zweiter Content-String
            
        Returns:
            float: Ähnlichkeit zwischen 0.0 und 1.0
        """
        # Tokenisierung (einfache Wort-Trennung)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        # Jaccard-Similarity: |intersection| / |union|
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _custom_health_check(self) -> Optional[Dict[str, Any]]:
        """
        Hybrid-spezifische Health-Checks
        
        Returns:
            Dict[str, Any]: Health-Status der Sub-Komponenten
        """
        health_data = {
            'vector_retriever_available': self.vector_retriever is not None,
            'keyword_retriever_available': self.keyword_retriever is not None,
            'parallel_search_enabled': self.hybrid_config.parallel_search,
            'fusion_strategy': self.hybrid_config.fusion_strategy.value
        }
        
        # Sub-Retriever Health-Checks
        if self.vector_retriever:
            try:
                vector_health = self.vector_retriever.health_check()
                health_data['vector_retriever_status'] = vector_health.get('status', 'unknown')
            except Exception as e:
                health_data['vector_retriever_status'] = 'error'
                health_data['vector_retriever_error'] = str(e)
        
        if self.keyword_retriever:
            try:
                keyword_health = self.keyword_retriever.health_check()  
                health_data['keyword_retriever_status'] = keyword_health.get('status', 'unknown')
            except Exception as e:
                health_data['keyword_retriever_status'] = 'error'
                health_data['keyword_retriever_error'] = str(e)
        
        # Performance-Metriken
        health_data.update({
            'total_vector_queries': self._vector_queries,
            'total_keyword_queries': self._keyword_queries,
            'total_fusion_operations': self._fusion_operations,
            'fallback_count': self._fallback_count,
            'duplicates_removed': self._duplicate_removals
        })
        
        # Thread-Pool Status
        if self.executor:
            health_data['thread_pool_active'] = True
            health_data['max_workers'] = self.hybrid_config.max_workers
        else:
            health_data['thread_pool_active'] = False
        
        return health_data

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Erweiterte Performance-Statistiken für Hybrid Retriever
        
        Returns:
            Dict[str, Any]: Detaillierte Performance-Metriken
        """
        base_stats = super().get_performance_stats()
        
        hybrid_stats = {
            'vector_queries': self._vector_queries,
            'keyword_queries': self._keyword_queries,
            'fusion_operations': self._fusion_operations,
            'fallback_count': self._fallback_count,
            'duplicates_removed': self._duplicate_removals,
            'fusion_strategy': self.hybrid_config.fusion_strategy.value,
            'vector_weight': self.hybrid_config.vector_weight,
            'keyword_weight': self.hybrid_config.keyword_weight
        }
        
        # Fusion-Rate berechnen
        if self._total_queries > 0:
            hybrid_stats['fusion_rate'] = self._fusion_operations / self._total_queries
            hybrid_stats['fallback_rate'] = self._fallback_count / self._total_queries
        
        # Stats kombinieren
        base_stats.update(hybrid_stats)
        return base_stats

    def update_weights(self, vector_weight: float, keyword_weight: float):
        """
        Dynamische Anpassung der Fusion-Gewichte zur Laufzeit
        
        Args:
            vector_weight: Neue Gewichtung für Vector-Suche (0.0-1.0)
            keyword_weight: Neue Gewichtung für Keyword-Suche (0.0-1.0)
        """
        if not (0.0 <= vector_weight <= 1.0):
            raise ValidationError("vector_weight muss zwischen 0.0 und 1.0 liegen")
        if not (0.0 <= keyword_weight <= 1.0):
            raise ValidationError("keyword_weight muss zwischen 0.0 und 1.0 liegen")
        
        # Normalisierung
        total = vector_weight + keyword_weight
        if total > 0:
            self.hybrid_config.vector_weight = vector_weight / total
            self.hybrid_config.keyword_weight = keyword_weight / total
        
        self.logger.info(f"Gewichte aktualisiert: Vector={self.hybrid_config.vector_weight:.2f}, Keyword={self.hybrid_config.keyword_weight:.2f}")

    def set_fusion_strategy(self, strategy: FusionStrategy):
        """
        Wechselt Fusion-Strategie zur Laufzeit
        
        Args:
            strategy: Neue Fusion-Strategie
        """
        old_strategy = self.hybrid_config.fusion_strategy
        self.hybrid_config.fusion_strategy = strategy
        
        self.logger.info(f"Fusion-Strategie geändert: {old_strategy.value} -> {strategy.value}")

    def __del__(self):
        """Cleanup beim Zerstören der Instanz"""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)


# =============================================================================
# ADAPTIVE HYBRID RETRIEVER
# =============================================================================

class AdaptiveHybridRetriever(HybridRetriever):
    """
    Erweiterte Hybrid-Implementierung mit adaptiver Gewichtung
    
    Passt Vector/Keyword-Gewichtung basierend auf Query-Eigenschaften an:
    - Fachbegriffe -> Höhere Keyword-Gewichtung
    - Konzeptuelle Fragen -> Höhere Vector-Gewichtung  
    - Query-Länge -> Anpassung der Retrieval-Parameter
    - Historical Performance -> Lernende Gewichtung
    """
    
    def __init__(self, 
                 config: HybridRetrieverConfig,
                 vector_retriever=None,
                 keyword_retriever=None):
        super().__init__(config, vector_retriever, keyword_retriever)
        
        # Adaptive Parameter
        self._query_history: List[Dict[str, Any]] = []
        self._max_history = 1000
        self._adaptation_enabled = True
        
        self.logger.info(f"Adaptive Hybrid Retriever initialisiert: {config.name}")

    def _retrieve_impl(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Adaptive Retrieval-Implementierung mit Query-Analyse
        
        Args:
            query: Retrieval-Query
            
        Returns:
            RetrievalResult: Optimierte Hybrid-Ergebnisse
        """
        # Query-Analyse für adaptive Gewichtung
        if self._adaptation_enabled:
            self._adapt_weights_for_query(query)
        
        # Standard Hybrid-Retrieval ausführen
        result = super()._retrieve_impl(query)
        
        # Query-Performance für zukünftige Anpassungen speichern
        self._record_query_performance(query, result)
        
        return result

    def _adapt_weights_for_query(self, query: RetrievalQuery):
        """
        Passt Gewichtung basierend auf Query-Eigenschaften an
        
        Args:
            query: Zu analysierende Query
        """
        original_vector_weight = self.hybrid_config.vector_weight
        original_keyword_weight = self.hybrid_config.keyword_weight
        
        # Query-Eigenschaften analysieren
        query_analysis = self._analyze_query(query.text)
        
        # Gewichtungsanpassung basierend auf Analyse
        vector_weight = original_vector_weight
        keyword_weight = original_keyword_weight
        
        # Technische Begriffe -> Mehr Keyword-Gewichtung
        if query_analysis['technical_term_ratio'] > 0.3:
            keyword_weight *= 1.2
            vector_weight *= 0.8
        
        # Lange konzeptuelle Queries -> Mehr Vector-Gewichtung  
        if query_analysis['word_count'] > 10 and query_analysis['concept_indicators'] > 2:
            vector_weight *= 1.3
            keyword_weight *= 0.7
        
        # Kurze präzise Queries -> Mehr Keyword-Gewichtung
        if query_analysis['word_count'] <= 3:
            keyword_weight *= 1.4
            vector_weight *= 0.6
        
        # Normalisierung und Anwendung
        total = vector_weight + keyword_weight
        if total > 0:
            self.hybrid_config.vector_weight = vector_weight / total
            self.hybrid_config.keyword_weight = keyword_weight / total

    def _analyze_query(self, query_text: str) -> Dict[str, Any]:
        """
        Analysiert Query-Eigenschaften für adaptive Gewichtung
        
        Args:
            query_text: Query-Text
            
        Returns:
            Dict[str, Any]: Query-Analyse-Ergebnisse
        """
        words = query_text.lower().split()
        word_count = len(words)
        
        # Technische Begriffe (industrielle Fachterminologie)
        technical_terms = {
            'sps', 'plc', 'hmi', 'scada', 'profibus', 'profinet', 'modbus',
            'siemens', 'allen', 'bradley', 'schneider', 'omron', 'mitsubishi',
            'motor', 'antrieb', 'servo', 'frequenz', 'umrichter', 'encoder',
            'sensor', 'aktor', 'ventil', 'pneumatik', 'hydraulik',
            'temperatur', 'druck', 'durchfluss', 'level', 'position',
            'sicherheit', 'safety', 'kategorie', 'sil', 'pfd', 'mttf',
            'wartung', 'maintenance', 'diagnose', 'fehler', 'alarm', 'störung'
        }
        
        technical_word_count = sum(1 for word in words if word in technical_terms)
        technical_term_ratio = technical_word_count / word_count if word_count > 0 else 0
        
        # Konzeptuelle Indikatoren
        concept_indicators = {
            'warum', 'wieso', 'weshalb', 'wie', 'was', 'wann', 'wo',
            'why', 'how', 'what', 'when', 'where',
            'funktioniert', 'arbeitet', 'unterschied', 'vergleich',
            'prinzip', 'konzept', 'theorie', 'grundlagen'
        }
        
        concept_indicator_count = sum(1 for word in words if word in concept_indicators)
        
        # Fragetyp erkennen
        question_words = {'was', 'wie', 'warum', 'wann', 'wo', 'what', 'how', 'why', 'when', 'where'}
        is_question = any(word in question_words for word in words) or query_text.endswith('?')
        
        return {
            'word_count': word_count,
            'technical_term_ratio': technical_term_ratio,
            'concept_indicators': concept_indicator_count,
            'is_question': is_question,
            'query_length': len(query_text)
        }

    def _record_query_performance(self, query: RetrievalQuery, result: RetrievalResult):
        """
        Speichert Query-Performance für zukünftige Optimierungen
        
        Args:
            query: Ausgeführte Query
            result: Erhaltenes Ergebnis
        """
        performance_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'query_text': query.text,
            'query_length': len(query.text),
            'result_count': len(result.documents),
            'avg_score': result.average_score,
            'processing_time': result.processing_time_ms,
            'vector_weight_used': self.hybrid_config.vector_weight,
            'keyword_weight_used': self.hybrid_config.keyword_weight,
            'fusion_strategy': self.hybrid_config.fusion_strategy.value
        }
        
        self._query_history.append(performance_record)
        
        # History-Limit einhalten
        if len(self._query_history) > self._max_history:
            self._query_history.pop(0)

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """
        Statistiken zur adaptiven Gewichtung
        
        Returns:
            Dict[str, Any]: Adaptive Performance-Metriken
        """
        if not self._query_history:
            return {'adaptation_enabled': self._adaptation_enabled, 'query_history_size': 0}
        
        # Durchschnittliche Scores nach Gewichtung
        vector_heavy_queries = [q for q in self._query_history if q['vector_weight_used'] > 0.6]
        keyword_heavy_queries = [q for q in self._query_history if q['keyword_weight_used'] > 0.6]
        
        stats = {
            'adaptation_enabled': self._adaptation_enabled,
            'query_history_size': len(self._query_history),
            'vector_heavy_queries': len(vector_heavy_queries),
            'keyword_heavy_queries': len(keyword_heavy_queries)
        }
        
        if vector_heavy_queries:
            stats['avg_vector_heavy_score'] = sum(q['avg_score'] for q in vector_heavy_queries) / len(vector_heavy_queries)
        
        if keyword_heavy_queries:
            stats['avg_keyword_heavy_score'] = sum(q['avg_score'] for q in keyword_heavy_queries) / len(keyword_heavy_queries)
        
        return stats


# =============================================================================
# FACTORY UND REGISTRY INTEGRATION
# =============================================================================

def create_hybrid_retriever(config: Dict[str, Any], 
                           vector_retriever=None, 
                           keyword_retriever=None) -> HybridRetriever:
    """
    Factory-Funktion für Hybrid Retriever-Erstellung
    
    Args:
        config: Konfiguration als Dictionary
        vector_retriever: Vector-Retriever Instanz
        keyword_retriever: Keyword-Retriever Instanz
        
    Returns:
        HybridRetriever: Konfigurierte Hybrid-Retriever Instanz
    """
    # Config-Objekt erstellen
    hybrid_config = HybridRetrieverConfig(
        name=config.get('name', 'hybrid_retriever'),
        description=config.get('description', 'Hybrid Vector/Keyword Retriever'),
        fusion_strategy=FusionStrategy(config.get('fusion_strategy', 'reciprocal_rank_fusion')),
        vector_weight=config.get('vector_weight', 0.7),
        keyword_weight=config.get('keyword_weight', 0.3),
        parallel_search=config.get('parallel_search', True),
        **{k: v for k, v in config.items() if k not in ['name', 'description', 'fusion_strategy', 'vector_weight', 'keyword_weight', 'parallel_search']}
    )
    
    # Adaptive oder Standard Hybrid Retriever
    if config.get('adaptive', False):
        return AdaptiveHybridRetriever(hybrid_config, vector_retriever, keyword_retriever)
    else:
        return HybridRetriever(hybrid_config, vector_retriever, keyword_retriever)


# Registrierung im Retriever-Registry (wird von __init__.py aufgerufen)
def register_hybrid_retrievers():
    """Registriert Hybrid Retriever im globalen Registry"""
    from .base_retriever import RetrieverRegistry
    
    RetrieverRegistry.register('hybrid', HybridRetriever)
    RetrieverRegistry.register('adaptive_hybrid', AdaptiveHybridRetriever)