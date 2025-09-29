#!/usr/bin/env python3
"""
Semantic Retriever - Pure Vector-based Search
Industrielle RAG-Architektur - Module Layer

Spezialisierter Retriever für rein semantische, vector-basierte Suche
mit erweiterten Similarity-Algorithmen und intelligenter Query-Verarbeitung
für industrielle RAG-Anwendungen.

Features:
- Reine Embedding-basierte semantische Suche ohne Keyword-Komponenten
- Multiple Similarity-Metriken (Cosine, Dot-Product, Euclidean)
- Query-Expansion und semantische Augmentierung
- Kontext-bewusste Suche mit Multi-Vector-Strategien
- Production-Features: Caching, Performance-Monitoring, Batch-Processing

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

import numpy as np
import math
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
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
# SEMANTIC RETRIEVER KONFIGURATION
# =============================================================================

class SimilarityMetric(str, Enum):
    """Similarity-Metriken für Vector-Vergleiche"""
    COSINE = "cosine"                    # Cosine Similarity (Standard)
    DOT_PRODUCT = "dot_product"          # Dot Product (für normalisierte Vektoren)
    EUCLIDEAN = "euclidean"              # Euclidean Distance (invertiert)
    MANHATTAN = "manhattan"              # Manhattan Distance (invertiert)
    ANGULAR = "angular"                  # Angular Distance (Cosine-basiert)


class QueryExpansionStrategy(str, Enum):
    """Strategien für Query-Erweiterung"""
    NONE = "none"                        # Keine Erweiterung
    SYNONYMS = "synonyms"                # Synonym-basierte Erweiterung
    SEMANTIC_NEIGHBORS = "semantic_neighbors"  # Semantisch ähnliche Begriffe
    DOMAIN_SPECIFIC = "domain_specific"   # Industrielle Fachbegriffe
    MULTILINGUAL = "multilingual"         # Mehrsprachige Erweiterung


@dataclass
class SemanticRetrieverConfig(RetrieverConfig):
    """Erweiterte Konfiguration für Semantic Retriever"""
    
    # Similarity-Parameter
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    similarity_threshold: float = 0.1    # Minimale Ähnlichkeit für Ergebnisse
    normalize_vectors: bool = True       # Vector-Normalisierung aktivieren
    
    # Query-Processing
    query_expansion: QueryExpansionStrategy = QueryExpansionStrategy.NONE
    max_query_length: int = 512         # Maximale Query-Länge (Token)
    query_preprocessing: bool = True     # Query-Preprocessing aktivieren
    
    # Retrieval-Parameter
    reranking_enabled: bool = False      # Re-Ranking nach initialer Suche
    diversity_penalty: float = 0.0      # Diversitäts-Penalty (0.0-1.0)
    temporal_boost: bool = False        # Zeitliche Relevanz-Boost
    
    # Multi-Vector-Strategien
    multi_vector_strategy: str = "single"  # single, average, max, concat
    chunk_overlap_handling: str = "best"  # best, average, combine
    
    # Performance-Optimierung
    early_stopping: bool = True         # Early Stopping bei ausreichenden Ergebnissen
    batch_size: int = 32                # Batch-Größe für Embedding-Verarbeitung
    vector_cache_size: int = 10000      # Cache-Größe für Vector-Embeddings
    
    # Industrielle Features
    technical_term_boost: float = 1.2   # Boost für technische Begriffe
    safety_category_weight: float = 1.5 # Gewichtung für Safety-Kategorien
    domain_adaptation: bool = True      # Domain-spezifische Anpassungen

    def __post_init__(self):
        """Validierung der Semantic Retriever-Konfiguration"""
        super().__post_init__()
        
        # Similarity-Threshold Validierung
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValidationError("similarity_threshold muss zwischen 0.0 und 1.0 liegen")
        
        # Diversity-Penalty Validierung
        if not (0.0 <= self.diversity_penalty <= 1.0):
            raise ValidationError("diversity_penalty muss zwischen 0.0 und 1.0 liegen")
        
        # Boost-Parameter Validierung
        if self.technical_term_boost < 0.1:
            self.technical_term_boost = 1.0
        if self.safety_category_weight < 0.1:
            self.safety_category_weight = 1.0
        
        # Cache-Parameter
        if self.vector_cache_size < 100:
            self.vector_cache_size = 1000


# =============================================================================
# SEMANTIC RETRIEVER IMPLEMENTIERUNG  
# =============================================================================

class SemanticRetriever(BaseRetriever):
    """
    Semantic Retriever für reine Vector-basierte Suche
    
    Spezialisiert auf semantische Ähnlichkeitssuche ohne Keyword-Komponenten.
    Nutzt verschiedene Vector-Similarity-Metriken und erweiterte Query-Processing
    für optimale Ergebnisse in industriellen Anwendungen.
    
    Features:
    - Multiple Similarity-Metriken für verschiedene Anwendungsfälle
    - Query-Expansion für bessere semantische Abdeckung
    - Multi-Vector-Strategien für komplexe Dokumente
    - Industrielle Domain-Anpassungen
    - Re-Ranking und Diversitäts-Optimierung
    """
    
    def __init__(self, 
                 config: SemanticRetrieverConfig,
                 vector_store=None,
                 embedding_service=None):
        """
        Initialisiert Semantic Retriever
        
        Args:
            config: Semantic Retriever-Konfiguration
            vector_store: Vector Store für Embeddings (Injection)
            embedding_service: Embedding Service für Query-Verarbeitung (Injection)
        """
        super().__init__(config)
        self.semantic_config = config
        
        # Dependencies
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
        # Vector-Cache für Performance-Optimierung
        self._vector_cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Performance-Metriken  
        self._embedding_queries = 0
        self._similarity_calculations = 0
        self._reranking_operations = 0
        self._query_expansions = 0
        
        # Industrielle Terminologie
        self._technical_terms = self._load_technical_terminology()
        self._safety_keywords = self._load_safety_keywords()
        
        self.logger.info(f"Semantic Retriever initialisiert: {config.name}")
        self.logger.info(f"Similarity: {config.similarity_metric.value}, Expansion: {config.query_expansion.value}")

    def _retrieve_impl(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Implementiert semantische Retrieval-Logik
        
        Args:
            query: Retrieval-Query mit Parametern
            
        Returns:
            RetrievalResult: Semantisch ähnliche Dokumente mit Scores
        """
        try:
            # Query-Preprocessing und -Expansion
            processed_query = self._preprocess_query(query)
            expanded_queries = self._expand_query(processed_query) if self.semantic_config.query_expansion != QueryExpansionStrategy.NONE else [processed_query]
            
            # Query-Embeddings generieren
            query_embeddings = self._get_query_embeddings(expanded_queries)
            
            # Vector-Suche durchführen
            candidate_docs = self._vector_search(query_embeddings, query)
            
            # Similarity-Scores berechnen
            scored_documents = self._calculate_similarities(
                query_embeddings, candidate_docs, query
            )
            
            # Optional: Re-Ranking anwenden
            if self.semantic_config.reranking_enabled and len(scored_documents) > 1:
                scored_documents = self._rerank_results(scored_documents, query)
                self._reranking_operations += 1
            
            # Diversitäts-Filter anwenden
            if self.semantic_config.diversity_penalty > 0.0:
                scored_documents = self._apply_diversity_penalty(scored_documents)
            
            # Post-Processing
            final_results = self._post_process_semantic_results(scored_documents, query)
            
            return RetrievalResult(
                documents=final_results[:query.k],
                query=query,
                total_found=len(final_results),
                processing_time_ms=0.0,  # Wird von BaseRetriever gesetzt
                metadata={
                    'similarity_metric': self.semantic_config.similarity_metric.value,
                    'query_expansions': len(expanded_queries) - 1,
                    'candidates_found': len(candidate_docs),
                    'reranking_applied': self.semantic_config.reranking_enabled,
                    'diversity_penalty': self.semantic_config.diversity_penalty
                }
            )
            
        except Exception as e:
            error_context = create_error_context(
                operation="semantic_retrieve", 
                query=query.text,
                config=self.semantic_config.name
            )
            self.logger.error(f"Semantic Retrieval Fehler: {e}", extra=error_context)
            raise

    def _preprocess_query(self, query: RetrievalQuery) -> str:
        """
        Preprocesses Query-Text für optimale semantische Suche
        
        Args:
            query: Original Query
            
        Returns:
            str: Preprocessed Query-Text
        """
        if not self.semantic_config.query_preprocessing:
            return query.text
        
        text = query.text.strip().lower()
        
        # Längen-Limitierung
        if len(text) > self.semantic_config.max_query_length:
            text = text[:self.semantic_config.max_query_length]
            self.logger.warning(f"Query gekürzt auf {self.semantic_config.max_query_length} Zeichen")
        
        # Industrielle Domain-Anpassungen
        if self.semantic_config.domain_adaptation:
            text = self._adapt_for_industrial_domain(text)
        
        return text

    def _adapt_for_industrial_domain(self, text: str) -> str:
        """
        Passt Query-Text für industrielle Domäne an
        
        Args:
            text: Original Query-Text
            
        Returns:
            str: Domain-angepasster Text
        """
        # Technische Abkürzungen expandieren
        abbreviations = {
            'sps': 'speicherprogrammierbare steuerung',
            'hmi': 'human machine interface',
            'scada': 'supervisory control and data acquisition',
            'plc': 'programmable logic controller',
            'vfd': 'variable frequency drive',
            'io': 'input output',
            'ai': 'analog input',
            'di': 'digital input'
        }
        
        words = text.split()
        expanded_words = []
        
        for word in words:
            if word in abbreviations:
                expanded_words.extend([word, abbreviations[word]])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)

    def _expand_query(self, query_text: str) -> List[str]:
        """
        Erweitert Query mit zusätzlichen semantischen Begriffen
        
        Args:
            query_text: Basis Query-Text
            
        Returns:
            List[str]: Liste erweiterter Queries
        """
        expanded_queries = [query_text]  # Original Query immer enthalten
        
        if self.semantic_config.query_expansion == QueryExpansionStrategy.SYNONYMS:
            expanded_queries.extend(self._get_synonyms(query_text))
        elif self.semantic_config.query_expansion == QueryExpansionStrategy.DOMAIN_SPECIFIC:
            expanded_queries.extend(self._get_domain_expansions(query_text))
        elif self.semantic_config.query_expansion == QueryExpansionStrategy.MULTILINGUAL:
            expanded_queries.extend(self._get_multilingual_expansions(query_text))
        
        # Duplikate entfernen
        expanded_queries = list(set(expanded_queries))
        
        if len(expanded_queries) > 1:
            self._query_expansions += 1
            self.logger.debug(f"Query erweitert: {len(expanded_queries)} Varianten")
        
        return expanded_queries

    def _get_synonyms(self, query_text: str) -> List[str]:
        """Generiert Synonyme für Query-Expansion"""
        # Basis-Implementierung - kann durch externe Synonym-Datenbank erweitert werden
        industrial_synonyms = {
            'fehler': ['störung', 'defekt', 'problem', 'alarm'],
            'motor': ['antrieb', 'maschine', 'aggregat'],
            'sensor': ['messfühler', 'detektor', 'aufnehmer'],
            'ventil': ['absperrorgan', 'regelventil', 'stellglied'],
            'pumpe': ['verdichter', 'gebläse', 'kompressor'],
            'temperatur': ['wärme', 'hitze', 'grad'],
            'druck': ['kraft', 'belastung', 'spannung'],
            'sicherheit': ['safety', 'protection', 'schutz']
        }
        
        synonyms = []
        words = query_text.lower().split()
        
        for word in words:
            if word in industrial_synonyms:
                for synonym in industrial_synonyms[word]:
                    synonym_query = query_text.lower().replace(word, synonym)
                    synonyms.append(synonym_query)
        
        return synonyms[:3]  # Limitierung auf 3 Synonyme

    def _get_domain_expansions(self, query_text: str) -> List[str]:
        """Generiert domänen-spezifische Query-Erweiterungen"""
        expansions = []
        
        # Wenn technische Begriffe erkannt werden, füge Kontext hinzu
        for term in self._technical_terms:
            if term in query_text.lower():
                # Kontext-Erweiterung für technische Begriffe
                context_expansions = [
                    f"{query_text} wartung",
                    f"{query_text} konfiguration", 
                    f"{query_text} troubleshooting",
                    f"{query_text} spezifikation"
                ]
                expansions.extend(context_expansions)
                break
        
        return expansions[:2]  # Limitierung auf 2 Erweiterungen

    def _get_multilingual_expansions(self, query_text: str) -> List[str]:
        """Generiert mehrsprachige Query-Erweiterungen"""
        # Basis Deutsch-Englisch Mapping für industrielle Begriffe
        translations = {
            'motor': 'engine drive',
            'pumpe': 'pump',
            'ventil': 'valve',
            'sensor': 'sensor detector',
            'steuerung': 'control controller',
            'sicherheit': 'safety protection',
            'fehler': 'error fault alarm',
            'wartung': 'maintenance service'
        }
        
        expansions = []
        words = query_text.lower().split()
        
        for word in words:
            if word in translations:
                english_query = query_text.lower().replace(word, translations[word])
                expansions.append(english_query)
        
        return expansions

    def _get_query_embeddings(self, queries: List[str]) -> List[np.ndarray]:
        """
        Generiert Embeddings für Query-Liste
        
        Args:
            queries: Liste der Query-Strings
            
        Returns:
            List[np.ndarray]: Query-Embeddings
        """
        if not self.embedding_service:
            raise ValidationError("Embedding Service nicht verfügbar")
        
        embeddings = []
        
        for query_text in queries:
            # Cache-Check
            cache_key = f"query:{hash(query_text)}"
            if cache_key in self._vector_cache:
                embeddings.append(self._vector_cache[cache_key])
                self._cache_hits += 1
            else:
                # Embedding generieren
                try:
                    embedding = self.embedding_service.embed_query(query_text)
                    if isinstance(embedding, list):
                        embedding = np.array(embedding, dtype=np.float32)
                    
                    # Normalisierung falls aktiviert
                    if self.semantic_config.normalize_vectors:
                        embedding = self._normalize_vector(embedding)
                    
                    embeddings.append(embedding)
                    
                    # Cache-Update (mit Größen-Limitierung)
                    if len(self._vector_cache) < self.semantic_config.vector_cache_size:
                        self._vector_cache[cache_key] = embedding
                    
                    self._cache_misses += 1
                    self._embedding_queries += 1
                    
                except Exception as e:
                    self.logger.warning(f"Embedding-Generierung fehlgeschlagen für '{query_text}': {e}")
                    continue
        
        if not embeddings:
            raise ValidationError("Keine gültigen Query-Embeddings generiert")
        
        return embeddings

    def _vector_search(self, query_embeddings: List[np.ndarray], query: RetrievalQuery) -> List[Tuple[Document, np.ndarray]]:
        """
        Führt Vector-Suche im Vector Store durch
        
        Args:
            query_embeddings: Query-Embeddings
            query: Original Query für Kontext
            
        Returns:
            List[Tuple[Document, np.ndarray]]: Kandidaten-Dokumente mit Embeddings
        """
        if not self.vector_store:
            raise ValidationError("Vector Store nicht verfügbar")
        
        all_candidates = []
        
        # Für jedes Query-Embedding suchen
        for embedding in query_embeddings:
            try:
                # Vector Store Query (implementierungsabhängig)
                candidates = self.vector_store.similarity_search_with_embeddings(
                    embedding=embedding.tolist(),
                    k=min(query.k * 2, 100),  # Mehr Kandidaten für bessere Auswahl
                    filters=query.filters,
                    score_threshold=self.semantic_config.similarity_threshold
                )
                
                # Ergebnisse sammeln
                for doc, doc_embedding, score in candidates:
                    if isinstance(doc_embedding, list):
                        doc_embedding = np.array(doc_embedding, dtype=np.float32)
                    
                    all_candidates.append((doc, doc_embedding))
                    
            except Exception as e:
                self.logger.warning(f"Vector-Suche fehlgeschlagen: {e}")
                continue
        
        # Duplikate entfernen (basierend auf doc_id oder content)
        unique_candidates = self._deduplicate_candidates(all_candidates)
        
        self.logger.debug(f"Vector-Suche: {len(unique_candidates)} eindeutige Kandidaten gefunden")
        return unique_candidates

    def _deduplicate_candidates(self, candidates: List[Tuple[Document, np.ndarray]]) -> List[Tuple[Document, np.ndarray]]:
        """
        Entfernt Duplikate aus Kandidaten-Liste
        
        Args:
            candidates: Kandidaten mit potentiellen Duplikaten
            
        Returns:
            List[Tuple[Document, np.ndarray]]: Deduplizierte Kandidaten
        """
        seen_ids = set()
        seen_hashes = set()
        unique_candidates = []
        
        for doc, embedding in candidates:
            # ID-basierte Deduplication (bevorzugt)
            if doc.doc_id and doc.doc_id in seen_ids:
                continue
            
            # Content-basierte Deduplication
            content_hash = hash(doc.content[:200])  # Erste 200 Zeichen
            if content_hash in seen_hashes:
                continue
            
            # Als eindeutig markieren
            if doc.doc_id:
                seen_ids.add(doc.doc_id)
            seen_hashes.add(content_hash)
            
            unique_candidates.append((doc, embedding))
        
        return unique_candidates

    def _calculate_similarities(self, 
                              query_embeddings: List[np.ndarray],
                              candidates: List[Tuple[Document, np.ndarray]],
                              query: RetrievalQuery) -> List[Tuple[Document, float]]:
        """
        Berechnet Similarity-Scores zwischen Queries und Dokumenten
        
        Args:
            query_embeddings: Query-Embeddings
            candidates: Kandidaten-Dokumente mit Embeddings
            query: Original Query für Kontext
            
        Returns:
            List[Tuple[Document, float]]: Dokumente mit Similarity-Scores
        """
        scored_documents = []
        
        for doc, doc_embedding in candidates:
            max_similarity = 0.0
            
            # Beste Similarity über alle Query-Embeddings finden
            for query_embedding in query_embeddings:
                similarity = self._compute_similarity(
                    query_embedding, doc_embedding, self.semantic_config.similarity_metric
                )
                max_similarity = max(max_similarity, similarity)
                self._similarity_calculations += 1
            
            # Industrielle Boosts anwenden
            boosted_similarity = self._apply_industrial_boosts(doc, max_similarity, query)
            
            # Threshold-Filter
            if boosted_similarity >= self.semantic_config.similarity_threshold:
                scored_documents.append((doc, boosted_similarity))
        
        # Nach Score sortieren (absteigend)
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        
        return scored_documents

    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray, metric: SimilarityMetric) -> float:
        """
        Berechnet Similarity zwischen zwei Vektoren
        
        Args:
            vec1: Erster Vektor (Query)
            vec2: Zweiter Vektor (Dokument)
            metric: Zu verwendende Similarity-Metrik
            
        Returns:
            float: Similarity-Score zwischen 0.0 und 1.0
        """
        try:
            if metric == SimilarityMetric.COSINE:
                return self._cosine_similarity(vec1, vec2)
            elif metric == SimilarityMetric.DOT_PRODUCT:
                return max(0.0, np.dot(vec1, vec2))  # Clamp auf [0,1]
            elif metric == SimilarityMetric.EUCLIDEAN:
                return 1.0 / (1.0 + np.linalg.norm(vec1 - vec2))  # Invertierte Distance
            elif metric == SimilarityMetric.MANHATTAN:
                return 1.0 / (1.0 + np.sum(np.abs(vec1 - vec2)))  # Invertierte Distance
            elif metric == SimilarityMetric.ANGULAR:
                cosine_sim = self._cosine_similarity(vec1, vec2)
                return 1.0 - (math.acos(max(-1.0, min(1.0, cosine_sim))) / math.pi)
            else:
                return self._cosine_similarity(vec1, vec2)  # Fallback
                
        except Exception as e:
            self.logger.warning(f"Similarity-Berechnung fehlgeschlagen: {e}")
            return 0.0

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Berechnet Cosine Similarity zwischen zwei Vektoren"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return max(0.0, min(1.0, dot_product / (norm1 * norm2)))
        except:
            return 0.0

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalisiert Vektor auf Einheitslänge"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _apply_industrial_boosts(self, doc: Document, similarity: float, query: RetrievalQuery) -> float:
        """
        Wendet industrielle Domain-Boosts auf Similarity-Score an
        
        Args:
            doc: Dokument
            similarity: Basis-Similarity
            query: Original Query
            
        Returns:
            float: Geboostete Similarity
        """
        boosted_similarity = similarity
        
        # Technische Begriffe Boost
        if self._contains_technical_terms(doc.content):
            boosted_similarity *= self.semantic_config.technical_term_boost
        
        # Safety-Kategorien Boost
        if self._contains_safety_keywords(doc.content):
            boosted_similarity *= self.semantic_config.safety_category_weight
        
        # Zeitlicher Boost (falls aktiviert)
        if self.semantic_config.temporal_boost:
            boosted_similarity = self._apply_temporal_boost(doc, boosted_similarity)
        
        # Sicherstellen dass Score in [0,1] bleibt
        return min(1.0, boosted_similarity)

    def _contains_technical_terms(self, content: str) -> bool:
        """Prüft ob Content technische Begriffe enthält"""
        content_lower = content.lower()
        return any(term in content_lower for term in self._technical_terms)

    def _contains_safety_keywords(self, content: str) -> bool:
        """Prüft ob Content Safety-relevante Keywords enthält"""
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in self._safety_keywords)

    def _apply_temporal_boost(self, doc: Document, similarity: float) -> float:
        """Wendet zeitlichen Boost basierend auf Dokumenten-Alter an"""
        # Einfache Implementierung - kann erweitert werden
        if 'timestamp' in doc.metadata:
            try:
                # Neuere Dokumente erhalten leichten Boost
                # Implementation abhängig von Timestamp-Format
                return similarity * 1.05  # 5% Boost für zeitliche Relevanz
            except:
                pass
        return similarity

    def _rerank_results(self, scored_documents: List[Tuple[Document, float]], query: RetrievalQuery) -> List[Tuple[Document, float]]:
        """
        Re-Ranking der Ergebnisse für verbesserte Relevanz
        
        Args:
            scored_documents: Vorläufig bewertete Dokumente
            query: Original Query
            
        Returns:
            List[Tuple[Document, float]]: Re-gerankte Dokumente
        """
        if len(scored_documents) <= 1:
            return scored_documents
        
        # Einfaches Re-Ranking basierend auf Query-Länge und Content-Länge
        reranked = []
        
        for doc, score in scored_documents:
            # Content-Länge normalisieren (bevorzuge mittlere Längen)
            content_length = len(doc.content)
            length_factor = 1.0
            
            if content_length < 100:
                length_factor = 0.9  # Zu kurze Dokumente abwerten
            elif content_length > 5000:
                length_factor = 0.95  # Zu lange Dokumente leicht abwerten
            
            # Query-Terme in Content
            query_words = set(query.text.lower().split())
            content_words = set(doc.content.lower().split())
            term_overlap = len(query_words.intersection(content_words))
            term_factor = 1.0 + (term_overlap * 0.05)  # Boost für Term-Überlappung
            
            reranked_score = score * length_factor * term_factor
            reranked.append((doc, reranked_score))
        
        # Neu sortieren
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def _apply_diversity_penalty(self, scored_documents: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Wendet Diversitäts-Penalty an um redundante Ergebnisse zu reduzieren
        
        Args:
            scored_documents: Bewertete Dokumente
            
        Returns:
            List[Tuple[Document, float]]: Diversitäts-optimierte Ergebnisse
        """
        if self.semantic_config.diversity_penalty == 0.0 or len(scored_documents) <= 1:
            return scored_documents
        
        diverse_results = []
        selected_contents = []
        
        for doc, score in scored_documents:
            # Ähnlichkeit zu bereits ausgewählten Dokumenten berechnen
            max_content_similarity = 0.0
            
            for selected_content in selected_contents:
                content_sim = self._calculate_content_similarity(doc.content, selected_content)
                max_content_similarity = max(max_content_similarity, content_sim)
            
            # Penalty anwenden
            diversity_penalty = max_content_similarity * self.semantic_config.diversity_penalty
            penalized_score = score * (1.0 - diversity_penalty)
            
            diverse_results.append((doc, penalized_score))
            selected_contents.append(doc.content)
        
        # Nach penalisierten Scores neu sortieren
        diverse_results.sort(key=lambda x: x[1], reverse=True)
        return diverse_results

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Berechnet Content-Ähnlichkeit zwischen zwei Texten
        
        Args:
            content1: Erster Text
            content2: Zweiter Text
            
        Returns:
            float: Ähnlichkeit zwischen 0.0 und 1.0
        """
        # Einfache Jaccard-Similarity auf Wort-Ebene
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _post_process_semantic_results(self, 
                                     results: List[Tuple[Document, float]], 
                                     query: RetrievalQuery) -> List[Tuple[Document, float]]:
        """
        Post-Processing der semantischen Ergebnisse
        
        Args:
            results: Bewertete Ergebnisse
            query: Original Query
            
        Returns:
            List[Tuple[Document, float]]: Finalisierte Ergebnisse
        """
        processed_results = results
        
        # Early Stopping falls aktiviert
        if (self.semantic_config.early_stopping and 
            len(processed_results) >= query.k and 
            processed_results[0][1] > 0.8):  # Hohe Confidence
            
            processed_results = processed_results[:query.k]
        
        # Score-Threshold final anwenden
        processed_results = [
            (doc, score) for doc, score in processed_results
            if score >= self.semantic_config.similarity_threshold
        ]
        
        return processed_results

    def _load_technical_terminology(self) -> Set[str]:
        """Lädt industrielle Fachterminologie"""
        return {
            # SPS/PLC Begriffe
            'sps', 'plc', 'steuerung', 'programmierung', 'ladder', 'fbd', 'st', 'il',
            'profibus', 'profinet', 'modbus', 'ethernet', 'fieldbus', 'canbus',
            
            # HMI/SCADA Begriffe  
            'hmi', 'scada', 'bedienoberfläche', 'visualisierung', 'alarmierung',
            'trending', 'logging', 'archivierung', 'reporting',
            
            # Antriebstechnik
            'motor', 'antrieb', 'servo', 'frequenzumrichter', 'vfd', 'encoder',
            'resolver', 'getriebe', 'kupplung', 'bremse', 'lageregelung',
            
            # Sensorik/Aktorik
            'sensor', 'aktor', 'messfühler', 'transmitter', 'converter',
            'temperatur', 'druck', 'durchfluss', 'füllstand', 'position',
            'näherung', 'lichtschranke', 'ultraschall', 'radar', 'lidar',
            
            # Pneumatik/Hydraulik
            'pneumatik', 'hydraulik', 'zylinder', 'ventil', 'kompressor',
            'filter', 'regler', 'manometer', 'flowmeter',
            
            # Safety/Sicherheit
            'safety', 'sicherheit', 'notaus', 'schutztür', 'lichtgitter',
            'kategorie', 'sil', 'pfd', 'mttf', 'diagnostic', 'failsafe',
            
            # Wartung/Maintenance
            'wartung', 'instandhaltung', 'maintenance', 'diagnose', 'troubleshooting',
            'kalibrierung', 'justierung', 'verschleiß', 'lebensdauer'
        }

    def _load_safety_keywords(self) -> Set[str]:
        """Lädt Safety-relevante Keywords"""
        return {
            'safety', 'sicherheit', 'notaus', 'emergency', 'stop', 'schutz',
            'kategorie', 'category', 'sil', 'pfd', 'mttf', 'failsafe', 'fail-safe',
            'redundant', 'diverse', 'diagnostic', 'test', 'proof', 'gefahr',
            'danger', 'hazard', 'risk', 'risiko', 'norm', 'standard', 'ce',
            'iec', 'iso', 'din', 'en', 'ansi', 'osha', 'atex', 'explosion'
        }

    def _custom_health_check(self) -> Optional[Dict[str, Any]]:
        """
        Semantic-spezifische Health-Checks
        
        Returns:
            Dict[str, Any]: Health-Status der Semantic-Komponenten
        """
        health_data = {
            'vector_store_available': self.vector_store is not None,
            'embedding_service_available': self.embedding_service is not None,
            'similarity_metric': self.semantic_config.similarity_metric.value,
            'query_expansion': self.semantic_config.query_expansion.value
        }
        
        # Vector Store Health-Check
        if self.vector_store:
            try:
                vector_health = self.vector_store.health_check()
                health_data['vector_store_status'] = vector_health.get('status', 'unknown')
                health_data['vector_store_count'] = vector_health.get('document_count', 0)
            except Exception as e:
                health_data['vector_store_status'] = 'error'
                health_data['vector_store_error'] = str(e)
        
        # Embedding Service Health-Check
        if self.embedding_service:
            try:
                embedding_health = self.embedding_service.health_check()
                health_data['embedding_service_status'] = embedding_health.get('status', 'unknown')
            except Exception as e:
                health_data['embedding_service_status'] = 'error'
                health_data['embedding_service_error'] = str(e)
        
        # Cache-Statistiken
        health_data.update({
            'vector_cache_size': len(self._vector_cache),
            'vector_cache_hits': self._cache_hits,
            'vector_cache_misses': self._cache_misses,
            'cache_hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        })
        
        # Performance-Metriken
        health_data.update({
            'embedding_queries': self._embedding_queries,
            'similarity_calculations': self._similarity_calculations,
            'reranking_operations': self._reranking_operations,
            'query_expansions': self._query_expansions
        })
        
        return health_data

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Erweiterte Performance-Statistiken für Semantic Retriever
        
        Returns:
            Dict[str, Any]: Detaillierte Performance-Metriken
        """
        base_stats = super().get_performance_stats()
        
        semantic_stats = {
            'embedding_queries': self._embedding_queries,
            'similarity_calculations': self._similarity_calculations,
            'reranking_operations': self._reranking_operations,
            'query_expansions': self._query_expansions,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'similarity_metric': self.semantic_config.similarity_metric.value,
            'query_expansion_strategy': self.semantic_config.query_expansion.value
        }
        
        # Cache-Effizienz
        total_cache_requests = self._cache_hits + self._cache_misses
        if total_cache_requests > 0:
            semantic_stats['cache_hit_rate'] = self._cache_hits / total_cache_requests
            semantic_stats['cache_efficiency'] = 'good' if semantic_stats['cache_hit_rate'] > 0.7 else 'poor'
        
        # Query-Expansion-Rate
        if self._total_queries > 0:
            semantic_stats['expansion_rate'] = self._query_expansions / self._total_queries
        
        # Re-Ranking-Rate
        if self._total_queries > 0:
            semantic_stats['reranking_rate'] = self._reranking_operations / self._total_queries
        
        # Stats kombinieren
        base_stats.update(semantic_stats)
        return base_stats

    def clear_vector_cache(self):
        """Leert den Vector-Cache"""
        self._vector_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self.logger.info(f"Vector-Cache für {self.semantic_config.name} geleert")

    def update_similarity_metric(self, metric: SimilarityMetric):
        """
        Ändert Similarity-Metrik zur Laufzeit
        
        Args:
            metric: Neue Similarity-Metrik
        """
        old_metric = self.semantic_config.similarity_metric
        self.semantic_config.similarity_metric = metric
        
        # Cache leeren da sich Berechnungen ändern
        self.clear_vector_cache()
        
        self.logger.info(f"Similarity-Metrik geändert: {old_metric.value} -> {metric.value}")

    def set_query_expansion(self, strategy: QueryExpansionStrategy):
        """
        Ändert Query-Expansion-Strategie zur Laufzeit
        
        Args:
            strategy: Neue Query-Expansion-Strategie
        """
        old_strategy = self.semantic_config.query_expansion
        self.semantic_config.query_expansion = strategy
        
        self.logger.info(f"Query-Expansion-Strategie geändert: {old_strategy.value} -> {strategy.value}")

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Informationen über Vector-Cache-Status
        
        Returns:
            Dict[str, Any]: Cache-Informationen
        """
        total_requests = self._cache_hits + self._cache_misses
        
        return {
            'cache_size': len(self._vector_cache),
            'max_cache_size': self.semantic_config.vector_cache_size,
            'cache_usage': len(self._vector_cache) / self.semantic_config.vector_cache_size,
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_rate': self._cache_hits / max(1, total_requests),
            'total_requests': total_requests
        }


# =============================================================================
# MULTI-VECTOR SEMANTIC RETRIEVER
# =============================================================================

class MultiVectorSemanticRetriever(SemanticRetriever):
    """
    Erweiterte Semantic Retriever-Implementierung mit Multi-Vector-Unterstützung
    
    Unterstützt verschiedene Strategien für Dokumente mit mehreren Vector-Repräsentationen:
    - Chunk-Level Embeddings mit Aggregation
    - Multi-Aspect Embeddings (verschiedene semantische Aspekte)
    - Hierarchische Document-Representations
    """
    
    def __init__(self, 
                 config: SemanticRetrieverConfig,
                 vector_store=None,
                 embedding_service=None):
        super().__init__(config, vector_store, embedding_service)
        
        # Multi-Vector spezifische Metriken
        self._vector_aggregations = 0
        self._multi_vector_queries = 0
        
        self.logger.info(f"Multi-Vector Semantic Retriever initialisiert: {config.multi_vector_strategy}")

    def _calculate_similarities(self, 
                              query_embeddings: List[np.ndarray],
                              candidates: List[Tuple[Document, np.ndarray]],
                              query: RetrievalQuery) -> List[Tuple[Document, float]]:
        """
        Erweiterte Similarity-Berechnung mit Multi-Vector-Unterstützung
        
        Args:
            query_embeddings: Query-Embeddings
            candidates: Kandidaten-Dokumente mit Embeddings
            query: Original Query
            
        Returns:
            List[Tuple[Document, float]]: Dokumente mit aggregierten Similarity-Scores
        """
        if self.semantic_config.multi_vector_strategy == "single":
            return super()._calculate_similarities(query_embeddings, candidates, query)
        
        scored_documents = []
        
        for doc, doc_embeddings in candidates:
            # Multi-Vector-Aggregation
            if isinstance(doc_embeddings, list) and len(doc_embeddings) > 1:
                aggregated_score = self._aggregate_multi_vector_similarities(
                    query_embeddings, doc_embeddings, query
                )
                self._vector_aggregations += 1
            else:
                # Fallback auf Single-Vector-Berechnung
                single_embedding = doc_embeddings if isinstance(doc_embeddings, np.ndarray) else doc_embeddings[0]
                aggregated_score = self._calculate_single_similarity(query_embeddings, single_embedding)
            
            # Industrielle Boosts anwenden
            boosted_similarity = self._apply_industrial_boosts(doc, aggregated_score, query)
            
            if boosted_similarity >= self.semantic_config.similarity_threshold:
                scored_documents.append((doc, boosted_similarity))
        
        # Nach Score sortieren
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        self._multi_vector_queries += 1
        
        return scored_documents

    def _aggregate_multi_vector_similarities(self, 
                                           query_embeddings: List[np.ndarray],
                                           doc_embeddings: List[np.ndarray],
                                           query: RetrievalQuery) -> float:
        """
        Aggregiert Similarities zwischen Query und Multi-Vector Dokument
        
        Args:
            query_embeddings: Query-Embeddings
            doc_embeddings: Document-Embeddings (mehrere Vektoren)
            query: Original Query
            
        Returns:
            float: Aggregierte Similarity
        """
        strategy = self.semantic_config.multi_vector_strategy.lower()
        
        # Similarity-Matrix berechnen
        similarities = []
        for q_emb in query_embeddings:
            for d_emb in doc_embeddings:
                sim = self._compute_similarity(q_emb, d_emb, self.semantic_config.similarity_metric)
                similarities.append(sim)
                self._similarity_calculations += 1
        
        # Aggregation-Strategie anwenden
        if strategy == "average":
            return np.mean(similarities)
        elif strategy == "max":
            return np.max(similarities)
        elif strategy == "concat":
            # Für Concat-Strategie: Gewichteter Durchschnitt mit Positions-Bias
            weights = np.exp(-0.1 * np.arange(len(similarities)))  # Exponential decay
            return np.average(similarities, weights=weights)
        else:
            return np.max(similarities)  # Default: Max-Strategie

    def _calculate_single_similarity(self, query_embeddings: List[np.ndarray], doc_embedding: np.ndarray) -> float:
        """Berechnet Similarity für Single-Vector Dokument"""
        max_similarity = 0.0
        for query_embedding in query_embeddings:
            similarity = self._compute_similarity(
                query_embedding, doc_embedding, self.semantic_config.similarity_metric
            )
            max_similarity = max(max_similarity, similarity)
            self._similarity_calculations += 1
        return max_similarity

    def get_multi_vector_stats(self) -> Dict[str, Any]:
        """Multi-Vector spezifische Statistiken"""
        base_stats = self.get_performance_stats()
        
        multi_vector_stats = {
            'vector_aggregations': self._vector_aggregations,
            'multi_vector_queries': self._multi_vector_queries,
            'multi_vector_strategy': self.semantic_config.multi_vector_strategy
        }
        
        if self._multi_vector_queries > 0:
            multi_vector_stats['avg_aggregations_per_query'] = self._vector_aggregations / self._multi_vector_queries
        
        base_stats.update(multi_vector_stats)
        return base_stats


# =============================================================================
# FACTORY UND REGISTRY INTEGRATION
# =============================================================================

def create_semantic_retriever(config: Dict[str, Any], 
                             vector_store=None, 
                             embedding_service=None) -> SemanticRetriever:
    """
    Factory-Funktion für Semantic Retriever-Erstellung
    
    Args:
        config: Konfiguration als Dictionary
        vector_store: Vector Store Instanz
        embedding_service: Embedding Service Instanz
        
    Returns:
        SemanticRetriever: Konfigurierte Semantic-Retriever Instanz
    """
    # Config-Objekt erstellen
    semantic_config = SemanticRetrieverConfig(
        name=config.get('name', 'semantic_retriever'),
        description=config.get('description', 'Pure Vector-based Semantic Retriever'),
        similarity_metric=SimilarityMetric(config.get('similarity_metric', 'cosine')),
        query_expansion=QueryExpansionStrategy(config.get('query_expansion', 'none')),
        **{k: v for k, v in config.items() if k not in ['name', 'description', 'similarity_metric', 'query_expansion']}
    )
    
    # Multi-Vector oder Standard Semantic Retriever
    if config.get('multi_vector', False) or config.get('multi_vector_strategy', 'single') != 'single':
        return MultiVectorSemanticRetriever(semantic_config, vector_store, embedding_service)
    else:
        return SemanticRetriever(semantic_config, vector_store, embedding_service)


# Registrierung im Retriever-Registry (wird von __init__.py aufgerufen)
def register_semantic_retrievers():
    """Registriert Semantic Retriever im globalen Registry"""
    from .base_retriever import RetrieverRegistry
    
    RetrieverRegistry.register('semantic', SemanticRetriever)
    RetrieverRegistry.register('multi_vector_semantic', MultiVectorSemanticRetriever)