#!/usr/bin/env python3
"""
Base Vector Store - DocumentRecord und fehlende Klassen definiert
Industrielle RAG-Architektur - Vector Store Abstraktion

KRITISCHE BUGFIXES:
- DocumentRecord Klasse hinzugefügt
- Alle fehlenden Base-Klassen definiert
- Import-Konflikte behoben

Version: 4.0.0 - Service-orientierte Architektur
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid


# =============================================================================
# DOCUMENT RECORD - FEHLENDE KLASSE HINZUGEFÜGT
# =============================================================================

@dataclass
class DocumentRecord:
    """
    Document Record für Vector Store Operations - NEU HINZUGEFÜGT
    
    BUGFIX: Diese Klasse fehlte und verursachte Import-Fehler
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary"""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'embedding': self.embedding,
            'source': self.source,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentRecord':
        """Erstelle von Dictionary"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class VectorStoreResult:
    """
    Ergebnis einer Vector Store Operation - NEU HINZUGEFÜGT
    """
    success: bool = False
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    operation_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary"""
        return {
            'success': self.success,
            'data': self.data,
            'error': self.error,
            'metadata': self.metadata,
            'operation_time': self.operation_time
        }


@dataclass 
class SearchResult:
    """
    Such-Ergebnis für Vector Store Queries - NEU HINZUGEFÜGT
    """
    document: DocumentRecord
    score: float
    distance: Optional[float] = None
    rank: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiere zu Dictionary"""
        return {
            'document': self.document.to_dict(),
            'score': self.score,
            'distance': self.distance,
            'rank': self.rank
        }


# =============================================================================
# BASE VECTOR STORE INTERFACE
# =============================================================================

class BaseVectorStore(ABC):
    """
    Abstract Base Class für alle Vector Store Implementierungen
    
    Definiert die Standard-Schnittstelle für Vector Stores
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialisiere Vector Store
        
        Args:
            config: Konfiguration für den Vector Store
        """
        self.config = config or {}
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialisiere Vector Store
        
        Returns:
            bool: True wenn erfolgreich initialisiert
        """
        pass
    
    @abstractmethod
    def add_documents(
        self, 
        documents: List[DocumentRecord], 
        collection_name: Optional[str] = None
    ) -> VectorStoreResult:
        """
        Füge Dokumente zum Vector Store hinzu
        
        Args:
            documents: Liste von DocumentRecord Objekten
            collection_name: Name der Collection (optional)
            
        Returns:
            VectorStoreResult: Ergebnis der Operation
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        collection_name: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Suche ähnliche Dokumente
        
        Args:
            query_embedding: Query-Embedding als Vektor
            k: Anzahl der Ergebnisse
            collection_name: Name der Collection
            filter_metadata: Metadata-Filter
            
        Returns:
            List[SearchResult]: Liste der Such-Ergebnisse
        """
        pass
    
    @abstractmethod
    def delete_documents(
        self,
        document_ids: List[str],
        collection_name: Optional[str] = None
    ) -> VectorStoreResult:
        """
        Lösche Dokumente aus Vector Store
        
        Args:
            document_ids: Liste der Dokument-IDs
            collection_name: Name der Collection
            
        Returns:
            VectorStoreResult: Ergebnis der Operation
        """
        pass
    
    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Hole Informationen über eine Collection
        
        Args:
            collection_name: Name der Collection
            
        Returns:
            Dict: Collection-Informationen
        """
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Führe Health-Check durch
        
        Returns:
            Dict: Health-Status
        """
        pass
    
    def is_initialized(self) -> bool:
        """Prüfe ob Vector Store initialisiert ist"""
        return self._initialized
    
    def get_config(self) -> Dict[str, Any]:
        """Hole aktuelle Konfiguration"""
        return self.config.copy()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_document_record(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    embedding: Optional[List[float]] = None
) -> DocumentRecord:
    """
    Helper-Funktion zur Erstellung von DocumentRecord
    
    Args:
        content: Textinhalt
        metadata: Metadaten
        source: Quelle des Dokuments
        embedding: Pre-computed Embedding
        
    Returns:
        DocumentRecord: Neues DocumentRecord Objekt
    """
    return DocumentRecord(
        content=content,
        metadata=metadata or {},
        source=source,
        embedding=embedding
    )


def create_search_result(
    document: DocumentRecord,
    score: float,
    distance: Optional[float] = None,
    rank: Optional[int] = None
) -> SearchResult:
    """
    Helper-Funktion zur Erstellung von SearchResult
    
    Args:
        document: DocumentRecord
        score: Relevanz-Score
        distance: Vektor-Distanz
        rank: Ranking-Position
        
    Returns:
        SearchResult: Neues SearchResult Objekt
    """
    return SearchResult(
        document=document,
        score=score,
        distance=distance,
        rank=rank
    )


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    # Data Classes
    'DocumentRecord',
    'VectorStoreResult', 
    'SearchResult',
    
    # Base Classes
    'BaseVectorStore',
    
    # Utility Functions
    'create_document_record',
    'create_search_result'
]
    