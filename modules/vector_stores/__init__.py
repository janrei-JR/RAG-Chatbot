# modules/vector_stores/__init__.py - FACTORY METHOD BUGFIX
"""
Vector Store Factory - create_vector_store() Method hinzugefügt
BUGFIX: 'VectorStoreFactory' hat kein Attribut 'create_vector_store'
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VectorStoreResult:
    """Ergebnis einer Vector Store Operation"""
    documents: List[Any] = None
    distances: List[float] = None  
    metadatas: List[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None
    total_results: int = 0
    
    def __post_init__(self):
        if self.documents is None:
            self.documents = []
        if self.distances is None:
            self.distances = []
        if self.metadatas is None:
            self.metadatas = []

# Memory Vector Store als Fallback
class MemoryVectorStore:
    """In-Memory Vector Store Implementation"""
    
    def __init__(self, **kwargs):
        self.documents = {}
        self.embeddings = {}
        self.metadata = {}
        self.provider = "memory"
        
    def add_documents(self, texts, metadatas=None, ids=None):
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        if metadatas is None:
            metadatas = [{}] * len(texts)
            
        for i, (text, metadata, doc_id) in enumerate(zip(texts, metadatas, ids)):
            self.documents[doc_id] = text
            self.metadata[doc_id] = metadata
        return True
    
    def similarity_search(self, query, k=4):
        results = []
        for doc_id, text in self.documents.items():
            if query.lower() in text.lower():
                results.append({
                    'id': doc_id,
                    'content': text,
                    'metadata': self.metadata.get(doc_id, {}),
                    'distance': 0.5
                })
                if len(results) >= k:
                    break
        return results
    
    def health_check(self):
        return True

# Base Vector Store
class BaseVectorStore:
    def __init__(self, **kwargs):
        self.provider = kwargs.get('provider', 'unknown')
    
    def add_documents(self, texts, metadatas=None, ids=None):
        raise NotImplementedError
    
    def similarity_search(self, query, k=4):
        raise NotImplementedError
    
    def health_check(self):
        return True

# BUGFIX: VectorStoreFactory mit create_vector_store() Method
class VectorStoreFactory:
    """Factory für Vector Store Erstellung - BUGFIX"""
    
    @staticmethod
    def create_vector_store(provider: str = "memory", **kwargs):
        """
        BUGFIX: create_vector_store() Methode implementiert
        """
        logger.info(f"Creating vector store with provider: {provider}")
        
        try:
            if provider == "memory":
                return MemoryVectorStore(**kwargs)
            elif provider == "chroma":
                try:
                    from .chroma_store import ChromaVectorStore
                    return ChromaVectorStore(**kwargs)
                except ImportError:
                    logger.warning("ChromaDB nicht verfügbar, fallback auf Memory")
                    return MemoryVectorStore(**kwargs)
            else:
                logger.warning(f"Provider '{provider}' nicht verfügbar, fallback auf Memory")
                return MemoryVectorStore(**kwargs)
                
        except Exception as e:
            logger.error(f"Vector Store Erstellung fehlgeschlagen: {e}")
            return MemoryVectorStore(**kwargs)
    
    @staticmethod
    def create_vectorstore(provider: str = "memory", **kwargs):
        """Alternative Methoden-Name für Kompatibilität"""
        return VectorStoreFactory.create_vector_store(provider, **kwargs)
    
    @staticmethod  
    def get_available_providers():
        providers = ["memory"]
        try:
            import chromadb
            providers.append("chroma")
        except ImportError:
            pass
        return providers

# Factory Functions
def create_auto_vector_store(provider: str = "memory", **kwargs):
    """Auto VectorStore Creation"""
    return VectorStoreFactory.create_vector_store(provider, **kwargs)

def create_vector_store(provider: str = "memory", **kwargs):
    """Direkte Vector Store Erstellung"""
    return VectorStoreFactory.create_vector_store(provider, **kwargs)

def get_available_providers():
    """Verfügbare Provider auflisten"""
    return VectorStoreFactory.get_available_providers()

# Exports
__all__ = [
    'VectorStoreResult',
    'BaseVectorStore', 
    'MemoryVectorStore',
    'VectorStoreFactory',
    'create_vector_store',
    'create_auto_vector_store',
    'get_available_providers'
]

logger.info(f"Vector Store Module initialized with providers: {get_available_providers()}")
