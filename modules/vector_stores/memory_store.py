#!/usr/bin/env python3
"""
Memory Vector Store - Vereinfachte Version für schnellen Start
In-Memory Speicherung ohne externe Dependencies
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base_vectorstore import BaseVectorStore

class MemoryVectorStore(BaseVectorStore):
    """Einfacher In-Memory Vector Store"""
    
    def __init__(self, collection_name: str = "memory_collection"):
        self.collection_name = collection_name
        self.documents = []  # [{"id": str, "content": str, "metadata": dict, "embedding": list}]
        
    def add_documents(self, 
                     documents: List[str], 
                     embeddings: List[List[float]], 
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     ids: Optional[List[str]] = None) -> bool:
        """Dokumente hinzufügen"""
        try:
            if len(documents) != len(embeddings):
                return False
            
            # Standard-Werte setzen
            if ids is None:
                ids = [f"doc_{len(self.documents) + i}" for i in range(len(documents))]
            if metadatas is None:
                metadatas = [{} for _ in documents]
            
            # Dokumente speichern
            for i, (doc_id, content, embedding, metadata) in enumerate(
                zip(ids, documents, embeddings, metadatas)
            ):
                doc_entry = {
                    "id": doc_id,
                    "content": content,
                    "embedding": embedding,
                    "metadata": metadata,
                    "created_at": datetime.now().isoformat()
                }
                self.documents.append(doc_entry)
            
            print(f"Added {len(documents)} documents to memory store")
            return True
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    def similarity_search(self, 
                         query_embedding: List[float], 
                         k: int = 5, 
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Similarity Search mit einfacher Cosine Similarity"""
        try:
            if not self.documents:
                return []
            
            # Similarity Scores berechnen
            results = []
            query_vector = np.array(query_embedding)
            
            for doc in self.documents:
                doc_vector = np.array(doc["embedding"])
                
                # Cosine Similarity
                cos_sim = np.dot(query_vector, doc_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)
                )
                
                # Filter anwenden falls vorhanden
                if filter_dict:
                    match = all(
                        doc["metadata"].get(key) == value 
                        for key, value in filter_dict.items()
                    )
                    if not match:
                        continue
                
                result = {
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": float(cos_sim)
                }
                results.append(result)
            
            # Top-K sortiert zurückgeben
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:k]
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def get_documents(self, ids: List[str]) -> List[Dict[str, Any]]:
        """Dokumente nach IDs abrufen"""
        results = []
        for doc in self.documents:
            if doc["id"] in ids:
                results.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"]
                })
        return results
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Dokumente löschen"""
        try:
            original_count = len(self.documents)
            self.documents = [doc for doc in self.documents if doc["id"] not in ids]
            deleted_count = original_count - len(self.documents)
            print(f"Deleted {deleted_count} documents")
            return deleted_count > 0
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Collection-Info"""
        embedding_dim = len(self.documents[0]["embedding"]) if self.documents else 0
        return {
            "name": self.collection_name,
            "document_count": len(self.documents),
            "embedding_dimension": embedding_dim,
            "type": "memory"
        }


def create_memory_vector_store(collection_name: str = "memory_collection") -> MemoryVectorStore:
    """Factory für Memory Vector Store"""
    return MemoryVectorStore(collection_name)
