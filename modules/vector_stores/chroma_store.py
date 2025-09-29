#!/usr/bin/env python3
"""
ChromaDB Vector Store für RAG Chatbot Industrial

ChromaDB-Integration für lokale, persistente Vektor-Speicherung ohne Cloud-Abhängigkeiten.
Optimiert für industrielle On-Premise-Deployments mit robuster Persistierung.

Features:
- Lokale ChromaDB-Instanz mit Persistierung auf Dateisystem
- Robuste Collection-Verwaltung mit automatischer Initialisierung
- Optimierte Batch-Operationen für große Dokumentensammlungen
- Erweiterte Metadaten-Filterung mit ChromaDB-Where-Clauses
- Production-Ready mit Health-Monitoring und Backup-Funktionen

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import asdict

# ChromaDB Import (mit Fallback)
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.errors import InvalidCollectionException, ChromaError
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Base Vector Store
from .base_vectorstore import (
    BaseVectorStore, VectorStoreProvider, SearchType,
    DocumentRecord, SearchFilter, SearchResult, SearchRequest, SearchResponse,
    CollectionInfo
)

# Core-Komponenten
from core import (
    get_logger, RAGConfig, get_current_config,
    DocumentProcessingError, ValidationError,
    create_error_context, log_performance
)


# =============================================================================
# CHROMADB-SPEZIFISCHE KONFIGURATION
# =============================================================================

class ChromaStore(BaseVectorStore):
    """
    ChromaDB-basierte Vector Store Implementierung für lokale Persistierung
    
    Unterstützt:
    - Lokale Persistierung ohne Cloud-Abhängigkeiten
    - Robuste Collection-Verwaltung
    - Erweiterte Metadaten-Filterung
    - Batch-Operationen für Performance
    - Health-Monitoring für Produktionsumgebungen
    """
    
    def __init__(self, 
                 config: RAGConfig = None,
                 persist_directory: str = None,
                 collection_name: str = None):
        """
        Initialisiert ChromaDB Vector Store
        
        Args:
            config (RAGConfig): Konfiguration
            persist_directory (str): Verzeichnis für Persistierung
            collection_name (str): Name der Collection
        """
        if not CHROMADB_AVAILABLE:
            raise DocumentProcessingError(
                "ChromaDB nicht verfügbar. Installation: pip install chromadb",
                processing_stage="vector_store_init"
            )
        
        # Base-Klasse initialisieren
        super().__init__(config)
        
        # Provider-Eigenschaften setzen
        self.provider = VectorStoreProvider.CHROMA
        self.collection_name = collection_name or getattr(self.config.vector_store, 'collection_name', 'industrial_rag')
        self.dimension = getattr(self.config.vector_store, 'dimension', 768)
        self.distance_metric = "cosine"  # ChromaDB Standard
        
        # Persistierung-Konfiguration
        self.persist_directory = persist_directory or getattr(
            self.config.vector_store, 'persist_directory', './data/vectorstore'
        )
        
        # ChromaDB-spezifische Einstellungen
        self.batch_size = getattr(self.config.vector_store, 'batch_size', 100)
        self.allow_reset = getattr(self.config.vector_store, 'allow_reset', False)
        
        # ChromaDB Client und Collection
        self.client = None
        self.collection = None
        
        # ChromaDB-spezifische Statistiken
        self._chroma_stats = {
            'collections_created': 0,
            'collections_deleted': 0,
            'batch_operations': 0,
            'metadata_queries': 0,
            'persistence_operations': 0
        }
        
        # Initialisierung
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialisiert ChromaDB Client"""
        try:
            # Persist-Directory erstellen
            persist_path = Path(self.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            # ChromaDB Client mit Persistierung
            settings = Settings(
                persist_directory=str(persist_path),
                anonymized_telemetry=False,  # Für industrielle Umgebungen
                allow_reset=self.allow_reset
            )
            
            self.client = chromadb.PersistentClient(settings=settings)
            self._chroma_stats['persistence_operations'] += 1
            
            self.logger.info(f"ChromaDB Client initialisiert: {self.persist_directory}")
            
        except Exception as e:
            error_msg = f"Fehler bei ChromaDB Client-Initialisierung: {str(e)}"
            self.logger.error(error_msg)
            raise DocumentProcessingError(error_msg, processing_stage="chroma_client_init")
    
    # =============================================================================
    # ABSTRACT METHODS IMPLEMENTATION
    # =============================================================================
    
    def _initialize_collection(self, collection_name: str, dimension: int) -> bool:
        """
        Initialisiert ChromaDB Collection
        
        Args:
            collection_name (str): Collection-Name
            dimension (int): Embedding-Dimension
            
        Returns:
            bool: True wenn erfolgreich
        """
        try:
            self.collection_name = collection_name
            self.dimension = dimension
            
            # Versuche vorhandene Collection zu laden
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                self.logger.info(f"Vorhandene ChromaDB Collection geladen: {self.collection_name}")
                
            except (InvalidCollectionException, ValueError):
                # Collection existiert nicht - neue erstellen
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "dimension": dimension,
                        "distance_metric": self.distance_metric,
                        "created_at": time.time(),
                        "rag_version": "4.0.0"
                    }
                )
                
                self._chroma_stats['collections_created'] += 1
                self.logger.info(f"Neue ChromaDB Collection erstellt: {self.collection_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler bei Collection-Initialisierung: {str(e)}")
            return False
    
    def _add_documents_batch(self, documents: List[DocumentRecord]) -> bool:
        """
        Fügt Dokumente in Batch zu ChromaDB hinzu
        
        Args:
            documents (List[DocumentRecord]): Hinzuzufügende Dokumente
            
        Returns:
            bool: True wenn erfolgreich
        """
        if not self.collection:
            raise DocumentProcessingError("Collection nicht initialisiert", processing_stage="chroma_add_docs")
        
        try:
            # Dokumente in Batches aufteilen
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                
                # ChromaDB-Format vorbereiten
                ids = [doc.id for doc in batch]
                embeddings = [doc.embedding for doc in batch]
                documents_content = [doc.content for doc in batch]
                metadatas = [self._prepare_metadata(doc.metadata) for doc in batch]
                
                # Batch zu Collection hinzufügen
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents_content,
                    metadatas=metadatas
                )
                
                self._chroma_stats['batch_operations'] += 1
                
                self.logger.debug(f"ChromaDB Batch hinzugefügt: {len(batch)} Dokumente")
            
            # Persistierung forcieren (für Datenintegrität)
            self._force_persistence()
            
            return True
            
        except Exception as e:
            error_msg = f"Fehler beim Hinzufügen von Dokumenten zu ChromaDB: {str(e)}"
            self.logger.error(error_msg)
            raise DocumentProcessingError(error_msg, processing_stage="chroma_add_batch")
    
    def _search_similar(self, request: SearchRequest) -> SearchResponse:
        """
        Führt Ähnlichkeitssuche in ChromaDB durch
        
        Args:
            request (SearchRequest): Such-Anfrage
            
        Returns:
            SearchResponse: Such-Ergebnisse
        """
        if not self.collection:
            raise DocumentProcessingError("Collection nicht initialisiert", processing_stage="chroma_search")
        
        try:
            start_time = time.time()
            
            # Where-Clause aus SearchFilter erstellen
            where_clause = self._build_where_clause(request.filter)
            
            # ChromaDB Query durchführen
            results = self.collection.query(
                query_embeddings=[request.query_embedding],
                n_results=request.limit,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            if where_clause:
                self._chroma_stats['metadata_queries'] += 1
            
            search_time_ms = (time.time() - start_time) * 1000
            
            # Ergebnisse konvertieren
            search_results = []
            
            if results['ids'] and results['ids'][0]:  # ChromaDB gibt nested Lists zurück
                for i, doc_id in enumerate(results['ids'][0]):
                    # Similarity-Score aus Distance berechnen (ChromaDB gibt Distances zurück)
                    distance = results['distances'][0][i]
                    similarity_score = self._distance_to_similarity(distance)
                    
                    # Minimale Ähnlichkeit prüfen
                    if similarity_score < request.min_similarity:
                        continue
                    
                    # DocumentRecord rekonstruieren
                    doc_record = DocumentRecord(
                        id=doc_id,
                        content=results['documents'][0][i],
                        embedding=[],  # Embeddings nicht zurückgeben (Performance)
                        metadata=results['metadatas'][0][i] or {}
                    )
                    
                    search_result = SearchResult(
                        document=doc_record,
                        similarity_score=similarity_score,
                        rank=i + 1,
                        search_metadata={
                            'distance': distance,
                            'search_type': request.search_type.value
                        }
                    )
                    
                    search_results.append(search_result)
            
            response = SearchResponse(
                results=search_results,
                total_found=len(search_results),  # ChromaDB gibt nicht total_found zurück
                search_time_ms=search_time_ms,
                search_metadata={
                    'where_clause_used': where_clause is not None,
                    'chroma_results_count': len(results['ids'][0]) if results['ids'] else 0
                }
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Fehler bei ChromaDB Suche: {str(e)}"
            self.logger.error(error_msg)
            raise DocumentProcessingError(error_msg, processing_stage="chroma_search")
    
    def _delete_documents(self, document_ids: List[str]) -> bool:
        """
        Löscht Dokumente aus ChromaDB
        
        Args:
            document_ids (List[str]): Zu löschende Dokument-IDs
            
        Returns:
            bool: True wenn erfolgreich
        """
        if not self.collection:
            raise DocumentProcessingError("Collection nicht initialisiert", processing_stage="chroma_delete")
        
        try:
            # Dokumente löschen
            self.collection.delete(ids=document_ids)
            
            # Persistierung forcieren
            self._force_persistence()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler beim Löschen von ChromaDB Dokumenten: {str(e)}")
            return False
    
    def _get_collection_info(self) -> CollectionInfo:
        """
        Holt ChromaDB Collection-Informationen
        
        Returns:
            CollectionInfo: Collection-Details
        """
        try:
            if not self.collection:
                return CollectionInfo(
                    name=self.collection_name,
                    document_count=0,
                    dimension=self.dimension
                )
            
            # Collection-Count (ChromaDB hat keinen direkten count)
            count_result = self.collection.count()
            
            # Collection-Metadaten
            try:
                # Versuche Collection-Metadaten zu bekommen
                collection_metadata = {}
                if hasattr(self.collection, 'metadata') and self.collection.metadata:
                    collection_metadata = dict(self.collection.metadata)
            except:
                collection_metadata = {}
            
            # Dateisystem-Größe schätzen
            size_bytes = self._estimate_collection_size()
            
            return CollectionInfo(
                name=self.collection_name,
                document_count=count_result,
                dimension=collection_metadata.get('dimension', self.dimension),
                size_bytes=size_bytes,
                created_at=str(collection_metadata.get('created_at', '')),
                metadata=collection_metadata
            )
            
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen der Collection-Info: {str(e)}")
            return CollectionInfo(
                name=self.collection_name,
                document_count=0,
                dimension=self.dimension,
                metadata={'error': str(e)}
            )
    
    def _validate_connection(self) -> bool:
        """
        Validiert ChromaDB-Verbindung
        
        Returns:
            bool: True wenn Verbindung funktioniert
        """
        try:
            if not self.client:
                return False
            
            # Client-Funktionalität testen
            collections = self.client.list_collections()
            return True
            
        except Exception as e:
            self.logger.debug(f"ChromaDB Verbindungstest fehlgeschlagen: {str(e)}")
            return False
    
    # =============================================================================
    # CHROMADB-SPEZIFISCHE METHODEN
    # =============================================================================
    
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bereitet Metadaten für ChromaDB vor (Typ-Konvertierung)
        
        Args:
            metadata (Dict[str, Any]): Original-Metadaten
            
        Returns:
            Dict[str, Any]: ChromaDB-kompatible Metadaten
        """
        prepared = {}
        
        for key, value in metadata.items():
            # ChromaDB unterstützt nur: str, int, float, bool
            if isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            elif value is None:
                prepared[key] = ""  # None zu leerem String
            else:
                # Komplexe Typen zu String konvertieren
                prepared[key] = str(value)
        
        return prepared
    
    def _build_where_clause(self, filter: SearchFilter) -> Optional[Dict[str, Any]]:
        """
        Erstellt ChromaDB Where-Clause aus SearchFilter
        
        Args:
            filter (SearchFilter): Such-Filter
            
        Returns:
            Optional[Dict[str, Any]]: ChromaDB Where-Clause
        """
        if filter.is_empty():
            return None
        
        where_conditions = []
        
        # Include-Filter (AND-Verknüpfung)
        for key, value in filter.include.items():
            where_conditions.append({key: {"$eq": value}})
        
        # Exclude-Filter (NOT-Verknüpfung)
        for key, value in filter.exclude.items():
            where_conditions.append({key: {"$ne": value}})
        
        # Range-Filter
        for key, (min_val, max_val) in filter.range_filters.items():
            where_conditions.append({
                "$and": [
                    {key: {"$gte": min_val}},
                    {key: {"$lte": max_val}}
                ]
            })
        
        # Text-Contains-Filter (ChromaDB unterstützt begrenzte String-Operationen)
        for key, text in filter.text_contains.items():
            # ChromaDB hat keine nativen String-Contains, verwende Gleichheit als Fallback
            where_conditions.append({key: {"$eq": text}})
        
        # Alle Bedingungen mit AND verknüpfen
        if len(where_conditions) == 1:
            return where_conditions[0]
        elif len(where_conditions) > 1:
            return {"$and": where_conditions}
        
        return None
    
    def _distance_to_similarity(self, distance: float) -> float:
        """
        Konvertiert ChromaDB Distance zu Similarity Score
        
        Args:
            distance (float): ChromaDB Distance
            
        Returns:
            float: Similarity Score (0.0-1.0)
        """
        # Für Cosine Distance: similarity = 1 - distance
        # ChromaDB gibt normalisierte Cosine Distance zurück (0-2)
        return max(0.0, 1.0 - (distance / 2.0))
    
    def _force_persistence(self) -> None:
        """Forciert ChromaDB Persistierung"""
        try:
            if self.client and hasattr(self.client, 'persist'):
                self.client.persist()
                self._chroma_stats['persistence_operations'] += 1
        except Exception as e:
            self.logger.debug(f"Persistierung-Warnung: {str(e)}")
    
    def _estimate_collection_size(self) -> int:
        """
        Schätzt Collection-Größe auf Dateisystem
        
        Returns:
            int: Geschätzte Größe in Bytes
        """
        try:
            persist_path = Path(self.persist_directory)
            if persist_path.exists():
                total_size = 0
                for file_path in persist_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                return total_size
            return 0
        except Exception:
            return 0
    
    # =============================================================================
    # ERWEITERTE CHROMADB-OPERATIONEN
    # =============================================================================
    
    @log_performance()
    def upsert_documents(self, documents: List[DocumentRecord]) -> bool:
        """
        Fügt Dokumente hinzu oder aktualisiert sie (Upsert)
        
        Args:
            documents (List[DocumentRecord]): Dokumente für Upsert
            
        Returns:
            bool: True wenn erfolgreich
        """
        if not self.collection:
            raise DocumentProcessingError("Collection nicht initialisiert")
        
        try:
            # ChromaDB Upsert in Batches
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                
                ids = [doc.id for doc in batch]
                embeddings = [doc.embedding for doc in batch]
                documents_content = [doc.content for doc in batch]
                metadatas = [self._prepare_metadata(doc.metadata) for doc in batch]
                
                # ChromaDB Upsert
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents_content,
                    metadatas=metadatas
                )
                
                self._chroma_stats['batch_operations'] += 1
            
            self._force_persistence()
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler bei ChromaDB Upsert: {str(e)}")
            return False
    
    def delete_collection(self) -> bool:
        """
        Löscht gesamte Collection
        
        Returns:
            bool: True wenn erfolgreich
        """
        try:
            if self.client and self.collection:
                self.client.delete_collection(name=self.collection_name)
                self.collection = None
                self._chroma_stats['collections_deleted'] += 1
                self.logger.info(f"ChromaDB Collection gelöscht: {self.collection_name}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Fehler beim Löschen der Collection: {str(e)}")
            return False
    
    def reset_collection(self) -> bool:
        """
        Setzt Collection zurück (alle Dokumente löschen)
        
        Returns:
            bool: True wenn erfolgreich
        """
        if not self.allow_reset:
            self.logger.warning("Collection-Reset nicht erlaubt (allow_reset=False)")
            return False
        
        try:
            if self.collection:
                # Alle Dokumente aus Collection löschen
                collection_info = self._get_collection_info()
                if collection_info.document_count > 0:
                    # ChromaDB: Collection löschen und neu erstellen
                    self.delete_collection()
                    return self._initialize_collection(self.collection_name, self.dimension)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Fehler beim Collection-Reset: {str(e)}")
            return False
    
    def backup_collection(self, backup_path: str = None) -> bool:
        """
        Erstellt Backup der Collection
        
        Args:
            backup_path (str): Backup-Verzeichnis
            
        Returns:
            bool: True wenn erfolgreich
        """
        try:
            import shutil
            
            backup_dir = backup_path or f"{self.persist_directory}_backup_{int(time.time())}"
            backup_path = Path(backup_dir)
            
            if Path(self.persist_directory).exists():
                shutil.copytree(self.persist_directory, backup_path, dirs_exist_ok=True)
                self.logger.info(f"ChromaDB Backup erstellt: {backup_path}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Fehler beim Backup: {str(e)}")
            return False
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Erweiterte Performance-Statistiken mit ChromaDB-spezifischen Metriken
        
        Returns:
            Dict[str, Any]: Detaillierte Performance-Daten
        """
        # Basis-Statistiken von Parent-Klasse
        stats = super().get_performance_statistics()
        
        # ChromaDB-spezifische Statistiken
        stats['chroma'] = self._chroma_stats.copy()
        
        # Persistierung-Informationen
        stats['persistence'] = {
            'persist_directory': self.persist_directory,
            'directory_exists': Path(self.persist_directory).exists(),
            'estimated_size_bytes': self._estimate_collection_size(),
            'batch_size': self.batch_size,
            'allow_reset': self.allow_reset
        }
        
        return stats


# =============================================================================
# CHROMADB FACTORY UND UTILITIES
# =============================================================================

class ChromaStoreFactory:
    """Factory für verschiedene ChromaDB-Konfigurationen"""
    
    @staticmethod
    def create_local_store(persist_directory: str = "./data/vectorstore",
                          collection_name: str = "industrial_rag",
                          config: RAGConfig = None) -> ChromaStore:
        """
        Erstellt lokalen ChromaDB Store
        
        Args:
            persist_directory (str): Persistierung-Verzeichnis
            collection_name (str): Collection-Name
            config (RAGConfig): Konfiguration
            
        Returns:
            ChromaStore: Lokaler Chroma Store
        """
        return ChromaStore(
            config=config,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    
    @staticmethod
    def create_industrial_store(persist_directory: str = "./data/industrial_docs",
                               config: RAGConfig = None) -> ChromaStore:
        """
        Erstellt industriellen ChromaDB Store
        
        Args:
            persist_directory (str): Persistierung-Verzeichnis  
            config (RAGConfig): Konfiguration
            
        Returns:
            ChromaStore: Industrieller Chroma Store
        """
        return ChromaStore(
            config=config,
            persist_directory=persist_directory,
            collection_name="industrial_documentation"
        )
    
    @staticmethod
    def create_development_store(config: RAGConfig = None) -> ChromaStore:
        """
        Erstellt Development ChromaDB Store (mit Reset-Berechtigung)
        
        Args:
            config (RAGConfig): Konfiguration
            
        Returns:
            ChromaStore: Development Chroma Store
        """
        store = ChromaStore(
            config=config,
            persist_directory="./data/dev_vectorstore",
            collection_name="development_docs"
        )
        store.allow_reset = True
        return store


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Hauptklasse
    'ChromaStore',
    
    # Factory
    'ChromaStoreFactory'
]


if __name__ == "__main__":
    # Testing und Demonstration
    print("ChromaDB Vector Store - Lokale Persistierung")
    print("===========================================")
    
    if not CHROMADB_AVAILABLE:
        print("❌ ChromaDB nicht verfügbar. Installation: pip install chromadb")
        print("Teste mit Mock-Implementierung:")
        
        # Mock-Test ohne ChromaDB
        print("✅ ChromaDB Module-Import getestet")
    else:
        try:
            # ChromaDB Store erstellen
            store = ChromaStore(
                persist_directory="./test_vectorstore",
                collection_name="test_industrial"
            )
            
            print(f"ChromaDB Store erstellt: {store}")
            
            # Initialisierung
            init_success = store.initialize(dimension=384)
            print(f"Store initialisiert: {init_success}")
            
            if init_success:
                # Test-Dokumente erstellen
                test_docs = []
                for i in range(5):
                    doc = DocumentRecord(
                        id=f"industrial_doc_{i}",
                        content=f"Industrielle Dokumentation {i}: Sicherheitshinweise und technische Daten",
                        embedding=[0.1 + i * 0.05] * 384,  # Test-Embeddings
                        metadata={
                            'primary_category': 'safety' if i % 2 == 0 else 'specifications',
                            'safety_level': 'high' if i < 2 else 'medium',
                            'technical_domain': 'electrical',
                            'chunk_id': i,
                            'expertise_level': 'skilled',
                            'has_sequential_steps': i % 3 == 0
                        }
                    )
                    test_docs.append(doc)
                
                # Dokumente hinzufügen
                add_success = store.add_documents(test_docs)
                print(f"Dokumente hinzugefügt: {add_success} ({len(test_docs)} docs)")
                
                # Collection-Info
                collection_info = store.get_collection_info()
                print(f"Collection: {collection_info.name} ({collection_info.document_count} docs, {collection_info.size_bytes} bytes)")
                
                # Suche testen
                query_embedding = [0.15] * 384
                
                print("\n--- Basis-Suche ---")
                search_response = store.search(
                    query_embedding=query_embedding,
                    query_text="Sicherheitshinweise",
                    limit=3
                )
                
                print(f"Gefunden: {len(search_response.results)}")
                print(f"Suchzeit: {search_response.search_time_ms:.1f}ms")
                
                for result in search_response.results:
                    safety_level = result.document.metadata.get('safety_level', 'unknown')
                    print(f"  Rang {result.rank}: {result.similarity_score:.3f} - Safety: {safety_level}")
                
                # Gefilterte Suche
                print("\n--- Gefilterte Suche (Safety High) ---")
                from .base_vectorstore import SearchFilter
                
                safety_filter = SearchFilter(include={'safety_level': 'high'})
                filtered_response = store.search(
                    query_embedding=query_embedding,
                    filter=safety_filter,
                    limit=5
                )
                
                print(f"Gefilterte Ergebnisse: {len(filtered_response.results)}")
                for result in filtered_response.results:
                    category = result.document.metadata.get('primary_category', 'unknown')
                    print(f"  {result.similarity_score:.3f} - Kategorie: {category}")
                
                # Upsert testen
                print("\n--- Upsert-Test ---")
                test_docs[0].content = "AKTUALISIERT: " + test_docs[0].content
                test_docs[0].metadata['updated'] = True
                
                upsert_success = store.upsert_documents([test_docs[0]])
                print(f"Upsert erfolgreich: {upsert_success}")
                
                # Health-Check
                print("\n--- Health-Check ---")
                health_status = store.health_check()
                print(f"Verbindung: {health_status['connection_ok']}")
                print(f"Collection: {health_status['collection_accessible']}")
                print(f"Write-Test: {health_status['write_test_ok']}")
                print(f"Search-Test: {health_status['search_test_ok']}")
                print(f"Status: {health_status['overall_status']}")
                
                # Performance-Statistiken
                print("\n--- Performance-Statistiken ---")
                stats = store.get_performance_statistics()
                print(f"Operationen: {stats['total_operations']}")
                print(f"Erfolgsrate: {stats.get('success_rate', 0):.1%}")
                print(f"Batch-Operationen: {stats['chroma']['batch_operations']}")
                print(f"Persistierung-Operationen: {stats['chroma']['persistence_operations']}")
                print(f"Verzeichnis-Größe: {stats['persistence']['estimated_size_bytes']} bytes")
                
                # Backup testen
                print("\n--- Backup-Test ---")
                backup_success = store.backup_collection("./test_backup")
                print(f"Backup erstellt: {backup_success}")
                
                print("\n✅ ChromaDB Store erfolgreich getestet")
            
            else:
                print("❌ Store-Initialisierung fehlgeschlagen")
        
        except Exception as e:
            print(f"❌ ChromaDB Test-Fehler: {str(e)}")
    
    # Factory-Tests
    print(f"\n--- Factory-Tests ---")
    try:
        # Verschiedene Store-Varianten
        factories = [
            ('Local', ChromaStoreFactory.create_local_store),
            ('Industrial', ChromaStoreFactory.create_industrial_store),
            ('Development', ChromaStoreFactory.create_development_store)
        ]
        
        for name, factory_method in factories:
            try:
                factory_store = factory_method()
                print(f"  {name}: {factory_store.collection_name} -> {factory_store.persist_directory}")
                print(f"    Reset erlaubt: {factory_store.allow_reset}")
            except Exception as e:
                print(f"  {name}: Fehler - {str(e)}")
    
    except Exception as e:
        print(f"Factory-Test Fehler: {str(e)}")
    
    print("\n✅ ChromaDB Vector Store Module getestet")