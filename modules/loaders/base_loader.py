#!/usr/bin/env python3
"""
Abstract Base Loader für RAG Chatbot Industrial

Definiert das Plugin-Interface für alle Document-Loader-Implementierungen
mit standardisierten Methoden, Fehlerbehandlung und Metadaten-Management.

Features:
- Plugin-Interface für verschiedene Dokumentformate
- Standardisierte Datenstrukturen für Dokumente
- Robuste Fehlerbehandlung mit Context-Informationen
- Metadaten-Extraktion und -Validierung
- Performance-Monitoring für Loader-Operationen
- Konfigurierbare Verarbeitungsoptionen

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Module-Komponente
"""

import os
import hashlib
import mimetypes
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO
from dataclasses import dataclass, field

# Import der Core-Komponenten
from core import (
    get_logger, RAGConfig, get_current_config,
    RAGException, DocumentProcessingError, ValidationError,
    create_error_context, log_performance
)


# =============================================================================
# STANDARDISIERTE DATENSTRUKTUREN
# =============================================================================

@dataclass
class DocumentMetadata:
    """
    Standardisierte Metadaten-Struktur für alle Dokumente
    
    Attributes:
        source_path (str): Pfad zur ursprünglichen Datei
        file_name (str): Name der Datei ohne Pfad
        file_extension (str): Dateierweiterung (z.B. '.pdf')
        file_size (int): Dateigröße in Bytes
        mime_type (str): MIME-Type der Datei
        created_at (datetime): Erstellungszeitpunkt der Verarbeitung
        modified_at (Optional[datetime]): Letzte Änderung der Originaldatei
        file_hash (str): SHA-256 Hash der Datei für Duplikaterkennung
        language (Optional[str]): Erkannte Sprache des Inhalts
        encoding (str): Text-Encoding (Standard: 'utf-8')
        page_count (Optional[int]): Anzahl Seiten (falls zutreffend)
        word_count (Optional[int]): Geschätzte Wortanzahl
        character_count (int): Anzahl Zeichen im Text
        custom_metadata (Dict[str, Any]): Format-spezifische Metadaten
    """
    source_path: str
    file_name: str
    file_extension: str
    file_size: int
    mime_type: str
    created_at: datetime
    file_hash: str
    character_count: int
    modified_at: Optional[datetime] = None
    language: Optional[str] = None
    encoding: str = 'utf-8'
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert Metadaten zu Dictionary für Serialisierung
        
        Returns:
            Dict[str, Any]: Serialisierte Metadaten
        """
        return {
            'source_path': self.source_path,
            'file_name': self.file_name,
            'file_extension': self.file_extension,
            'file_size': self.file_size,
            'mime_type': self.mime_type,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat() if self.modified_at else None,
            'file_hash': self.file_hash,
            'language': self.language,
            'encoding': self.encoding,
            'page_count': self.page_count,
            'word_count': self.word_count,
            'character_count': self.character_count,
            'custom_metadata': self.custom_metadata
        }


@dataclass
class Document:
    """
    Standardisierte Dokument-Struktur für das RAG-System
    
    Attributes:
        content (str): Extrahierter Textinhalt
        metadata (DocumentMetadata): Detaillierte Metadaten
        doc_id (Optional[str]): Eindeutige Dokument-ID
        source (str): Quelldatei-Pfad (Alias für metadata.source_path)
    """
    content: str
    metadata: DocumentMetadata
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        """Generiert Dokument-ID falls nicht vorhanden"""
        if self.doc_id is None:
            # ID basierend auf Hash und Dateiname generieren
            content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()[:8]
            file_hash = self.metadata.file_hash[:8]
            self.doc_id = f"{self.metadata.file_name}_{file_hash}_{content_hash}"
    
    @property
    def source(self) -> str:
        """Alias für metadata.source_path"""
        return self.metadata.source_path
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert Document zu Dictionary für Serialisierung
        
        Returns:
            Dict[str, Any]: Serialisiertes Dokument
        """
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'metadata': self.metadata.to_dict(),
            'source': self.source
        }


@dataclass
class LoaderResult:
    """
    Ergebnis-Struktur für Loader-Operationen
    
    Attributes:
        success (bool): Erfolgsstatus der Operation
        documents (List[Document]): Liste der extrahierten Dokumente
        error_message (Optional[str]): Fehlermeldung bei Misserfolg
        processing_stats (Dict[str, Any]): Verarbeitungsstatistiken
        warnings (List[str]): Warnungen während der Verarbeitung
    """
    success: bool
    documents: List[Document] = field(default_factory=list)
    error_message: Optional[str] = None
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# ABSTRACT BASE LOADER
# =============================================================================

class BaseDocumentLoader(ABC):
    """
    Abstract Base Class für alle Document-Loader
    
    Definiert das Plugin-Interface mit standardisierten Methoden
    für Dokumenten-Loading, Validierung und Metadaten-Extraktion.
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialisiert Base-Loader
        
        Args:
            config (RAGConfig): Konfiguration für Loader-Verhalten
        """
        self.config = config or get_current_config()
        self.logger = get_logger(self.__class__.__name__.lower(), "modules.loaders")
        
        # Loader-spezifische Konfiguration
        self.max_file_size = self.config.app.max_file_size_mb * 1024 * 1024  # MB zu Bytes
        self.supported_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        # Performance-Tracking
        self._processing_stats = {
            'files_processed': 0,
            'total_processing_time_ms': 0,
            'average_processing_time_ms': 0,
            'errors_encountered': 0
        }
    
    @abstractmethod
    def supports_format(self, file_extension: str) -> bool:
        """
        Prüft ob Loader das Dateiformat unterstützt
        
        Args:
            file_extension (str): Dateierweiterung (mit oder ohne Punkt)
            
        Returns:
            bool: True wenn Format unterstützt wird
        """
        pass
    
    @abstractmethod
    def _extract_content(self, file_path: str) -> str:
        """
        Extrahiert Textinhalt aus Datei (Implementierung in Subclass)
        
        Args:
            file_path (str): Pfad zur Datei
            
        Returns:
            str: Extrahierter Textinhalt
            
        Raises:
            DocumentProcessingError: Bei Extraktionsfehlern
        """
        pass
    
    @abstractmethod
    def _extract_custom_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extrahiert format-spezifische Metadaten (Implementierung in Subclass)
        
        Args:
            file_path (str): Pfad zur Datei
            
        Returns:
            Dict[str, Any]: Format-spezifische Metadaten
        """
        pass
    
    @log_performance()
    def load_document(self, file_path: Union[str, Path]) -> LoaderResult:
        """
        Hauptmethode zum Laden eines Dokuments
        
        Args:
            file_path (Union[str, Path]): Pfad zur zu ladenden Datei
            
        Returns:
            LoaderResult: Ergebnis mit extrahierten Dokumenten oder Fehlern
        """
        file_path = str(file_path)
        
        try:
            # Pre-Processing Validierung
            validation_result = self._validate_file(file_path)
            if not validation_result.success:
                return validation_result
            
            # Metadaten extrahieren
            metadata = self._extract_metadata(file_path)
            
            # Content extrahieren
            content = self._extract_content(file_path)
            
            # Post-Processing Validierung
            if not content or len(content.strip()) == 0:
                return LoaderResult(
                    success=False,
                    error_message=f"Keine Textinhalte in {file_path} gefunden"
                )
            
            # Dokument erstellen
            document = Document(
                content=content,
                metadata=metadata
            )
            
            # Statistiken aktualisieren
            self._update_processing_stats(success=True)
            
            # Erfolgreiches Ergebnis
            result = LoaderResult(
                success=True,
                documents=[document],
                processing_stats=self._get_current_stats()
            )
            
            self.logger.info(
                f"Dokument erfolgreich geladen: {metadata.file_name}",
                extra={
                    'extra_data': {
                        'file_path': file_path,
                        'file_size': metadata.file_size,
                        'content_length': len(content),
                        'loader_type': self.__class__.__name__
                    }
                }
            )
            
            return result
            
        except Exception as e:
            self._update_processing_stats(success=False)
            
            # Strukturierte Fehlerbehandlung
            error_context = create_error_context(
                component=f"modules.loaders.{self.__class__.__name__}",
                operation="load_document",
                file_path=file_path
            )
            
            if isinstance(e, RAGException):
                # RAG-Exception weiterleiten
                return LoaderResult(
                    success=False,
                    error_message=str(e),
                    processing_stats=self._get_current_stats()
                )
            else:
                # Standard-Exception wrappen
                wrapped_error = DocumentProcessingError(
                    message=f"Fehler beim Laden von {file_path}: {str(e)}",
                    document_path=file_path,
                    processing_stage="document_loading",
                    context=error_context,
                    original_exception=e
                )
                
                return LoaderResult(
                    success=False,
                    error_message=str(wrapped_error),
                    processing_stats=self._get_current_stats()
                )
    
    def load_multiple_documents(self, file_paths: List[Union[str, Path]]) -> LoaderResult:
        """
        Lädt mehrere Dokumente in einem Batch
        
        Args:
            file_paths (List[Union[str, Path]]): Liste der Dateipfade
            
        Returns:
            LoaderResult: Aggregiertes Ergebnis aller Dokumente
        """
        all_documents = []
        all_warnings = []
        error_count = 0
        
        for file_path in file_paths:
            result = self.load_document(file_path)
            
            if result.success:
                all_documents.extend(result.documents)
                all_warnings.extend(result.warnings)
            else:
                error_count += 1
                all_warnings.append(f"Fehler bei {file_path}: {result.error_message}")
        
        # Gesamtergebnis bestimmen
        success = len(all_documents) > 0
        
        return LoaderResult(
            success=success,
            documents=all_documents,
            error_message=f"{error_count} von {len(file_paths)} Dateien konnten nicht geladen werden" if error_count > 0 else None,
            processing_stats=self._get_current_stats(),
            warnings=all_warnings
        )
    
    def _validate_file(self, file_path: str) -> LoaderResult:
        """
        Validiert Datei vor der Verarbeitung
        
        Args:
            file_path (str): Pfad zur Datei
            
        Returns:
            LoaderResult: Validierungsergebnis
        """
        # Datei existiert
        if not os.path.exists(file_path):
            return LoaderResult(
                success=False,
                error_message=f"Datei nicht gefunden: {file_path}"
            )
        
        # Dateigröße prüfen
        file_size = os.path.getsize(file_path)
        if file_size > self.max_file_size:
            return LoaderResult(
                success=False,
                error_message=f"Datei zu groß: {file_size / 1024 / 1024:.1f}MB > {self.max_file_size / 1024 / 1024}MB"
            )
        
        if file_size == 0:
            return LoaderResult(
                success=False,
                error_message=f"Datei ist leer: {file_path}"
            )
        
        # Format unterstützt
        file_extension = Path(file_path).suffix.lower()
        if not self.supports_format(file_extension):
            return LoaderResult(
                success=False,
                error_message=f"Nicht unterstütztes Dateiformat: {file_extension}"
            )
        
        return LoaderResult(success=True)
    
    def _extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extrahiert vollständige Metadaten aus Datei
        
        Args:
            file_path (str): Pfad zur Datei
            
        Returns:
            DocumentMetadata: Extrahierte Metadaten
        """
        path_obj = Path(file_path)
        
        # Basis-Dateiinformationen
        file_stats = os.stat(file_path)
        
        # Hash berechnen
        file_hash = self._calculate_file_hash(file_path)
        
        # MIME-Type ermitteln
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = 'application/octet-stream'
        
        # Format-spezifische Metadaten
        custom_metadata = self._extract_custom_metadata(file_path)
        
        return DocumentMetadata(
            source_path=file_path,
            file_name=path_obj.name,
            file_extension=path_obj.suffix.lower(),
            file_size=file_stats.st_size,
            mime_type=mime_type,
            created_at=datetime.now(),
            modified_at=datetime.fromtimestamp(file_stats.st_mtime),
            file_hash=file_hash,
            character_count=0,  # Wird nach Content-Extraktion aktualisiert
            custom_metadata=custom_metadata
        )
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Berechnet SHA-256 Hash der Datei
        
        Args:
            file_path (str): Pfad zur Datei
            
        Returns:
            str: SHA-256 Hash als Hex-String
        """
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            # Datei in Chunks lesen für große Dateien
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _estimate_word_count(self, content: str) -> int:
        """
        Schätzt Wortanzahl im Text
        
        Args:
            content (str): Textinhalt
            
        Returns:
            int: Geschätzte Wortanzahl
        """
        # Einfache Wortanzahl-Schätzung
        words = content.split()
        return len([word for word in words if len(word.strip()) > 0])
    
    def _detect_language(self, content: str) -> Optional[str]:
        """
        Erkennt Sprache des Textinhalts
        
        Args:
            content (str): Textinhalt
            
        Returns:
            Optional[str]: ISO-Sprachcode oder None
        """
        # Einfache Sprach-Heuristik (kann durch externe Bibliothek ersetzt werden)
        sample = content[:1000].lower()
        
        # Deutsche Indikatoren
        german_indicators = ['der', 'die', 'das', 'und', 'oder', 'mit', 'von', 'zu', 'für']
        german_count = sum(1 for word in german_indicators if word in sample)
        
        # Englische Indikatoren
        english_indicators = ['the', 'and', 'or', 'with', 'from', 'to', 'for', 'of', 'in']
        english_count = sum(1 for word in english_indicators if word in sample)
        
        if german_count > english_count and german_count > 2:
            return 'de'
        elif english_count > german_count and english_count > 2:
            return 'en'
        
        return None
    
    def _update_processing_stats(self, success: bool) -> None:
        """Aktualisiert interne Verarbeitungsstatistiken"""
        self._processing_stats['files_processed'] += 1
        
        if not success:
            self._processing_stats['errors_encountered'] += 1
    
    def _get_current_stats(self) -> Dict[str, Any]:
        """Holt aktuelle Verarbeitungsstatistiken"""
        stats = self._processing_stats.copy()
        
        if stats['files_processed'] > 0:
            stats['error_rate'] = stats['errors_encountered'] / stats['files_processed']
            stats['success_rate'] = 1.0 - stats['error_rate']
        else:
            stats['error_rate'] = 0.0
            stats['success_rate'] = 0.0
        
        return stats
    
    def get_supported_formats(self) -> List[str]:
        """
        Holt Liste der unterstützten Dateiformate
        
        Returns:
            List[str]: Liste der Dateierweiterungen (mit Punkt)
        """
        # Subklassen sollten diese Methode überschreiben
        return []
    
    def get_loader_info(self) -> Dict[str, Any]:
        """
        Holt Informationen über den Loader
        
        Returns:
            Dict[str, Any]: Loader-Metadaten und Statistiken
        """
        return {
            'loader_name': self.__class__.__name__,
            'supported_formats': self.get_supported_formats(),
            'max_file_size_mb': self.max_file_size / 1024 / 1024,
            'processing_stats': self._get_current_stats(),
            'config': {
                'supported_encodings': self.supported_encodings
            }
        }


# =============================================================================
# LOADER REGISTRY FÜR PLUGIN-MANAGEMENT
# =============================================================================

class LoaderRegistry:
    """
    Registry für Document-Loader-Plugins
    
    Verwaltet verfügbare Loader und ermöglicht automatische
    Format-Erkennung und Loader-Selection.
    """
    
    def __init__(self):
        self._loaders: Dict[str, BaseDocumentLoader] = {}
        self.logger = get_logger("loader_registry", "modules.loaders")
    
    def register_loader(self, loader: BaseDocumentLoader) -> None:
        """
        Registriert einen Loader im Registry
        
        Args:
            loader (BaseDocumentLoader): Zu registrierender Loader
        """
        loader_name = loader.__class__.__name__
        self._loaders[loader_name] = loader
        
        supported_formats = loader.get_supported_formats()
        self.logger.info(
            f"Loader registriert: {loader_name}",
            extra={
                'extra_data': {
                    'loader_name': loader_name,
                    'supported_formats': supported_formats
                }
            }
        )
    
    def get_loader_for_file(self, file_path: str) -> Optional[BaseDocumentLoader]:
        """
        Findet passenden Loader für Datei
        
        Args:
            file_path (str): Pfad zur Datei
            
        Returns:
            Optional[BaseDocumentLoader]: Passender Loader oder None
        """
        file_extension = Path(file_path).suffix.lower()
        
        for loader in self._loaders.values():
            if loader.supports_format(file_extension):
                return loader
        
        return None
    
    def get_all_supported_formats(self) -> List[str]:
        """
        Holt alle unterstützten Formate aller registrierten Loader
        
        Returns:
            List[str]: Liste aller unterstützten Dateierweiterungen
        """
        all_formats = set()
        
        for loader in self._loaders.values():
            all_formats.update(loader.get_supported_formats())
        
        return list(all_formats)
    
    def get_registry_info(self) -> Dict[str, Any]:
        """
        Holt Informationen über das Registry
        
        Returns:
            Dict[str, Any]: Registry-Status und Statistiken
        """
        return {
            'registered_loaders': list(self._loaders.keys()),
            'total_loaders': len(self._loaders),
            'supported_formats': self.get_all_supported_formats(),
            'loader_details': {
                name: loader.get_loader_info()
                for name, loader in self._loaders.items()
            }
        }


# =============================================================================
# GLOBALES LOADER REGISTRY
# =============================================================================

# Singleton-Pattern für globales Registry
_loader_registry: Optional[LoaderRegistry] = None


def get_loader_registry() -> LoaderRegistry:
    """
    Holt globale LoaderRegistry-Instanz
    
    Returns:
        LoaderRegistry: Globale Registry-Instanz
    """
    global _loader_registry
    
    if _loader_registry is None:
        _loader_registry = LoaderRegistry()
    
    return _loader_registry


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Datenstrukturen
    'DocumentMetadata', 'Document', 'LoaderResult',
    
    # Base-Klassen
    'BaseDocumentLoader',
    
    # Registry
    'LoaderRegistry', 'get_loader_registry'
]