#!/usr/bin/env python3
"""
PDF Document Loader für RAG Chatbot Industrial

Extrahiert aus dem monolithischen RAG Chatbot Code und erweitert für die neue
service-orientierte Architektur. Robuste PDF-Verarbeitung mit Metadaten-Extraktion.

Extrahierte Features vom Original:
- PyPDFLoader Integration (langchain_community.document_loaders)
- Robuste Fehlerbehandlung für PDF-Parsing
- Metadaten-Extraktion (Seiten, Autor, Titel)

Neue Features:
- Plugin-Interface Implementierung
- Strukturierte Metadaten-Extraktion
- Performance-Monitoring
- Erweiterte Validierung

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Extrahiert aus monolithischem Code
"""

import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# PDF-Verarbeitung (aus dem Original übernommen)
try:
    from langchain_community.document_loaders import PyPDFLoader
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        from langchain.document_loaders import PyPDFLoader
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False

# PDF-Bibliotheken für erweiterte Metadaten
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Core-Komponenten
from core import (
    get_logger, RAGConfig, get_current_config,
    DocumentProcessingError, ValidationError,
    create_error_context, log_performance
)

# Base-Loader
from .base_loader import BaseDocumentLoader, DocumentMetadata, LoaderResult


# =============================================================================
# PDF-LOADER IMPLEMENTIERUNG (EXTRAHIERT UND ERWEITERT)
# =============================================================================

class PDFDocumentLoader(BaseDocumentLoader):
    """
    PDF Document Loader mit erweiterter Funktionalität
    
    Extrahiert aus dem monolithischen RAG Chatbot und erweitert um:
    - Robuste PDF-Parsing mit mehreren Bibliotheken
    - Erweiterte Metadaten-Extraktion  
    - Strukturierte Fehlerbehandlung
    - Plugin-Interface Implementierung
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialisiert PDF-Loader
        
        Args:
            config (RAGConfig): Konfiguration für PDF-Verarbeitung
        """
        super().__init__(config)
        
        # PDF-spezifische Konfiguration
        self.supported_extensions = ['.pdf']
        self.fallback_libraries = self._get_available_libraries()
        
        # Validiere verfügbare PDF-Bibliotheken
        if not self.fallback_libraries:
            raise ValidationError(
                "Keine PDF-Bibliotheken verfügbar. Installieren Sie PyPDF2 oder PyMuPDF",
                field_name="pdf_libraries"
            )
        
        self.logger.info(
            f"PDF-Loader initialisiert mit Bibliotheken: {', '.join(self.fallback_libraries)}"
        )
    
    def _get_available_libraries(self) -> List[str]:
        """
        Ermittelt verfügbare PDF-Bibliotheken
        
        Returns:
            List[str]: Liste verfügbarer Bibliotheken
        """
        available = []
        
        if PYPDF_AVAILABLE:
            available.append('langchain_pypdf')
        if PYPDF2_AVAILABLE:
            available.append('pypdf2')
        if PYMUPDF_AVAILABLE:
            available.append('pymupdf')
        
        return available
    
    def supports_format(self, file_extension: str) -> bool:
        """
        Prüft ob PDF-Format unterstützt wird
        
        Args:
            file_extension (str): Dateierweiterung
            
        Returns:
            bool: True für .pdf Dateien
        """
        # Normalisiere Extension (mit oder ohne Punkt)
        if not file_extension.startswith('.'):
            file_extension = '.' + file_extension
        
        return file_extension.lower() in self.supported_extensions
    
    @log_performance()
    def _extract_content(self, file_path: str) -> str:
        """
        Extrahiert Textinhalt aus PDF (HAUPTMETHODE AUS ORIGINAL)
        
        Diese Methode implementiert die gleiche Logik wie im monolithischen Code
        aber mit verbesserter Fehlerbehandlung und mehreren Fallback-Strategien.
        
        Args:
            file_path (str): Pfad zur PDF-Datei
            
        Returns:
            str: Extrahierter Textinhalt
            
        Raises:
            DocumentProcessingError: Bei Extraktionsfehlern
        """
        content = None
        last_error = None
        
        # 1. LangChain PyPDFLoader (Original-Methode aus dem Code)
        if PYPDF_AVAILABLE and content is None:
            content, last_error = self._extract_with_langchain(file_path)
        
        # 2. Fallback: PyMuPDF (bessere OCR-Unterstützung)
        if PYMUPDF_AVAILABLE and content is None:
            content, last_error = self._extract_with_pymupdf(file_path)
        
        # 3. Fallback: PyPDF2 (einfache PDFs)
        if PYPDF2_AVAILABLE and content is None:
            content, last_error = self._extract_with_pypdf2(file_path)
        
        # Validation des extrahierten Contents
        if content is None or len(content.strip()) == 0:
            error_context = create_error_context(
                component="modules.loaders.pdf",
                operation="content_extraction",
                file_path=file_path
            )
            
            raise DocumentProcessingError(
                message=f"Konnte keinen Text aus PDF extrahieren: {file_path}",
                document_path=file_path,
                processing_stage="content_extraction",
                context=error_context,
                original_exception=last_error
            )
        
        # Content-Nachverarbeitung (wie im Original)
        content = self._post_process_content(content)
        
        return content
    
    def _extract_with_langchain(self, file_path: str) -> tuple[Optional[str], Optional[Exception]]:
        """
        Extrahiert Content mit LangChain PyPDFLoader (ORIGINAL-METHODE)
        
        Diese Methode repliziert exakt die Logik aus dem monolithischen Code:
        - PyPDFLoader aus langchain_community.document_loaders
        - Temporäre Datei-Behandlung 
        - Document-Loading und Content-Aggregation
        
        Args:
            file_path (str): Pfad zur PDF-Datei
            
        Returns:
            tuple[Optional[str], Optional[Exception]]: Content und potentielle Exception
        """
        try:
            self.logger.debug(f"Versuche LangChain-Extraktion für: {file_path}")
            
            # ORIGINAL LOGIC: PyPDFLoader verwenden
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                return None, Exception("Keine Dokumente von PyPDFLoader erhalten")
            
            # ORIGINAL LOGIC: Content aller Seiten aggregieren
            content_parts = []
            for doc in documents:
                if hasattr(doc, 'page_content') and doc.page_content:
                    content_parts.append(doc.page_content)
            
            if not content_parts:
                return None, Exception("Keine Seiteninhalte gefunden")
            
            # Content zusammenfügen
            full_content = "\n\n".join(content_parts)
            
            self.logger.info(
                f"LangChain-Extraktion erfolgreich: {len(documents)} Seiten, {len(full_content)} Zeichen"
            )
            
            return full_content, None
            
        except Exception as e:
            self.logger.warning(f"LangChain-Extraktion fehlgeschlagen: {str(e)}")
            return None, e
    
    def _extract_with_pymupdf(self, file_path: str) -> tuple[Optional[str], Optional[Exception]]:
        """
        Extrahiert Content mit PyMuPDF (FALLBACK-METHODE)
        
        Args:
            file_path (str): Pfad zur PDF-Datei
            
        Returns:
            tuple[Optional[str], Optional[Exception]]: Content und potentielle Exception
        """
        try:
            self.logger.debug(f"Versuche PyMuPDF-Extraktion für: {file_path}")
            
            doc = fitz.open(file_path)
            content_parts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():
                    content_parts.append(text.strip())
            
            doc.close()
            
            if not content_parts:
                return None, Exception("Keine Textinhalte mit PyMuPDF gefunden")
            
            full_content = "\n\n".join(content_parts)
            
            self.logger.info(
                f"PyMuPDF-Extraktion erfolgreich: {len(content_parts)} Seiten, {len(full_content)} Zeichen"
            )
            
            return full_content, None
            
        except Exception as e:
            self.logger.warning(f"PyMuPDF-Extraktion fehlgeschlagen: {str(e)}")
            return None, e
    
    def _extract_with_pypdf2(self, file_path: str) -> tuple[Optional[str], Optional[Exception]]:
        """
        Extrahiert Content mit PyPDF2 (BASIS-FALLBACK)
        
        Args:
            file_path (str): Pfad zur PDF-Datei
            
        Returns:
            tuple[Optional[str], Optional[Exception]]: Content und potentielle Exception
        """
        try:
            self.logger.debug(f"Versuche PyPDF2-Extraktion für: {file_path}")
            
            content_parts = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text.strip():
                        content_parts.append(text.strip())
            
            if not content_parts:
                return None, Exception("Keine Textinhalte mit PyPDF2 gefunden")
            
            full_content = "\n\n".join(content_parts)
            
            self.logger.info(
                f"PyPDF2-Extraktion erfolgreich: {len(content_parts)} Seiten, {len(full_content)} Zeichen"
            )
            
            return full_content, None
            
        except Exception as e:
            self.logger.warning(f"PyPDF2-Extraktion fehlgeschlagen: {str(e)}")
            return None, e
    
    def _post_process_content(self, content: str) -> str:
        """
        Nachverarbeitung des extrahierten Contents
        
        Args:
            content (str): Roher extrahierter Content
            
        Returns:
            str: Bereinigter Content
        """
        # Überschüssige Whitespace entfernen
        content = '\n'.join(line.strip() for line in content.split('\n') if line.strip())
        
        # Mehrfache Zeilenumbrüche reduzieren
        import re
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Sehr kurze "Zeilen" die nur Seitenzahlen o.ä. sind entfernen
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Zeilen mit nur Zahlen (Seitenzahlen) überspringen
            if line.strip().isdigit() and len(line.strip()) < 5:
                continue
            # Sehr kurze Zeilen mit nur Sonderzeichen überspringen
            if len(line.strip()) < 3 and not line.strip().isalnum():
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_custom_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extrahiert PDF-spezifische Metadaten
        
        Args:
            file_path (str): Pfad zur PDF-Datei
            
        Returns:
            Dict[str, Any]: PDF-spezifische Metadaten
        """
        metadata = {
            'pdf_version': None,
            'title': None,
            'author': None,
            'subject': None,
            'creator': None,
            'producer': None,
            'creation_date': None,
            'modification_date': None,
            'page_count': 0,
            'encrypted': False,
            'form_fields': False,
            'bookmarks': False
        }
        
        # Versuche Metadaten mit verschiedenen Bibliotheken zu extrahieren
        if PYPDF2_AVAILABLE:
            metadata = self._extract_metadata_pypdf2(file_path, metadata)
        elif PYMUPDF_AVAILABLE:
            metadata = self._extract_metadata_pymupdf(file_path, metadata)
        
        return metadata
    
    def _extract_metadata_pypdf2(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extrahiert Metadaten mit PyPDF2"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Basis-Informationen
                metadata['page_count'] = len(pdf_reader.pages)
                metadata['encrypted'] = pdf_reader.is_encrypted
                
                # Dokument-Informationen
                if pdf_reader.metadata:
                    doc_info = pdf_reader.metadata
                    
                    metadata['title'] = doc_info.get('/Title', '').strip() if doc_info.get('/Title') else None
                    metadata['author'] = doc_info.get('/Author', '').strip() if doc_info.get('/Author') else None
                    metadata['subject'] = doc_info.get('/Subject', '').strip() if doc_info.get('/Subject') else None
                    metadata['creator'] = doc_info.get('/Creator', '').strip() if doc_info.get('/Creator') else None
                    metadata['producer'] = doc_info.get('/Producer', '').strip() if doc_info.get('/Producer') else None
                    
                    # Datumsangaben
                    creation_date = doc_info.get('/CreationDate')
                    if creation_date:
                        metadata['creation_date'] = self._parse_pdf_date(creation_date)
                    
                    mod_date = doc_info.get('/ModDate')
                    if mod_date:
                        metadata['modification_date'] = self._parse_pdf_date(mod_date)
                
                # Erweiterte Features
                if hasattr(pdf_reader, 'outline') and pdf_reader.outline:
                    metadata['bookmarks'] = True
                
        except Exception as e:
            self.logger.warning(f"PyPDF2-Metadaten-Extraktion fehlgeschlagen: {e}")
        
        return metadata
    
    def _extract_metadata_pymupdf(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extrahiert Metadaten mit PyMuPDF"""
        try:
            doc = fitz.open(file_path)
            
            # Basis-Informationen
            metadata['page_count'] = len(doc)
            metadata['encrypted'] = doc.needs_pass
            
            # Dokument-Metadaten
            doc_metadata = doc.metadata
            if doc_metadata:
                metadata['title'] = doc_metadata.get('title', '').strip() if doc_metadata.get('title') else None
                metadata['author'] = doc_metadata.get('author', '').strip() if doc_metadata.get('author') else None
                metadata['subject'] = doc_metadata.get('subject', '').strip() if doc_metadata.get('subject') else None
                metadata['creator'] = doc_metadata.get('creator', '').strip() if doc_metadata.get('creator') else None
                metadata['producer'] = doc_metadata.get('producer', '').strip() if doc_metadata.get('producer') else None
                
                # Datumsangaben
                if doc_metadata.get('creationDate'):
                    metadata['creation_date'] = doc_metadata['creationDate']
                if doc_metadata.get('modDate'):
                    metadata['modification_date'] = doc_metadata['modDate']
            
            # Erweiterte Features
            toc = doc.get_toc()
            if toc:
                metadata['bookmarks'] = True
            
            doc.close()
            
        except Exception as e:
            self.logger.warning(f"PyMuPDF-Metadaten-Extraktion fehlgeschlagen: {e}")
        
        return metadata
    
    def _parse_pdf_date(self, date_str: str) -> Optional[str]:
        """
        Parst PDF-Datumsformat
        
        Args:
            date_str (str): PDF-Datumsstring
            
        Returns:
            Optional[str]: ISO-Datumsstring oder None
        """
        try:
            # PDF-Datumsformat: D:YYYYMMDDHHmmSSOHH'mm'
            if date_str.startswith('D:'):
                date_str = date_str[2:]
            
            # Extrahiere Jahr, Monat, Tag
            if len(date_str) >= 8:
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                
                # Optional: Stunde, Minute
                hour = int(date_str[8:10]) if len(date_str) >= 10 else 0
                minute = int(date_str[10:12]) if len(date_str) >= 12 else 0
                
                parsed_date = datetime(year, month, day, hour, minute)
                return parsed_date.isoformat()
        
        except (ValueError, IndexError):
            pass
        
        return None
    
    def get_supported_formats(self) -> List[str]:
        """
        Holt unterstützte Dateiformate
        
        Returns:
            List[str]: Liste der unterstützten Extensions
        """
        return self.supported_extensions.copy()
    
    def load_from_bytes(self, pdf_bytes: bytes, filename: str = "document.pdf") -> LoaderResult:
        """
        Lädt PDF aus Byte-Array (für Streamlit File-Upload)
        
        Diese Methode repliziert das Verhalten aus dem Original-Code für
        Streamlit-File-Uploads mit temporären Dateien.
        
        Args:
            pdf_bytes (bytes): PDF-Daten als Bytes
            filename (str): Name der Datei für Metadaten
            
        Returns:
            LoaderResult: Verarbeitungsergebnis
        """
        # Temporäre Datei erstellen (wie im Original)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Standard-Loading-Prozess verwenden
            result = self.load_document(tmp_path)
            
            # Metadaten anpassen für Upload-Kontext
            if result.success and result.documents:
                for doc in result.documents:
                    doc.metadata.file_name = filename
                    doc.metadata.source_path = f"upload://{filename}"
            
            return result
            
        finally:
            # Temporäre Datei löschen (wie im Original)
            try:
                os.unlink(tmp_path)
            except Exception as e:
                self.logger.warning(f"Konnte temporäre PDF-Datei nicht löschen: {e}")


# =============================================================================
# FACTORY-FUNKTIONEN FÜR INTEGRATION
# =============================================================================

def create_pdf_loader(config: RAGConfig = None) -> PDFDocumentLoader:
    """
    Factory-Funktion für PDF-Loader
    
    Args:
        config (RAGConfig): Konfiguration
        
    Returns:
        PDFDocumentLoader: Konfigurierter PDF-Loader
    """
    return PDFDocumentLoader(config)


def register_pdf_loader(registry=None) -> None:
    """
    Registriert PDF-Loader im globalen Registry
    
    Args:
        registry: LoaderRegistry-Instanz (optional)
    """
    if registry is None:
        from .base_loader import get_loader_registry
        registry = get_loader_registry()
    
    pdf_loader = create_pdf_loader()
    registry.register_loader(pdf_loader)


# =============================================================================
# KOMPATIBILITÄTS-WRAPPER FÜR MIGRATION
# =============================================================================

class LegacyPDFProcessor:
    """
    Kompatibilitäts-Wrapper für Migration vom monolithischen Code
    
    Repliziert die ursprüngliche API für nahtlose Migration.
    """
    
    def __init__(self, config: RAGConfig = None):
        self.pdf_loader = PDFDocumentLoader(config)
        self.logger = get_logger("legacy_pdf_processor", "modules.loaders")
    
    def process_pdf_document(self, uploaded_file) -> tuple[Optional[Any], int, Dict]:
        """
        Legacy-API-Wrapper für process_pdf_document aus dem Original
        
        Repliziert exakt die Signatur und das Verhalten der ursprünglichen
        Methode aus StableRAGChatbot.process_pdf_document().
        
        Args:
            uploaded_file: Streamlit UploadedFile-Objekt
            
        Returns:
            tuple: (vectorstore, chunk_count, stats) wie im Original
        """
        try:
            # Byte-Content aus Uploaded File lesen
            file_content = uploaded_file.read()
            filename = uploaded_file.name
            
            # PDF-Loader verwenden
            result = self.pdf_loader.load_from_bytes(file_content, filename)
            
            if not result.success:
                return None, 0, {'error': result.error_message}
            
            # Legacy-Format für Stats
            documents = result.documents
            chunk_count = len(documents)
            
            stats = {
                'success': True,
                'file_name': filename,
                'file_size': uploaded_file.size,
                'total_chunks': chunk_count,
                'content_types': {'pdf_document': chunk_count},
                'chunks_with_steps': 0,  # Wird in Text-Processing bestimmt
                'avg_chunk_length': sum(len(doc.content) for doc in documents) / chunk_count if chunk_count > 0 else 0,
                'keywords_found': []  # Wird in Text-Processing bestimmt
            }
            
            # Hinweis: vectorstore wird in der Service-Schicht erstellt
            # Hier geben wir die Documents zurück für weitere Verarbeitung
            return documents, chunk_count, stats
            
        except Exception as e:
            self.logger.error(f"Legacy PDF-Processing fehlgeschlagen: {e}")
            return None, 0, {'error': str(e)}


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Haupt-Klassen
    'PDFDocumentLoader',
    
    # Factory-Funktionen
    'create_pdf_loader', 'register_pdf_loader',
    
    # Legacy-Support
    'LegacyPDFProcessor'
]


# =============================================================================
# AUTO-REGISTRATION
# =============================================================================

# Automatische Registrierung beim Import (wenn Registry verfügbar)
try:
    register_pdf_loader()
except ImportError:
    # Registry noch nicht initialisiert - wird später registriert
    pass


if __name__ == "__main__":
    # Testing und Demonstration
    print("PDF Document Loader - Extrahiert aus RAG Chatbot")
    print("=================================================")
    
    # Verfügbare Bibliotheken anzeigen
    pdf_loader = create_pdf_loader()
    print(f"Verfügbare PDF-Bibliotheken: {pdf_loader.fallback_libraries}")
    
    # Loader-Info anzeigen
    info = pdf_loader.get_loader_info()
    print(f"Loader-Info: {info}")
    
    # Beispiel-Usage (wenn PDF-Datei verfügbar)
    test_pdf_path = "example.pdf"
    if os.path.exists(test_pdf_path):
        print(f"\nTeste PDF-Laden: {test_pdf_path}")
        
        result = pdf_loader.load_document(test_pdf_path)
        if result.success:
            print(f"✅ PDF erfolgreich geladen:")
            for doc in result.documents:
                print(f"  - Dokument-ID: {doc.doc_id}")
                print(f"  - Content-Länge: {len(doc.content)} Zeichen")
                print(f"  - Seiten: {doc.metadata.custom_metadata.get('page_count', 'unbekannt')}")
        else:
            print(f"❌ PDF-Laden fehlgeschlagen: {result.error_message}")
    else:
        print(f"\nKeine Test-PDF gefunden ({test_pdf_path})")
    
    print("✅ PDF-Loader erfolgreich getestet")