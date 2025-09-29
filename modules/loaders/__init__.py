#!/usr/bin/env python3
"""
Document Loaders Module für RAG Chatbot Industrial

Sammelt alle Document-Loader-Implementierungen und stellt einheitliche
Schnittstelle für das Laden verschiedener Dokumentformate bereit.

Aktuell implementierte Loader:
- PDF-Loader (extrahiert aus monolithischem Code)

Geplante Erweiterungen:
- DOCX-Loader für Word-Dokumente
- Markdown-Loader für .md Dateien
- TXT-Loader für einfache Textdateien
- Web-Scraper für Online-Inhalte

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Phase 2 Migration
"""

from typing import List, Optional, Dict, Any

# Core-Komponenten
from core import get_logger, RAGConfig, get_current_config

# Base-Klassen und Registry
from .base_loader import (
    BaseDocumentLoader, DocumentMetadata, Document, LoaderResult,
    LoaderRegistry, get_loader_registry
)

# Konkrete Loader-Implementierungen
from .pdf_loader import (
    PDFDocumentLoader, create_pdf_loader, register_pdf_loader,
    LegacyPDFProcessor
)


# =============================================================================
# LOADER FACTORY UND MANAGEMENT
# =============================================================================

class LoaderFactory:
    """
    Factory für automatische Loader-Erstellung basierend auf Dateiformaten
    
    Vereinfacht die Verwendung verschiedener Loader durch automatische
    Format-Erkennung und Loader-Selection.
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialisiert Loader-Factory
        
        Args:
            config (RAGConfig): Konfiguration für Loader
        """
        self.config = config or get_current_config()
        self.logger = get_logger("loader_factory", "modules.loaders")
        self.registry = get_loader_registry()
        
        # Alle verfügbaren Loader registrieren
        self._register_all_loaders()
    
    def _register_all_loaders(self) -> None:
        """Registriert alle verfügbaren Loader im Registry"""
        try:
            # PDF-Loader registrieren
            register_pdf_loader(self.registry)
            
            # TODO: Weitere Loader hier registrieren
            # register_docx_loader(self.registry)
            # register_markdown_loader(self.registry)
            # register_txt_loader(self.registry)
            
            registered_loaders = self.registry.get_registry_info()
            self.logger.info(
                f"Loader-Factory initialisiert mit {registered_loaders['total_loaders']} Loadern",
                extra={
                    'extra_data': {
                        'registered_loaders': registered_loaders['registered_loaders'],
                        'supported_formats': registered_loaders['supported_formats']
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Fehler bei Loader-Registrierung: {e}")
    
    def create_loader_for_file(self, file_path: str) -> Optional[BaseDocumentLoader]:
        """
        Erstellt passenden Loader für Datei
        
        Args:
            file_path (str): Pfad zur Datei
            
        Returns:
            Optional[BaseDocumentLoader]: Passender Loader oder None
        """
        return self.registry.get_loader_for_file(file_path)
    
    def load_document(self, file_path: str) -> LoaderResult:
        """
        Lädt Dokument mit automatischer Loader-Selection
        
        Args:
            file_path (str): Pfad zur zu ladenden Datei
            
        Returns:
            LoaderResult: Verarbeitungsergebnis
        """
        loader = self.create_loader_for_file(file_path)
        
        if loader is None:
            from pathlib import Path
            file_extension = Path(file_path).suffix.lower()
            
            return LoaderResult(
                success=False,
                error_message=f"Kein Loader für Format '{file_extension}' verfügbar"
            )
        
        return loader.load_document(file_path)
    
    def load_multiple_documents(self, file_paths: List[str]) -> LoaderResult:
        """
        Lädt mehrere Dokumente mit automatischer Loader-Selection
        
        Args:
            file_paths (List[str]): Liste der Dateipfade
            
        Returns:
            LoaderResult: Aggregiertes Verarbeitungsergebnis
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
        
        return LoaderResult(
            success=len(all_documents) > 0,
            documents=all_documents,
            error_message=f"{error_count} von {len(file_paths)} Dateien konnten nicht geladen werden" if error_count > 0 else None,
            warnings=all_warnings
        )
    
    def get_supported_formats(self) -> List[str]:
        """
        Holt alle unterstützten Dateiformate
        
        Returns:
            List[str]: Liste aller unterstützten Extensions
        """
        return self.registry.get_all_supported_formats()
    
    def get_factory_info(self) -> Dict[str, Any]:
        """
        Holt Factory-Informationen
        
        Returns:
            Dict[str, Any]: Factory-Status und verfügbare Loader
        """
        registry_info = self.registry.get_registry_info()
        
        return {
            'factory_name': 'LoaderFactory',
            'total_loaders': registry_info['total_loaders'],
            'supported_formats': registry_info['supported_formats'],
            'registered_loaders': registry_info['registered_loaders'],
            'loader_details': registry_info['loader_details']
        }


# =============================================================================
# GLOBALE FACTORY-INSTANZ
# =============================================================================

_loader_factory: Optional[LoaderFactory] = None


def get_loader_factory(config: RAGConfig = None) -> LoaderFactory:
    """
    Holt globale LoaderFactory-Instanz (Singleton)
    
    Args:
        config (RAGConfig): Konfiguration für Factory
        
    Returns:
        LoaderFactory: Globale Factory-Instanz
    """
    global _loader_factory
    
    if _loader_factory is None:
        _loader_factory = LoaderFactory(config)
    
    return _loader_factory


# =============================================================================
# CONVENIENCE-FUNKTIONEN
# =============================================================================

def load_document(file_path: str) -> LoaderResult:
    """
    Convenience-Funktion zum Laden eines Dokuments
    
    Args:
        file_path (str): Pfad zur Datei
        
    Returns:
        LoaderResult: Verarbeitungsergebnis
    """
    factory = get_loader_factory()
    return factory.load_document(file_path)


def load_multiple_documents(file_paths: List[str]) -> LoaderResult:
    """
    Convenience-Funktion zum Laden mehrerer Dokumente
    
    Args:
        file_paths (List[str]): Liste der Dateipfade
        
    Returns:
        LoaderResult: Aggregiertes Ergebnis
    """
    factory = get_loader_factory()
    return factory.load_multiple_documents(file_paths)


def get_supported_formats() -> List[str]:
    """
    Convenience-Funktion für unterstützte Formate
    
    Returns:
        List[str]: Liste der unterstützten Extensions
    """
    factory = get_loader_factory()
    return factory.get_supported_formats()


def load_pdf_from_bytes(pdf_bytes: bytes, filename: str = "document.pdf") -> LoaderResult:
    """
    Convenience-Funktion für PDF aus Byte-Array (Streamlit-Upload)
    
    Args:
        pdf_bytes (bytes): PDF-Daten
        filename (str): Dateiname für Metadaten
        
    Returns:
        LoaderResult: Verarbeitungsergebnis
    """
    pdf_loader = create_pdf_loader()
    return pdf_loader.load_from_bytes(pdf_bytes, filename)


# =============================================================================
# MIGRATION-SUPPORT
# =============================================================================

def create_legacy_processor(config: RAGConfig = None) -> LegacyPDFProcessor:
    """
    Erstellt Legacy-Processor für Migration vom monolithischen Code
    
    Args:
        config (RAGConfig): Konfiguration
        
    Returns:
        LegacyPDFProcessor: Legacy-kompatible API
    """
    return LegacyPDFProcessor(config)


# =============================================================================
# MODUL-VALIDIERUNG UND HEALTH-CHECK
# =============================================================================

def validate_loader_module() -> Dict[str, Any]:
    """
    Validiert Loader-Modul und verfügbare Abhängigkeiten
    
    Returns:
        Dict[str, Any]: Validierungsergebnis
    """
def validate_loader_module() -> Dict[str, Any]:
    """
    Validiert Loader-Modul und verfügbare Abhängigkeiten
    
    Returns:
        Dict[str, Any]: Validierungsergebnis
    """
    validation_result = {
        'module_status': 'healthy',
        'available_loaders': [],
        'missing_dependencies': [],
        'warnings': []
    }
    
    # PDF-Loader Abhängigkeiten prüfen
    pdf_dependencies = []
    try:
        from langchain_community.document_loaders import PyPDFLoader
        pdf_dependencies.append('langchain_pypdf')
    except ImportError:
        validation_result['missing_dependencies'].append('langchain_community')
    
    try:
        import PyPDF2
        pdf_dependencies.append('pypdf2')
    except ImportError:
        validation_result['missing_dependencies'].append('PyPDF2')
    
    try:
        import fitz
        pdf_dependencies.append('pymupdf')
    except ImportError:
        validation_result['missing_dependencies'].append('PyMuPDF')
    
    if pdf_dependencies:
        validation_result['available_loaders'].append({
            'type': 'PDFDocumentLoader',
            'dependencies': pdf_dependencies,
            'formats': ['.pdf']
        })
    else:
        validation_result['warnings'].append('Keine PDF-Bibliotheken verfügbar')
        validation_result['module_status'] = 'degraded'
    
    # Factory-Status prüfen
    try:
        factory = get_loader_factory()
        factory_info = factory.get_factory_info()
        validation_result['factory_status'] = 'available'
        validation_result['total_supported_formats'] = len(factory_info['supported_formats'])
    except Exception as e:
        validation_result['warnings'].append(f'Factory-Initialisierung fehlgeschlagen: {e}')
        validation_result['module_status'] = 'degraded'
    
    return validation_result


def get_module_health() -> Dict[str, Any]:
    """
    Holt Gesundheitsstatus des Loader-Moduls
    
    Returns:
        Dict[str, Any]: Health-Status mit Metriken
    """
    try:
        factory = get_loader_factory()
        registry = get_loader_registry()
        
        factory_info = factory.get_factory_info()
        registry_info = registry.get_registry_info()
        
        return {
            'status': 'healthy',
            'factory_available': True,
            'registry_available': True,
            'total_loaders': registry_info['total_loaders'],
            'supported_formats': registry_info['supported_formats'],
            'loader_details': registry_info['loader_details']
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'factory_available': False,
            'registry_available': False
        }


# =============================================================================
# AUTO-INITIALISIERUNG
# =============================================================================

def _initialize_loader_module():
    """Initialisiert Loader-Modul beim Import"""
    try:
        # Factory initialisieren (registriert automatisch alle Loader)
        factory = get_loader_factory()
        
        # Validierung durchführen
        validation = validate_loader_module()
        
        logger = get_logger("loaders_init", "modules.loaders")
        logger.info(
            f"Loader-Modul initialisiert: {validation['module_status']}",
            extra={
                'extra_data': {
                    'available_loaders': len(validation['available_loaders']),
                    'missing_dependencies': validation['missing_dependencies'],
                    'warnings': validation['warnings']
                }
            }
        )
        
        return True
        
    except Exception as e:
        # Stille Initialisierung - Fehler erst bei Nutzung anzeigen
        return False


# Automatische Initialisierung beim Modul-Import
_initialization_success = _initialize_loader_module()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Base-Klassen und Datenstrukturen
    'BaseDocumentLoader', 'DocumentMetadata', 'Document', 'LoaderResult',
    
    # Registry und Factory
    'LoaderRegistry', 'LoaderFactory', 'get_loader_registry', 'get_loader_factory',
    
    # Konkrete Loader
    'PDFDocumentLoader', 'create_pdf_loader',
    
    # Convenience-Funktionen
    'load_document', 'load_multiple_documents', 'get_supported_formats',
    'load_pdf_from_bytes',
    
    # Migration-Support
    'LegacyPDFProcessor', 'create_legacy_processor',
    
    # Validierung und Health
    'validate_loader_module', 'get_module_health'
]


if __name__ == "__main__":
    # Modul-Testing und Demonstration
    print("Document Loaders Module - Phase 2 Migration")
    print("===========================================")
    
    # Initialisierungsstatus
    if _initialization_success:
        print("✅ Modul erfolgreich initialisiert")
    else:
        print("⚠️ Modul-Initialisierung mit Problemen")
    
    # Validierung durchführen
    validation = validate_loader_module()
    print(f"\nModul-Status: {validation['module_status']}")
    print(f"Verfügbare Loader: {len(validation['available_loaders'])}")
    
    for loader_info in validation['available_loaders']:
        print(f"  - {loader_info['type']}: {loader_info['formats']}")
        print(f"    Abhängigkeiten: {', '.join(loader_info['dependencies'])}")
    
    if validation['missing_dependencies']:
        print(f"\nFehlende Abhängigkeiten: {', '.join(validation['missing_dependencies'])}")
    
    if validation['warnings']:
        print(f"\nWarnungen:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    # Factory-Info anzeigen
    try:
        factory = get_loader_factory()
        factory_info = factory.get_factory_info()
        
        print(f"\nFactory-Info:")
        print(f"  - Registrierte Loader: {factory_info['total_loaders']}")
        print(f"  - Unterstützte Formate: {', '.join(factory_info['supported_formats'])}")
        
    except Exception as e:
        print(f"\n❌ Factory-Error: {e}")
    
    # Health-Check
    health = get_module_health()
    print(f"\nHealth-Status: {health['status']}")
    
    # Beispiel-Usage (falls möglich)
    supported_formats = get_supported_formats()
    if supported_formats:
        print(f"\nUnterstützte Formate für automatisches Loading:")
        for fmt in supported_formats:
            print(f"  - {fmt}")
        
        # Test-Datei laden (falls vorhanden)
        import os
        test_files = ["example.pdf", "test.pdf", "sample.pdf"]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"\n📄 Teste Laden von: {test_file}")
                
                result = load_document(test_file)
                if result.success:
                    print(f"  ✅ Erfolgreich: {len(result.documents)} Dokument(e)")
                    if result.documents:
                        doc = result.documents[0]
                        print(f"  📝 Content-Länge: {len(doc.content)} Zeichen")
                        print(f"  📋 Metadaten: {doc.metadata.file_name}")
                else:
                    print(f"  ❌ Fehlgeschlagen: {result.error_message}")
                break
        else:
            print(f"\nKeine Test-Dateien gefunden ({', '.join(test_files)})")
    
    print("\n🎯 Loader-Modul Testing abgeschlossen")