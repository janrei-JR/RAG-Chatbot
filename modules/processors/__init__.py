#!/usr/bin/env python3
"""
Text Processors Module für RAG Chatbot Industrial

Sammelt alle Text-Processing-Komponenten extrahiert aus dem monolithischen
RAG Chatbot Code und integriert sie in die service-orientierte Architektur.

Implementierte Prozessoren:
- TextAnalyzer: Intelligente Content-Klassifikation (extrahiert)
- StableTextSplitter: Bewährte Text-Segmentierung (extrahiert)

Kernverbesserungen gegenüber Original:
- Alle Regex-Pattern als Raw Strings gefixt
- Erweiterte Metadaten-Anreicherung
- Chroma-kompatible Datenstrukturen
- Performance-Monitoring und Statistiken

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Phase 2 Migration
"""

from typing import List, Dict, Any, Optional, Union

# Core-Komponenten
from core import get_logger, RAGConfig, get_current_config

# Text-Analyzer (extrahiert)
from .text_analyzer import (
    TextAnalyzer, BatchTextAnalyzer, AnalyzerFactory,
    ContentType, TextAnalysisResult
)

# Text-Splitter (extrahiert)
from .text_splitter import (
    StableTextSplitter, StructuralTextSplitter, IndustrialTextSplitter,
    TextSplitterFactory, LegacyTextSplitter,
    ChunkMetadata, TextChunk, SplittingResult, SplittingStrategy
)


# =============================================================================
# INTEGRIERTE PROCESSING-PIPELINE
# =============================================================================

class IntegratedTextProcessor:
    """
    Integrierte Text-Processing-Pipeline
    
    Kombiniert TextAnalyzer und TextSplitter für die vollständige
    Text-Verarbeitung wie sie im ursprünglichen monolithischen Code
    in der _analyze_chunks() Methode implementiert war.
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialisiert integrierten Text-Processor
        
        Args:
            config (RAGConfig): Konfiguration für Processing
        """
        self.config = config or get_current_config()
        self.logger = get_logger("integrated_text_processor", "modules.processors")
        
        # Komponenten initialisieren
        self.analyzer = TextAnalyzer(self.config)
        self.splitter = StableTextSplitter(self.config)
        
        # Processing-Statistiken
        self._processing_stats = {
            'documents_processed': 0,
            'total_chunks_created': 0,
            'chunks_with_steps': 0,
            'content_type_distribution': {},
            'avg_confidence_score': 0.0
        }
    
    def process_documents(self, documents: List) -> List[TextChunk]:
        """
        Vollständige Dokument-Verarbeitung (wie im Original)
        
        Repliziert die Funktionalität der ursprünglichen _analyze_chunks()
        Methode aus dem monolithischen Code.
        
        Args:
            documents (List): Liste der zu verarbeitenden Dokumente
            
        Returns:
            List[TextChunk]: Vollständig verarbeitete Text-Chunks
        """
        if not documents:
            return []
        
        try:
            # PHASE 1: Text-Splitting (aus Original split_documents())
            chunks = self.splitter.split_documents(documents)
            
            # PHASE 2: Erweiterte Analyse und Metadaten-Anreicherung
            enhanced_chunks = []
            
            for chunk in chunks:
                # Detaillierte Analyse pro Chunk
                analysis = self.analyzer.analyze_text(chunk.page_content)
                
                # Original-Metadaten mit Analyse-Ergebnissen erweitern
                chunk.metadata.confidence_score = analysis.confidence_score
                
                # Zusätzliche technische Begriffe hinzufügen
                if analysis.technical_terms:
                    existing_keywords = chunk.metadata.keywords.split(', ') if chunk.metadata.keywords else []
                    all_keywords = list(set(existing_keywords + analysis.technical_terms))
                    chunk.metadata.keywords = ', '.join(all_keywords)
                
                enhanced_chunks.append(chunk)
            
            # Statistiken aktualisieren (wie im Original _analyze_chunks())
            self._update_processing_stats(enhanced_chunks)
            
            self.logger.info(
                f"Dokumente vollständig verarbeitet: {len(documents)} Docs -> {len(enhanced_chunks)} Chunks",
                extra={
                    'extra_data': {
                        'input_documents': len(documents),
                        'output_chunks': len(enhanced_chunks),
                        'chunks_with_steps': sum(1 for c in enhanced_chunks if c.metadata.has_sequential_steps),
                        'avg_confidence': sum(c.metadata.confidence_score for c in enhanced_chunks) / len(enhanced_chunks) if enhanced_chunks else 0
                    }
                }
            )
            
            return enhanced_chunks
            
        except Exception as e:
            self.logger.error(f"Fehler bei integrierter Text-Verarbeitung: {str(e)}")
            raise
    
    def process_single_text(self, text: str, source: str = "direct_input") -> List[TextChunk]:
        """
        Verarbeitet einzelnen Text-String
        
        Args:
            text (str): Zu verarbeitender Text
            source (str): Quell-Identifikation
            
        Returns:
            List[TextChunk]: Verarbeitete Text-Chunks
        """
        # Pseudo-Dokument erstellen
        pseudo_doc = type('Document', (), {
            'page_content': text,
            'metadata': {'source': source}
        })()
        
        return self.process_documents([pseudo_doc])
    
    def _update_processing_stats(self, chunks: List[TextChunk]) -> None:
        """Aktualisiert Processing-Statistiken (wie im Original)"""
        self._processing_stats['documents_processed'] += 1
        self._processing_stats['total_chunks_created'] += len(chunks)
        
        steps_count = sum(1 for c in chunks if c.metadata.has_sequential_steps)
        self._processing_stats['chunks_with_steps'] += steps_count
        
        # Content-Type-Verteilung
        for chunk in chunks:
            content_type = chunk.metadata.content_type
            self._processing_stats['content_type_distribution'][content_type] = \
                self._processing_stats['content_type_distribution'].get(content_type, 0) + 1
        
        # Durchschnittliche Konfidenz
        if chunks:
            total_confidence = sum(c.metadata.confidence_score for c in chunks)
            self._processing_stats['avg_confidence_score'] = total_confidence / len(chunks)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Holt Processing-Statistiken (repliziert Original _analyze_chunks() Stats)
        
        Returns:
            Dict[str, Any]: Statistiken im Original-Format
        """
        stats = self._processing_stats.copy()
        
        # Original-Format für Kompatibilität
        if stats['total_chunks_created'] > 0:
            stats['success'] = True
            stats['chunks_with_steps'] = stats['chunks_with_steps']
            stats['content_types'] = stats['content_type_distribution']
            stats['avg_chunk_length'] = 0  # Wird vom Splitter berechnet
            
        return stats


# =============================================================================
# PROCESSOR-FACTORY FÜR VERSCHIEDENE ANWENDUNGSFÄLLE
# =============================================================================

class ProcessorFactory:
    """
    Factory für verschiedene Text-Processing-Konfigurationen
    """
    
    @staticmethod
    def create_industrial_processor(config: RAGConfig = None) -> IntegratedTextProcessor:
        """
        Erstellt Processor optimiert für industrielle Dokumentation
        
        Args:
            config (RAGConfig): Konfiguration
            
        Returns:
            IntegratedTextProcessor: Industriell optimierter Processor
        """
        processor = IntegratedTextProcessor(config)
        
        # Industriellen Analyzer verwenden
        processor.analyzer = AnalyzerFactory.create_industrial_analyzer(config)
        
        # Industriellen Splitter verwenden
        processor.splitter = TextSplitterFactory.create_splitter(
            SplittingStrategy.INDUSTRIAL, config
        )
        
        return processor
    
    @staticmethod
    def create_legacy_processor(config: RAGConfig = None) -> IntegratedTextProcessor:
        """
        Erstellt Processor mit exakter Original-Kompatibilität
        
        Args:
            config (RAGConfig): Konfiguration
            
        Returns:
            IntegratedTextProcessor: Legacy-kompatibler Processor
        """
        processor = IntegratedTextProcessor(config)
        
        # Standard-Komponenten (wie im Original)
        processor.analyzer = TextAnalyzer(config)
        processor.splitter = StableTextSplitter(config)
        
        return processor


# =============================================================================
# CONVENIENCE-FUNKTIONEN
# =============================================================================

def analyze_text(text: str, config: RAGConfig = None) -> TextAnalysisResult:
    """
    Convenience-Funktion für Text-Analyse
    
    Args:
        text (str): Zu analysierender Text
        config (RAGConfig): Konfiguration
        
    Returns:
        TextAnalysisResult: Analyse-Ergebnis
    """
    analyzer = TextAnalyzer(config)
    return analyzer.analyze_text(text)


def split_text(text: str, config: RAGConfig = None) -> List[TextChunk]:
    """
    Convenience-Funktion für Text-Splitting
    
    Args:
        text (str): Zu segmentierender Text
        config (RAGConfig): Konfiguration
        
    Returns:
        List[TextChunk]: Text-Chunks
    """
    splitter = StableTextSplitter(config)
    return splitter.split_text(text)


def process_text_complete(text: str, config: RAGConfig = None) -> List[TextChunk]:
    """
    Convenience-Funktion für komplette Text-Verarbeitung
    
    Args:
        text (str): Zu verarbeitender Text
        config (RAGConfig): Konfiguration
        
    Returns:
        List[TextChunk]: Vollständig verarbeitete Chunks
    """
    processor = IntegratedTextProcessor(config)
    return processor.process_single_text(text)


# =============================================================================
# LEGACY-KOMPATIBILITÄT FÜR MIGRATION
# =============================================================================

def create_legacy_text_splitter(config: RAGConfig = None) -> LegacyTextSplitter:
    """
    Erstellt Legacy-TextSplitter für nahtlose Migration
    
    Args:
        config (RAGConfig): Konfiguration
        
    Returns:
        LegacyTextSplitter: Legacy-kompatible API
    """
    return LegacyTextSplitter(config)


def replicate_original_analyze_chunks(chunks: List) -> Dict[str, Any]:
    """
    Repliziert die ursprüngliche _analyze_chunks() Methode
    
    Diese Funktion stellt die exakte API der ursprünglichen _analyze_chunks()
    Methode aus dem monolithischen Code bereit für Migration.
    
    Args:
        chunks (List): Text-Chunks aus split_documents()
        
    Returns:
        Dict[str, Any]: Statistiken im Original-Format
    """
    stats = {
        'total_chunks': len(chunks),
        'content_types': {},
        'chunks_with_steps': 0,
        'avg_chunk_length': 0,
        'keywords_found': set()
    }
    
    total_length = 0
    analyzer = TextAnalyzer()
    
    for chunk in chunks:
        # Original-Logic für Metadaten-Extraktion
        if hasattr(chunk, 'metadata'):
            metadata = chunk.metadata
        else:
            metadata = {}
        
        # Content-Type aus Metadaten oder analysieren
        content_type = metadata.get('content_type', 'unknown')
        if content_type == 'unknown':
            # Fallback auf Analyse
            try:
                analysis = analyzer.analyze_text(chunk.page_content)
                content_type = analysis.content_type.value
            except:
                content_type = 'general'
        
        stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
        
        # Schritte zählen
        if metadata.get('has_sequential_steps', False):
            stats['chunks_with_steps'] += 1
        
        # Chunk-Länge
        chunk_length = metadata.get('chunk_length', len(chunk.page_content))
        total_length += chunk_length
        
        # Keywords sammeln
        keywords = metadata.get('keywords', '')
        if keywords:
            stats['keywords_found'].update(keywords.split(', '))
    
    # Durchschnittliche Chunk-Länge
    stats['avg_chunk_length'] = total_length / len(chunks) if chunks else 0
    stats['keywords_found'] = list(stats['keywords_found'])
    
    return stats


# =============================================================================
# MODUL-VALIDIERUNG
# =============================================================================

def validate_processors_module() -> Dict[str, Any]:
    """
    Validiert Processors-Modul und Abhängigkeiten
    
    Returns:
        Dict[str, Any]: Validierungsergebnis
    """
    validation_result = {
        'module_status': 'healthy',
        'available_processors': [],
        'missing_dependencies': [],
        'warnings': []
    }
    
    # TextAnalyzer-Verfügbarkeit
    try:
        analyzer = TextAnalyzer()
        validation_result['available_processors'].append('TextAnalyzer')
    except Exception as e:
        validation_result['warnings'].append(f'TextAnalyzer-Problem: {e}')
    
    # LangChain-Abhängigkeiten für Splitter
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        validation_result['available_processors'].append('StableTextSplitter (LangChain)')
    except ImportError:
        validation_result['missing_dependencies'].append('langchain')
        validation_result['warnings'].append('LangChain nicht verfügbar - Fallback-Splitting aktiv')
    
    # Integrierte Pipeline
    try:
        processor = IntegratedTextProcessor()
        validation_result['available_processors'].append('IntegratedTextProcessor')
    except Exception as e:
        validation_result['warnings'].append(f'Pipeline-Problem: {e}')
        validation_result['module_status'] = 'degraded'
    
    return validation_result


def get_module_health() -> Dict[str, Any]:
    """
    Holt Gesundheitsstatus des Processors-Moduls
    
    Returns:
        Dict[str, Any]: Health-Status
    """
    try:
        # Test-Pipeline erstellen
        processor = IntegratedTextProcessor()
        
        # Einfachen Test durchführen
        test_text = "1. Test Schritt\n2. Zweiter Schritt\nTechnische Daten: 24V"
        chunks = processor.process_single_text(test_text)
        
        # Statistiken sammeln
        stats = processor.get_processing_statistics()
        
        return {
            'status': 'healthy',
            'test_chunks_created': len(chunks),
            'processors_available': ['TextAnalyzer', 'StableTextSplitter', 'IntegratedTextProcessor'],
            'processing_stats': stats
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'processors_available': []
        }


# =============================================================================
# AUTO-INITIALISIERUNG
# =============================================================================

def _initialize_processors_module():
    """Initialisiert Processors-Modul beim Import"""
    try:
        # Validierung durchführen
        validation = validate_processors_module()
        
        logger = get_logger("processors_init", "modules.processors")
        logger.info(
            f"Processors-Modul initialisiert: {validation['module_status']}",
            extra={
                'extra_data': {
                    'available_processors': validation['available_processors'],
                    'warnings': validation['warnings'],
                    'missing_dependencies': validation['missing_dependencies']
                }
            }
        )
        
        return True
        
    except Exception as e:
        return False


# Automatische Initialisierung
_initialization_success = _initialize_processors_module()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Text-Analyzer
    'TextAnalyzer', 'BatchTextAnalyzer', 'AnalyzerFactory',
    'ContentType', 'TextAnalysisResult',
    
    # Text-Splitter
    'StableTextSplitter', 'StructuralTextSplitter', 'IndustrialTextSplitter',
    'TextSplitterFactory', 'LegacyTextSplitter',
    'ChunkMetadata', 'TextChunk', 'SplittingResult', 'SplittingStrategy',
    
    # Integrierte Pipeline
    'IntegratedTextProcessor', 'ProcessorFactory',
    
    # Convenience-Funktionen
    'analyze_text', 'split_text', 'process_text_complete',
    
    # Legacy-Support
    'create_legacy_text_splitter', 'replicate_original_analyze_chunks',
    
    # Validierung
    'validate_processors_module', 'get_module_health'
]


if __name__ == "__main__":
    # Modul-Testing und Demonstration
    print("Text Processors Module - Phase 2 Migration")
    print("=========================================")
    
    # Initialisierungsstatus
    if _initialization_success:
        print("✅ Modul erfolgreich initialisiert")
    else:
        print("⚠️ Modul-Initialisierung mit Problemen")
    
    # Validierung
    validation = validate_processors_module()
    print(f"\nModul-Status: {validation['module_status']}")
    print(f"Verfügbare Prozessoren: {len(validation['available_processors'])}")
    
    for processor in validation['available_processors']:
        print(f"  - {processor}")
    
    if validation['missing_dependencies']:
        print(f"\nFehlende Abhängigkeiten: {', '.join(validation['missing_dependencies'])}")
    
    if validation['warnings']:
        print(f"\nWarnungen:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    # Test-Text für Demonstration
    test_text = """
    # Installationsanleitung
    
    1. Software herunterladen
    2. setup.exe als Administrator ausführen  
    3. Lizenzvereinbarung akzeptieren
    4. Installationspfad wählen
    
    WARNUNG: Stellen Sie sicher, dass alle Anwendungen geschlossen sind.
    
    Technische Voraussetzungen:
    - Windows 10 oder höher
    - 4 GB RAM minimum
    - 500 MB freier Speicherplatz
    """
    
    # Einzelne Komponenten testen
    print(f"\n--- Komponenten-Tests ---")
    
    # Text-Analyzer
    try:
        analysis = analyze_text(test_text)
        print(f"TextAnalyzer: {analysis.content_type.value} (Konfidenz: {analysis.confidence_score:.2f})")
        print(f"  Keywords: {analysis.keywords[:3]}")
        print(f"  Hat Schritte: {analysis.has_sequential_steps}")
    except Exception as e:
        print(f"TextAnalyzer-Fehler: {e}")
    
    # Text-Splitter
    try:
        chunks = split_text(test_text)
        print(f"StableTextSplitter: {len(chunks)} Chunks erstellt")
        if chunks:
            print(f"  Erster Chunk-Typ: {chunks[0].metadata.content_type}")
    except Exception as e:
        print(f"TextSplitter-Fehler: {e}")
    
    # Integrierte Pipeline
    try:
        integrated_chunks = process_text_complete(test_text)
        print(f"IntegratedProcessor: {len(integrated_chunks)} Chunks")
        
        steps_count = sum(1 for c in integrated_chunks if c.metadata.has_sequential_steps)
        print(f"  Chunks mit Schritten: {steps_count}")
    except Exception as e:
        print(f"IntegratedProcessor-Fehler: {e}")
    
    # Health-Check
    health = get_module_health()
    print(f"\nHealth-Status: {health['status']}")
    
    if health.get('test_chunks_created'):
        print(f"Test-Chunks erstellt: {health['test_chunks_created']}")
    
    print("\n🎯 Processors-Modul Testing abgeschlossen")