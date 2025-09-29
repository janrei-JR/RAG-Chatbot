#!/usr/bin/env python3
"""
Text Splitter für RAG Chatbot Industrial

Extrahiert aus dem monolithischen RAG Chatbot Code und erweitert für die neue
service-orientierte Architektur. Intelligente Text-Segmentierung mit Metadaten.

Extrahierte Features vom Original:
- StableTextSplitter-Klasse mit bewährten Separatoren
- RecursiveCharacterTextSplitter Integration (LangChain)
- Metadaten-Anreicherung für Chunks
- Regex-Pattern für Listen und Strukturen (ALLE RAW STRINGS GEFIXT)

Neue Features:
- Konfigurierbare Splitting-Strategien
- Erweiterte Chunk-Metadaten
- Performance-optimierte Verarbeitung
- Plugin-Interface für verschiedene Splitter-Arten

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Extrahiert aus monolithischem Code
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# LangChain Text-Splitter (aus Original übernommen)
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False

# Core-Komponenten
from core import (
    get_logger, RAGConfig, get_current_config,
    DocumentProcessingError, ValidationError,
    create_error_context, log_performance
)

# Text-Analyzer Integration
from .text_analyzer import TextAnalyzer, TextAnalysisResult


# =============================================================================
# CHUNK-METADATEN UND DATENSTRUKTUREN
# =============================================================================

@dataclass
class ChunkMetadata:
    """
    Erweiterte Metadaten für Text-Chunks
    
    Attributes:
        chunk_id (int): Eindeutige Chunk-ID
        content_type (str): Content-Typ aus TextAnalyzer
        keywords (str): Keywords als String (für Chroma-Kompatibilität)
        has_sequential_steps (bool): Enthält sequentielle Schritte
        chunk_length (int): Länge des Chunks in Zeichen
        word_count (int): Anzahl Wörter im Chunk
        step_count (int): Anzahl erkannter Schritte
        confidence_score (float): Vertrauen in Content-Klassifikation
        source_document (str): Ursprungsdokument
        chunk_index (int): Position im Originaldokument
        overlap_info (Dict[str, Any]): Informationen über Chunk-Überlappungen
    """
    chunk_id: int
    content_type: str = "general"
    keywords: str = ""  # String statt Liste für Chroma-Kompatibilität!
    has_sequential_steps: bool = False
    chunk_length: int = 0
    word_count: int = 0
    step_count: int = 0
    confidence_score: float = 0.0
    source_document: str = ""
    chunk_index: int = 0
    overlap_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert Metadaten zu Dictionary (Chroma-kompatibel)
        
        WICHTIG: Alle Werte müssen Chroma-kompatible Typen sein (str, int, float, bool)
        
        Returns:
            Dict[str, Any]: Chroma-kompatible Metadaten
        """
        return {
            'chunk_id': self.chunk_id,
            'content_type': self.content_type,
            'keywords': self.keywords,  # String, nicht Liste!
            'has_sequential_steps': self.has_sequential_steps,
            'chunk_length': self.chunk_length,
            'word_count': self.word_count,
            'step_count': self.step_count,
            'confidence_score': self.confidence_score,
            'source_document': self.source_document,
            'chunk_index': self.chunk_index
        }


@dataclass
class TextChunk:
    """
    Text-Chunk mit erweiterten Metadaten
    
    Kompatibel mit LangChain Document-Interface aber erweitert um
    industrielle Metadaten und Analyse-Ergebnisse.
    """
    page_content: str  # LangChain-kompatibel
    metadata: ChunkMetadata
    
    @property
    def content(self) -> str:
        """Alias für LangChain-Kompatibilität"""
        return self.page_content


@dataclass
class SplittingResult:
    """Ergebnis der Text-Segmentierung"""
    chunks: List[TextChunk]
    statistics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)


class SplittingStrategy(str, Enum):
    """Verfügbare Splitting-Strategien"""
    RECURSIVE = "recursive"           # Standard rekursive Segmentierung
    STRUCTURAL = "structural"         # Struktur-basierte Segmentierung
    SEMANTIC = "semantic"            # Semantik-basierte Segmentierung
    INDUSTRIAL = "industrial"        # Industrielle Dokumente optimiert


# =============================================================================
# STABLE TEXT SPLITTER (AUS ORIGINAL EXTRAHIERT)
# =============================================================================

class StableTextSplitter:
    """
    Stabilisierte Text-Splitter Implementierung extrahiert aus dem monolithischen Code
    
    Diese Klasse repliziert exakt die StableTextSplitter-Klasse aus dem Original
    mit allen bewährten Separatoren und der TextAnalyzer-Integration.
    
    ALLE REGEX-SEPARATOREN WURDEN ALS RAW STRINGS GEFIXT!
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialisiert Stable Text Splitter (ORIGINAL-KONSTRUKTOR)
        
        Args:
            config (RAGConfig): Konfiguration für Text-Splitting
        """
        self.config = config or get_current_config()
        self.logger = get_logger("stable_text_splitter", "modules.processors")
        
        # TextAnalyzer für Metadaten-Anreicherung (aus Original)
        self.analyzer = TextAnalyzer(self.config)
        
        # ORIGINAL SEPARATORS - ALLE ALS RAW STRINGS GEFIXT!
        self.separators = [
            "\n\n# ",          # Hauptüberschriften
            "\n\n## ",         # Unterüberschriften
            "\n\n### ",        # Sub-Überschriften
            "\n\n",            # Absätze
            r"\n\d+\. ",       # Nummerierte Listen - GEFIXT: Raw String!
            "\n- ",            # Bullet Points
            "\n• ",            # Alternative Bullets
            "\n",              # Zeilenumbrüche
            ". ",              # Sätze
            " ",               # Wörter
            ""                 # Zeichen
        ]
        
        # Performance-Tracking
        self._splitting_stats = {
            'documents_processed': 0,
            'total_chunks_created': 0,
            'avg_chunk_length': 0.0,
            'content_type_distribution': {}
        }
    
    @log_performance()
    def split_documents(self, documents: List) -> List[TextChunk]:
        """
        Teilt Dokumente intelligent in Chunks auf (ORIGINAL-METHODE)
        
        Diese Methode repliziert die split_documents() Logik aus dem monolithischen
        Code mit der bewährten RecursiveCharacterTextSplitter-Integration.
        
        Args:
            documents (List): Liste der zu segmentierenden Dokumente
            
        Returns:
            List[TextChunk]: Liste der erweiterten Text-Chunks
        """
        if not LANGCHAIN_AVAILABLE:
            raise DocumentProcessingError(
                "LangChain nicht verfügbar für Text-Splitting",
                processing_stage="text_splitting"
            )
        
        if not documents:
            return []
        
        try:
            # ORIGINAL LOGIC: RecursiveCharacterTextSplitter mit bewährten Parametern
            base_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.processing.chunk_size,
                chunk_overlap=self.config.processing.chunk_overlap,
                length_function=len,
                separators=self.separators,  # GEFIXT: Raw Strings verwenden
                keep_separator=True
            )
            
            # ORIGINAL LOGIC: Basis-Chunks erstellen
            base_chunks = base_splitter.split_documents(documents)
            
            # ORIGINAL LOGIC: Chunks mit erweiterten Metadaten anreichern
            enhanced_chunks = []
            
            for i, chunk in enumerate(base_chunks):
                # TextAnalyzer auf Chunk-Content anwenden (wie im Original)
                analysis_result = self.analyzer.analyze_text(chunk.page_content)
                
                # ORIGINAL METADATA-LOGIC mit Erweiterungen
                chunk_metadata = ChunkMetadata(
                    chunk_id=i,
                    content_type=analysis_result.content_type.value,
                    keywords=', '.join(analysis_result.keywords) if analysis_result.keywords else '',  # String!
                    has_sequential_steps=analysis_result.has_sequential_steps,
                    chunk_length=len(chunk.page_content),
                    word_count=len(chunk.page_content.split()),
                    step_count=analysis_result.step_count,
                    confidence_score=analysis_result.confidence_score,
                    source_document=chunk.metadata.get('source', 'unknown'),
                    chunk_index=i
                )
                
                # Enhanced TextChunk erstellen
                enhanced_chunk = TextChunk(
                    page_content=chunk.page_content,
                    metadata=chunk_metadata
                )
                
                enhanced_chunks.append(enhanced_chunk)
            
            # Statistiken aktualisieren
            self._update_splitting_stats(enhanced_chunks)
            
            self.logger.info(
                f"Dokumente erfolgreich segmentiert: {len(documents)} Docs -> {len(enhanced_chunks)} Chunks",
                extra={
                    'extra_data': {
                        'input_documents': len(documents),
                        'output_chunks': len(enhanced_chunks),
                        'avg_chunk_length': sum(c.metadata.chunk_length for c in enhanced_chunks) / len(enhanced_chunks) if enhanced_chunks else 0,
                        'chunks_with_steps': sum(1 for c in enhanced_chunks if c.metadata.has_sequential_steps)
                    }
                }
            )
            
            return enhanced_chunks
            
        except Exception as e:
            error_context = create_error_context(
                component="modules.processors.stable_text_splitter",
                operation="split_documents",
                document_count=len(documents)
            )
            
            raise DocumentProcessingError(
                message=f"Fehler bei Dokumenten-Segmentierung: {str(e)}",
                processing_stage="text_splitting",
                context=error_context,
                original_exception=e
            )
    
    def split_text(self, text: str, source_document: str = "direct_input") -> List[TextChunk]:
        """
        Segmentiert einzelnen Text-String
        
        Args:
            text (str): Zu segmentierender Text
            source_document (str): Quell-Dokument für Metadaten
            
        Returns:
            List[TextChunk]: Segmentierte Text-Chunks
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Pseudo-Dokument für split_documents erstellen
        pseudo_doc = type('Document', (), {
            'page_content': text,
            'metadata': {'source': source_document}
        })()
        
        return self.split_documents([pseudo_doc])
    
    def _update_splitting_stats(self, chunks: List[TextChunk]) -> None:
        """Aktualisiert interne Splitting-Statistiken"""
        self._splitting_stats['documents_processed'] += 1
        self._splitting_stats['total_chunks_created'] += len(chunks)
        
        if chunks:
            total_length = sum(c.metadata.chunk_length for c in chunks)
            self._splitting_stats['avg_chunk_length'] = total_length / len(chunks)
            
            # Content-Type-Verteilung
            for chunk in chunks:
                content_type = chunk.metadata.content_type
                self._splitting_stats['content_type_distribution'][content_type] = \
                    self._splitting_stats['content_type_distribution'].get(content_type, 0) + 1
    
    def get_splitting_statistics(self) -> Dict[str, Any]:
        """
        Holt Splitting-Statistiken
        
        Returns:
            Dict[str, Any]: Detaillierte Statistiken
        """
        return self._splitting_stats.copy()


# =============================================================================
# ERWEITERTE SPLITTER-IMPLEMENTIERUNGEN
# =============================================================================

class StructuralTextSplitter:
    """
    Struktureller Text-Splitter für Dokumente mit klaren Hierarchien
    
    Optimiert für technische Dokumentation mit Überschriften und Listen.
    """
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or get_current_config()
        self.logger = get_logger("structural_text_splitter", "modules.processors")
        self.analyzer = TextAnalyzer(self.config)
    
    def split_by_structure(self, text: str) -> List[TextChunk]:
        """
        Segmentiert Text basierend auf strukturellen Elementen
        
        Args:
            text (str): Zu segmentierender Text
            
        Returns:
            List[TextChunk]: Strukturell segmentierte Chunks
        """
        # Strukturelle Breakpoints finden
        breakpoints = self._find_structural_breakpoints(text)
        
        chunks = []
        for i, (start, end, structure_type) in enumerate(breakpoints):
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) > 0:
                analysis = self.analyzer.analyze_text(chunk_text)
                
                metadata = ChunkMetadata(
                    chunk_id=i,
                    content_type=analysis.content_type.value,
                    keywords=', '.join(analysis.keywords),
                    has_sequential_steps=analysis.has_sequential_steps,
                    chunk_length=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    step_count=analysis.step_count,
                    chunk_index=i,
                    overlap_info={'structure_type': structure_type}
                )
                
                chunks.append(TextChunk(
                    page_content=chunk_text,
                    metadata=metadata
                ))
        
        return chunks
    
    def _find_structural_breakpoints(self, text: str) -> List[Tuple[int, int, str]]:
        """Findet strukturelle Breakpoints im Text"""
        breakpoints = []
        
        # Überschriften-Pattern (Raw Strings!)
        header_patterns = [
            (r'\n#{1,3}\s+.+\n', 'header'),
            (r'\n\d+\.\s+.+\n', 'numbered_list'),
            (r'\n[-•]\s+.+\n', 'bullet_list'),
            (r'\n\n.+\n\n', 'paragraph')
        ]
        
        for pattern, structure_type in header_patterns:
            matches = list(re.finditer(pattern, text, re.MULTILINE))
            for match in matches:
                breakpoints.append((match.start(), match.end(), structure_type))
        
        # Sortieren nach Position
        breakpoints.sort(key=lambda x: x[0])
        
        # Lücken füllen
        if breakpoints:
            filled_breakpoints = []
            
            # Erster Chunk vom Anfang bis zum ersten Breakpoint
            if breakpoints[0][0] > 0:
                filled_breakpoints.append((0, breakpoints[0][0], 'intro'))
            
            # Breakpoints übernehmen
            filled_breakpoints.extend(breakpoints)
            
            # Letzter Chunk bis zum Ende
            if breakpoints[-1][1] < len(text):
                filled_breakpoints.append((breakpoints[-1][1], len(text), 'outro'))
            
            return filled_breakpoints
        else:
            return [(0, len(text), 'single_block')]


class IndustrialTextSplitter:
    """
    Splitter optimiert für industrielle Dokumentation
    
    Berücksichtigt technische Listen, Spezifikationen und Anleitungen.
    """
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or get_current_config()
        self.logger = get_logger("industrial_text_splitter", "modules.processors")
        self.analyzer = TextAnalyzer(self.config)
        
        # Industrielle Separatoren
        self.industrial_separators = [
            "\n\n# ",                    # Hauptüberschriften
            "\n\n## ",                   # Unterüberschriften
            "\n\nTechnische Daten:",     # Spezifikations-Blöcke
            "\n\nWARNUNG:",              # Sicherheitshinweise
            "\n\nHINWEIS:",              # Wichtige Hinweise
            "\n\nSchritt ",              # Anleitungsschritte
            r"\n\d+\.\s+",              # Nummerierte Listen
            "\n\n",                      # Standard-Absätze
            "\n",                        # Zeilenumbrüche
            ". ",                        # Sätze
            " "                          # Wörter
        ]
    
    def split_industrial_document(self, text: str) -> List[TextChunk]:
        """
        Segmentiert industrielle Dokumentation
        
        Args:
            text (str): Industrieller Dokumententext
            
        Returns:
            List[TextChunk]: Optimiert segmentierte Chunks
        """
        # Basis-Splitter mit industriellen Separatoren
        if LANGCHAIN_AVAILABLE:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.processing.chunk_size,
                chunk_overlap=self.config.processing.chunk_overlap,
                length_function=len,
                separators=self.industrial_separators,
                keep_separator=True
            )
            
            # Pseudo-Dokument erstellen
            pseudo_doc = type('Document', (), {
                'page_content': text,
                'metadata': {'source': 'industrial_document'}
            })()
            
            base_chunks = splitter.split_documents([pseudo_doc])
        else:
            # Fallback: Einfache Segmentierung
            base_chunks = self._simple_split(text, self.config.processing.chunk_size)
        
        # Industrielle Metadaten anreichern
        enhanced_chunks = []
        
        for i, chunk in enumerate(base_chunks):
            analysis = self.analyzer.analyze_text(chunk.page_content)
            
            # Industrielle Kategorisierung
            industrial_category = self._categorize_industrial_content(chunk.page_content)
            
            metadata = ChunkMetadata(
                chunk_id=i,
                content_type=analysis.content_type.value,
                keywords=', '.join(analysis.keywords),
                has_sequential_steps=analysis.has_sequential_steps,
                chunk_length=len(chunk.page_content),
                word_count=len(chunk.page_content.split()),
                step_count=analysis.step_count,
                confidence_score=analysis.confidence_score,
                chunk_index=i,
                overlap_info={'industrial_category': industrial_category}
            )
            
            enhanced_chunks.append(TextChunk(
                page_content=chunk.page_content,
                metadata=metadata
            ))
        
        return enhanced_chunks
    
    def _categorize_industrial_content(self, text: str) -> str:
        """Kategorisiert industrielle Inhalte"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['warnung', 'gefahr', 'achtung', 'warning', 'danger']):
            return 'safety'
        elif any(term in text_lower for term in ['technische daten', 'spezifikation', 'parameter']):
            return 'specifications'
        elif any(term in text_lower for term in ['installation', 'montage', 'aufbau']):
            return 'installation'
        elif any(term in text_lower for term in ['bedienung', 'betrieb', 'anwendung']):
            return 'operation'
        elif any(term in text_lower for term in ['wartung', 'instandhaltung', 'service']):
            return 'maintenance'
        else:
            return 'general'
    
    def _simple_split(self, text: str, chunk_size: int) -> List:
        """Einfache Fallback-Segmentierung ohne LangChain"""
        chunks = []
        words = text.split()
        
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                pseudo_chunk = type('Chunk', (), {
                    'page_content': chunk_text,
                    'metadata': {}
                })()
                chunks.append(pseudo_chunk)
                
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        # Letzten Chunk hinzufügen
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            pseudo_chunk = type('Chunk', (), {
                'page_content': chunk_text,
                'metadata': {}
            })()
            chunks.append(pseudo_chunk)
        
        return chunks


# =============================================================================
# SPLITTER-FACTORY UND REGISTRY
# =============================================================================

class TextSplitterFactory:
    """
    Factory für verschiedene Text-Splitter-Strategien
    """
    
    @staticmethod
    def create_splitter(strategy: SplittingStrategy, config: RAGConfig = None):
        """
        Erstellt Splitter basierend auf Strategie
        
        Args:
            strategy (SplittingStrategy): Gewünschte Splitter-Strategie
            config (RAGConfig): Konfiguration
            
        Returns:
            Entsprechende Splitter-Instanz
        """
        if strategy == SplittingStrategy.RECURSIVE:
            return StableTextSplitter(config)
        elif strategy == SplittingStrategy.STRUCTURAL:
            return StructuralTextSplitter(config)
        elif strategy == SplittingStrategy.INDUSTRIAL:
            return IndustrialTextSplitter(config)
        else:
            # Fallback auf bewährten Stable Splitter
            return StableTextSplitter(config)
    
    @staticmethod
    def create_stable_splitter(config: RAGConfig = None) -> StableTextSplitter:
        """
        Erstellt den bewährten Stable Text Splitter (aus Original)
        
        Args:
            config (RAGConfig): Konfiguration
            
        Returns:
            StableTextSplitter: Bewährte Splitter-Instanz aus dem Original
        """
        return StableTextSplitter(config)


# =============================================================================
# LEGACY-KOMPATIBILITÄT
# =============================================================================

class LegacyTextSplitter:
    """
    Legacy-Wrapper für Migration vom monolithischen Code
    
    Stellt die gleiche API bereit wie die ursprüngliche StableTextSplitter-Klasse.
    """
    
    def __init__(self, config: RAGConfig = None):
        self.splitter = StableTextSplitter(config)
    
    def split_documents(self, documents: List) -> List:
        """
        Legacy-API-Wrapper für split_documents
        
        Args:
            documents (List): LangChain-Dokumente
            
        Returns:
            List: Enhanced Chunks mit Original-API-Kompatibilität
        """
        enhanced_chunks = self.splitter.split_documents(documents)
        
        # Zurück zu LangChain-Format konvertieren für Kompatibilität
        legacy_chunks = []
        
        for chunk in enhanced_chunks:
            # LangChain Document-ähnliches Objekt erstellen
            legacy_chunk = type('Document', (), {
                'page_content': chunk.page_content,
                'metadata': chunk.metadata.to_dict()  # Chroma-kompatible Metadaten
            })()
            
            legacy_chunks.append(legacy_chunk)
        
        return legacy_chunks


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Datenstrukturen
    'ChunkMetadata', 'TextChunk', 'SplittingResult', 'SplittingStrategy',
    
    # Hauptklassen  
    'StableTextSplitter', 'StructuralTextSplitter', 'IndustrialTextSplitter',
    
    # Factory
    'TextSplitterFactory',
    
    # Legacy-Support
    'LegacyTextSplitter'
]


if __name__ == "__main__":
    # Testing und Demonstration
    print("Text Splitter - Extrahiert aus RAG Chatbot")
    print("==========================================")
    
    # Test-Text mit verschiedenen Strukturen
    test_text = """
    # Benutzerhandbuch
    
    ## 1. Installation
    
    1. Laden Sie die Software herunter
    2. Führen Sie setup.exe aus
    3. Folgen Sie den Anweisungen
    
    ## 2. Erste Schritte
    
    WARNUNG: Stellen Sie sicher, dass alle Kabel korrekt angeschlossen sind.
    
    Technische Daten:
    - Spannung: 24V DC
    - Leistung: 150W  
    - Betriebstemperatur: -10°C bis +60°C
    
    ## 3. Bedienung
    
    Die Bedienung erfolgt über das HMI-Panel.
    Klicken Sie auf "Start" um die Demo zu beginnen.
    
    Bei Problemen kontaktieren Sie den Support.
    """
    
    # Stable Text Splitter testen (Original-Replika)
    print("--- Stable Text Splitter (Original) ---")
    stable_splitter = StableTextSplitter()
    stable_chunks = stable_splitter.split_text(test_text)
    
    print(f"Chunks erstellt: {len(stable_chunks)}")
    for i, chunk in enumerate(stable_chunks[:3]):  # Erste 3 zeigen
        print(f"\nChunk {i+1}:")
        print(f"  Content-Type: {chunk.metadata.content_type}")
        print(f"  Hat Schritte: {chunk.metadata.has_sequential_steps}")
        print(f"  Keywords: {chunk.metadata.keywords}")
        print(f"  Text: {chunk.page_content[:100]}...")
    
    # Industrieller Splitter testen
    print(f"\n--- Industrial Text Splitter ---")
    industrial_splitter = IndustrialTextSplitter()
    industrial_chunks = industrial_splitter.split_industrial_document(test_text)
    
    print(f"Industrielle Chunks: {len(industrial_chunks)}")
    for chunk in industrial_chunks[:2]:  # Erste 2
        industrial_cat = chunk.metadata.overlap_info.get('industrial_category', 'unknown')
        print(f"  Kategorie: {industrial_cat}")
    
    # Factory testen
    print(f"\n--- Splitter Factory ---")
    factory_splitter = TextSplitterFactory.create_splitter(SplittingStrategy.RECURSIVE)
    factory_chunks = factory_splitter.split_text(test_text)
    print(f"Factory-Chunks: {len(factory_chunks)}")
    
    # Statistiken
    stats = stable_splitter.get_splitting_statistics()
    print(f"\nStatistiken: {stats}")
    
    print("\n✅ Text-Splitter erfolgreich getestet")