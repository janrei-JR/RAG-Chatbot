#!/usr/bin/env python3
"""
Metadata Enhancer für RAG Chatbot Industrial

Metadaten-Anreicherung für Text-Chunks durch Kombination und Veredelung der
Ergebnisse von TextAnalyzer, TextSplitter und ContentClassifier.

Features:
- Konsolidierung von Metadaten aus allen Processor-Modulen
- Erweiterte Metadaten-Anreicherung (Qualitäts-Scores, Kontext-Informationen)
- Chroma-kompatible Metadaten-Normalisierung
- Industrielle Metadaten-Anreicherung (Sicherheitsstufen, Fachbereich-Tags)
- Performance-optimierte Batch-Verarbeitung

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Integriert alle Processor-Module
"""

import re
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from collections import defaultdict

# Core-Komponenten
from core import (
    get_logger, RAGConfig, get_current_config,
    DocumentProcessingError, ValidationError,
    create_error_context, log_performance
)

# Processor-Module Integration
from .text_analyzer import TextAnalyzer, TextAnalysisResult, ContentType
from .content_classifier import (
    ContentClassifier, ClassificationResult, IndustrialCategory, ContentFlag
)
from .text_splitter import TextChunk, ChunkMetadata


# =============================================================================
# ERWEITERTE METADATEN-STRUKTUREN
# =============================================================================

class QualityLevel(str, Enum):
    """Qualitätsstufen für Text-Chunks"""
    EXCELLENT = "excellent"    # > 0.8 Confidence, vollständige Metadaten
    GOOD = "good"             # > 0.6 Confidence, meiste Metadaten
    MODERATE = "moderate"     # > 0.4 Confidence, Basis-Metadaten
    POOR = "poor"            # < 0.4 Confidence, minimale Metadaten


class SafetyLevel(str, Enum):
    """Sicherheitsstufen für industrielle Inhalte"""
    CRITICAL = "critical"     # Lebensgefahr, Hochspannung, etc.
    HIGH = "high"            # Wichtige Sicherheitshinweise
    MEDIUM = "medium"        # Standard-Sicherheitsmaßnahmen
    LOW = "low"             # Allgemeine Hinweise
    NONE = "none"           # Keine Sicherheitsrelevanz


class ExpertiseLevel(str, Enum):
    """Erforderliche Fachkenntnisse"""
    EXPERT = "expert"         # Spezialist erforderlich
    SKILLED = "skilled"       # Fachkraft erforderlich
    TRAINED = "trained"       # Eingewiesene Person
    BASIC = "basic"          # Grundkenntnisse ausreichend
    ANYONE = "anyone"        # Für jeden verständlich


@dataclass
class EnhancedMetadata:
    """
    Erweiterte, konsolidierte Metadaten für Text-Chunks
    
    Kombiniert Ergebnisse aller Processor-Module und fügt zusätzliche
    industrielle Metadaten hinzu.
    
    WICHTIG: Alle Werte müssen Chroma-kompatible Typen sein!
    """
    # Basis-Informationen
    chunk_id: int
    content_hash: str                      # SHA-256 Hash des Contents
    processing_timestamp: str              # ISO-Format Timestamp
    
    # Content-Klassifikation (aus ContentClassifier)
    primary_category: str                  # IndustrialCategory.value
    secondary_categories: str              # Comma-separated Liste
    content_flags: str                     # Comma-separated ContentFlags
    
    # Text-Analyse (aus TextAnalyzer)
    base_content_type: str                 # ContentType.value
    keywords: str                          # Comma-separated Keywords
    has_sequential_steps: bool
    step_count: int
    
    # Qualitäts- und Sicherheitsbewertung
    quality_level: str                     # QualityLevel.value
    safety_level: str                      # SafetyLevel.value
    expertise_level: str                   # ExpertiseLevel.value
    overall_confidence: float              # 0.0-1.0
    
    # Chunk-Eigenschaften
    chunk_length: int
    word_count: int
    sentence_count: int
    chunk_index: int
    
    # Industrielle Kategorisierung
    technical_domain: str                  # Fachbereich (electrical, mechanical, etc.)
    document_section: str                  # Dokumenten-Abschnitt
    language_primary: str                  # Hauptsprache
    
    # Kontext-Informationen
    source_document: str
    related_chunks: str                    # IDs verwandter Chunks (comma-separated)
    prerequisite_knowledge: str            # Erforderliches Vorwissen
    
    # Such- und Filter-Optimierung
    search_boost: float = 1.0             # Boost-Faktor für Suche (0.1-2.0)
    filter_tags: str = ""                 # Additional filter tags
    
    def to_chroma_metadata(self) -> Dict[str, Any]:
        """
        Konvertiert zu Chroma-kompatiblen Metadaten
        
        Returns:
            Dict[str, Any]: Chroma-kompatible Metadaten (nur str, int, float, bool)
        """
        return {
            'chunk_id': self.chunk_id,
            'content_hash': self.content_hash,
            'processing_timestamp': self.processing_timestamp,
            'primary_category': self.primary_category,
            'secondary_categories': self.secondary_categories,
            'content_flags': self.content_flags,
            'base_content_type': self.base_content_type,
            'keywords': self.keywords,
            'has_sequential_steps': self.has_sequential_steps,
            'step_count': self.step_count,
            'quality_level': self.quality_level,
            'safety_level': self.safety_level,
            'expertise_level': self.expertise_level,
            'overall_confidence': self.overall_confidence,
            'chunk_length': self.chunk_length,
            'word_count': self.word_count,
            'sentence_count': self.sentence_count,
            'chunk_index': self.chunk_index,
            'technical_domain': self.technical_domain,
            'document_section': self.document_section,
            'language_primary': self.language_primary,
            'source_document': self.source_document,
            'related_chunks': self.related_chunks,
            'prerequisite_knowledge': self.prerequisite_knowledge,
            'search_boost': self.search_boost,
            'filter_tags': self.filter_tags
        }


@dataclass
class EnhancedTextChunk:
    """
    Text-Chunk mit erweiterten, konsolidierten Metadaten
    """
    content: str
    metadata: EnhancedMetadata
    
    # LangChain-Kompatibilität
    @property
    def page_content(self) -> str:
        """LangChain-kompatible Eigenschaft"""
        return self.content


# =============================================================================
# METADATA ENHANCER HAUPTKLASSE
# =============================================================================

class MetadataEnhancer:
    """
    Erweiterte Metadaten-Anreicherung für industrielle Text-Chunks
    
    Konsolidiert und veredelt Metadaten aus TextAnalyzer, ContentClassifier
    und TextSplitter. Fügt industrielle Metadaten hinzu.
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialisiert Metadata Enhancer
        
        Args:
            config (RAGConfig): Konfiguration
        """
        self.config = config or get_current_config()
        self.logger = get_logger("metadata_enhancer", "modules.processors")
        
        # Processor-Module
        self.text_analyzer = TextAnalyzer(self.config)
        self.content_classifier = ContentClassifier(self.config)
        
        # Technische Domänen-Klassifikation
        self._init_domain_patterns()
        
        # Performance-Tracking
        self._enhancement_stats = {
            'chunks_enhanced': 0,
            'avg_enhancement_time_ms': 0.0,
            'quality_distribution': defaultdict(int),
            'safety_distribution': defaultdict(int),
            'domain_distribution': defaultdict(int)
        }
    
    def _init_domain_patterns(self) -> None:
        """Initialisiert Muster für technische Domänen-Erkennung"""
        
        self.domain_patterns = {
            'electrical': [
                # Keywords
                ['spannung', 'voltage', 'strom', 'current', 'leistung', 'power', 
                 'widerstand', 'resistance', 'kondensator', 'capacitor'],
                # Patterns (Raw Strings!)
                [r'\b\d+\s*(?:V|A|W|Ω|F|H)\b', r'\b(?:AC|DC)\b', r'\bkV\b']
            ],
            
            'mechanical': [
                # Keywords
                ['drehmoment', 'torque', 'drehzahl', 'rpm', 'lager', 'bearing',
                 'kupplung', 'coupling', 'getriebe', 'gearbox'],
                # Patterns
                [r'\b\d+\s*(?:rpm|min⁻¹|Nm|bar)\b', r'\bM\d+\s*(?:schrauben|bolts)']
            ],
            
            'hydraulic': [
                # Keywords  
                ['hydraulik', 'hydraulic', 'druck', 'pressure', 'pumpe', 'pump',
                 'ventil', 'valve', 'zylinder', 'cylinder'],
                # Patterns
                [r'\b\d+\s*(?:bar|psi|l/min|gpm)\b', r'\bhydraulik(?:öl|oil)']
            ],
            
            'pneumatic': [
                # Keywords
                ['pneumatik', 'pneumatic', 'druckluft', 'compressed air',
                 'kompressor', 'compressor'],
                # Patterns
                [r'\b\d+\s*bar\s*(?:luft|air)', r'\bdruckluft']
            ],
            
            'automation': [
                # Keywords
                ['sps', 'plc', 'hmi', 'scada', 'sensor', 'aktor', 'actuator',
                 'steuerung', 'control', 'regelung', 'regulation'],
                # Patterns
                [r'\b(?:I/O|E/A)\b', r'\bfieldbus\b', r'\bprofibus\b', r'\bprofinet\b']
            ],
            
            'safety': [
                # Keywords
                ['sicherheit', 'safety', 'schutz', 'protection', 'not-aus',
                 'emergency stop', 'lichtvorhang', 'light curtain'],
                # Patterns
                [r'\bSIL\s*[0-4]', r'\b(?:CE|ATEX)\b', r'\bkategorie\s*[1-4]']
            ]
        }
    
    @log_performance()
    def enhance_chunk_metadata(self, 
                              chunk: Union[TextChunk, str], 
                              source_document: str = "unknown",
                              chunk_index: int = 0) -> EnhancedTextChunk:
        """
        Erweitert Metadaten eines einzelnen Text-Chunks
        
        Args:
            chunk (Union[TextChunk, str]): Text-Chunk oder String
            source_document (str): Quell-Dokument
            chunk_index (int): Index im Dokument
            
        Returns:
            EnhancedTextChunk: Chunk mit erweiterten Metadaten
        """
        # Input normalisieren
        if isinstance(chunk, str):
            content = chunk
            base_metadata = None
        else:
            content = chunk.page_content
            base_metadata = getattr(chunk, 'metadata', None)
        
        if not content or len(content.strip()) == 0:
            raise ValidationError("Chunk-Content darf nicht leer sein", field_name="content")
        
        try:
            # 1. Basis-Analysen durchführen
            text_analysis = self.text_analyzer.analyze_text(content)
            classification_result = self.content_classifier.classify_content(content)
            
            # 2. Chunk-Eigenschaften berechnen
            chunk_properties = self._calculate_chunk_properties(content)
            
            # 3. Qualitätsbewertung
            quality_assessment = self._assess_quality(text_analysis, classification_result, content)
            
            # 4. Sicherheitsbewertung
            safety_assessment = self._assess_safety(classification_result, content)
            
            # 5. Expertise-Level bestimmen
            expertise_assessment = self._assess_expertise_level(classification_result, content)
            
            # 6. Technische Domäne ermitteln
            technical_domain = self._classify_technical_domain(content)
            
            # 7. Dokumenten-Abschnitt erkennen
            document_section = self._identify_document_section(content, base_metadata)
            
            # 8. Sprache ermitteln
            primary_language = self._detect_primary_language(text_analysis, content)
            
            # 9. Such-Boost berechnen
            search_boost = self._calculate_search_boost(quality_assessment, safety_assessment, classification_result)
            
            # 10. Erweiterte Metadaten zusammenstellen
            enhanced_metadata = EnhancedMetadata(
                # Basis-Informationen
                chunk_id=chunk_index,
                content_hash=self._generate_content_hash(content),
                processing_timestamp=datetime.now(timezone.utc).isoformat(),
                
                # Content-Klassifikation
                primary_category=classification_result.primary_category.value,
                secondary_categories=','.join([cat.value for cat in classification_result.secondary_categories]),
                content_flags=','.join([flag.value for flag in classification_result.content_flags]),
                
                # Text-Analyse
                base_content_type=text_analysis.content_type.value,
                keywords=','.join(text_analysis.keywords) if text_analysis.keywords else '',
                has_sequential_steps=text_analysis.has_sequential_steps,
                step_count=text_analysis.step_count,
                
                # Qualitäts- und Sicherheitsbewertung
                quality_level=quality_assessment.value,
                safety_level=safety_assessment.value,
                expertise_level=expertise_assessment.value,
                overall_confidence=(text_analysis.confidence_score + classification_result.total_confidence) / 2,
                
                # Chunk-Eigenschaften
                chunk_length=chunk_properties['length'],
                word_count=chunk_properties['word_count'],
                sentence_count=chunk_properties['sentence_count'],
                chunk_index=chunk_index,
                
                # Industrielle Kategorisierung
                technical_domain=technical_domain,
                document_section=document_section,
                language_primary=primary_language,
                
                # Kontext-Informationen
                source_document=source_document,
                related_chunks='',  # Wird später von Services gefüllt
                prerequisite_knowledge=self._determine_prerequisite_knowledge(classification_result, expertise_assessment),
                
                # Such-Optimierung
                search_boost=search_boost,
                filter_tags=self._generate_filter_tags(classification_result, technical_domain, safety_assessment)
            )
            
            # Enhanced Chunk erstellen
            enhanced_chunk = EnhancedTextChunk(
                content=content,
                metadata=enhanced_metadata
            )
            
            # Statistiken aktualisieren
            self._update_enhancement_stats(enhanced_metadata)
            
            self.logger.debug(
                f"Chunk-Metadaten erweitert: {quality_assessment.value} quality, {safety_assessment.value} safety",
                extra={
                    'extra_data': {
                        'chunk_index': chunk_index,
                        'primary_category': enhanced_metadata.primary_category,
                        'technical_domain': technical_domain,
                        'quality_level': quality_assessment.value,
                        'overall_confidence': enhanced_metadata.overall_confidence
                    }
                }
            )
            
            return enhanced_chunk
            
        except Exception as e:
            error_context = create_error_context(
                component="modules.processors.metadata_enhancer",
                operation="enhance_chunk_metadata",
                chunk_index=chunk_index,
                content_length=len(content)
            )
            
            raise DocumentProcessingError(
                message=f"Fehler bei Metadaten-Erweiterung: {str(e)}",
                processing_stage="metadata_enhancement",
                context=error_context,
                original_exception=e
            )
    
    @log_performance()
    def enhance_chunk_batch(self, 
                           chunks: List[Union[TextChunk, str]], 
                           source_document: str = "unknown") -> List[EnhancedTextChunk]:
        """
        Erweitert Metadaten für Batch von Text-Chunks
        
        Args:
            chunks (List[Union[TextChunk, str]]): Liste von Chunks
            source_document (str): Quell-Dokument
            
        Returns:
            List[EnhancedTextChunk]: Liste erweiterter Chunks
        """
        if not chunks:
            return []
        
        enhanced_chunks = []
        errors = []
        
        for i, chunk in enumerate(chunks):
            try:
                enhanced_chunk = self.enhance_chunk_metadata(
                    chunk=chunk,
                    source_document=source_document,
                    chunk_index=i
                )
                enhanced_chunks.append(enhanced_chunk)
                
            except Exception as e:
                self.logger.warning(f"Fehler bei Chunk {i}: {str(e)}")
                errors.append((i, str(e)))
                
                # Fallback-Chunk erstellen
                content = chunk if isinstance(chunk, str) else chunk.page_content
                fallback_metadata = EnhancedMetadata(
                    chunk_id=i,
                    content_hash=self._generate_content_hash(content),
                    processing_timestamp=datetime.now(timezone.utc).isoformat(),
                    primary_category=IndustrialCategory.DOCUMENTATION.value,
                    secondary_categories='',
                    content_flags='',
                    base_content_type=ContentType.GENERAL.value,
                    keywords='',
                    has_sequential_steps=False,
                    step_count=0,
                    quality_level=QualityLevel.POOR.value,
                    safety_level=SafetyLevel.NONE.value,
                    expertise_level=ExpertiseLevel.ANYONE.value,
                    overall_confidence=0.0,
                    chunk_length=len(content),
                    word_count=len(content.split()),
                    sentence_count=len([s for s in content.split('.') if s.strip()]),
                    chunk_index=i,
                    technical_domain='unknown',
                    document_section='unknown',
                    language_primary='unknown',
                    source_document=source_document,
                    related_chunks='',
                    prerequisite_knowledge='',
                    search_boost=0.5
                )
                
                enhanced_chunks.append(EnhancedTextChunk(
                    content=content,
                    metadata=fallback_metadata
                ))
        
        # Post-Processing: Related Chunks identifizieren
        self._identify_related_chunks(enhanced_chunks)
        
        self.logger.info(
            f"Batch-Metadaten-Erweiterung abgeschlossen: {len(chunks)} Chunks, {len(errors)} Fehler",
            extra={
                'extra_data': {
                    'total_chunks': len(chunks),
                    'successful_enhancements': len(chunks) - len(errors),
                    'errors': len(errors),
                    'source_document': source_document
                }
            }
        )
        
        return enhanced_chunks
    
    # =============================================================================
    # BEWERTUNGS-METHODEN
    # =============================================================================
    
    def _calculate_chunk_properties(self, content: str) -> Dict[str, int]:
        """Berechnet Basis-Eigenschaften des Chunks"""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        return {
            'length': len(content),
            'word_count': len(content.split()),
            'sentence_count': len(sentences)
        }
    
    def _assess_quality(self, 
                       text_analysis: TextAnalysisResult,
                       classification_result: ClassificationResult,
                       content: str) -> QualityLevel:
        """Bewertet Qualität des Chunks"""
        
        # Basis-Score aus Analyse-Confidence
        base_score = (text_analysis.confidence_score + classification_result.total_confidence) / 2
        
        # Qualitäts-Boosts
        quality_boosts = 0.0
        
        # Content-Vollständigkeit
        if len(content) > 200:  # Ausreichend Inhalt
            quality_boosts += 0.1
        
        if text_analysis.keywords and len(text_analysis.keywords) >= 3:  # Gute Keywords
            quality_boosts += 0.1
        
        if text_analysis.has_sequential_steps:  # Strukturierter Inhalt
            quality_boosts += 0.1
        
        if classification_result.content_flags:  # Spezielle Merkmale erkannt
            quality_boosts += 0.05
        
        # Gesamt-Score
        total_score = min(1.0, base_score + quality_boosts)
        
        # Qualitätslevel bestimmen
        if total_score >= 0.8:
            return QualityLevel.EXCELLENT
        elif total_score >= 0.6:
            return QualityLevel.GOOD
        elif total_score >= 0.4:
            return QualityLevel.MODERATE
        else:
            return QualityLevel.POOR
    
    def _assess_safety(self, 
                      classification_result: ClassificationResult,
                      content: str) -> SafetyLevel:
        """Bewertet Sicherheitsrelevanz des Chunks"""
        
        content_lower = content.lower()
        
        # Kritische Sicherheitshinweise
        critical_patterns = [
            r'\b(?:lebensgefahr|danger|hochspannung|high\s+voltage)\b',
            r'\b\d+\s*kV\b',  # Hochspannung
            r'⚠️|⚡|☢️|☣️',   # Warnsymbole
        ]
        
        for pattern in critical_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return SafetyLevel.CRITICAL
        
        # Hohe Sicherheitsrelevanz
        if ContentFlag.HAS_WARNINGS in classification_result.content_flags:
            high_keywords = ['warnung', 'warning', 'achtung', 'caution', 'vorsicht']
            if any(keyword in content_lower for keyword in high_keywords):
                return SafetyLevel.HIGH
        
        # Mittlere Sicherheitsrelevanz
        if classification_result.primary_category == IndustrialCategory.SAFETY:
            return SafetyLevel.MEDIUM
        
        # Niedrige Sicherheitsrelevanz
        safety_related = ['sicherheit', 'safety', 'schutz', 'protection']
        if any(keyword in content_lower for keyword in safety_related):
            return SafetyLevel.LOW
        
        return SafetyLevel.NONE
    
    def _assess_expertise_level(self, 
                               classification_result: ClassificationResult,
                               content: str) -> ExpertiseLevel:
        """Bestimmt erforderliches Expertise-Level"""
        
        content_lower = content.lower()
        
        # Experten-Level erforderlich
        expert_indicators = [
            r'\b(?:kalibrierung|calibration|justierung|adjustment)\b',
            r'\bservice\s*(?:modus|mode)\b',
            r'\b(?:reparatur|repair|instandsetzung)\b'
        ]
        
        for pattern in expert_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                return ExpertiseLevel.EXPERT
        
        # Fachkraft erforderlich
        if ContentFlag.REQUIRES_EXPERTISE in classification_result.content_flags:
            return ExpertiseLevel.SKILLED
        
        if classification_result.primary_category in [
            IndustrialCategory.MAINTENANCE, 
            IndustrialCategory.TROUBLESHOOTING,
            IndustrialCategory.INSTALLATION
        ]:
            return ExpertiseLevel.SKILLED
        
        # Eingewiesene Person
        if classification_result.primary_category == IndustrialCategory.OPERATION:
            return ExpertiseLevel.TRAINED
        
        # Grundkenntnisse
        technical_terms = ['parameter', 'einstellung', 'konfiguration', 'setting']
        if any(term in content_lower for term in technical_terms):
            return ExpertiseLevel.BASIC
        
        # Für jeden verständlich
        return ExpertiseLevel.ANYONE
    
    def _classify_technical_domain(self, content: str) -> str:
        """Klassifiziert technische Domäne des Contents"""
        
        content_lower = content.lower()
        domain_scores = defaultdict(float)
        
        for domain, (keywords, patterns) in self.domain_patterns.items():
            score = 0.0
            
            # Keyword-Matches
            for keyword in keywords:
                if keyword in content_lower:
                    score += 1.0
            
            # Pattern-Matches
            for pattern in patterns:
                try:
                    matches = len(re.findall(pattern, content, re.IGNORECASE))
                    score += matches * 1.5  # Pattern schwerer gewichtet
                except re.error:
                    continue
            
            domain_scores[domain] = score
        
        # Beste Domäne bestimmen
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] >= 1.0:  # Mindest-Score
                return best_domain
        
        return 'general'
    
    def _identify_document_section(self, content: str, base_metadata: Any = None) -> str:
        """Identifiziert Dokumenten-Abschnitt"""
        
        content_lower = content.lower()
        
        # Überschriften-Pattern
        section_patterns = {
            'introduction': r'\b(?:einleitung|introduction|übersicht|overview)\b',
            'installation': r'\b(?:installation|montage|aufbau|setup)\b',
            'operation': r'\b(?:bedienung|operation|betrieb|anwendung)\b',
            'maintenance': r'\b(?:wartung|maintenance|instandhaltung|service)\b',
            'troubleshooting': r'\b(?:troubleshooting|fehlerbehebung|störung|problem)\b',
            'specifications': r'\b(?:technische\s+daten|technical\s+data|spezifikation)\b',
            'safety': r'\b(?:sicherheit|safety|sicherheitshinweise|safety\s+instructions)\b',
            'appendix': r'\b(?:anhang|appendix|anlage)\b'
        }
        
        for section, pattern in section_patterns.items():
            if re.search(pattern, content_lower):
                return section
        
        # Fallback aus base_metadata
        if base_metadata and hasattr(base_metadata, 'to_dict'):
            metadata_dict = base_metadata.to_dict()
            if 'document_section' in metadata_dict:
                return metadata_dict['document_section']
        
        return 'content'
    
    def _detect_primary_language(self, text_analysis: TextAnalysisResult, content: str) -> str:
        """Erkennt Primärsprache des Contents"""
        
        if text_analysis.language_indicators:
            dominant = text_analysis.language_indicators.get('dominant_language', 'unknown')
            if dominant in ['german', 'english']:
                return dominant
        
        # Einfache Fallback-Erkennung
        german_indicators = content.lower().count('der') + content.lower().count('und') + content.lower().count('ist')
        english_indicators = content.lower().count('the') + content.lower().count('and') + content.lower().count('is')
        
        if german_indicators > english_indicators:
            return 'german'
        elif english_indicators > german_indicators:
            return 'english'
        else:
            return 'mixed'
    
    def _calculate_search_boost(self, 
                               quality: QualityLevel,
                               safety: SafetyLevel,
                               classification: ClassificationResult) -> float:
        """Berechnet Such-Boost-Faktor"""
        
        # Basis-Boost basierend auf Qualität
        quality_boost = {
            QualityLevel.EXCELLENT: 1.5,
            QualityLevel.GOOD: 1.2,
            QualityLevel.MODERATE: 1.0,
            QualityLevel.POOR: 0.7
        }
        
        # Sicherheits-Boost
        safety_boost = {
            SafetyLevel.CRITICAL: 1.8,
            SafetyLevel.HIGH: 1.4,
            SafetyLevel.MEDIUM: 1.1,
            SafetyLevel.LOW: 1.0,
            SafetyLevel.NONE: 1.0
        }
        
        # Content-Boost
        content_boost = 1.0
        if ContentFlag.HAS_PROCEDURES in classification.content_flags:
            content_boost += 0.2
        if ContentFlag.HAS_SPECIFICATIONS in classification.content_flags:
            content_boost += 0.1
        
        # Gesamt-Boost (begrenzt auf 0.1-2.0)
        total_boost = quality_boost[quality] * safety_boost[safety] * content_boost
        return max(0.1, min(2.0, total_boost))
    
    def _determine_prerequisite_knowledge(self, 
                                        classification: ClassificationResult,
                                        expertise: ExpertiseLevel) -> str:
        """Bestimmt erforderliches Vorwissen"""
        
        prerequisites = []
        
        # Expertise-basierte Voraussetzungen
        expertise_prereqs = {
            ExpertiseLevel.EXPERT: ['Fachausbildung', 'Zertifizierung', 'Berufserfahrung'],
            ExpertiseLevel.SKILLED: ['Einweisung', 'Grundkenntnisse'],
            ExpertiseLevel.TRAINED: ['Schulung'],
            ExpertiseLevel.BASIC: ['Grundverständnis'],
            ExpertiseLevel.ANYONE: []
        }
        
        prerequisites.extend(expertise_prereqs[expertise])
        
        # Kategorie-spezifische Voraussetzungen
        if classification.primary_category == IndustrialCategory.SAFETY:
            prerequisites.append('Sicherheitsunterweisung')
        elif classification.primary_category == IndustrialCategory.MAINTENANCE:
            prerequisites.append('Wartungsschulung')
        elif classification.primary_category == IndustrialCategory.TROUBLESHOOTING:
            prerequisites.append('Systemkenntnisse')
        
        return ', '.join(prerequisites) if prerequisites else 'Keine'
    
    def _generate_filter_tags(self, 
                            classification: ClassificationResult,
                            technical_domain: str,
                            safety_level: SafetyLevel) -> str:
        """Generiert zusätzliche Filter-Tags"""
        
        tags = []
        
        # Domäne als Tag
        if technical_domain != 'general':
            tags.append(f'domain:{technical_domain}')
        
        # Sicherheits-Tags
        if safety_level != SafetyLevel.NONE:
            tags.append(f'safety:{safety_level.value}')
        
        # Content-Flag-basierte Tags
        for flag in classification.content_flags:
            if flag in [ContentFlag.HAS_WARNINGS, ContentFlag.HAS_PROCEDURES, ContentFlag.HAS_SPECIFICATIONS]:
                tags.append(flag.value.replace('has_', ''))
        
        # Kategorie-Tags
        tags.append(f'category:{classification.primary_category.value}')
        
        return ', '.join(tags)
    
    def _generate_content_hash(self, content: str) -> str:
        """Generiert SHA-256 Hash für Content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]  # Erste 16 Zeichen
    
    def _identify_related_chunks(self, enhanced_chunks: List[EnhancedTextChunk]) -> None:
        """Identifiziert verwandte Chunks basierend auf Ähnlichkeit"""
        
        for i, chunk_a in enumerate(enhanced_chunks):
            related_ids = []
            
            for j, chunk_b in enumerate(enhanced_chunks):
                if i == j:
                    continue
                
                # Ähnlichkeitskriterien
                similarity_score = 0.0
                
                # Gleiche Kategorie
                if chunk_a.metadata.primary_category == chunk_b.metadata.primary_category:
                    similarity_score += 0.3
                
                # Ähnliche Keywords
                keywords_a = set(chunk_a.metadata.keywords.split(', ')) if chunk_a.metadata.keywords else set()
                keywords_b = set(chunk_b.metadata.keywords.split(', ')) if chunk_b.metadata.keywords else set()
                
                if keywords_a and keywords_b:
                    common_keywords = keywords_a.intersection(keywords_b)
                    keyword_similarity = len(common_keywords) / max(len(keywords_a), len(keywords_b))
                    similarity_score += keyword_similarity * 0.4
                
                # Gleiche technische Domäne
                if chunk_a.metadata.technical_domain == chunk_b.metadata.technical_domain:
                    similarity_score += 0.2
                
                # Nachbarschaft im Dokument
                if abs(chunk_a.metadata.chunk_index - chunk_b.metadata.chunk_index) <= 2:
                    similarity_score += 0.1
                
                # Ähnlichkeits-Schwellwert
                if similarity_score >= 0.5:
                    related_ids.append(str(j))
            
            # Related Chunks aktualisieren
            chunk_a.metadata.related_chunks = ', '.join(related_ids[:5])  # Max 5 verwandte Chunks
    
    def _update_enhancement_stats(self, metadata: EnhancedMetadata) -> None:
        """Aktualisiert interne Enhancement-Statistiken"""
        
        self._enhancement_stats['chunks_enhanced'] += 1
        self._enhancement_stats['quality_distribution'][metadata.quality_level] += 1
        self._enhancement_stats['safety_distribution'][metadata.safety_level] += 1
        self._enhancement_stats['domain_distribution'][metadata.technical_domain] += 1
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """
        Holt Enhancement-Statistiken
        
        Returns:
            Dict[str, Any]: Detaillierte Statistiken
        """
        stats = dict(self._enhancement_stats)
        
        if stats['chunks_enhanced'] > 0:
            total = stats['chunks_enhanced']
            
            # Prozentuale Verteilungen
            stats['quality_percentages'] = {
                level: (count / total) * 100
                for level, count in stats['quality_distribution'].items()
            }
            
            stats['safety_percentages'] = {
                level: (count / total) * 100
                for level, count in stats['safety_distribution'].items()
            }
            
            stats['domain_percentages'] = {
                domain: (count / total) * 100
                for domain, count in stats['domain_distribution'].items()
            }
        
        return stats


# =============================================================================
# BATCH-PROCESSOR FÜR PERFORMANCE
# =============================================================================

class BatchMetadataEnhancer:
    """
    Batch-Processor für effiziente Verarbeitung großer Dokumentensammlungen
    """
    
    def __init__(self, config: RAGConfig = None):
        self.enhancer = MetadataEnhancer(config)
        self.logger = get_logger("batch_metadata_enhancer", "modules.processors")
    
    @log_performance()
    def process_document_batch(self, 
                              documents: List[Dict[str, Any]], 
                              batch_size: int = 50) -> Dict[str, List[EnhancedTextChunk]]:
        """
        Verarbeitet Batch von Dokumenten mit Chunks
        
        Args:
            documents (List[Dict]): Dokumente mit 'name' und 'chunks' Keys
            batch_size (int): Batch-Größe für Verarbeitung
            
        Returns:
            Dict[str, List[EnhancedTextChunk]]: Ergebnisse pro Dokument
        """
        results = {}
        total_chunks = sum(len(doc.get('chunks', [])) for doc in documents)
        processed_chunks = 0
        
        for document in documents:
            doc_name = document.get('name', 'unknown')
            chunks = document.get('chunks', [])
            
            if not chunks:
                results[doc_name] = []
                continue
            
            try:
                # Chunks in Batches verarbeiten
                enhanced_chunks = []
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    
                    enhanced_batch = self.enhancer.enhance_chunk_batch(
                        chunks=batch,
                        source_document=doc_name
                    )
                    
                    enhanced_chunks.extend(enhanced_batch)
                    processed_chunks += len(batch)
                    
                    # Progress-Logging
                    progress = (processed_chunks / total_chunks) * 100
                    if processed_chunks % 100 == 0:  # Alle 100 Chunks
                        self.logger.info(f"Batch-Verarbeitung: {progress:.1f}% ({processed_chunks}/{total_chunks})")
                
                results[doc_name] = enhanced_chunks
                
            except Exception as e:
                self.logger.error(f"Fehler bei Dokument {doc_name}: {str(e)}")
                results[doc_name] = []
        
        self.logger.info(
            f"Batch-Verarbeitung abgeschlossen: {len(documents)} Dokumente, {processed_chunks} Chunks"
        )
        
        return results


# =============================================================================
# ENHANCER-FACTORY
# =============================================================================

class EnhancerFactory:
    """Factory für verschiedene Enhancer-Konfigurationen"""
    
    @staticmethod
    def create_industrial_enhancer(config: RAGConfig = None) -> MetadataEnhancer:
        """
        Erstellt Enhancer optimiert für industrielle Dokumentation
        
        Args:
            config (RAGConfig): Konfiguration
            
        Returns:
            MetadataEnhancer: Industriell optimierter Enhancer
        """
        enhancer = MetadataEnhancer(config)
        
        # Zusätzliche industrielle Domänen
        enhancer.domain_patterns.update({
            'process_control': [
                ['prozess', 'process', 'regelung', 'control', 'pid', 'setpoint'],
                [r'\bPID\b', r'\bsetpoint\b', r'\bcontroller\b']
            ],
            'quality_assurance': [
                ['qualität', 'quality', 'prüfung', 'inspection', 'test', 'messung'],
                [r'\bQS\b', r'\bISO\s+\d+', r'\bmessprotokoll\b']
            ]
        })
        
        return enhancer
    
    @staticmethod
    def create_lightweight_enhancer(config: RAGConfig = None) -> MetadataEnhancer:
        """
        Erstellt Performance-optimierten Enhancer mit reduzierten Features
        
        Args:
            config (RAGConfig): Konfiguration
            
        Returns:
            MetadataEnhancer: Lightweight Enhancer
        """
        # Hier könnte eine vereinfachte Version implementiert werden
        # Für jetzt verwenden wir Standard-Enhancer
        return MetadataEnhancer(config)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums und Datenstrukturen
    'QualityLevel', 'SafetyLevel', 'ExpertiseLevel',
    'EnhancedMetadata', 'EnhancedTextChunk',
    
    # Hauptklassen
    'MetadataEnhancer', 'BatchMetadataEnhancer',
    
    # Factory
    'EnhancerFactory'
]


if __name__ == "__main__":
    # Testing und Demonstration
    print("Metadata Enhancer - Erweiterte Metadaten-Anreicherung")
    print("====================================================")
    
    # Enhancer erstellen
    enhancer = MetadataEnhancer()
    
    # Test-Texte verschiedener Kategorien
    test_chunks = [
        # Sicherheitskritischer Text
        """
        WARNUNG: Lebensgefahr durch Hochspannung!
        Arbeiten an der 24kV Anlage nur im spannungsfreien Zustand.
        Vor Beginn der Arbeiten fünf Sicherheitsregeln befolgen:
        1. Freischalten
        2. Gegen Wiedereinschalten sichern
        3. Spannungsfreiheit feststellen
        4. Erden und kurzschließen
        5. Benachbarte unter Spannung stehende Teile abdecken
        """,
        
        # Technische Spezifikation
        """
        Technische Daten Frequenzumrichter FU-2000:
        - Eingangsspannung: 3x400V AC ±10%
        - Ausgangsspannung: 0-400V AC variabel
        - Nennstrom: 25A
        - Leistung: 15kW
        - Schutzart: IP65
        - Umgebungstemperatur: -10°C bis +50°C
        Siehe Tabelle 12 für weitere Parameter.
        """,
        
        # Wartungsanweisung
        """
        Wartungsintervall: Alle 3 Monate
        1. Sichtprüfung aller Komponenten
        2. Reinigung der Lüftungsschlitze
        3. Kontrolle der Verschraubungen (10 Nm)
        4. Funktionstest der Notabschaltung
        Wartung nur durch Elektrofachkraft durchführen.
        Verschleißteile nach 5000 Betriebsstunden ersetzen.
        """
    ]
    
    # Einzelne Chunks erweitern
    print("--- Einzelne Chunk-Erweiterung ---")
    for i, chunk_text in enumerate(test_chunks):
        print(f"\nChunk {i+1}:")
        print(f"Text: {chunk_text.strip()[:80]}...")
        
        enhanced_chunk = enhancer.enhance_chunk_metadata(
            chunk=chunk_text,
            source_document=f"test_doc_{i+1}",
            chunk_index=i
        )
        
        metadata = enhanced_chunk.metadata
        print(f"Kategorie: {metadata.primary_category}")
        print(f"Qualität: {metadata.quality_level}")
        print(f"Sicherheit: {metadata.safety_level}")
        print(f"Expertise: {metadata.expertise_level}")
        print(f"Domäne: {metadata.technical_domain}")
        print(f"Confidence: {metadata.overall_confidence:.2f}")
        print(f"Search-Boost: {metadata.search_boost:.2f}")
        
        if metadata.content_flags:
            print(f"Flags: {metadata.content_flags}")
    
    # Batch-Verarbeitung testen
    print(f"\n--- Batch-Verarbeitung ---")
    enhanced_batch = enhancer.enhance_chunk_batch(
        chunks=test_chunks,
        source_document="industrial_manual_v1.0"
    )
    
    print(f"Batch verarbeitet: {len(enhanced_batch)} Chunks")
    
    # Qualitäts-Verteilung
    quality_dist = {}
    for chunk in enhanced_batch:
        quality = chunk.metadata.quality_level
        quality_dist[quality] = quality_dist.get(quality, 0) + 1
    
    print(f"Qualitäts-Verteilung: {quality_dist}")
    
    # Related Chunks anzeigen
    for i, chunk in enumerate(enhanced_batch):
        if chunk.metadata.related_chunks:
            print(f"Chunk {i} verwandt mit: {chunk.metadata.related_chunks}")
    
    # Chroma-Kompatibilität testen
    print(f"\n--- Chroma-Metadaten Test ---")
    sample_metadata = enhanced_batch[0].metadata.to_chroma_metadata()
    print(f"Chroma-Metadaten Keys: {list(sample_metadata.keys())}")
    print(f"Metadaten-Typen: {[(k, type(v).__name__) for k, v in sample_metadata.items()]}")
    
    # Statistiken
    stats = enhancer.get_enhancement_statistics()
    print(f"\n--- Enhancement-Statistiken ---")
    print(f"Verarbeitete Chunks: {stats['chunks_enhanced']}")
    if 'quality_percentages' in stats:
        print("Qualitäts-Verteilung:")
        for quality, percent in stats['quality_percentages'].items():
            print(f"  {quality}: {percent:.1f}%")
    
    print("\n✅ Metadata-Enhancer erfolgreich getestet")