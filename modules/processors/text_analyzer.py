#!/usr/bin/env python3
"""
Text Analyzer für RAG Chatbot Industrial

Extrahiert aus dem monolithischen RAG Chatbot Code und erweitert für die neue
service-orientierte Architektur. Intelligente Textanalyse und Content-Klassifizierung.

Extrahierte Features vom Original:
- Content-Type-Erkennung (step_by_step_guide, demo_instructions, etc.)
- Keyword-Extraktion mit technischen Begriffen
- Sequential-Step-Detection für Anleitungen
- Regex-Pattern-Erkennung (ALLE RAW STRINGS GEFIXT)

Neue Features:
- Erweiterte Metadaten-Anreicherung
- Konfigurierbare Analyse-Strategien
- Performance-optimierte Pattern-Matching
- Strukturierte Analyse-Ergebnisse

Autor: KI-Consultant für industrielle Automatisierung  
Version: 4.0.0 - Extrahiert aus monolithischem Code
"""

import re
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Core-Komponenten
from core import (
    get_logger, RAGConfig, get_current_config,
    DocumentProcessingError, ValidationError,
    create_error_context, log_performance
)


# =============================================================================
# CONTENT-TYPE KLASSIFIKATION (AUS ORIGINAL EXTRAHIERT)
# =============================================================================

class ContentType(str, Enum):
    """
    Content-Typen aus dem ursprünglichen TextAnalyzer extrahiert
    
    Diese Enum repliziert exakt die Content-Types aus detect_content_type()
    """
    STEP_BY_STEP_GUIDE = "step_by_step_guide"      # Nummerierte Anleitungen
    DEMO_INSTRUCTIONS = "demo_instructions"        # Demo/Start Anleitungen  
    INSTALLATION_GUIDE = "installation_guide"      # Installation/Setup
    TECHNICAL_SPECIFICATIONS = "technical_specifications"  # Tech. Daten
    TROUBLESHOOTING = "troubleshooting"            # Fehlerbehebung
    GENERAL = "general"                            # Allgemeine Inhalte


@dataclass
class TextAnalysisResult:
    """
    Strukturiertes Ergebnis der Textanalyse
    
    Attributes:
        content_type (ContentType): Erkannter Content-Typ
        keywords (List[str]): Extrahierte Keywords
        has_sequential_steps (bool): Enthält nummerierte Schritte
        step_count (int): Anzahl erkannter Schritte
        technical_terms (List[str]): Technische Fachbegriffe
        language_indicators (Dict[str, int]): Spracherkennungs-Score
        confidence_score (float): Vertrauen in die Klassifikation (0.0-1.0)
        metadata (Dict[str, Any]): Zusätzliche Analyse-Metadaten
    """
    content_type: ContentType
    keywords: List[str] = field(default_factory=list)
    has_sequential_steps: bool = False
    step_count: int = 0
    technical_terms: List[str] = field(default_factory=list)
    language_indicators: Dict[str, int] = field(default_factory=dict)
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# TEXT ANALYZER HAUPTKLASSE (EXTRAHIERT UND ERWEITERT)
# =============================================================================

class TextAnalyzer:
    """
    Intelligenter Textanalyser extrahiert aus dem monolithischen RAG Chatbot
    
    Diese Klasse repliziert und erweitert die ursprüngliche TextAnalyzer-Klasse
    mit allen ihren Methoden:
    - detect_content_type() -> EXAKTE REPLIKATION
    - extract_keywords() -> EXAKTE REPLIKATION  
    - has_sequential_steps() -> EXAKTE REPLIKATION
    
    Alle Regex-Pattern wurden als Raw Strings gefixt (ursprüngliches Problem).
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialisiert Text-Analyzer
        
        Args:
            config (RAGConfig): Konfiguration für Analyse-Verhalten
        """
        self.config = config or get_current_config()
        self.logger = get_logger("text_analyzer", "modules.processors")
        
        # ORIGINAL PATTERN - ALLE ALS RAW STRINGS GEFIXT
        self._init_analysis_patterns()
        
        # Performance-Tracking
        self._analysis_stats = {
            'texts_analyzed': 0,
            'content_type_distribution': {},
            'avg_processing_time_ms': 0.0
        }
    
    def _init_analysis_patterns(self) -> None:
        """Initialisiert alle Analyse-Pattern (ORIGINAL-LOGIC mit Raw String Fix)"""
        
        # GEFIXT: Alle Regex-Pattern als Raw Strings (war das Problem im Original!)
        self.step_patterns = [
            r'\d+\.\s+\w+',           # "1. Schritt" Pattern
            r'\d+\)\s+\w+',           # "1) Schritt" Pattern  
            r'Schritt\s+\d+',         # "Schritt 1" Pattern
            r'Step\s+\d+',            # "Step 1" Pattern
        ]
        
        # ORIGINAL: Demo/Start Keywords (case-insensitive)
        self.demo_keywords = [
            'demo', 'start', 'starten', 'anleitung', 'begin', 'beginnen',
            'launch', 'öffnen', 'aktivieren', 'initialisieren'
        ]
        
        # ORIGINAL: Step-Indicators für sequentielle Anleitungen
        self.step_indicators = [
            'schritt', 'klicken', 'öffnen', 'drücken', 'wählen', 'auswählen',
            'step', 'click', 'open', 'press', 'select', 'choose'
        ]
        
        # ORIGINAL: Installation Keywords
        self.installation_keywords = ['installation', 'setup', 'install', 'configure']
        
        # ORIGINAL: Technical Keywords
        self.technical_keywords = ['spannung', 'leistung', 'technische daten', 'voltage', 'power', 'specifications']
        
        # ORIGINAL: Troubleshooting Keywords  
        self.troubleshooting_keywords = ['fehler', 'problem', 'troubleshooting', 'error', 'issue', 'fix']
        
        # ERWEITERT: Technische Begriffe Pattern (Raw Strings!)
        self.tech_patterns = [
            r'\b[A-Z]{2,}\b',                    # Abkürzungen (CPU, RAM, etc.)
            r'\b\w+(?:spannung|leistung|strom)\b',  # Technische Parameter
            r'\b(?:demo|start|installation|setup)\b',  # Aktionen
            r'\d+(?:\.\d+)?\s*(?:V|A|W|Hz|MHz|GHz)\b',  # Technische Einheiten
            r'\b(?:IP|TCP|UDP|HTTP|FTP)\b',      # Netzwerk-Protokolle
        ]
    
    @log_performance()
    def analyze_text(self, text: str) -> TextAnalysisResult:
        """
        Führt komplette Textanalyse durch (HAUPTMETHODE)
        
        Args:
            text (str): Zu analysierender Text
            
        Returns:
            TextAnalysisResult: Strukturiertes Analyse-Ergebnis
        """
        if not text or len(text.strip()) == 0:
            raise ValidationError("Text für Analyse darf nicht leer sein", field_name="text")
        
        try:
            # ORIGINAL METHODS - EXAKT REPLIZIERT
            content_type = self.detect_content_type(text)
            keywords = self.extract_keywords(text) 
            has_steps = self.has_sequential_steps(text)
            
            # Schritt-Anzahl ermitteln
            step_count = self._count_steps(text)
            
            # Technische Begriffe extrahieren (erweitert)
            technical_terms = self._extract_technical_terms(text)
            
            # Sprach-Indikatoren analysieren (erweitert)
            language_indicators = self._analyze_language_indicators(text)
            
            # Konfidenz-Score berechnen
            confidence_score = self._calculate_confidence(
                content_type, keywords, has_steps, step_count, technical_terms
            )
            
            # Ergebnis zusammenstellen
            result = TextAnalysisResult(
                content_type=content_type,
                keywords=keywords,
                has_sequential_steps=has_steps,
                step_count=step_count,
                technical_terms=technical_terms,
                language_indicators=language_indicators,
                confidence_score=confidence_score,
                metadata={
                    'text_length': len(text),
                    'word_count': len(text.split()),
                    'sentence_count': len([s for s in text.split('.') if s.strip()])
                }
            )
            
            # Statistiken aktualisieren
            self._update_analysis_stats(result)
            
            self.logger.debug(
                f"Text analysiert: {content_type.value} (Konfidenz: {confidence_score:.2f})",
                extra={
                    'extra_data': {
                        'content_type': content_type.value,
                        'keyword_count': len(keywords),
                        'has_steps': has_steps,
                        'step_count': step_count,
                        'confidence': confidence_score
                    }
                }
            )
            
            return result
            
        except Exception as e:
            error_context = create_error_context(
                component="modules.processors.text_analyzer",
                operation="analyze_text",
                text_length=len(text)
            )
            
            raise DocumentProcessingError(
                message=f"Fehler bei Textanalyse: {str(e)}",
                processing_stage="text_analysis",
                context=error_context,
                original_exception=e
            )
    
    def detect_content_type(self, text: str) -> ContentType:
        """
        Erkennt Content-Typ (EXAKTE REPLIKATION DER ORIGINAL-METHODE)
        
        Diese Methode repliziert 1:1 die ursprüngliche detect_content_type()
        Logik aus der TextAnalyzer-Klasse im monolithischen Code.
        
        Args:
            text (str): Zu analysierender Text
            
        Returns:
            ContentType: Erkannter Content-Typ
        """
        text_lower = text.lower()
        
        # ORIGINAL LOGIC: Schritt-für-Schritt Anleitungen - GEFIXT: Raw String
        if re.search(r'\d+\.\s+\w+', text) and len(re.findall(r'\d+\.', text)) >= 2:
            return ContentType.STEP_BY_STEP_GUIDE
        
        # ORIGINAL LOGIC: Demo/Start Anleitungen
        if any(keyword in text_lower for keyword in self.demo_keywords):
            if any(step in text_lower for step in self.step_indicators):
                return ContentType.DEMO_INSTRUCTIONS
        
        # ORIGINAL LOGIC: Installation/Setup
        if any(keyword in text_lower for keyword in self.installation_keywords):
            return ContentType.INSTALLATION_GUIDE
        
        # ORIGINAL LOGIC: Technische Spezifikationen
        if any(spec in text_lower for spec in self.technical_keywords):
            return ContentType.TECHNICAL_SPECIFICATIONS
        
        # ORIGINAL LOGIC: Fehlerbehebung
        if any(trouble in text_lower for trouble in self.troubleshooting_keywords):
            return ContentType.TROUBLESHOOTING
        
        # ORIGINAL LOGIC: Fallback
        return ContentType.GENERAL
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extrahiert Keywords (EXAKTE REPLIKATION DER ORIGINAL-METHODE)
        
        Diese Methode repliziert 1:1 die ursprüngliche extract_keywords()
        Logik mit allen Original-Pattern.
        
        Args:
            text (str): Zu analysierender Text
            
        Returns:
            List[str]: Liste der extrahierten Keywords
        """
        keywords = []
        
        # ORIGINAL LOGIC: Technische Begriffe - GEFIXT: Raw Strings
        for pattern in self.tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend(matches)
        
        # ORIGINAL LOGIC: Zahlen und Schritte - GEFIXT: Raw String
        step_numbers = re.findall(r'\d+\.', text)
        if step_numbers:
            keywords.append('numbered_steps')
        
        # ORIGINAL LOGIC: Duplikate entfernen
        return list(set(keywords))
    
    def has_sequential_steps(self, text: str) -> bool:
        """
        Prüft auf sequentielle Schritte (EXAKTE REPLIKATION DER ORIGINAL-METHODE)
        
        Diese Methode repliziert 1:1 die ursprüngliche has_sequential_steps()
        Logik aus der TextAnalyzer-Klasse.
        
        Args:
            text (str): Zu analysierender Text
            
        Returns:
            bool: True wenn sequentielle Schritte vorhanden
        """
        # ORIGINAL LOGIC - GEFIXT: Raw String
        steps = re.findall(r'(\d+)\.\s+', text)
        if len(steps) < 2:
            return False
        
        try:
            step_numbers = [int(step) for step in steps]
            return step_numbers == list(range(min(step_numbers), max(step_numbers) + 1))
        except:
            return False
    
    # =============================================================================
    # ERWEITERTE ANALYSE-METHODEN (NEU)
    # =============================================================================
    
    def _count_steps(self, text: str) -> int:
        """Zählt Anzahl der erkannten Schritte"""
        step_count = 0
        
        for pattern in self.step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            step_count = max(step_count, len(matches))
        
        return step_count
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extrahiert technische Fachbegriffe"""
        technical_terms = []
        
        # Erweiterte technische Pattern
        extended_tech_patterns = [
            r'\b(?:API|REST|JSON|XML|SQL|HTTP|HTTPS|FTP|SSH)\b',  # Protokolle
            r'\b(?:CPU|GPU|RAM|SSD|HDD|USB|HDMI|VGA)\b',          # Hardware
            r'\b(?:Windows|Linux|macOS|Android|iOS)\b',           # Systeme
            r'\b\d+(?:\.\d+)?\s*(?:GB|MB|TB|GHz|MHz|V|A|W)\b',   # Einheiten
        ]
        
        for pattern in extended_tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technical_terms.extend(matches)
        
        return list(set(technical_terms))
    
    def _analyze_language_indicators(self, text: str) -> Dict[str, int]:
        """Analysiert Sprach-Indikatoren"""
        sample = text[:1000].lower()
        
        # Deutsche Indikatoren
        german_indicators = [
            'der', 'die', 'das', 'und', 'oder', 'mit', 'von', 'zu', 'für',
            'ist', 'sind', 'wird', 'werden', 'haben', 'sein', 'auf', 'in'
        ]
        
        # Englische Indikatoren
        english_indicators = [
            'the', 'and', 'or', 'with', 'from', 'to', 'for', 'of', 'in',
            'is', 'are', 'will', 'have', 'be', 'on', 'at', 'by'
        ]
        
        german_count = sum(1 for word in german_indicators if word in sample)
        english_count = sum(1 for word in english_indicators if word in sample)
        
        return {
            'german': german_count,
            'english': english_count,
            'dominant_language': 'german' if german_count > english_count else 'english'
        }
    
    def _calculate_confidence(self, 
                            content_type: ContentType,
                            keywords: List[str], 
                            has_steps: bool,
                            step_count: int,
                            technical_terms: List[str]) -> float:
        """Berechnet Konfidenz-Score für Klassifikation"""
        
        base_confidence = 0.5
        
        # Content-Type spezifische Boosts
        if content_type == ContentType.STEP_BY_STEP_GUIDE:
            if has_steps and step_count >= 3:
                base_confidence += 0.4
            elif has_steps:
                base_confidence += 0.2
                
        elif content_type == ContentType.TECHNICAL_SPECIFICATIONS:
            if len(technical_terms) >= 3:
                base_confidence += 0.3
            elif technical_terms:
                base_confidence += 0.1
                
        elif content_type == ContentType.DEMO_INSTRUCTIONS:
            if keywords and any('demo' in k.lower() for k in keywords):
                base_confidence += 0.3
        
        # Keyword-basierte Boosts
        if len(keywords) >= 5:
            base_confidence += 0.1
        elif len(keywords) >= 3:
            base_confidence += 0.05
        
        # Normalisierung auf 0.0-1.0
        return min(1.0, base_confidence)
    
    def _update_analysis_stats(self, result: TextAnalysisResult) -> None:
        """Aktualisiert interne Analyse-Statistiken"""
        self._analysis_stats['texts_analyzed'] += 1
        
        content_type = result.content_type.value
        self._analysis_stats['content_type_distribution'][content_type] = \
            self._analysis_stats['content_type_distribution'].get(content_type, 0) + 1
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Holt Analyse-Statistiken
        
        Returns:
            Dict[str, Any]: Detaillierte Statistiken
        """
        stats = self._analysis_stats.copy()
        
        if stats['texts_analyzed'] > 0:
            # Prozentuale Verteilung berechnen
            total = stats['texts_analyzed']
            stats['content_type_percentages'] = {
                content_type: (count / total) * 100
                for content_type, count in stats['content_type_distribution'].items()
            }
        
        return stats


# =============================================================================
# BATCH-ANALYZER FÜR PERFORMANCE
# =============================================================================

class BatchTextAnalyzer:
    """
    Batch-Analyzer für effiziente Verarbeitung mehrerer Texte
    """
    
    def __init__(self, config: RAGConfig = None):
        self.analyzer = TextAnalyzer(config)
        self.logger = get_logger("batch_text_analyzer", "modules.processors")
    
    @log_performance()
    def analyze_batch(self, texts: List[str]) -> List[TextAnalysisResult]:
        """
        Analysiert Batch von Texten
        
        Args:
            texts (List[str]): Liste der zu analysierenden Texte
            
        Returns:
            List[TextAnalysisResult]: Liste der Analyse-Ergebnisse
        """
        if not texts:
            return []
        
        results = []
        errors = []
        
        for i, text in enumerate(texts):
            try:
                result = self.analyzer.analyze_text(text)
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Fehler bei Text {i}: {str(e)}")
                errors.append((i, str(e)))
                
                # Fallback-Ergebnis erstellen
                fallback_result = TextAnalysisResult(
                    content_type=ContentType.GENERAL,
                    confidence_score=0.0,
                    metadata={'error': str(e), 'batch_index': i}
                )
                results.append(fallback_result)
        
        self.logger.info(
            f"Batch-Analyse abgeschlossen: {len(texts)} Texte, {len(errors)} Fehler",
            extra={
                'extra_data': {
                    'total_texts': len(texts),
                    'successful_analyses': len(texts) - len(errors),
                    'errors': len(errors)
                }
            }
        )
        
        return results


# =============================================================================
# ANALYZER-FACTORY FÜR KONFIGURABLE STRATEGIEN
# =============================================================================

class AnalyzerFactory:
    """Factory für verschiedene Analyzer-Konfigurationen"""
    
    @staticmethod
    def create_industrial_analyzer(config: RAGConfig = None) -> TextAnalyzer:
        """
        Erstellt Analyzer optimiert für industrielle Dokumentation
        
        Args:
            config (RAGConfig): Konfiguration
            
        Returns:
            TextAnalyzer: Industriell optimierter Analyzer
        """
        analyzer = TextAnalyzer(config)
        
        # Erweiterte industrielle Keywords
        analyzer.technical_keywords.extend([
            'maschine', 'anlage', 'steuerung', 'sensor', 'aktor',
            'sps', 'hmi', 'scada', 'feldbus', 'profibus', 'profinet'
        ])
        
        analyzer.troubleshooting_keywords.extend([
            'störung', 'ausfall', 'wartung', 'instandhaltung', 'reparatur',
            'diagnose', 'maintenance', 'repair', 'malfunction'
        ])
        
        return analyzer
    
    @staticmethod
    def create_software_analyzer(config: RAGConfig = None) -> TextAnalyzer:
        """
        Erstellt Analyzer optimiert für Software-Dokumentation
        
        Args:
            config (RAGConfig): Konfiguration
            
        Returns:
            TextAnalyzer: Software-optimierter Analyzer
        """
        analyzer = TextAnalyzer(config)
        
        # Software-spezifische Keywords
        analyzer.technical_keywords.extend([
            'api', 'database', 'frontend', 'backend', 'framework',
            'library', 'package', 'module', 'class', 'function'
        ])
        
        analyzer.installation_keywords.extend([
            'npm install', 'pip install', 'composer install', 'deployment',
            'build', 'compile', 'docker', 'kubernetes'
        ])
        
        return analyzer


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums und Datenstrukturen
    'ContentType', 'TextAnalysisResult',
    
    # Hauptklassen
    'TextAnalyzer', 'BatchTextAnalyzer',
    
    # Factory
    'AnalyzerFactory'
]


if __name__ == "__main__":
    # Testing und Demonstration
    print("Text Analyzer - Extrahiert aus RAG Chatbot")
    print("==========================================")
    
    # Analyzer erstellen
    analyzer = TextAnalyzer()
    
    # Test-Texte aus dem Original-Kontext
    test_texts = [
        """
        1. Öffnen Sie die Demo-Anwendung
        2. Klicken Sie auf "Start"
        3. Wählen Sie die gewünschte Konfiguration
        4. Drücken Sie OK um fortzufahren
        """,
        
        """
        Installation der Software:
        Stellen Sie sicher, dass alle Systemanforderungen erfüllt sind.
        Führen Sie setup.exe als Administrator aus.
        """,
        
        """
        Technische Daten:
        Spannung: 24V DC
        Leistung: 150W
        Betriebstemperatur: -10°C bis +60°C
        """,
        
        """
        Bei Problemen mit der Verbindung prüfen Sie:
        - Netzwerkkabel korrekt angeschlossen
        - IP-Adresse korrekt konfiguriert
        - Firewall-Einstellungen
        """
    ]
    
    # Einzelne Texte analysieren
    for i, text in enumerate(test_texts):
        print(f"\n--- Test {i+1} ---")
        print(f"Text: {text.strip()[:100]}...")
        
        result = analyzer.analyze_text(text)
        
        print(f"Content-Type: {result.content_type.value}")
        print(f"Keywords: {result.keywords[:5]}")  # Erste 5
        print(f"Hat Schritte: {result.has_sequential_steps}")
        print(f"Schritt-Anzahl: {result.step_count}")
        print(f"Konfidenz: {result.confidence_score:.2f}")
        
        if result.technical_terms:
            print(f"Technische Begriffe: {result.technical_terms[:3]}")
    
    # Batch-Analyse testen
    print(f"\n--- Batch-Analyse ---")
    batch_analyzer = BatchTextAnalyzer()
    batch_results = batch_analyzer.analyze_batch(test_texts)
    
    content_types = [r.content_type.value for r in batch_results]
    print(f"Batch-Ergebnisse: {content_types}")
    
    # Statistiken anzeigen
    stats = analyzer.get_analysis_statistics()
    print(f"\nStatistiken: {stats}")
    
    print("\n✅ Text-Analyzer erfolgreich getestet")