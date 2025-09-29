#!/usr/bin/env python3
"""
Content Classifier für RAG Chatbot Industrial

Erweiterte Content-Klassifikation für industrielle Dokumentation aufbauend auf
dem extrahierten TextAnalyzer. Multi-Label-Klassifikation mit industriellen
Kategorien und konfigurierbaren Classifier-Strategien.

Features:
- Multi-Label-Klassifikation über Basis-ContentTypes hinaus
- Industrielle Kategorien (Safety, Specifications, Maintenance, etc.)
- Confidence-Scoring für alle Klassifikationen
- Plugin-Interface für domänenspezifische Classifier
- Konfigurierbare Regel-Sets und Machine Learning Integration

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Erweitert TextAnalyzer-Funktionalität
"""

import re
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict

# Core-Komponenten
from core import (
    get_logger, RAGConfig, get_current_config,
    DocumentProcessingError, ValidationError,
    create_error_context, log_performance
)

# Text-Analyzer Integration
from .text_analyzer import TextAnalyzer, TextAnalysisResult, ContentType


# =============================================================================
# ERWEITERTE KATEGORIEN UND KLASSIFIKATIONEN
# =============================================================================

class IndustrialCategory(str, Enum):
    """Industrielle Haupt-Kategorien für technische Dokumentation"""
    SAFETY = "safety"                      # Sicherheitshinweise, Warnungen
    SPECIFICATIONS = "specifications"       # Technische Daten, Parameter
    INSTALLATION = "installation"          # Montage, Installation, Setup
    OPERATION = "operation"                # Bedienung, Betrieb
    MAINTENANCE = "maintenance"            # Wartung, Instandhaltung
    TROUBLESHOOTING = "troubleshooting"    # Fehlerbehebung, Diagnose
    COMPLIANCE = "compliance"              # Normen, Zertifizierungen
    DOCUMENTATION = "documentation"        # Meta-Dokumentation


class ContentFlag(str, Enum):
    """Spezielle Content-Flags für detaillierte Klassifikation"""
    HAS_WARNINGS = "has_warnings"          # Enthält Sicherheitswarnungen
    HAS_PROCEDURES = "has_procedures"      # Enthält Arbeitsanweisungen
    HAS_SPECIFICATIONS = "has_specifications"  # Enthält technische Daten
    HAS_DIAGRAMS = "has_diagrams"          # Verweise auf Diagramme/Bilder
    HAS_TABLES = "has_tables"              # Enthält tabellarische Daten
    HAS_CALCULATIONS = "has_calculations"  # Enthält Berechnungen/Formeln
    IS_MULTILINGUAL = "is_multilingual"    # Mehrsprachiger Inhalt
    REQUIRES_EXPERTISE = "requires_expertise"  # Erfordert Fachkenntnisse


@dataclass
class ClassificationResult:
    """
    Ergebnis der erweiterten Content-Klassifikation
    
    Attributes:
        primary_category (IndustrialCategory): Haupt-Kategorie
        secondary_categories (List[IndustrialCategory]): Zusätzliche Kategorien
        content_flags (Set[ContentFlag]): Spezielle Content-Merkmale
        confidence_scores (Dict[str, float]): Vertrauen pro Kategorie (0.0-1.0)
        rule_matches (Dict[str, List[str]]): Welche Regeln getroffen haben
        base_analysis (TextAnalysisResult): Ursprüngliche TextAnalyzer-Ergebnisse
        metadata (Dict[str, Any]): Zusätzliche Klassifikations-Metadaten
    """
    primary_category: IndustrialCategory
    secondary_categories: List[IndustrialCategory] = field(default_factory=list)
    content_flags: Set[ContentFlag] = field(default_factory=set)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    rule_matches: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    base_analysis: Optional[TextAnalysisResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def all_categories(self) -> List[IndustrialCategory]:
        """Alle erkannten Kategorien (primär + sekundär)"""
        categories = [self.primary_category]
        categories.extend(self.secondary_categories)
        return list(set(categories))  # Duplikate entfernen
    
    @property
    def total_confidence(self) -> float:
        """Durchschnittliches Vertrauen über alle Kategorien"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)


# =============================================================================
# REGEL-BASIERTE KLASSIFIKATION
# =============================================================================

@dataclass
class ClassificationRule:
    """
    Regel für Content-Klassifikation
    
    Attributes:
        name (str): Eindeutiger Regel-Name
        category (IndustrialCategory): Ziel-Kategorie
        keywords (List[str]): Schlüsselwörter (case-insensitive)
        patterns (List[str]): Regex-Pattern (als Raw Strings!)
        flags (List[ContentFlag]): Zu setzende Flags
        weight (float): Gewichtung der Regel (0.0-1.0)
        requires_all (bool): Alle Keywords/Pattern erforderlich (UND) vs. mindestens eins (ODER)
    """
    name: str
    category: IndustrialCategory
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    flags: List[ContentFlag] = field(default_factory=list)
    weight: float = 1.0
    requires_all: bool = False
    
    def matches(self, text: str) -> Tuple[bool, List[str]]:
        """
        Prüft ob Regel auf Text zutrifft
        
        Args:
            text (str): Zu prüfender Text
            
        Returns:
            Tuple[bool, List[str]]: (Match gefunden, Liste der Matches)
        """
        text_lower = text.lower()
        matches_found = []
        
        # Keyword-Matches
        keyword_matches = []
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                keyword_matches.append(f"keyword:{keyword}")
        
        # Pattern-Matches (Raw Strings verwenden!)
        pattern_matches = []
        for pattern in self.patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    pattern_matches.append(f"pattern:{pattern}")
            except re.error as e:
                # Pattern-Fehler loggen aber nicht abbrechen
                continue
        
        all_matches = keyword_matches + pattern_matches
        
        if self.requires_all:
            # Alle Keywords UND alle Pattern müssen matchen
            required_matches = len(self.keywords) + len(self.patterns)
            has_match = len(all_matches) >= required_matches
        else:
            # Mindestens ein Match reicht
            has_match = len(all_matches) > 0
        
        return has_match, all_matches


class RuleBasedClassifier:
    """
    Regel-basierter Classifier für industrielle Content-Klassifikation
    
    Verwendet konfigurierbare Regel-Sets für robuste, nachvollziehbare
    Klassifikation technischer Dokumentation.
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialisiert Rule-Based Classifier
        
        Args:
            config (RAGConfig): Konfiguration
        """
        self.config = config or get_current_config()
        self.logger = get_logger("rule_based_classifier", "modules.processors")
        
        # Standard-Regel-Sets laden
        self.rules = self._init_industrial_rules()
        
        # Performance-Tracking
        self._classification_stats = {
            'texts_classified': 0,
            'rule_hit_count': defaultdict(int),
            'category_distribution': defaultdict(int)
        }
    
    def _init_industrial_rules(self) -> List[ClassificationRule]:
        """Initialisiert Standard-Regel-Set für industrielle Dokumentation"""
        
        rules = []
        
        # =============================================================================
        # SAFETY RULES - Sicherheitshinweise erkennen
        # =============================================================================
        
        rules.append(ClassificationRule(
            name="safety_warnings",
            category=IndustrialCategory.SAFETY,
            keywords=[
                'warnung', 'warning', 'achtung', 'caution', 'gefahr', 'danger',
                'vorsicht', 'hinweis', 'notice', 'sicherheit', 'safety'
            ],
            patterns=[
                r'\b(?:WARNUNG|WARNING|ACHTUNG|CAUTION|GEFAHR|DANGER)\b',
                r'⚠️|⚡|☢️|☣️',  # Unicode-Warnsymbole
                r'\brisiko\b|\brisk\b',
                r'\bverletzung\b|\binjury\b'
            ],
            flags=[ContentFlag.HAS_WARNINGS, ContentFlag.REQUIRES_EXPERTISE],
            weight=0.9
        ))
        
        # =============================================================================
        # SPECIFICATIONS RULES - Technische Daten
        # =============================================================================
        
        rules.append(ClassificationRule(
            name="technical_specifications",
            category=IndustrialCategory.SPECIFICATIONS,
            keywords=[
                'technische daten', 'technical data', 'spezifikation', 'specification',
                'parameter', 'eigenschaften', 'properties', 'kennwerte', 'ratings'
            ],
            patterns=[
                r'\b\d+(?:\.\d+)?\s*(?:V|A|W|Hz|MHz|GHz|°C|bar|psi|rpm)\b',  # Technische Einheiten
                r'\b(?:spannung|voltage|strom|current|leistung|power)\s*[:=]\s*\d+',
                r'\b(?:min|max|typ)\.\s*\d+',
                r'\btabelle\s+\d+|\btable\s+\d+'
            ],
            flags=[ContentFlag.HAS_SPECIFICATIONS, ContentFlag.HAS_TABLES],
            weight=0.8
        ))
        
        # =============================================================================
        # INSTALLATION RULES - Montage und Setup
        # =============================================================================
        
        rules.append(ClassificationRule(
            name="installation_procedures",
            category=IndustrialCategory.INSTALLATION,
            keywords=[
                'installation', 'montage', 'aufbau', 'setup', 'konfiguration',
                'configuration', 'anschluss', 'connection', 'verdrahtung', 'wiring'
            ],
            patterns=[
                r'\b(?:schritt|step)\s+\d+',
                r'\b\d+\.\s+(?:installieren|install|montieren|mount)',
                r'\banschlussplan|\bwiring\s+diagram',
                r'\bpin\s+\d+|\bklemme\s+\d+'
            ],
            flags=[ContentFlag.HAS_PROCEDURES, ContentFlag.HAS_DIAGRAMS],
            weight=0.7
        ))
        
        # =============================================================================
        # OPERATION RULES - Bedienung und Betrieb
        # =============================================================================
        
        rules.append(ClassificationRule(
            name="operation_procedures",
            category=IndustrialCategory.OPERATION,
            keywords=[
                'bedienung', 'operation', 'betrieb', 'anwendung', 'usage',
                'steuerung', 'control', 'start', 'stop', 'betätigen', 'activate'
            ],
            patterns=[
                r'\bdrücken\s+sie|\bpress\s+the',
                r'\bklicken\s+auf|\bclick\s+on',
                r'\bmenü\s+→|\bmenu\s+→',
                r'\bhmi\b|\bscada\b|\bsps\b'  # Industrielle Bedienelemente
            ],
            flags=[ContentFlag.HAS_PROCEDURES],
            weight=0.6
        ))
        
        # =============================================================================
        # MAINTENANCE RULES - Wartung und Instandhaltung
        # =============================================================================
        
        rules.append(ClassificationRule(
            name="maintenance_procedures",
            category=IndustrialCategory.MAINTENANCE,
            keywords=[
                'wartung', 'maintenance', 'instandhaltung', 'service', 'inspektion',
                'inspection', 'reinigung', 'cleaning', 'schmierung', 'lubrication',
                'austausch', 'replacement', 'reparatur', 'repair'
            ],
            patterns=[
                r'\b(?:alle|every)\s+\d+\s+(?:stunden|hours|tage|days|monate|months)',
                r'\bwartungsintervall|\bmaintenance\s+interval',
                r'\bverschleißteil|\bwear\s+part',
                r'\bölwechsel|\boil\s+change'
            ],
            flags=[ContentFlag.HAS_PROCEDURES, ContentFlag.REQUIRES_EXPERTISE],
            weight=0.7
        ))
        
        # =============================================================================
        # TROUBLESHOOTING RULES - Fehlerbehebung
        # =============================================================================
        
        rules.append(ClassificationRule(
            name="troubleshooting_procedures",
            category=IndustrialCategory.TROUBLESHOOTING,
            keywords=[
                'troubleshooting', 'fehlerbehebung', 'störung', 'problem',
                'fehler', 'error', 'alarm', 'diagnose', 'diagnosis',
                'lösung', 'solution', 'behebung', 'fix'
            ],
            patterns=[
                r'\bfehlermeldung|\berror\s+message|\balarm\s+\d+',
                r'\bprüfen\s+sie|\bcheck\s+the',
                r'\bursache|\bcause|\bgrund|\breason',
                r'\b(?:led|status)\s+(?:rot|red|grün|green|gelb|yellow)'
            ],
            flags=[ContentFlag.HAS_PROCEDURES, ContentFlag.REQUIRES_EXPERTISE],
            weight=0.8
        ))
        
        # =============================================================================
        # COMPLIANCE RULES - Normen und Zertifizierungen
        # =============================================================================
        
        rules.append(ClassificationRule(
            name="compliance_standards",
            category=IndustrialCategory.COMPLIANCE,
            keywords=[
                'norm', 'standard', 'richtlinie', 'directive', 'zertifikat',
                'certificate', 'konformität', 'compliance', 'ce-kennzeichnung',
                'ce marking', 'atex', 'iec', 'iso', 'din'
            ],
            patterns=[
                r'\b(?:EN|IEC|ISO|DIN|ANSI|IEEE)\s+\d+',
                r'\bce\s*[-\s]*mark',
                r'\batex\s+zone|\bexplosion\s+protection',
                r'\bzertifikat\s+nr\.|\bcertificate\s+no\.'
            ],
            weight=0.6
        ))
        
        # =============================================================================
        # SPECIAL CONTENT FLAGS - Zusätzliche Merkmale
        # =============================================================================
        
        # Diagramm-Verweise
        rules.append(ClassificationRule(
            name="has_diagrams",
            category=IndustrialCategory.DOCUMENTATION,  # Niedrige Priorität
            keywords=[],
            patterns=[
                r'\b(?:abbildung|figure|abb\.|fig\.)\s+\d+',
                r'\b(?:diagramm|diagram|schema|schematic)',
                r'\b(?:siehe|see)\s+(?:bild|image|foto|photo)'
            ],
            flags=[ContentFlag.HAS_DIAGRAMS],
            weight=0.3
        ))
        
        # Berechnungen und Formeln
        rules.append(ClassificationRule(
            name="has_calculations",
            category=IndustrialCategory.SPECIFICATIONS,
            keywords=[],
            patterns=[
                r'[=×÷+\-]\s*\d+(?:\.\d+)?',  # Mathematische Operationen
                r'\b(?:formel|formula|berechnung|calculation)',
                r'\b[A-Z]\s*=\s*\d+',  # Formel-Zuweisungen
                r'√|∑|∫|π'  # Mathematische Symbole
            ],
            flags=[ContentFlag.HAS_CALCULATIONS, ContentFlag.REQUIRES_EXPERTISE],
            weight=0.4
        ))
        
        return rules
    
    def classify(self, text: str, base_analysis: TextAnalysisResult = None) -> ClassificationResult:
        """
        Klassifiziert Text mit regel-basiertem Ansatz
        
        Args:
            text (str): Zu klassifizierender Text
            base_analysis (TextAnalysisResult): Basis-Analyse vom TextAnalyzer
            
        Returns:
            ClassificationResult: Detaillierte Klassifikation
        """
        if not text or len(text.strip()) == 0:
            raise ValidationError("Text für Klassifikation darf nicht leer sein", field_name="text")
        
        # Regel-Matches sammeln
        category_scores = defaultdict(float)
        all_flags = set()
        all_rule_matches = defaultdict(list)
        
        for rule in self.rules:
            has_match, matches = rule.matches(text)
            
            if has_match:
                # Score für Kategorie erhöhen
                category_scores[rule.category] += rule.weight
                
                # Flags sammeln
                all_flags.update(rule.flags)
                
                # Rule-Matches dokumentieren
                all_rule_matches[rule.category.value].extend(matches)
                
                # Statistik aktualisieren
                self._classification_stats['rule_hit_count'][rule.name] += 1
        
        # Primäre Kategorie bestimmen (höchster Score)
        if category_scores:
            primary_category = max(category_scores, key=category_scores.get)
            
            # Sekundäre Kategorien (alle anderen mit Score > 0.3)
            secondary_categories = [
                cat for cat, score in category_scores.items() 
                if cat != primary_category and score >= 0.3
            ]
        else:
            # Fallback: Basis-Analyse verwenden
            if base_analysis and base_analysis.content_type != ContentType.GENERAL:
                primary_category = self._map_base_content_type(base_analysis.content_type)
            else:
                primary_category = IndustrialCategory.DOCUMENTATION
            secondary_categories = []
        
        # Confidence-Scores normalisieren
        max_possible_score = sum(rule.weight for rule in self.rules if rule.category == primary_category)
        confidence_scores = {}
        
        for category, score in category_scores.items():
            max_score_for_category = sum(rule.weight for rule in self.rules if rule.category == category)
            if max_score_for_category > 0:
                confidence_scores[category.value] = min(1.0, score / max_score_for_category)
        
        # Ergebnis zusammenstellen
        result = ClassificationResult(
            primary_category=primary_category,
            secondary_categories=secondary_categories,
            content_flags=all_flags,
            confidence_scores=confidence_scores,
            rule_matches=dict(all_rule_matches),
            base_analysis=base_analysis,
            metadata={
                'total_rules_matched': len(all_rule_matches),
                'text_length': len(text),
                'classification_method': 'rule_based'
            }
        )
        
        # Statistiken aktualisieren
        self._update_classification_stats(result)
        
        self.logger.debug(
            f"Text klassifiziert: {primary_category.value} (Confidence: {result.total_confidence:.2f})",
            extra={
                'extra_data': {
                    'primary_category': primary_category.value,
                    'secondary_categories': [cat.value for cat in secondary_categories],
                    'flags_count': len(all_flags),
                    'rules_matched': len(all_rule_matches),
                    'total_confidence': result.total_confidence
                }
            }
        )
        
        return result
    
    def _map_base_content_type(self, content_type: ContentType) -> IndustrialCategory:
        """Mappt TextAnalyzer ContentType zu IndustrialCategory"""
        mapping = {
            ContentType.STEP_BY_STEP_GUIDE: IndustrialCategory.OPERATION,
            ContentType.DEMO_INSTRUCTIONS: IndustrialCategory.OPERATION,
            ContentType.INSTALLATION_GUIDE: IndustrialCategory.INSTALLATION,
            ContentType.TECHNICAL_SPECIFICATIONS: IndustrialCategory.SPECIFICATIONS,
            ContentType.TROUBLESHOOTING: IndustrialCategory.TROUBLESHOOTING,
            ContentType.GENERAL: IndustrialCategory.DOCUMENTATION
        }
        return mapping.get(content_type, IndustrialCategory.DOCUMENTATION)
    
    def _update_classification_stats(self, result: ClassificationResult) -> None:
        """Aktualisiert interne Klassifikations-Statistiken"""
        self._classification_stats['texts_classified'] += 1
        self._classification_stats['category_distribution'][result.primary_category.value] += 1
        
        for cat in result.secondary_categories:
            self._classification_stats['category_distribution'][cat.value] += 1
    
    def add_custom_rule(self, rule: ClassificationRule) -> None:
        """
        Fügt benutzerdefinierte Regel hinzu
        
        Args:
            rule (ClassificationRule): Neue Regel
        """
        self.rules.append(rule)
        self.logger.info(f"Custom Rule hinzugefügt: {rule.name} -> {rule.category.value}")
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """
        Holt Klassifikations-Statistiken
        
        Returns:
            Dict[str, Any]: Detaillierte Statistiken
        """
        stats = dict(self._classification_stats)
        
        if stats['texts_classified'] > 0:
            # Prozentuale Verteilung
            total = stats['texts_classified']
            stats['category_percentages'] = {
                category: (count / total) * 100
                for category, count in stats['category_distribution'].items()
            }
            
            # Top-Rules
            stats['top_rules'] = dict(
                sorted(stats['rule_hit_count'].items(), key=lambda x: x[1], reverse=True)[:10]
            )
        
        return stats


# =============================================================================
# CONTENT CLASSIFIER HAUPTKLASSE
# =============================================================================

class ContentClassifier:
    """
    Erweiterte Content-Klassifikation für industrielle Dokumentation
    
    Kombiniert TextAnalyzer-Basis-Analyse mit erweiterten Klassifikations-Strategien
    für detaillierte, multi-label Content-Kategorisierung.
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialisiert Content Classifier
        
        Args:
            config (RAGConfig): Konfiguration
        """
        self.config = config or get_current_config()
        self.logger = get_logger("content_classifier", "modules.processors")
        
        # TextAnalyzer für Basis-Analyse
        self.text_analyzer = TextAnalyzer(self.config)
        
        # Rule-Based Classifier
        self.rule_classifier = RuleBasedClassifier(self.config)
        
        # Performance-Tracking
        self._processing_stats = {
            'documents_classified': 0,
            'avg_processing_time_ms': 0.0
        }
    
    @log_performance()
    def classify_content(self, text: str) -> ClassificationResult:
        """
        Führt erweiterte Content-Klassifikation durch
        
        Args:
            text (str): Zu klassifizierender Text
            
        Returns:
            ClassificationResult: Detaillierte Klassifikations-Ergebnisse
        """
        try:
            # 1. Basis-Analyse mit TextAnalyzer
            base_analysis = self.text_analyzer.analyze_text(text)
            
            # 2. Erweiterte Klassifikation mit Rules
            classification_result = self.rule_classifier.classify(text, base_analysis)
            
            # 3. Ergebnis-Anreicherung
            classification_result.metadata.update({
                'base_content_type': base_analysis.content_type.value,
                'base_confidence': base_analysis.confidence_score,
                'base_keywords': base_analysis.keywords,
                'processing_version': '4.0.0'
            })
            
            # Statistiken aktualisieren
            self._processing_stats['documents_classified'] += 1
            
            return classification_result
            
        except Exception as e:
            error_context = create_error_context(
                component="modules.processors.content_classifier",
                operation="classify_content",
                text_length=len(text)
            )
            
            raise DocumentProcessingError(
                message=f"Fehler bei Content-Klassifikation: {str(e)}",
                processing_stage="content_classification",
                context=error_context,
                original_exception=e
            )
    
    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Klassifiziert Batch von Texten
        
        Args:
            texts (List[str]): Liste zu klassifizierender Texte
            
        Returns:
            List[ClassificationResult]: Klassifikations-Ergebnisse
        """
        if not texts:
            return []
        
        results = []
        errors = []
        
        for i, text in enumerate(texts):
            try:
                result = self.classify_content(text)
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Fehler bei Text {i}: {str(e)}")
                errors.append((i, str(e)))
                
                # Fallback-Klassifikation
                fallback_result = ClassificationResult(
                    primary_category=IndustrialCategory.DOCUMENTATION,
                    confidence_scores={'documentation': 0.0},
                    metadata={'error': str(e), 'batch_index': i}
                )
                results.append(fallback_result)
        
        self.logger.info(
            f"Batch-Klassifikation abgeschlossen: {len(texts)} Texte, {len(errors)} Fehler"
        )
        
        return results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Holt Processing-Statistiken
        
        Returns:
            Dict[str, Any]: Kombinierte Statistiken
        """
        stats = {
            'content_classifier': self._processing_stats.copy(),
            'text_analyzer': self.text_analyzer.get_analysis_statistics(),
            'rule_classifier': self.rule_classifier.get_classification_statistics()
        }
        
        return stats


# =============================================================================
# CLASSIFIER FACTORY
# =============================================================================

class ClassifierFactory:
    """Factory für verschiedene Classifier-Konfigurationen"""
    
    @staticmethod
    def create_industrial_classifier(config: RAGConfig = None) -> ContentClassifier:
        """
        Erstellt Classifier optimiert für industrielle Dokumentation
        
        Args:
            config (RAGConfig): Konfiguration
            
        Returns:
            ContentClassifier: Industriell optimierter Classifier
        """
        classifier = ContentClassifier(config)
        
        # Zusätzliche industrielle Regeln
        classifier.rule_classifier.add_custom_rule(ClassificationRule(
            name="machine_controls",
            category=IndustrialCategory.OPERATION,
            keywords=['sps', 'plc', 'hmi', 'scada', 'steuerung', 'control'],
            patterns=[r'\bI/O\s+module', r'\bfieldbus\b', r'\bprofibus\b', r'\bprofinet\b'],
            weight=0.8
        ))
        
        classifier.rule_classifier.add_custom_rule(ClassificationRule(
            name="electrical_safety",
            category=IndustrialCategory.SAFETY,
            keywords=['hochspannung', 'high voltage', 'lebensgefahr', 'electrical hazard'],
            patterns=[r'\b\d+\s*kV\b', r'\b(?:not|nicht)\s+(?:hot|spannungsführend)'],
            flags=[ContentFlag.HAS_WARNINGS, ContentFlag.REQUIRES_EXPERTISE],
            weight=0.95
        ))
        
        return classifier
    
    @staticmethod
    def create_software_classifier(config: RAGConfig = None) -> ContentClassifier:
        """
        Erstellt Classifier optimiert für Software-Dokumentation
        
        Args:
            config (RAGConfig): Konfiguration
            
        Returns:
            ContentClassifier: Software-optimierter Classifier
        """
        classifier = ContentClassifier(config)
        
        # Software-spezifische Regeln
        classifier.rule_classifier.add_custom_rule(ClassificationRule(
            name="api_documentation",
            category=IndustrialCategory.SPECIFICATIONS,
            keywords=['api', 'endpoint', 'rest', 'json', 'xml', 'parameter'],
            patterns=[r'\b(?:GET|POST|PUT|DELETE)\s+/', r'\b\{.*\}', r'\bHTTP\s+\d{3}'],
            weight=0.8
        ))
        
        return classifier


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums und Datenstrukturen
    'IndustrialCategory', 'ContentFlag', 'ClassificationResult', 'ClassificationRule',
    
    # Hauptklassen
    'ContentClassifier', 'RuleBasedClassifier',
    
    # Factory
    'ClassifierFactory'
]


if __name__ == "__main__":
    # Testing und Demonstration
    print("Content Classifier - Erweiterte Klassifikation")
    print("==============================================")
    
    # Classifier erstellen
    classifier = ContentClassifier()
    
    # Test-Texte für verschiedene Kategorien
    test_texts = [
        # Safety-Test
        """
        WARNUNG: Hochspannungsgefahr!
        Vor Arbeiten an der Anlage unbedingt die Hauptsicherung ausschalten.
        Lebensgefahr durch elektrischen Schlag bei unsachgemäßer Handhabung.
        ⚠️ Nur von Elektrofachkräften zu bedienen.
        """,
        
        # Specifications-Test  
        """
        Technische Daten Motor XB-2000:
        - Nennspannung: 400V AC
        - Nennstrom: 12.5A
        - Leistung: 5.5 kW
        - Drehzahl: 1450 rpm
        - Schutzart: IP65
        Siehe Tabelle 3 für weitere Parameter.
        """,
        
        # Installation-Test
        """
        Installation des Sensors:
        1. Montage an der Halterung mit 4x M6 Schrauben
        2. Anschluss des Kabels an Klemme X1
        3. Konfiguration über HMI-Panel
        Siehe Anschlussplan Abb. 5 für Verdrahtung.
        """,
        
        # Troubleshooting-Test
        """
        Fehlerbehebung Alarm A001:
        Ursache: Temperatursensor defekt
        Prüfen Sie:
        - LED Status (rot = Fehler)  
        - Verkabelung zu Sensor
        - Sensorwiderstand (sollte 100Ω sein)
        Lösung: Sensor austauschen wenn Widerstand > 150Ω
        """,
        
        # Maintenance-Test
        """
        Wartungsintervall: Alle 6 Monate
        1. Reinigung des Filters
        2. Kontrolle der Verschraubungen  
        3. Ölwechsel (SAE 10W-40)
        4. Inspektion der Verschleißteile
        Wartung nur durch geschultes Personal durchführen.
        """
    ]
    
    # Einzelne Texte klassifizieren
    for i, text in enumerate(test_texts):
        print(f"\n--- Test {i+1} ---")
        print(f"Text: {text.strip()[:80]}...")
        
        result = classifier.classify_content(text)
        
        print(f"Primäre Kategorie: {result.primary_category.value}")
        if result.secondary_categories:
            print(f"Sekundäre Kategorien: {[cat.value for cat in result.secondary_categories]}")
        if result.content_flags:
            print(f"Content-Flags: {[flag.value for flag in result.content_flags]}")
        print(f"Gesamt-Confidence: {result.total_confidence:.2f}")
        
        # Top-3 Confidence Scores
        top_scores = sorted(result.confidence_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_scores:
            print(f"Top Confidence: {top_scores}")
    
    # Batch-Klassifikation testen  
    print(f"\n--- Batch-Klassifikation ---")
    batch_results = classifier.classify_batch(test_texts)
    
    categories = [r.primary_category.value for r in batch_results]
    flags_total = sum(len(r.content_flags) for r in batch_results)
    
    print(f"Batch-Ergebnisse: {categories}")
    print(f"Gesamt Content-Flags: {flags_total}")
    
    # Industrieller Classifier testen
    print(f"\n--- Industrieller Classifier ---")
    industrial_classifier = ClassifierFactory.create_industrial_classifier()
    
    electrical_test = """
    ACHTUNG: Hochspannung 24kV!  
    Arbeiten nur bei not-hot Zustand. 
    SPS-Steuerung über PROFIBUS-Fieldbus.
    I/O Module in IP67 Gehäuse montiert.
    """
    
    industrial_result = industrial_classifier.classify_content(electrical_test)
    print(f"Industrielle Klassifikation: {industrial_result.primary_category.value}")
    print(f"Flags: {[flag.value for flag in industrial_result.content_flags]}")
    
    # Statistiken anzeigen
    stats = classifier.get_processing_statistics()
    print(f"\n--- Statistiken ---")
    print(f"Verarbeitete Dokumente: {stats['content_classifier']['documents_classified']}")
    
    rule_stats = stats['rule_classifier']
    if 'category_percentages' in rule_stats:
        print("Kategorien-Verteilung:")
        for cat, percent in rule_stats['category_percentages'].items():
            print(f"  {cat}: {percent:.1f}%")
    
    print("\n✅ Content-Classifier erfolgreich getestet")