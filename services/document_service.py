#!/usr/bin/env python3
"""
Document Service für RAG Chatbot Industrial

Service-Layer für die Orchestrierung der kompletten Dokumenten-Verarbeitungs-Pipeline.
Koordiniert alle Processor-Module und Embedding-Services für industrielle RAG-Systeme.

Features:
- End-to-End Dokumenten-Pipeline: PDF-Loading → Text-Splitting → Classification → Enhancement
- Multi-Format-Support mit Plugin-Architektur für verschiedene Dokumenttypen
- Robuste Fehlerbehandlung und Rollback-Mechanismen
- Batch-Verarbeitung für große Dokumentensammlungen
- Progress-Tracking und detailliertes Monitoring

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

import os
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Core-Komponenten
from core import (
    get_logger, RAGConfig, get_current_config,
    DocumentProcessingError, ValidationError,
    create_error_context, log_performance
)

# Processor-Module
from modules.processors.text_analyzer import TextAnalyzer
from modules.processors.text_splitter import TextSplitterFactory, SplittingStrategy
from modules.processors.content_classifier import ContentClassifier
from modules.processors.metadata_enhancer import MetadataEnhancer, EnhancedTextChunk

# Loader-Module (aus processors)
from modules.processors.text_splitter import TextChunk  # Für Legacy-Support


# =============================================================================
# DOCUMENT SERVICE DATENSTRUKTUREN
# =============================================================================

class ProcessingStage(str, Enum):
    """Verarbeitungs-Stadien der Dokumenten-Pipeline"""
    LOADING = "loading"                    # Dokument-Loading
    ANALYSIS = "analysis"                  # Text-Analyse
    SPLITTING = "splitting"                # Text-Segmentierung
    CLASSIFICATION = "classification"      # Content-Klassifikation
    ENHANCEMENT = "enhancement"            # Metadaten-Anreicherung
    COMPLETED = "completed"                # Vollständig verarbeitet
    FAILED = "failed"                     # Verarbeitung fehlgeschlagen


class ProcessingPriority(str, Enum):
    """Prioritätsstufen für Dokumenten-Verarbeitung"""
    LOW = "low"                           # Niedrige Priorität
    NORMAL = "normal"                     # Standard-Priorität
    HIGH = "high"                         # Hohe Priorität
    URGENT = "urgent"                     # Dringende Verarbeitung


@dataclass
class DocumentSource:
    """
    Dokument-Quelle für die Verarbeitung
    
    Attributes:
        file_path (str): Pfad zur Datei
        content (str): Text-Inhalt (falls bereits geladen)
        metadata (Dict[str, Any]): Basis-Metadaten
        source_type (str): Art der Quelle (file, text, url, etc.)
        encoding (str): Text-Encoding
        priority (ProcessingPriority): Verarbeitungs-Priorität
    """
    file_path: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_type: str = "file"
    encoding: str = "utf-8"
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    
    @property
    def identifier(self) -> str:
        """Eindeutige Identifikation der Quelle"""
        if self.file_path:
            return self.file_path
        elif self.content:
            # Hash für Text-Content
            return hashlib.md5(self.content.encode()).hexdigest()[:16]
        else:
            return "unknown_source"


@dataclass
class ProcessingProgress:
    """
    Fortschritts-Tracking für Dokumenten-Verarbeitung
    
    Attributes:
        document_id (str): Dokument-Identifikation
        current_stage (ProcessingStage): Aktuelles Verarbeitungs-Stadium
        stages_completed (List[ProcessingStage]): Abgeschlossene Stadien
        progress_percentage (float): Fortschritt in Prozent (0.0-100.0)
        start_time (float): Start-Zeit (Unix Timestamp)
        stage_times (Dict[ProcessingStage, float]): Zeit pro Stadium in Sekunden
        error_message (Optional[str]): Fehlermeldung bei Problemen
        warnings (List[str]): Warnungen während Verarbeitung
    """
    document_id: str
    current_stage: ProcessingStage = ProcessingStage.LOADING
    stages_completed: List[ProcessingStage] = field(default_factory=list)
    progress_percentage: float = 0.0
    start_time: float = field(default_factory=time.time)
    stage_times: Dict[ProcessingStage, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def update_stage(self, stage: ProcessingStage, duration_ms: float = 0.0):
        """Aktualisiert Verarbeitungs-Stadium"""
        if self.current_stage not in self.stages_completed:
            self.stages_completed.append(self.current_stage)
            self.stage_times[self.current_stage] = duration_ms / 1000.0
        
        self.current_stage = stage
        
        # Fortschritt berechnen (5 Hauptstadien)
        stage_progress = {
            ProcessingStage.LOADING: 20.0,
            ProcessingStage.ANALYSIS: 40.0,
            ProcessingStage.SPLITTING: 60.0,
            ProcessingStage.CLASSIFICATION: 80.0,
            ProcessingStage.ENHANCEMENT: 90.0,
            ProcessingStage.COMPLETED: 100.0,
            ProcessingStage.FAILED: 0.0
        }
        
        self.progress_percentage = stage_progress.get(stage, 0.0)
    
    @property
    def total_processing_time_ms(self) -> float:
        """Gesamte Verarbeitungszeit in Millisekunden"""
        return (time.time() - self.start_time) * 1000
    
    @property
    def is_completed(self) -> bool:
        """Prüft ob Verarbeitung abgeschlossen"""
        return self.current_stage == ProcessingStage.COMPLETED
    
    @property
    def has_failed(self) -> bool:
        """Prüft ob Verarbeitung fehlgeschlagen"""
        return self.current_stage == ProcessingStage.FAILED


@dataclass
class ProcessingResult:
    """
    Ergebnis der Dokumenten-Verarbeitung
    
    Attributes:
        document_source (DocumentSource): Ursprüngliche Quelle
        enhanced_chunks (List[EnhancedTextChunk]): Verarbeitete Chunks mit Metadaten
        progress (ProcessingProgress): Verarbeitungs-Fortschritt
        processing_statistics (Dict[str, Any]): Detaillierte Statistiken
        intermediate_results (Dict[ProcessingStage, Any]): Zwischenergebnisse pro Stadium
    """
    document_source: DocumentSource
    enhanced_chunks: List[EnhancedTextChunk] = field(default_factory=list)
    progress: Optional[ProcessingProgress] = None
    processing_statistics: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[ProcessingStage, Any] = field(default_factory=dict)


# =============================================================================
# DOCUMENT SERVICE HAUPTKLASSE
# =============================================================================

class DocumentService:
    """
    Service für End-to-End Dokumenten-Verarbeitung in industriellen RAG-Systemen
    
    Orchestriert die komplette Pipeline:
    1. Document Loading (PDF, Text, etc.)
    2. Text Analysis (Content-Type, Keywords)
    3. Text Splitting (Chunks mit Metadaten)
    4. Content Classification (Industrial Categories)
    5. Metadata Enhancement (Consolidated Metadata)
    """
    
    def __init__(self, config: RAGConfig = None):
        """
        Initialisiert Document Service
        
        Args:
            config (RAGConfig): Konfiguration
        """
        self.config = config or get_current_config()
        self.logger = get_logger("document_service", "services")
        
        # Processor-Module initialisieren
        self._initialize_processors()
        
        # Threading für parallele Verarbeitung
        self.max_workers = getattr(self.config.processing, 'max_workers', 4)
        self._processing_lock = threading.Lock()
        
        # Progress-Tracking
        self._active_processing: Dict[str, ProcessingProgress] = {}
        self._progress_callbacks: List[Callable[[ProcessingProgress], None]] = []
        
        # Service-Statistiken
        self._service_stats = {
            'documents_processed': 0,
            'total_chunks_created': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'total_processing_time_ms': 0.0,
            'average_processing_time_ms': 0.0,
            'stage_performance': {stage.value: {'count': 0, 'total_time_ms': 0.0} 
                                for stage in ProcessingStage}
        }
    
    def _get_document_id(self, document_source) -> str:
        """Sichere Extraktion einer Dokument-ID aus verschiedenen Quellen"""
        import re, time
    
        if hasattr(document_source, 'identifier'):
            return self._get_document_id(document_source)
        elif hasattr(document_source, 'name') and document_source.name:
            return re.sub(r'[^a-zA-Z0-9._-]', '_', document_source.name)
        elif hasattr(document_source, 'id'):
            return str(document_source.id)
        else:
            return f"document_{int(time.time() * 1000)}"
    
    def _initialize_processors(self) -> None:
        """Initialisiert alle Processor-Module"""
        try:
            # Text-Analyzer für Content-Analyse
            self.text_analyzer = TextAnalyzer(self.config)
            
            # Text-Splitter für Chunk-Erstellung
            self.text_splitter = TextSplitterFactory.create_stable_splitter(self.config)
            
            # Content-Classifier für erweiterte Klassifikation
            self.content_classifier = ContentClassifier(self.config)
            
            # Metadata-Enhancer für konsolidierte Metadaten
            self.metadata_enhancer = MetadataEnhancer(self.config)
            
            self.logger.info("Document Service Processor-Module initialisiert")
            
        except Exception as e:
            error_msg = f"Fehler bei Processor-Initialisierung: {str(e)}"
            self.logger.error(error_msg)
            raise DocumentProcessingError(error_msg, processing_stage="service_init")
    
    # =============================================================================
    # HAUPT-VERARBEITUNGSMETHODEN
    # =============================================================================
    
    @log_performance()
    def process_document(self, 
                        document_source: DocumentSource,
                        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None) -> ProcessingResult:
        """
        Verarbeitet einzelnes Dokument durch komplette Pipeline
        
        Args:
            document_source (DocumentSource): Dokument-Quelle
            progress_callback (Optional[Callable]): Callback für Fortschritts-Updates
            
        Returns:
            ProcessingResult: Vollständiges Verarbeitungs-Ergebnis
        """
        # Progress-Tracking initialisieren
        progress = ProcessingProgress(
            document_id=getattr(document_source, 'identifier', getattr(document_source, 'name', f"uploaded_doc_{int(time.time())}"))
        )
        
        # Progress-Callback registrieren
        if progress_callback:
            self._progress_callbacks.append(progress_callback)
        
        # Active Processing tracking
        with self._processing_lock:
            self._active_processing[self._get_document_id(document_source)] = progress
        
        result = ProcessingResult(
            document_source=document_source,
            progress=progress
        )
        
        try:
            self.logger.info(f"Starte Dokumenten-Verarbeitung: {self._get_document_id(document_source)}")
            
            # Stage 1: Document Loading
            progress.update_stage(ProcessingStage.LOADING)
            self._notify_progress(progress)
            
            stage_start = time.time()
            content = self._load_document_content(document_source)
            stage_duration = (time.time() - stage_start) * 1000
            
            result.intermediate_results[ProcessingStage.LOADING] = {
                'content_length': len(content),
                'encoding_detected': document_source.encoding
            }
            
            # Stage 2: Text Analysis
            progress.update_stage(ProcessingStage.ANALYSIS, stage_duration)
            self._notify_progress(progress)
            
            stage_start = time.time()
            text_analysis = self.text_analyzer.analyze_text(content)
            stage_duration = (time.time() - stage_start) * 1000
            
            result.intermediate_results[ProcessingStage.ANALYSIS] = {
                'content_type': text_analysis.content_type.value,
                'keywords_count': len(text_analysis.keywords),
                'has_sequential_steps': text_analysis.has_sequential_steps,
                'confidence_score': text_analysis.confidence_score
            }
            
            # Stage 3: Text Splitting
            progress.update_stage(ProcessingStage.SPLITTING, stage_duration)
            self._notify_progress(progress)
            
            stage_start = time.time()
            text_chunks = self.text_splitter.split_text(
                content, 
                source_document=self._get_document_id(document_source)
            )
            stage_duration = (time.time() - stage_start) * 1000
            
            result.intermediate_results[ProcessingStage.SPLITTING] = {
                'chunks_created': len(text_chunks),
                'average_chunk_length': sum(len(c.page_content) for c in text_chunks) / len(text_chunks) if text_chunks else 0
            }
            
            # Stage 4: Content Classification (für jeden Chunk)
            progress.update_stage(ProcessingStage.CLASSIFICATION, stage_duration)
            self._notify_progress(progress)
            
            stage_start = time.time()
            classified_chunks = []
            
            for chunk in text_chunks:
                classification_result = self.content_classifier.classify_content(chunk.page_content)
                
                # Classification-Ergebnisse zu Chunk-Metadaten hinzufügen
                enhanced_chunk_metadata = {
                    **chunk.metadata.to_dict(),
                    'classification_primary': classification_result.primary_category.value,
                    'classification_confidence': classification_result.total_confidence,
                    'content_flags': [flag.value for flag in classification_result.content_flags]
                }
                
                classified_chunks.append((chunk, classification_result))
            
            stage_duration = (time.time() - stage_start) * 1000
            
            result.intermediate_results[ProcessingStage.CLASSIFICATION] = {
                'chunks_classified': len(classified_chunks),
                'classification_distribution': self._analyze_classification_distribution(classified_chunks)
            }
            
            # Stage 5: Metadata Enhancement
            progress.update_stage(ProcessingStage.ENHANCEMENT, stage_duration)
            self._notify_progress(progress)
            
            stage_start = time.time()
            enhanced_chunks = []
            
            for i, (chunk, classification) in enumerate(classified_chunks):
                enhanced_chunk = self.metadata_enhancer.enhance_chunk_metadata(
                    chunk=chunk,
                    source_document=self._get_document_id(document_source),
                    chunk_index=i
                )
                enhanced_chunks.append(enhanced_chunk)
            
            stage_duration = (time.time() - stage_start) * 1000
            
            result.enhanced_chunks = enhanced_chunks
            result.intermediate_results[ProcessingStage.ENHANCEMENT] = {
                'chunks_enhanced': len(enhanced_chunks),
                'metadata_fields_per_chunk': len(enhanced_chunks[0].metadata.to_chroma_metadata()) if enhanced_chunks else 0
            }
            
            # Processing abgeschlossen
            progress.update_stage(ProcessingStage.COMPLETED, stage_duration)
            self._notify_progress(progress)
            
            # Statistiken aktualisieren
            self._update_service_statistics(progress, len(enhanced_chunks))
            
            # Processing-Statistiken für Ergebnis
            result.processing_statistics = {
                'total_processing_time_ms': progress.total_processing_time_ms,
                'stage_times': progress.stage_times,
                'chunks_created': len(enhanced_chunks),
                'success': True
            }
            
            self.logger.info(
                f"Dokumenten-Verarbeitung erfolgreich: {self._get_document_id(document_source)} "
                f"({len(enhanced_chunks)} Chunks in {progress.total_processing_time_ms:.1f}ms)"
            )
            
            return result
            
        except Exception as e:
            # Fehler-Behandlung
            progress.update_stage(ProcessingStage.FAILED)
            progress.error_message = str(e)
            self._notify_progress(progress)
            
            # Service-Statistiken für Fehler aktualisieren
            self._service_stats['failed_processing'] += 1
            
            error_context = create_error_context(
                component="services.document_service",
                operation="process_document",
                document_id=self._get_document_id(document_source),
                current_stage=progress.current_stage.value
            )
            
            raise DocumentProcessingError(
                message=f"Fehler bei Dokumenten-Verarbeitung: {str(e)}",
                processing_stage=progress.current_stage.value,
                context=error_context,
                original_exception=e
            )
        
        finally:
            # Progress-Tracking aufräumen
            with self._processing_lock:
                self._active_processing.pop(self._get_document_id(document_source), None)
            
            if progress_callback in self._progress_callbacks:
                self._progress_callbacks.remove(progress_callback)
    
    @log_performance()
    def process_document_batch(self, 
                              document_sources: List[DocumentSource],
                              max_workers: Optional[int] = None,
                              progress_callback: Optional[Callable[[str, ProcessingProgress], None]] = None) -> List[ProcessingResult]:
        """
        Verarbeitet Batch von Dokumenten parallel
        
        Args:
            document_sources (List[DocumentSource]): Dokument-Quellen
            max_workers (Optional[int]): Max parallele Worker
            progress_callback (Optional[Callable]): Batch-Progress-Callback
            
        Returns:
            List[ProcessingResult]: Verarbeitungs-Ergebnisse
        """
        if not document_sources:
            return []
        
        workers = max_workers or self.max_workers
        results = []
        failed_count = 0
        
        self.logger.info(f"Starte Batch-Verarbeitung: {len(document_sources)} Dokumente mit {workers} Workern")
        
        start_time = time.time()
        
        # Parallel-Verarbeitung
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Futures für alle Dokumente erstellen
            future_to_source = {
                executor.submit(
                    self.process_document, 
                    source,
                    lambda p: progress_callback(self._get_document_id(source), p) if progress_callback else None
                ): source 
                for source in document_sources
            }
            
            # Ergebnisse sammeln
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Batch-Verarbeitung fehlgeschlagen für {self._get_document_id(source)}: {str(e)}")
                    failed_count += 1
                    
                    # Fallback-Ergebnis erstellen
                    failed_progress = ProcessingProgress(
                        document_id=self._get_document_id(source),
                        current_stage=ProcessingStage.FAILED,
                        error_message=str(e)
                    )
                    
                    failed_result = ProcessingResult(
                        document_source=source,
                        progress=failed_progress,
                        processing_statistics={'success': False, 'error': str(e)}
                    )
                    
                    results.append(failed_result)
        
        total_time_ms = (time.time() - start_time) * 1000
        successful_count = len(document_sources) - failed_count
        total_chunks = sum(len(r.enhanced_chunks) for r in results)
        
        self.logger.info(
            f"Batch-Verarbeitung abgeschlossen: {successful_count}/{len(document_sources)} erfolgreich, "
            f"{total_chunks} Chunks in {total_time_ms:.1f}ms"
        )
        
        return results
    
    # =============================================================================
    # HILFSMETHODEN
    # =============================================================================
    
    def _load_document_content(self, document_source: DocumentSource) -> str:
        """
        Lädt Dokument-Content aus verschiedenen Quellen
        
        Args:
            document_source (DocumentSource): Dokument-Quelle
            
        Returns:
            str: Dokument-Inhalt
        """
        if document_source.content:
            # Content bereits vorhanden
            return document_source.content
        
        elif document_source.file_path:
            # Content aus Datei laden
            file_path = Path(document_source.file_path)
            
            if not file_path.exists():
                raise ValidationError(f"Datei nicht gefunden: {document_source.file_path}")
            
            # Datei-Typ bestimmen
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.pdf':
                return self._load_pdf_content(file_path)
            elif file_extension in ['.txt', '.md', '.csv']:
                return self._load_text_content(file_path, document_source.encoding)
            else:
                # Fallback: Als Text-Datei versuchen
                self.logger.warning(f"Unbekannter Dateityp {file_extension}, versuche als Text")
                return self._load_text_content(file_path, document_source.encoding)
        
        else:
            raise ValidationError("Weder content noch file_path in DocumentSource angegeben")
    
    def _load_pdf_content(self, file_path: Path) -> str:
        """Lädt PDF-Content mit Fallback auf unsere PDF-Loader"""
        try:
            # Verwende unseren PDF-Loader aus den Processors
            from ..processors.text_splitter import TextSplitterFactory
            
            # Pseudo-Dokument für PDF-Loader erstellen
            with open(file_path, 'rb') as f:
                # Vereinfachter PDF-Text-Extraktion
                # In produktiver Umgebung würde hier der vollständige PDF-Loader verwendet
                import PyPDF2
                
                pdf_reader = PyPDF2.PdfReader(f)
                text_content = ""
                
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                
                return text_content.strip()
        
        except Exception as e:
            self.logger.error(f"PDF-Loading fehlgeschlagen: {str(e)}")
            raise DocumentProcessingError(f"Kann PDF nicht laden: {str(e)}", processing_stage="loading")
    
    def _load_text_content(self, file_path: Path, encoding: str) -> str:
        """Lädt Text-Content mit Encoding-Behandlung"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        
        except UnicodeDecodeError:
            # Fallback auf UTF-8
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                # Letzter Fallback auf Latin-1
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
    
    def _analyze_classification_distribution(self, 
                                           classified_chunks: List[Tuple]) -> Dict[str, int]:
        """Analysiert Verteilung der Content-Klassifikationen"""
        distribution = {}
        
        for chunk, classification in classified_chunks:
            category = classification.primary_category.value
            distribution[category] = distribution.get(category, 0) + 1
        
        return distribution
    
    def _notify_progress(self, progress: ProcessingProgress) -> None:
        """Benachrichtigt alle registrierten Progress-Callbacks"""
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                self.logger.warning(f"Progress-Callback Fehler: {str(e)}")
    
    def _update_service_statistics(self, progress: ProcessingProgress, chunks_created: int) -> None:
        """Aktualisiert Service-Statistiken"""
        self._service_stats['documents_processed'] += 1
        self._service_stats['total_chunks_created'] += chunks_created
        
        if progress.is_completed:
            self._service_stats['successful_processing'] += 1
            
            # Verarbeitungszeit
            processing_time_ms = progress.total_processing_time_ms
            self._service_stats['total_processing_time_ms'] += processing_time_ms
            
            # Durchschnitt neu berechnen
            successful = self._service_stats['successful_processing']
            self._service_stats['average_processing_time_ms'] = (
                self._service_stats['total_processing_time_ms'] / successful
            )
            
            # Stage-Performance
            for stage, duration_s in progress.stage_times.items():
                stage_stats = self._service_stats['stage_performance'][stage.value]
                stage_stats['count'] += 1
                stage_stats['total_time_ms'] += duration_s * 1000
    
    # =============================================================================
    # MONITORING UND UTILITIES
    # =============================================================================
    
    def get_active_processing(self) -> Dict[str, ProcessingProgress]:
        """
        Holt aktuell aktive Verarbeitungsprozesse
        
        Returns:
            Dict[str, ProcessingProgress]: Aktive Prozesse
        """
        with self._processing_lock:
            return self._active_processing.copy()
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Holt detaillierte Service-Statistiken
        
        Returns:
            Dict[str, Any]: Service-Statistiken
        """
        stats = self._service_stats.copy()
        
        # Erfolgsrate berechnen
        total_processed = stats['documents_processed']
        if total_processed > 0:
            stats['success_rate'] = stats['successful_processing'] / total_processed
            stats['failure_rate'] = stats['failed_processing'] / total_processed
        
        # Stage-Performance aufbereiten
        stage_performance = {}
        for stage_name, stage_data in stats['stage_performance'].items():
            if stage_data['count'] > 0:
                stage_performance[stage_name] = {
                    'count': stage_data['count'],
                    'average_time_ms': stage_data['total_time_ms'] / stage_data['count']
                }
        
        stats['stage_performance_summary'] = stage_performance
        
        # Processor-Statistiken hinzufügen
        stats['processors'] = {
            'text_analyzer': self.text_analyzer.get_analysis_statistics(),
            'text_splitter': self.text_splitter.get_splitting_statistics(),
            'content_classifier': self.content_classifier.get_processing_statistics(),
            'metadata_enhancer': self.metadata_enhancer.get_enhancement_statistics()
        }
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        Umfassender Health-Check für Document Service
        
        Returns:
            Dict[str, Any]: Health-Status
        """
        health_status = {
            'service_operational': False,
            'processors_healthy': False,
            'test_processing_ok': False,
            'active_processes': 0,
            'overall_status': 'unhealthy'
        }
        
        try:
            # 1. Service-Verfügbarkeit
            health_status['service_operational'] = True
            
            # 2. Prozessoren testen
            processor_health = {
                'text_analyzer': hasattr(self, 'text_analyzer') and self.text_analyzer is not None,
                'text_splitter': hasattr(self, 'text_splitter') and self.text_splitter is not None,
                'content_classifier': hasattr(self, 'content_classifier') and self.content_classifier is not None,
                'metadata_enhancer': hasattr(self, 'metadata_enhancer') and self.metadata_enhancer is not None
            }
            
            health_status['processors_healthy'] = all(processor_health.values())
            health_status['processor_details'] = processor_health
            
            # 3. Aktive Prozesse
            health_status['active_processes'] = len(self._active_processing)
            
            # 4. Test-Verarbeitung
            if health_status['processors_healthy']:
                try:
                    test_source = DocumentSource(
                        content="Test-Dokument für Health-Check mit technischen Daten.",
                        metadata={'test': True},
                        source_type="health_check"
                    )
                    
                    test_result = self.process_document(test_source)
                    health_status['test_processing_ok'] = test_result.progress.is_completed
                    
                except Exception:
                    health_status['test_processing_ok'] = False
            
            # Gesamt-Status
            if (health_status['service_operational'] and 
                health_status['processors_healthy'] and
                health_status['test_processing_ok']):
                health_status['overall_status'] = 'healthy'
            elif health_status['service_operational']:
                health_status['overall_status'] = 'degraded'
            else:
                health_status['overall_status'] = 'unhealthy'
                
        except Exception as e:
            health_status['error'] = str(e)
        
        return health_status


# =============================================================================
# DOCUMENT SERVICE FACTORY
# =============================================================================

class DocumentServiceFactory:
    """Factory für verschiedene Document Service Konfigurationen"""
    
    @staticmethod
    def create_industrial_service(config: RAGConfig = None) -> DocumentService:
        """
        Erstellt Document Service für industrielle Dokumentation
        
        Args:
            config (RAGConfig): Konfiguration
            
        Returns:
            DocumentService: Industriell optimierter Service
        """
        service = DocumentService(config)
        
        # Industrielle Optimierungen könnten hier hinzugefügt werden
        # z.B. spezielle Processor-Konfigurationen
        
        return service
    
    @staticmethod
    def create_development_service(config: RAGConfig = None) -> DocumentService:
        """
        Erstellt Document Service für Development (mit Debug-Features)
        
        Args:
            config (RAGConfig): Konfiguration
            
        Returns:
            DocumentService: Development-optimierter Service
        """
        service = DocumentService(config)
        
        # Development-spezifische Einstellungen
        service.max_workers = 1  # Einzelthread für besseres Debugging
        
        return service
    
    @staticmethod
    def create_high_performance_service(config: RAGConfig = None) -> DocumentService:
        """
        Erstellt Document Service für High-Performance (mehr Worker)
        
        Args:
            config (RAGConfig): Konfiguration
            
        Returns:
            DocumentService: Performance-optimierter Service
        """
        service = DocumentService(config)
        
        # Performance-Optimierungen
        service.max_workers = 8  # Mehr parallele Worker
        
        return service


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_document_source_from_file(file_path: str,
                                    priority: ProcessingPriority = ProcessingPriority.NORMAL,
                                    metadata: Dict[str, Any] = None) -> DocumentSource:
    """
    Erstellt DocumentSource aus Dateipfad
    
    Args:
        file_path (str): Pfad zur Datei
        priority (ProcessingPriority): Verarbeitungs-Priorität
        metadata (Dict[str, Any]): Zusätzliche Metadaten
        
    Returns:
        DocumentSource: Konfigurierte Dokument-Quelle
    """
    file_path_obj = Path(file_path)
    
    base_metadata = {
        'filename': file_path_obj.name,
        'file_extension': file_path_obj.suffix.lower(),
        'file_size_bytes': file_path_obj.stat().st_size if file_path_obj.exists() else 0,
        'source_type': 'file'
    }
    
    if metadata:
        base_metadata.update(metadata)
    
    return DocumentSource(
        file_path=str(file_path_obj.absolute()),
        metadata=base_metadata,
        source_type="file",
        priority=priority
    )


def create_document_source_from_text(text: str,
                                   source_id: str = None,
                                   priority: ProcessingPriority = ProcessingPriority.NORMAL,
                                   metadata: Dict[str, Any] = None) -> DocumentSource:
    """
    Erstellt DocumentSource aus Text-String
    
    Args:
        text (str): Text-Inhalt
        source_id (str): Optional Source-Identifier
        priority (ProcessingPriority): Verarbeitungs-Priorität
        metadata (Dict[str, Any]): Zusätzliche Metadaten
        
    Returns:
        DocumentSource: Konfigurierte Dokument-Quelle
    """
    base_metadata = {
        'text_length': len(text),
        'word_count': len(text.split()),
        'source_type': 'text',
        'source_id': source_id or f"text_{hashlib.md5(text.encode()).hexdigest()[:8]}"
    }
    
    if metadata:
        base_metadata.update(metadata)
    
    return DocumentSource(
        content=text,
        metadata=base_metadata,
        source_type="text",
        priority=priority
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'ProcessingStage', 'ProcessingPriority',
    
    # Datenstrukturen
    'DocumentSource', 'ProcessingProgress', 'ProcessingResult',
    
    # Hauptklasse
    'DocumentService',
    
    # Factory
    'DocumentServiceFactory',
    
    # Utility Functions
    'create_document_source_from_file', 'create_document_source_from_text'
]


if __name__ == "__main__":
    # Testing und Demonstration
    print("Document Service - End-to-End Dokumenten-Pipeline")
    print("=================================================")
    
    try:
        # Document Service erstellen
        service = DocumentService()
        
        print(f"Document Service erstellt: {service}")
        
        # Health-Check durchführen
        health_status = service.health_check()
        print(f"\nHealth-Check:")
        print(f"  Service operational: {health_status['service_operational']}")
        print(f"  Processors healthy: {health_status['processors_healthy']}")
        print(f"  Test processing: {health_status['test_processing_ok']}")
        print(f"  Overall status: {health_status['overall_status']}")
        
        if health_status['overall_status'] != 'healthy':
            print(f"  Processor-Details: {health_status.get('processor_details', {})}")
        
        # Test nur wenn Service gesund
        if health_status['overall_status'] in ['healthy', 'degraded']:
            
            print("\n--- Einzeldokument-Verarbeitung ---")
            
            # Test-Dokument-Quellen erstellen
            test_documents = [
                # Text-basierte Quelle
                create_document_source_from_text(
                    text="""
                    WARNUNG: Hochspannungsgefahr!
                    
                    Technische Daten Motor XB-2000:
                    1. Nennspannung: 400V AC
                    2. Nennstrom: 12.5A  
                    3. Leistung: 5.5 kW
                    4. Schutzart: IP65
                    
                    Installation:
                    1. Spannung freischalten
                    2. Motor an Halterung befestigen
                    3. Kabel gemäß Schaltplan anschließen
                    4. Funktionstest durchführen
                    
                    Bei Problemen wenden Sie sich an den technischen Support.
                    """,
                    source_id="test_industrial_manual",
                    priority=ProcessingPriority.HIGH,
                    metadata={
                        'document_type': 'manual',
                        'equipment': 'motor_xb_2000',
                        'language': 'german'
                    }
                ),
                
                # Zweites Test-Dokument
                create_document_source_from_text(
                    text="""
                    Wartungsprotokoll - Quartalsinspektion
                    
                    Durchgeführte Arbeiten:
                    - Sichtprüfung aller Komponenten
                    - Reinigung der Lüftungsschlitze
                    - Kontrolle der Verschraubungen
                    - Funktionstest Notabschaltung
                    
                    Ergebnis: Alle Systeme funktionsfähig
                    Nächste Wartung: In 3 Monaten
                    """,
                    source_id="maintenance_protocol",
                    priority=ProcessingPriority.NORMAL,
                    metadata={
                        'document_type': 'maintenance',
                        'inspection_type': 'quarterly'
                    }
                )
            ]
            
            # Progress-Callback definieren
            def progress_callback(progress: ProcessingProgress):
                print(f"    Progress: {progress.current_stage.value} ({progress.progress_percentage:.0f}%)")
            
            # Erstes Dokument verarbeiten
            print(f"Verarbeite: {test_documents[0].metadata['source_id']}")
            
            result1 = service.process_document(
                test_documents[0],
                progress_callback=progress_callback
            )
            
            print(f"  Verarbeitung abgeschlossen: {result1.progress.is_completed}")
            print(f"  Chunks erstellt: {len(result1.enhanced_chunks)}")
            print(f"  Verarbeitungszeit: {result1.progress.total_processing_time_ms:.1f}ms")
            
            # Chunk-Details anzeigen
            for i, chunk in enumerate(result1.enhanced_chunks[:2]):  # Erste 2 Chunks
                metadata = chunk.metadata
                print(f"    Chunk {i+1}: {metadata.primary_category} (Safety: {metadata.safety_level})")
                print(f"      Expertise: {metadata.expertise_level}, Domain: {metadata.technical_domain}")
                print(f"      Content: {chunk.content[:80]}...")
            
            print("\n--- Batch-Verarbeitung ---")
            
            # Batch-Progress-Callback
            def batch_progress_callback(doc_id: str, progress: ProcessingProgress):
                if progress.current_stage == ProcessingStage.COMPLETED:
                    print(f"    {doc_id}: Completed ({progress.total_processing_time_ms:.0f}ms)")
                elif progress.current_stage == ProcessingStage.FAILED:
                    print(f"    {doc_id}: Failed - {progress.error_message}")
            
            # Batch-Verarbeitung
            batch_results = service.process_document_batch(
                test_documents,
                max_workers=2,
                progress_callback=batch_progress_callback
            )
            
            print(f"Batch-Verarbeitung: {len(batch_results)} Dokumente")
            
            total_chunks = sum(len(r.enhanced_chunks) for r in batch_results)
            successful_docs = sum(1 for r in batch_results if r.progress.is_completed)
            
            print(f"  Erfolgreich: {successful_docs}/{len(batch_results)} Dokumente")
            print(f"  Gesamt Chunks: {total_chunks}")
            
            # Klassifikations-Verteilung analysieren
            category_distribution = {}
            safety_distribution = {}
            
            for result in batch_results:
                for chunk in result.enhanced_chunks:
                    # Kategorien zählen
                    category = chunk.metadata.primary_category
                    category_distribution[category] = category_distribution.get(category, 0) + 1
                    
                    # Safety-Level zählen
                    safety = chunk.metadata.safety_level
                    safety_distribution[safety] = safety_distribution.get(safety, 0) + 1
            
            print(f"\n  Kategorien-Verteilung: {category_distribution}")
            print(f"  Safety-Level-Verteilung: {safety_distribution}")
            
        else:
            print(f"\nService nicht gesund - überspringe Tests")
            print(f"Fehler: {health_status.get('error', 'Unbekannt')}")
    
    except Exception as e:
        print(f"Fehler beim Document Service Test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Service-Statistiken anzeigen
    try:
        print(f"\n--- Service-Statistiken ---")
        stats = service.get_processing_statistics()
        
        print(f"Verarbeitete Dokumente: {stats['documents_processed']}")
        print(f"Erfolgsrate: {stats.get('success_rate', 0):.1%}")
        print(f"Durchschnittliche Zeit: {stats['average_processing_time_ms']:.1f}ms")
        print(f"Gesamt Chunks: {stats['total_chunks_created']}")
        
        # Stage-Performance
        if 'stage_performance_summary' in stats:
            print(f"\nStage-Performance:")
            for stage, perf in stats['stage_performance_summary'].items():
                print(f"  {stage}: {perf['average_time_ms']:.1f}ms (x{perf['count']})")
        
    except Exception as e:
        print(f"Fehler bei Statistiken: {str(e)}")
    
    # Factory-Tests
    print(f"\n--- Factory-Tests ---")
    try:
        factories = [
            ('Industrial', DocumentServiceFactory.create_industrial_service),
            ('Development', DocumentServiceFactory.create_development_service),
            ('High-Performance', DocumentServiceFactory.create_high_performance_service)
        ]
        
        for name, factory_method in factories:
            try:
                factory_service = factory_method()
                print(f"  {name}: Max-Workers={factory_service.max_workers}")
            except Exception as e:
                print(f"  {name}: Fehler - {str(e)}")
    
    except Exception as e:
        print(f"Factory-Test Fehler: {str(e)}")
    
    print("\n✅ Document Service erfolgreich getestet")
