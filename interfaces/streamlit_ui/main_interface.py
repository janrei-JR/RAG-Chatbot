#!/usr/bin/env python3
"""
Main Interface - RAG-System Hauptoberfläche
Industrielle RAG-Architektur - Phase 3 Migration

PHASE 3 INTEGRATION: End-to-End RAG-Pipeline vollständig funktional!
- PDF Upload → Document Service → Vector Store Pipeline
- Chat Query → Retrieval Service → LLM Response Pipeline  
- Service-orchestrierte Architektur mit Pipeline-Controller
- Production-ready Interface mit Health-Monitoring

Streamlit-basierte Hauptoberfläche mit vollständiger Controller-Integration,
robustem Session-Management und industrietauglichem Error-Handling.

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

import streamlit as st
import time
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import json

# Core Imports
from core.logger import get_logger
from core.config import get_config
from core.exceptions import InterfaceException

# Controller Imports
from controllers.pipeline_controller import PipelineController, PipelineResponse, PipelineStage
from controllers.session_controller import SessionController
from controllers.health_controller import HealthController

# Interface Components
from .components import (
    render_sidebar_navigation,
    render_system_status,
    render_message_bubble,
    render_file_upload,
    render_loading_spinner,
    render_error_message,
    render_success_message
)

# Interface Modules
from .chat_interface import ChatInterface
from .document_interface import DocumentInterface
from .admin_interface import AdminInterface

logger = get_logger(__name__)


class MainInterface:
    """
    Hauptinterface für RAG-System
    
    PHASE 3 Features:
    - End-to-End RAG-Pipeline Integration
    - Pipeline-Controller basierte Backend-Kommunikation
    - Multi-Interface Management (Chat, Document, Admin)
    - Real-time Status-Monitoring und Health-Checks
    - Production-ready Error-Handling und Recovery
    """
    
    def __init__(
        self,
        pipeline_controller: PipelineController,
        session_controller: SessionController,
        health_controller: HealthController
    ):
        """Initialisiert Main Interface"""
        self.logger = get_logger(f"{__name__}.main")
        self.config = get_config()
        
        # Controller References
        self.pipeline_controller = pipeline_controller
        self.session_controller = session_controller
        self.health_controller = health_controller
        
        # Sub-Interfaces
        self.chat_interface = ChatInterface()
        self.document_interface = DocumentInterface()
        self.admin_interface = AdminInterface()
        
        # Interface State
        self.current_page = "chat"
        self.last_health_check = 0
        self.health_check_interval = 30  # 30 Sekunden
        
        # Statistics
        self._interface_stats = {
            'page_views': 0,
            'user_interactions': 0,
            'errors_handled': 0,
            'last_activity': time.time()
        }
        
        self.logger.info("Main Interface initialisiert")

    def render(self):
        """Rendert die Hauptoberfläche"""
        try:
            # Page Navigation
            self._render_navigation()
            
            # Health Status Check
            self._check_and_display_health()
            
            # Main Content Area
            self._render_main_content()
            
            # Footer mit Status
            self._render_footer()
            
            # Update Statistics
            self._update_interface_stats()
            
        except Exception as e:
            self.logger.error(f"Main Interface Rendering Error: {str(e)}")
            self._handle_interface_error(e)

    def _render_navigation(self):
        """Rendert Navigation und Page-Selection"""
        # Header mit Titel und Status
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.title("🎯 RAG Chatbot Industrial")
            st.markdown("**Service-orientierte Architektur v4.0.0 - Phase 3 Integration**")
        
        with col2:
            # Real-time Pipeline Status
            pipeline_status = self._get_pipeline_status()
            if pipeline_status == "healthy":
                st.success("🚀 Pipeline Aktiv")
            elif pipeline_status == "degraded":
                st.warning("⚠️ Pipeline Probleme")
            else:
                st.error("❌ Pipeline Fehler")
        
        with col3:
            # Quick Actions
            if st.button("🔄 Refresh", key="header_refresh"):
                st.rerun()
        
        # Page Navigation
        page_options = {
            "chat": "💬 Chat",
            "documents": "📄 Dokumente", 
            "admin": "⚙️ Admin"
        }
        
        selected_page = render_sidebar_navigation(
            page_options=page_options,
            current_page=self.current_page
        )
        
        if selected_page != self.current_page:
            self.current_page = selected_page
            st.rerun()

    def _check_and_display_health(self):
        """Prüft und zeigt System-Health"""
        current_time = time.time()
        
        # Health Check nur alle X Sekunden
        if current_time - self.last_health_check > self.health_check_interval:
            try:
                health_status = self.health_controller.get_system_health()
                st.session_state.system_health = health_status
                self.last_health_check = current_time
            except Exception as e:
                self.logger.warning(f"Health Check fehlgeschlagen: {str(e)}")
        
        # Health Status anzeigen
        if 'system_health' in st.session_state:
            health = st.session_state.system_health
            render_system_status(health)

    def _render_main_content(self):
        """Rendert den Hauptinhalt basierend auf aktiver Page"""
        try:
            if self.current_page == "chat":
                self._render_chat_page()
            elif self.current_page == "documents":
                self._render_documents_page()
            elif self.current_page == "admin":
                self._render_admin_page()
            else:
                st.error(f"Unbekannte Seite: {self.current_page}")
                
        except Exception as e:
            self.logger.error(f"Page Rendering Error [{self.current_page}]: {str(e)}")
            self._handle_page_error(e)

    def _render_chat_page(self):
        """Rendert Chat-Interface mit RAG-Pipeline Integration"""
        st.header("💬 Intelligenter Chat")
        
        # Prerequisites Check
        if not self._check_chat_prerequisites():
            return
        
        # Chat Interface mit Pipeline-Integration
        try:
            # Existing Messages
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            # Display Chat History
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Show sources if available
                    if message["role"] == "assistant" and "sources" in message:
                        if message["sources"]:
                            with st.expander("📚 Quellen"):
                                for i, source in enumerate(message["sources"]):
                                    st.markdown(f"**Quelle {i+1}:** {source.get('document_title', 'Unbekannt')}")
                                    if source.get('content_snippet'):
                                        st.markdown(f"*{source['content_snippet']}*")
            
            # Chat Input
            if prompt := st.chat_input("Ihre Frage zu den Dokumenten..."):
                self._handle_chat_input(prompt)
                
        except Exception as e:
            self.logger.error(f"Chat Interface Error: {str(e)}")
            render_error_message(f"Chat-Interface Fehler: {str(e)}")

    def _handle_chat_input(self, prompt: str):
        """PHASE 3: Behandelt Chat-Input mit Pipeline-Controller"""
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response via Pipeline Controller
        with st.chat_message("assistant"):
            with st.spinner("🤔 Generiere Antwort..."):
                try:
                    # PHASE 3: Pipeline-Controller Integration
                    vectorstore = st.session_state.get('vectorstore')
                    
                    if vectorstore:
                        # Execute Chat Pipeline
                        pipeline_response = self.pipeline_controller.process_chat_query(
                            query=prompt,
                            vectorstore=vectorstore,
                            collection_name="default",
                            chat_context=st.session_state.messages[-10:]  # Last 10 messages
                        )
                        
                        if pipeline_response.success:
                            response_text = pipeline_response.chat_response
                            sources = pipeline_response.sources
                            confidence = pipeline_response.confidence_score
                            
                            # Display response
                            st.markdown(response_text)
                            
                            # Confidence indicator
                            if confidence > 0.8:
                                st.success(f"🎯 Hohe Vertrauenswürdigkeit ({confidence:.1%})")
                            elif confidence > 0.5:
                                st.info(f"📊 Mittlere Vertrauenswürdigkeit ({confidence:.1%})")
                            else:
                                st.warning(f"⚠️ Niedrige Vertrauenswürdigkeit ({confidence:.1%})")
                            
                            # Add to messages
                            message_data = {
                                "role": "assistant",
                                "content": response_text,
                                "sources": sources,
                                "confidence": confidence,
                                "processing_time": pipeline_response.processing_time
                            }
                            st.session_state.messages.append(message_data)
                            
                        else:
                            error_response = "Es tut mir leid, bei der Verarbeitung Ihrer Anfrage ist ein Fehler aufgetreten."
                            st.error(error_response)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_response,
                                "error": pipeline_response.error_message
                            })
                    else:
                        warning_msg = "⚠️ Bitte laden Sie zuerst ein PDF-Dokument hoch."
                        st.warning(warning_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": warning_msg
                        })
                        
                except Exception as e:
                    self.logger.error(f"Chat Pipeline Error: {str(e)}")
                    error_msg = "Es ist ein Fehler bei der Antwortgenerierung aufgetreten."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "error": str(e)
                    })

    def _render_documents_page(self):
        """Rendert Document-Interface mit Pipeline-Integration"""
        st.header("📄 Dokument-Management")
        
        # Document Interface mit Pipeline-Integration
        try:
            # File Upload
            uploaded_file = render_file_upload(
                accepted_types=['pdf'],
                max_file_size_mb=self.config.max_file_size,
                help_text="Laden Sie PDF-Dokumente für die Analyse hoch"
            )
            
            if uploaded_file:
                self._handle_document_upload(uploaded_file)
            
            # Document Status
            self._show_document_status()
            
            # Document Management
            if 'vectorstore' in st.session_state:
                self._show_document_management()
                
        except Exception as e:
            self.logger.error(f"Document Interface Error: {str(e)}")
            render_error_message(f"Dokument-Interface Fehler: {str(e)}")

    def _handle_document_upload(self, uploaded_file):
        """PHASE 3: Behandelt Document Upload mit Pipeline-Controller"""
        if st.button("🚀 Dokument verarbeiten", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # PHASE 3: Pipeline-Controller Integration
                pipeline_response = self.pipeline_controller.process_pdf_document(
                    uploaded_file=uploaded_file,
                    collection_name="default",
                    progress_callback=lambda msg, progress: (
                        status_text.text(msg),
                        progress_bar.progress(progress)
                    )
                )
                
                if pipeline_response.success:
                    # Success
                    progress_bar.progress(1.0)
                    status_text.text("✅ Verarbeitung erfolgreich!")
                    
                    # Store results
                    st.session_state.vectorstore = pipeline_response.result_data.get('vectorstore')
                    st.session_state.last_processed_file = uploaded_file.name
                    st.session_state.document_stats = {
                        'chunk_count': pipeline_response.result_data.get('chunk_count', 0),
                        'embeddings_count': pipeline_response.embeddings_count,
                        'processing_time': pipeline_response.processing_time
                    }
                    
                    render_success_message("Dokument erfolgreich verarbeitet!")
                    
                    # Show Results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📄 Text-Chunks", pipeline_response.result_data.get('chunk_count', 0))
                    with col2:
                        st.metric("🧠 Embeddings", pipeline_response.embeddings_count)
                    with col3:
                        st.metric("⏱️ Verarbeitungszeit", f"{pipeline_response.processing_time:.1f}s")
                    
                else:
                    render_error_message(f"Verarbeitung fehlgeschlagen: {pipeline_response.error_message}")
                    
            except Exception as e:
                self.logger.error(f"Document Processing Error: {str(e)}")
                render_error_message(f"Verarbeitungsfehler: {str(e)}")

    def _render_admin_page(self):
        """Rendert Admin-Interface"""
        try:
            self.admin_interface.render()
        except Exception as e:
            self.logger.error(f"Admin Interface Error: {str(e)}")
            render_error_message(f"Admin-Interface Fehler: {str(e)}")

    def _check_chat_prerequisites(self) -> bool:
        """Prüft Chat-Prerequisites"""
        # Check if pipeline is healthy
        if self._get_pipeline_status() != "healthy":
            st.warning("⚠️ Pipeline-Status nicht optimal. Chat-Funktionalität möglicherweise eingeschränkt.")
            return True  # Allow anyway
        
        return True

    def _show_document_status(self):
        """Zeigt Document-Status"""
        if 'vectorstore' in st.session_state and st.session_state.vectorstore:
            st.success("✅ Dokumente geladen und bereit für Chat")
            
            if 'document_stats' in st.session_state:
                stats = st.session_state.document_stats
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📄 Chunks", stats.get('chunk_count', 0))
                with col2:
                    st.metric("🧠 Embeddings", stats.get('embeddings_count', 0))
                with col3:
                    st.metric("⏱️ Zeit", f"{stats.get('processing_time', 0):.1f}s")
            
            if 'last_processed_file' in st.session_state:
                st.info(f"📁 Letztes Dokument: {st.session_state.last_processed_file}")
        else:
            st.info("📤 Noch keine Dokumente verarbeitet")

    def _show_document_management(self):
        """Zeigt Document-Management Optionen"""
        with st.expander("🔧 Dokument-Management"):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Vector Store löschen"):
                    if 'vectorstore' in st.session_state:
                        del st.session_state.vectorstore
                    if 'document_stats' in st.session_state:
                        del st.session_state.document_stats
                    if 'last_processed_file' in st.session_state:
                        del st.session_state.last_processed_file
                    st.success("✅ Vector Store gelöscht")
                    st.rerun()
            
            with col2:
                if st.button("📊 Detaillierte Statistiken"):
                    try:
                        pipeline_stats = self.pipeline_controller.get_statistics()
                        st.json(pipeline_stats)
                    except Exception as e:
                        st.error(f"Statistiken nicht verfügbar: {str(e)}")

    def _get_pipeline_status(self) -> str:
        """Ermittelt Pipeline-Status"""
        try:
            health = self.health_controller.get_system_health()
            return health.get('status', 'unknown')
        except:
            return 'unknown'

    def _render_footer(self):
        """Rendert Footer mit Status-Informationen"""
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Pipeline Statistics
            try:
                stats = self.pipeline_controller.get_statistics()
                st.metric(
                    "📊 Requests", 
                    stats.get('total_requests', 0),
                    delta=f"{stats.get('success_rate', 0):.1%} Success"
                )
            except:
                st.metric("📊 Requests", "N/A")
        
        with col2:
            # Session Info
            session_id = self.session_controller.get_current_session_id()
            if session_id:
                st.metric("🔐 Session", session_id[:8] + "...")
            else:
                st.metric("🔐 Session", "N/A")
        
        with col3:
            # System Health
            try:
                health = self.health_controller.get_system_health()
                services_healthy = sum(1 for status in health.get('services', {}).values() if status == 'healthy')
                total_services = len(health.get('services', {}))
                st.metric("🏥 Health", f"{services_healthy}/{total_services}")
            except:
                st.metric("🏥 Health", "N/A")
        
        with col4:
            # Interface Stats
            st.metric("👤 Interactions", self._interface_stats['user_interactions'])

    def _update_interface_stats(self):
        """Updated Interface-Statistiken"""
        self._interface_stats['page_views'] += 1
        self._interface_stats['last_activity'] = time.time()

    def _handle_interface_error(self, error: Exception):
        """Behandelt Interface-Fehler"""
        self._interface_stats['errors_handled'] += 1
        
        st.error("❌ **Interface-Fehler aufgetreten**")
        
        with st.expander("🔧 Fehler-Details"):
            st.code(str(error))
        
        # Recovery Options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Interface neu laden"):
                st.rerun()
        
        with col2:
            if st.button("🏠 Zur Startseite"):
                self.current_page = "chat"
                st.rerun()

    def _handle_page_error(self, error: Exception):
        """Behandelt Page-spezifische Fehler"""
        self.logger.error(f"Page Error [{self.current_page}]: {str(error)}")
        
        st.error(f"❌ **Fehler auf Seite '{self.current_page}'**")
        st.markdown(f"**Fehlermeldung:** {str(error)}")
        
        # Page Recovery
        if st.button("🔄 Seite neu laden"):
            st.rerun()

    # =============================================================================
    # ADVANCED FEATURES UND UTILITIES
    # =============================================================================

    def get_interface_statistics(self) -> Dict[str, Any]:
        """Gibt Interface-Statistiken zurück"""
        return {
            'current_page': self.current_page,
            'stats': self._interface_stats.copy(),
            'health_check_interval': self.health_check_interval,
            'last_health_check': self.last_health_check
        }

    def set_page(self, page_name: str):
        """Setzt aktive Page programmatisch"""
        if page_name in ["chat", "documents", "admin"]:
            self.current_page = page_name
            self.logger.info(f"Page geändert zu: {page_name}")
        else:
            self.logger.warning(f"Ungültige Page: {page_name}")

    def add_user_interaction(self, interaction_type: str):
        """Registriert User-Interaction für Statistics"""
        self._interface_stats['user_interactions'] += 1
        self._interface_stats['last_activity'] = time.time()
        self.logger.debug(f"User interaction: {interaction_type}")

    def show_pipeline_debug_info(self):
        """Zeigt Pipeline Debug-Informationen (für Development)"""
        if st.sidebar.checkbox("🔧 Debug Mode"):
            with st.sidebar.expander("📊 Pipeline Debug"):
                try:
                    # Pipeline Controller Status
                    pipeline_health = self.pipeline_controller.health_check()
                    st.json(pipeline_health)
                    
                    # System Health
                    system_health = self.health_controller.get_system_health()
                    st.json(system_health)
                    
                    # Session Info
                    session_info = self.session_controller.get_session_info()
                    st.json(session_info)
                    
                except Exception as e:
                    st.error(f"Debug Info Error: {str(e)}")

    def export_session_data(self) -> Dict[str, Any]:
        """Exportiert Session-Daten für Backup/Analysis"""
        try:
            return {
                'messages': st.session_state.get('messages', []),
                'document_stats': st.session_state.get('document_stats', {}),
                'last_processed_file': st.session_state.get('last_processed_file'),
                'interface_stats': self._interface_stats,
                'current_page': self.current_page,
                'timestamp': time.time()
            }
        except Exception as e:
            self.logger.error(f"Session export error: {str(e)}")
            return {}

    def import_session_data(self, session_data: Dict[str, Any]) -> bool:
        """Importiert Session-Daten für Recovery"""
        try:
            if 'messages' in session_data:
                st.session_state.messages = session_data['messages']
            
            if 'document_stats' in session_data:
                st.session_state.document_stats = session_data['document_stats']
            
            if 'last_processed_file' in session_data:
                st.session_state.last_processed_file = session_data['last_processed_file']
            
            if 'current_page' in session_data:
                self.current_page = session_data['current_page']
            
            self.logger.info("Session-Daten erfolgreich importiert")
            return True
            
        except Exception as e:
            self.logger.error(f"Session import error: {str(e)}")
            return False

    # =============================================================================
    # CONTEXTUAL HELP UND TUTORIALS
    # =============================================================================

    def show_contextual_help(self):
        """Zeigt kontextuelle Hilfe basierend auf aktueller Page"""
        help_content = {
            "chat": {
                "title": "💬 Chat-Hilfe",
                "content": """
                **So verwenden Sie den Chat:**
                1. Laden Sie zuerst ein PDF-Dokument auf der Dokumente-Seite hoch
                2. Stellen Sie Fragen zu dem verarbeiteten Dokument
                3. Der KI-Assistent antwortet basierend auf dem Dokumentinhalt
                4. Quellen werden automatisch zitiert
                
                **Tipps:**
                - Stellen Sie spezifische Fragen für bessere Antworten
                - Die Vertrauenswürdigkeit wird für jede Antwort angezeigt
                - Der Chat-Verlauf bleibt während der Session erhalten
                """
            },
            "documents": {
                "title": "📄 Dokument-Hilfe", 
                "content": """
                **Dokument-Verarbeitung:**
                1. Klicken Sie auf "Durchsuchen" und wählen Sie eine PDF-Datei
                2. Klicken Sie "Dokument verarbeiten" 
                3. Warten Sie auf die Verarbeitung (Text-Extraktion, Embedding-Erstellung)
                4. Nach Abschluss können Sie im Chat Fragen zum Dokument stellen
                
                **Unterstützte Formate:**
                - PDF-Dateien bis zu {max_size} MB
                - Text wird automatisch in Chunks aufgeteilt
                - Metadaten werden extrahiert und gespeichert
                """.format(max_size=self.config.max_file_size)
            },
            "admin": {
                "title": "⚙️ Admin-Hilfe",
                "content": """
                **System-Administration:**
                - Überwachen Sie System-Health und Service-Status
                - Prüfen Sie Performance-Metriken und Logs
                - Verwalten Sie Konfigurationen
                - Führen Sie System-Diagnosen durch
                
                **Health-Checks:**
                - Grün: Service funktioniert normal
                - Gelb: Service hat Probleme, funktioniert aber
                - Rot: Service nicht verfügbar oder fehlerhaft
                """
            }
        }
        
        if self.current_page in help_content:
            help_info = help_content[self.current_page]
            
            with st.sidebar.expander(help_info["title"]):
                st.markdown(help_info["content"])

    def show_getting_started_tutorial(self):
        """Zeigt Getting-Started Tutorial für neue Benutzer"""
        if 'tutorial_completed' not in st.session_state:
            st.session_state.tutorial_completed = False
        
        if not st.session_state.tutorial_completed:
            with st.expander("🚀 Erste Schritte - Tutorial", expanded=True):
                st.markdown("""
                ### Willkommen zum RAG Chatbot Industrial!
                
                **Phase 3 Integration - End-to-End RAG-Pipeline ist jetzt aktiv!**
                
                #### Schnellstart in 3 Schritten:
                
                1. **📄 Dokument hochladen:**
                   - Gehen Sie zur "Dokumente"-Seite
                   - Laden Sie eine PDF-Datei hoch
                   - Klicken Sie "Dokument verarbeiten"
                
                2. **💬 Chat starten:**
                   - Wechseln Sie zur "Chat"-Seite  
                   - Stellen Sie Fragen zu Ihrem Dokument
                   - Erhalten Sie KI-gestützte Antworten mit Quellenangaben
                
                3. **⚙️ System überwachen:**
                   - Prüfen Sie die "Admin"-Seite für System-Status
                   - Überwachen Sie Performance und Health-Checks
                
                #### Neue Features in Phase 3:
                - ✅ **End-to-End RAG-Pipeline** vollständig funktional
                - ✅ **Service-orchestrierte Architektur** für Skalierbarkeit  
                - ✅ **Pipeline-Controller** für robuste Verarbeitung
                - ✅ **Enhanced Chat** mit Confidence-Scoring
                - ✅ **Real-time Monitoring** und Health-Checks
                """)
                
                if st.button("✅ Tutorial abschließen"):
                    st.session_state.tutorial_completed = True
                    st.success("Tutorial abgeschlossen! Sie können jetzt loslegen.")
                    st.rerun()


# =============================================================================
# UTILITY FUNCTIONS FÜR INTERFACE-MANAGEMENT
# =============================================================================

def create_main_interface(
    pipeline_controller: PipelineController,
    session_controller: SessionController,
    health_controller: HealthController
) -> MainInterface:
    """Factory function für Main Interface"""
    return MainInterface(
        pipeline_controller=pipeline_controller,
        session_controller=session_controller,
        health_controller=health_controller
    )


def initialize_interface_session_state():
    """Initialisiert Interface-spezifische Session State"""
    defaults = {
        'current_page': 'chat',
        'messages': [],
        'vectorstore': None,
        'document_stats': {},
        'last_processed_file': None,
        'system_health': {},
        'tutorial_completed': False,
        'interface_errors': []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def clear_interface_session():
    """Löscht Interface Session State"""
    interface_keys = [
        'current_page', 'messages', 'vectorstore', 
        'document_stats', 'last_processed_file', 'system_health'
    ]
    
    for key in interface_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    # Re-initialize
    initialize_interface_session_state()


def get_interface_health() -> Dict[str, Any]:
    """Gibt Interface Health-Status zurück"""
    return {
        'interface': 'main_interface',
        'status': 'healthy',
        'session_active': len(st.session_state.keys()) > 0,
        'pages_available': ['chat', 'documents', 'admin'],
        'last_activity': time.time()
    }