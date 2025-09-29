# interfaces/streamlit_ui/chat_interface.py
"""
Chat Interface - Modernisierte Chat-Komponente
Industrielle RAG-Architektur - Phase 4 Migration

Robuste Chat-UI mit Controller-Integration, verbessertem UX,
Source-Tracking und industrietauglichem Message-Handling.
"""

import streamlit as st
import time
from typing import Dict, Any, Optional, List
import json

# Core Imports
from core.logger import get_logger
from core.exceptions import InterfaceException

# Controller Imports
from controllers import (
    get_session_controller,
    get_pipeline_controller,
    process_query_request,
    create_query_request,
    PipelineResponse
)

# UI Components
from .components import (
    render_message_bubble,
    render_source_citations,
    render_loading_spinner,
    render_error_message,
    render_success_message,
    render_chat_input,
    render_chat_settings
)

logger = get_logger(__name__)


class ChatInterface:
    """
    Modernisierte Chat-Komponente für RAG-System
    
    Features:
    - Controller-basierte Backend-Integration
    - Robustes Session-Management für Conversation-History
    - Source-Citations mit Dokumenten-Links
    - Streaming-Responses mit Progress-Tracking
    - Erweiterte Chat-Einstellungen und Konfiguration
    - Industrial-Grade Error-Handling und Recovery
    """
    
    def __init__(self):
        """Initialisiert Chat Interface"""
        self.logger = get_logger(f"{__name__}.chat")
        
        # Controller References
        self.session_controller = get_session_controller()
        self.pipeline_controller = get_pipeline_controller()
        
        # Chat Configuration
        self.default_config = {
            'max_tokens': 1000,
            'temperature': 0.7,
            'top_k': 5,
            'include_sources': True,
            'stream_response': True,
            'auto_save_history': True
        }
        
        self.logger.info("Chat Interface initialisiert")

    def render(self, session_id: Optional[str] = None):
        """
        Rendert Chat-Interface
        
        Args:
            session_id: Optional Session-ID für History-Management
        """
        try:
            # Chat-Container
            chat_container = st.container()
            
            with chat_container:
                # Chat-Settings (Collapsible)
                self._render_chat_settings()
                
                # Chat-History
                self._render_chat_history(session_id)
                
                # Chat-Input
                self._render_chat_input(session_id)
                
                # Chat-Actions
                self._render_chat_actions(session_id)
        
        except Exception as e:
            self.logger.error(f"Chat-Interface Rendering-Fehler: {str(e)}", exc_info=True)
            render_error_message(
                "Chat-Fehler",
                f"Chat-Interface konnte nicht geladen werden: {str(e)}",
                details={'session_id': session_id}
            )

    def _render_chat_settings(self):
        """Rendert Chat-Konfiguration"""
        with st.expander("⚙️ Chat-Einstellungen", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Retrieval-Einstellungen
                st.markdown("**📋 Retrieval**")
                
                top_k = st.slider(
                    "Anzahl Quellen",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.get('chat_top_k', 5),
                    help="Anzahl der relevantesten Dokumente für die Antwort"
                )
                st.session_state.chat_top_k = top_k
                
                include_sources = st.checkbox(
                    "Quellen anzeigen",
                    value=st.session_state.get('chat_include_sources', True),
                    help="Zeigt Quellenangaben unter den Antworten"
                )
                st.session_state.chat_include_sources = include_sources
                
                # Index-Auswahl
                available_indices = self._get_available_indices()
                
                if available_indices:
                    selected_index = st.selectbox(
                        "Wissensbasis",
                        options=available_indices,
                        index=0,
                        help="Auswahl der zu durchsuchenden Dokumentensammlung"
                    )
                    st.session_state.chat_index = selected_index
                else:
                    st.warning("⚠️ Keine Indizes verfügbar. Laden Sie zuerst Dokumente hoch.")
                    st.session_state.chat_index = "default"
            
            with col2:
                # LLM-Einstellungen
                st.markdown("**🤖 LLM-Parameter**")
                
                temperature = st.slider(
                    "Kreativität",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get('chat_temperature', 0.7),
                    step=0.1,
                    help="Höhere Werte = kreativere, niedrigere = präzisere Antworten"
                )
                st.session_state.chat_temperature = temperature
                
                max_tokens = st.slider(
                    "Max. Antwort-Länge",
                    min_value=100,
                    max_value=2000,
                    value=st.session_state.get('chat_max_tokens', 1000),
                    step=100,
                    help="Maximale Länge der generierten Antworten"
                )
                st.session_state.chat_max_tokens = max_tokens
                
                # Erweiterte Optionen
                stream_response = st.checkbox(
                    "Streaming-Antworten",
                    value=st.session_state.get('chat_stream', True),
                    help="Zeigt Antworten während der Generierung an"
                )
                st.session_state.chat_stream = stream_response
            
            # Chat-Konfiguration speichern
            if st.button("💾 Einstellungen speichern", key="save_chat_settings"):
                self._save_chat_settings_to_session()

    def _render_chat_history(self, session_id: Optional[str]):
        """Rendert Conversation-History"""
        try:
            # Messages aus Session State und Controller laden
            messages = self._load_conversation_history(session_id)
            
            if not messages:
                # Willkommens-Nachricht
                self._render_welcome_message()
                return
            
            # Chat-Messages Container
            chat_messages_container = st.container()
            
            with chat_messages_container:
                for i, message in enumerate(messages):
                    self._render_message(message, message_index=i)
                
                # Auto-Scroll zum Ende (simuliert)
                if len(messages) > 0:
                    st.empty()  # Spacer für Auto-Scroll-Effekt
        
        except Exception as e:
            self.logger.error(f"Chat-History Rendering-Fehler: {str(e)}")
            st.error(f"❌ Fehler beim Laden der Chat-History: {str(e)}")

    def _render_welcome_message(self):
        pass
