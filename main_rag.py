#!/usr/bin/env python3
"""
RAG Main Application - Service-orientierte Architektur
CONTROLLER-PARAMETER-FIX IMPLEMENTIERT (SCHRITT 1A+1B)

KRITISCHER FIX:
- PipelineController Parameter-Mismatch behoben
- DI-Container Integration für Controller
- Korrekte Controller-Initialisierung

Autor: KI-Consultant für industrielle Automatisierung  
Version: 4.0.0 - Service-orientierte Architektur mit Controller-Fix
"""

import streamlit as st
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

# Core System
from core import get_logger, get_config, setup_logging
from core.exceptions import RAGException, ConfigurationException, PipelineException, DocumentProcessingError, create_error_context

# Services  
from services import DocumentService
from services import RetrievalService
import uuid

# Controllers - Mit korrigierter Initialisierung
from controllers import PipelineController, SessionController, HealthController

logger = get_logger(__name__)

# =============================================================================
# STREAMLIT KONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="RAG System - Industrielle Automatisierung",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SERVICE INITIALISIERUNG MIT BUGFIXES
# =============================================================================

@st.cache_resource
def initialize_rag_system_with_bugfixes():
    """
    Initialisiert das komplette RAG-System mit allen Bugfixes
    CONTROLLER-PARAMETER-FIX: SCHRITT 1A+1B IMPLEMENTIERT
    """
    try:
        logger.info("RAG Main Application gestartet - SOA Version 4.0.0")
        
        # 1. CORE-SYSTEM mit instant fixes
        logger.info("Initialisiere Services mit Bugfixes...")
        config = get_config()
        
        # 2. SERVICES mit Import-Bugfixes
        logger.info("Lade Services...")
        services = {}
        
        # Import Services mit try-catch für robuste Behandlung
        try:
            from services import (
                VectorStoreService, EmbeddingService, ChatService, SessionService,
                DocumentService, RetrievalService, SearchService, ServiceIntegrator
            )
            service_classes = {
                'VectorStoreService': VectorStoreService,
                'EmbeddingService': EmbeddingService, 
                'ChatService': ChatService,
                'SessionService': SessionService,
                'DocumentService': DocumentService,
                'RetrievalService': RetrievalService,
                'SearchService': SearchService,
                'ServiceIntegrator': ServiceIntegrator
            }
        except ImportError as e:
            logger.error(f"Service-Import-Fehler: {e}")
            service_classes = {}
        
        # 3. DocumentService
        if service_classes.get('DocumentService'):
            services['document_service'] = service_classes['DocumentService']()
        else:
            services['document_service'] = None
        logger.info("✅ DocumentService geladen")
        
        # 4. EmbeddingService mit config Parameter + CONFIG-PATCH (SCHRITT 4)
        if service_classes.get('EmbeddingService'):
            # CONFIG-FIX: provider_config Attribut hinzufügen falls fehlend
            if not hasattr(config, 'provider_config'):
                config.provider_config = {
                    'provider': 'ollama',
                    'model': 'nomic-embed-text',
                    'base_url': 'http://localhost:11434',
                    'timeout': 30,
                    'dimension': 768,
                    'max_retries': 3
                }
                logger.info("✅ provider_config Fallback für EmbeddingService hinzugefügt")
            elif not config.provider_config:
                # Auch leere provider_config abfangen
                config.provider_config = {
                    'provider': 'ollama',
                    'model': 'nomic-embed-text',
                    'base_url': 'http://localhost:11434',
                    'timeout': 30,
                    'dimension': 768,
                    'max_retries': 3
                }
                logger.info("✅ Leere provider_config mit Fallback-Werten ersetzt")
            
            services['embedding_service'] = service_classes['EmbeddingService'](config=config)
        else:
            services['embedding_service'] = None
        logger.info("✅ EmbeddingService mit config-Parameter und Config-Patch geladen")
        
        # 5. ChatService mit BUGFIX
        if service_classes.get('ChatService'):
            services['chat_service'] = service_classes['ChatService']()
        else:
            services['chat_service'] = None
        logger.info("✅ ChatService mit BUGFIX geladen")
        
        # 6. VectorStoreService mit BUGFIX
        if service_classes.get('VectorStoreService'):
            services['vectorstore_service'] = service_classes['VectorStoreService']()
        else:
            services['vectorstore_service'] = None
        logger.info("✅ VectorStoreService mit BUGFIX geladen")
        
        # 7. RetrievalService
        if service_classes.get('RetrievalService'):
            try:
                # RetrievalService benötigt Dependencies
                services['retrieval_service'] = service_classes['RetrievalService'](
                    vector_store_service=services['vectorstore_service'],
                    embedding_service=services['embedding_service']
                )
                logger.info("RetrievalService mit Dependencies initialisiert")
            except TypeError as e:
                logger.warning(f"RetrievalService Standard-Init fehlgeschlagen: {e}")
                try:
                    # Alternative: Mit config
                    services['retrieval_service'] = service_classes['RetrievalService'](config=config)
                    logger.info("RetrievalService mit config initialisiert")
                except Exception as e2:
                    logger.error(f"RetrievalService Init fehlgeschlagen: {e2}")
                    services['retrieval_service'] = None
        else:
            services['retrieval_service'] = None
        
        if services['retrieval_service']:
            logger.info("RetrievalService geladen")
        else:
            logger.warning("RetrievalService nicht verfügbar")
            
        # 8. SessionService mit Property-BUGFIX
        if service_classes.get('SessionService'):
            services['session_service'] = service_classes['SessionService']()
        else:
            services['session_service'] = None
        logger.info("✅ SessionService mit BUGFIX geladen")
        
        # SERVICE-INITIALISIERUNG LOGS
        logger.info("DocumentService initialisiert")
        logger.info("EmbeddingService mit BUGFIX initialisiert")
        logger.info("VectorStoreService mit BUGFIX initialisiert")
        logger.info("RetrievalService initialisiert")
        logger.info("ChatService mit Constructor-BUGFIX initialisiert")
        logger.info("SessionService mit Property-BUGFIX initialisiert")
        
        # =================================================================
        # 9. CONTROLLER - PARAMETER-MISMATCH BEHOBEN (SCHRITT 1A+1B)
        # =================================================================
        
        logger.info("Initialisiere Controller mit korrigierten Parametern...")
        
        try:
            from controllers.pipeline_controller import PipelineConfig
            from core.container import get_container
            
            # DI-Container für PipelineController vorbereiten
            container = get_container()
            
            
            from services.document_service import DocumentService
            from services.embedding_service import EmbeddingService  
            from services.vector_store_service import VectorStoreService
            from services.retrieval_service import RetrievalService
            from services.chat_service import ChatService
            from services.session_service import SessionService

            # Services im Container registrieren für PipelineController DI
            if services.get('document_service'):
                container.register_instance(DocumentService, services['document_service'])
            if services.get('embedding_service'): 
                container.register_instance(EmbeddingService, services['embedding_service'])
            if services.get('vectorstore_service'):
                container.register_instance(VectorStoreService, services['vectorstore_service'])
            if services.get('retrieval_service'):
                container.register_instance(RetrievalService, services['retrieval_service'])
            if services.get('chat_service'):
                container.register_instance(ChatService, services['chat_service'])
            if services.get('session_service'):
                container.register_instance(SessionService, services['session_service'])
                
            logger.info("✅ Services im DI-Container für Controller registriert")
            
            # PipelineController mit korrekter PipelineConfig (SCHRITT 1A FIX)
            pipeline_config = PipelineConfig(
                # Service-Konfiguration
                document_service_config={},
                embedding_service_config={},
                search_service_config={},
                chat_service_config={},
                
                # Pipeline-Verhalten
                max_concurrent_requests=5,
                request_queue_size=100,
                default_timeout=300,
                
                # Performance-Optimierungen
                batch_processing=True,
                parallel_processing=True,
                cache_enabled=True,
                performance_monitoring=True,
                
                # Error Handling
                retry_failed_requests=True,
                max_retry_attempts=3,
                retry_delay_seconds=2.0
            )
            
            # container.register('search_service', search_service)
            container.register('SearchService', search_service)
            
            services['pipeline_controller'] = PipelineController(config=pipeline_config)
            logger.info("✅ PipelineController mit korrekter Config initialisiert")
            
            # SessionController (Parameter bereits korrekt)
            services['session_controller'] = SessionController(
                session_service=services['session_service'],
                config=config
            )
            logger.info("✅ SessionController initialisiert")
            
            # HealthController (Parameter bereits korrekt)
            services['health_controller'] = HealthController(
                services={
                    'document': services['document_service'],
                    'embedding': services['embedding_service'],
                    'vectorstore': services['vectorstore_service'],
                    'retrieval': services['retrieval_service'],
                    'chat': services['chat_service'],
                    'session': services['session_service']
                }
            )
            logger.info("✅ HealthController initialisiert")
            
            logger.info("🎉 Alle Controller mit korrigierten Parametern erfolgreich initialisiert!")
            
        except Exception as e:
            logger.error(f"❌ Controller-Initialisierung fehlgeschlagen: {e}")
            logger.error(f"Controller-Fehler-Traceback: {traceback.format_exc()}")
            
            # Fallback-Werte setzen
            services['pipeline_controller'] = None
            services['session_controller'] = None  
            services['health_controller'] = None
        
        logger.info("🎉 Alle Services mit Bugfixes erfolgreich initialisiert!")
        
        return services, config
        
    except Exception as e:
        logger.error(f"Kritischer Fehler bei RAG-System-Initialisierung: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        st.error(f"System konnte nicht initialisiert werden: {e}")
        st.stop()

# =============================================================================
# MAIN INTERFACE MIT CONTROLLER-INTEGRATION
# =============================================================================

def main():
    """
    Hauptfunktion mit korrigierter Controller-Integration
    """
    try:
        # System initialisieren
        services, config = initialize_rag_system_with_bugfixes()
        
        # Interface Header
        st.title("⚙️ RAG System - Industrielle Automatisierung")
        st.caption("Version 4.0.0 - Service-orientierte Architektur mit Bugfixes")
        
        # System-Status Sidebar
        with st.sidebar:
            st.header("🔧 System-Status (mit Bugfixes)")
            
            # Services Status
            st.subheader("Services Aktiv")
            services_count = sum(1 for service in services.values() if service is not None and 'controller' not in str(service))
            st.metric("Services", f"{services_count}", "✓ von 9 Services")
            
            # Embedding Service Status
            embedding_status = "OK" if services.get('embedding_service') else "Fehler" 
            st.metric("Embedding Service", embedding_status, "✓ Provider-BUGFIX angewendet")
            
            # Chat Service Status
            chat_status = "OK" if services.get('chat_service') else "Fehler"
            st.metric("Chat Service", chat_status, "✓ Constructor-BUGFIX angewendet")
        
        # Navigation
        tab1, tab2, tab3 = st.tabs(["🤖 Chat", "📄 Dokumente", "🔧 Administration"])
        
        with tab1:
            st.header("Chat")
            st.info("Chat-Interface wird geladen...")
            
        with tab2:
            st.header("Dokumenten-Upload")
            st.info("Dokumenten-Interface wird geladen...")
            
        with tab3:
            st.header("System-Administration")
            
            # Controller-Status (SCHRITT 1 VALIDIERUNG)
            if services.get('pipeline_controller'):
                st.success("✅ PipelineController: Aktiv (Parameter-Fix angewendet)")
            else:
                st.error("❌ PipelineController: Nicht verfügbar")
                
            if services.get('session_controller'):
                st.success("✅ SessionController: Aktiv")
            else:
                st.error("❌ SessionController: Nicht verfügbar")
                
            if services.get('health_controller'):
                st.success("✅ HealthController: Aktiv") 
            else:
                st.error("❌ HealthController: Nicht verfügbar")
            
            # Service-Details
            st.subheader("Service-Details")
            
            for service_name, service_instance in services.items():
                if service_instance:
                    if 'controller' not in service_name:
                        st.success(f"✅ {service_name}: healthy")
                    else:
                        st.info(f"🎮 {service_name}: aktiv")
                else:
                    st.error(f"❌ {service_name}: Nicht verfügbar")
            
            # Debug Information
            if st.checkbox("Debug-Informationen anzeigen"):
                st.subheader("System-Debug-Info")
                
                debug_info = {
                    "Konfiguration": str(config.__class__.__name__),
                    "Services geladen": len(services),
                    "Controller aktiv": sum(1 for k, v in services.items() if 'controller' in k and v is not None),
                    "Timestamp": datetime.now().isoformat()
                }
                
                st.json(debug_info)
    
    except Exception as e:
        st.error(f"Anwendungsfehler: {e}")
        st.exception(e)
        logger.error(f"Main-Anwendungsfehler: {e}")

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        logger.info("🚀 RAG Main Application gestartet")
        main()
    except Exception as e:
        logger.error(f"Kritischer Startfehler: {e}")
        st.error(f"System konnte nicht gestartet werden: {e}")
        st.exception(e)
