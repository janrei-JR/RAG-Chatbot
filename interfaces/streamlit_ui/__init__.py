#!/usr/bin/env python3
"""
Streamlit UI Package - Interface Layer Integration
Industrielle RAG-Architektur - Phase 4 Migration

Sammelt alle Streamlit-UI-Komponenten mit korrigierten Imports
und robuster Controller-Integration für die service-orientierte Architektur.

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

from typing import Dict, Any, Optional
import logging

# Core Imports - korrigierte Versionen
from core import RAGConfig as Config, get_logger, ServiceContainer, RAGSystemException
from controllers.pipeline_controller import PipelineController
from controllers.session_controller import SessionController
from controllers.health_controller import HealthController

logger = get_logger(__name__)


# =============================================================================
# UI KOMPONENTEN IMPORTS
# =============================================================================

try:
    # Main Interface Components
    from .main_interface import MainInterface
    from .chat_interface import ChatInterface
    from .document_interface import DocumentInterface
    
    # Admin Interface (falls vorhanden)
    try:
        from .admin_interface import AdminInterface
        ADMIN_INTERFACE_AVAILABLE = True
    except ImportError:
        logger.warning("AdminInterface nicht verfügbar - wird erstellt sobald implementiert")
        AdminInterface = None
        ADMIN_INTERFACE_AVAILABLE = False
    
    # UI Components
    from .components import (
        render_sidebar_navigation,
        render_system_status,
        render_message_bubble,
        render_file_upload,
        render_loading_spinner,
        render_error_message,
        render_success_message,
        setup_page_config
    )
    
    COMPONENTS_LOADED = True
    
except ImportError as e:
    logger.error(f"Fehler beim Laden der UI-Komponenten: {e}")
    COMPONENTS_LOADED = False


# =============================================================================
# INTERFACE FACTORY
# =============================================================================

class InterfaceFactory:
    """
    Factory für UI-Interface-Erstellung mit Dependency Injection
    """
    
    def __init__(self, service_container: ServiceContainer):
        self.container = service_container
        self.logger = get_logger(f"{__name__}.factory")
    
    def create_main_interface(self) -> Optional[Any]:
        """
        Erstellt MainInterface mit injizierten Dependencies
        
        Returns:
            MainInterface oder None bei Fehlern
        """
        try:
            # Controller aus Container holen
            pipeline_controller = self.container.get_service("pipeline_controller")
            session_controller = self.container.get_service("session_controller") 
            health_controller = self.container.get_service("health_controller")
            
            # MainInterface erstellen
            return MainInterface(
                pipeline_controller=pipeline_controller,
                session_controller=session_controller,
                health_controller=health_controller
            )
            
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des MainInterface: {e}")
            return None
    
    def create_chat_interface(self) -> Optional[ChatInterface]:
        """
        Erstellt ChatInterface mit injizierten Dependencies
        
        Returns:
            ChatInterface oder None bei Fehlern
        """
        try:
            return ChatInterface()
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des ChatInterface: {e}")
            return None
    
    def create_document_interface(self) -> Optional[DocumentInterface]:
        """
        Erstellt DocumentInterface mit injizierten Dependencies
        
        Returns:
            DocumentInterface oder None bei Fehlern
        """
        try:
            return DocumentInterface()
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des DocumentInterface: {e}")
            return None
    
    def create_admin_interface(self) -> Optional[AdminInterface]:
        """
        Erstellt AdminInterface falls verfügbar
        
        Returns:
            AdminInterface oder None wenn nicht verfügbar
        """
        if not ADMIN_INTERFACE_AVAILABLE:
            self.logger.warning("AdminInterface nicht verfügbar")
            return None
        
        try:
            return AdminInterface()
        except Exception as e:
            self.logger.error(f"Fehler beim Erstellen des AdminInterface: {e}")
            return None


# =============================================================================
# INTERFACE MANAGER
# =============================================================================

class InterfaceManager:
    """
    Zentraler Manager für alle UI-Interfaces mit Lifecycle-Management
    """
    
    def __init__(self, service_container: ServiceContainer):
        self.container = service_container
        self.factory = InterfaceFactory(service_container)
        self.logger = get_logger(f"{__name__}.manager")
        
        # Interface-Instances
        self._main_interface = None
        self._chat_interface = None
        self._document_interface = None
        self._admin_interface = None
        
        # Status-Tracking
        self._initialized = False
        self._health_status = {}
    
    def initialize(self) -> bool:
        """
        Initialisiert alle verfügbaren Interfaces
        
        Returns:
            bool: True wenn mindestens MainInterface erfolgreich
        """
        try:
            self.logger.info("Initialisiere UI-Interface-Manager...")
            
            # MainInterface ist kritisch
            self._main_interface = self.factory.create_main_interface()
            if not self._main_interface:
                self.logger.error("Kritischer Fehler: MainInterface konnte nicht erstellt werden")
                return False
            
            # Andere Interfaces sind optional
            self._chat_interface = self.factory.create_chat_interface()
            self._document_interface = self.factory.create_document_interface()
            
            if ADMIN_INTERFACE_AVAILABLE:
                self._admin_interface = self.factory.create_admin_interface()
            
            self._initialized = True
            
            # Status-Update
            self._update_health_status()
            
            self.logger.info("UI-Interface-Manager erfolgreich initialisiert")
            return True
            
        except Exception as e:
            self.logger.error(f"Fehler bei Interface-Manager-Initialisierung: {e}")
            return False
    
    def get_main_interface(self) -> Optional[MainInterface]:
        """Gibt MainInterface zurück"""
        return self._main_interface
    
    def get_chat_interface(self) -> Optional[ChatInterface]:
        """Gibt ChatInterface zurück"""
        return self._chat_interface
    
    def get_document_interface(self) -> Optional[DocumentInterface]:
        """Gibt DocumentInterface zurück"""
        return self._document_interface
    
    def get_admin_interface(self) -> Optional[AdminInterface]:
        """Gibt AdminInterface zurück"""
        return self._admin_interface
    
    def _update_health_status(self) -> None:
        """Aktualisiert Health-Status aller Interfaces"""
        self._health_status = {
            "main_interface": self._main_interface is not None,
            "chat_interface": self._chat_interface is not None,
            "document_interface": self._document_interface is not None,
            "admin_interface": self._admin_interface is not None,
            "components_loaded": COMPONENTS_LOADED,
            "admin_available": ADMIN_INTERFACE_AVAILABLE,
            "initialized": self._initialized
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Gibt aktuellen Health-Status zurück
        
        Returns:
            Dict mit Health-Status aller Komponenten
        """
        self._update_health_status()
        return self._health_status.copy()
    
    def is_healthy(self) -> bool:
        """
        Prüft ob Interface-Manager gesund ist
        
        Returns:
            bool: True wenn MainInterface verfügbar
        """
        return self._initialized and self._main_interface is not None


# =============================================================================
# GLOBALE FACTORY FUNCTIONS
# =============================================================================

def create_interface_manager(service_container: ServiceContainer) -> InterfaceManager:
    """
    Erstellt InterfaceManager mit Service-Container
    
    Args:
        service_container: Konfigurierter ServiceContainer
    
    Returns:
        InterfaceManager: Konfigurierter Interface-Manager
    """
    manager = InterfaceManager(service_container)
    
    if not manager.initialize():
        logger.warning("Interface-Manager konnte nicht vollständig initialisiert werden")
    
    return manager


def get_available_interfaces() -> Dict[str, bool]:
    """
    Gibt Verfügbarkeit aller Interface-Komponenten zurück
    
    Returns:
        Dict mit Verfügbarkeitsstatus
    """
    return {
        "main_interface": MainInterface is not None,
        "chat_interface": ChatInterface is not None,
        "document_interface": DocumentInterface is not None,
        "admin_interface": ADMIN_INTERFACE_AVAILABLE,
        "components": COMPONENTS_LOADED
    }


# =============================================================================
# FALLBACK IMPLEMENTATIONS
# =============================================================================

class DummyAdminInterface:
    """
    Dummy-Implementation für AdminInterface falls nicht verfügbar
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.dummy_admin")
        self.logger.warning("DummyAdminInterface verwendet - AdminInterface nicht implementiert")
    
    def render(self):
        """Dummy-Render-Methode"""
        import streamlit as st
        st.warning("🚧 Admin-Interface ist noch nicht implementiert")
        st.info("Das Admin-Interface wird in einer zukünftigen Version verfügbar sein.")


# Fallback falls AdminInterface nicht verfügbar
if not ADMIN_INTERFACE_AVAILABLE:
    AdminInterface = DummyAdminInterface


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main Classes
    'MainInterface', 'ChatInterface', 'DocumentInterface', 'AdminInterface',
    
    # Factory und Manager
    'InterfaceFactory', 'InterfaceManager',
    
    # Factory Functions
    'create_interface_manager', 'get_available_interfaces',
    
    # Status Constants
    'COMPONENTS_LOADED', 'ADMIN_INTERFACE_AVAILABLE',
    
    # Components (wenn verfügbar)
]

# Conditional Exports für Components
if COMPONENTS_LOADED:
    __all__.extend([
        'render_sidebar_navigation', 'render_system_status',
        'render_message_bubble', 'render_file_upload',
        'render_loading_spinner', 'render_error_message',
        'render_success_message', 'setup_page_config'
    ])


# =============================================================================
# LOGGING STATUS
# =============================================================================

if __name__ != "__main__":
    logger.info(f"Streamlit UI Package geladen - Components: {COMPONENTS_LOADED}, Admin: {ADMIN_INTERFACE_AVAILABLE}")


# =============================================================================
# MAININTERFACE FALLBACK - HOTFIX
# =============================================================================

class MainInterface:
    """Fallback MainInterface für UI-Kompatibilität"""
    
    def __init__(self):
        self.initialized = False
    
    def render(self):
        """Minimale Render-Funktion"""
        import streamlit as st
        st.warning("MainInterface im Fallback-Modus")
        st.info("Chat-Interface verfügbar über Navigation")
