#!/usr/bin/env python3
"""
Services Package - Clean Imports ohne Fallbacks
Version: 4.1.0 - Simplified (TODO #5)

ARCHITEKTUR:
- Fail Fast: Wenn Services fehlen → App crasht sofort
- Keine Fallbacks: Klare Fehlermeldungen statt Silent Failures
- 30 Zeilen statt 300: Maximale Einfachheit

AKTUELLER STAND (30.09.2025):
- ✅ Alle Core-Services importiert
- ✅ Fail-Fast-Pattern implementiert
- ✅ ServiceIntegrator optional
- ✅ get_services_status() Utility

BEKANNTE ISSUES:
- Keine
"""

from typing import Dict, Any

# Core imports
from core import get_logger

logger = get_logger("services")

# =============================================================================
# SERVICE IMPORTS - FAIL FAST (kein Fallback!)
# =============================================================================

try:
    from .document_service import DocumentService
    from .embedding_service import EmbeddingService
    from .vector_store_service import VectorStoreService
    from .retrieval_service import RetrievalService
    from .search_service import SearchService
    from .chat_service import ChatService
    from .session_service import SessionService
    
    logger.info("✅ Alle Services erfolgreich importiert")
    
except ImportError as e:
    logger.error(f"🔴 KRITISCHER SERVICE-IMPORT FEHLER: {e}")
    logger.error("Services können nicht geladen werden - App wird beendet")
    raise SystemExit(f"Service-Import fehlgeschlagen: {e}")

# ServiceIntegrator (optional)
try:
    from .service_integration import ServiceIntegrator
except ImportError:
    logger.warning("⚠️ ServiceIntegrator nicht verfügbar")
    ServiceIntegrator = None

# =============================================================================
# SERVICE STATUS
# =============================================================================

def get_services_status() -> Dict[str, Any]:
    """Liefert Status aller importierten Services"""
    return {
        'available_services': [
            'DocumentService',
            'EmbeddingService', 
            'VectorStoreService',
            'RetrievalService',
            'SearchService',
            'ChatService',
            'SessionService'
        ],
        'all_services_loaded': True,
        'integrator_available': ServiceIntegrator is not None
    }

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core Services
    'DocumentService',
    'EmbeddingService',
    'VectorStoreService',
    'RetrievalService',
    'SearchService',
    'ChatService',
    'SessionService',
    
    # Optional
    'ServiceIntegrator',
    
    # Utilities
    'get_services_status'
]

logger.info(f"Services module ready - {len(__all__) - 1} services verfügbar")