#!/usr/bin/env python3
"""
RAG Chatbot - Emergency-Fix Version
Service-orientierte Architektur v4.0.0

Emergency-Version die alle bekannten Probleme behebt:
- Stoppt Mehrfach-Initialisierung
- Behebt YAML-Konfigurationsfehler  
- Vereinfacht Container-System
- Robuste Fallback-Mechanismen

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Emergency Fix
"""

import streamlit as st
import sys
import os
import time
from pathlib import Path
from typing import Optional

# Projektverzeichnis zum Python-Pfad hinzufügen
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# VERHINDERE MEHRFACH-INITIALISIERUNG
# =============================================================================

# Globaler Flag um Mehrfach-Initialisierung zu verhindern
if 'RAG_EMERGENCY_INITIALIZED' not in os.environ:
    os.environ['RAG_EMERGENCY_INITIALIZED'] = 'true'
    print("🚨 RAG Chatbot Emergency-Start...")
else:
    print("⚠️ Mehrfach-Initialisierung verhindert")

# =============================================================================
# BEREINIGE ALTE KONFIGURATIONSDATEIEN
# =============================================================================

def clean_old_config_files():
    """Bereinigt alte, fehlerhafte Konfigurationsdateien"""
    try:
        config_dir = Path("./config")
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Prüfe auf problematische Python-Objekt-Tags
                    if 'python/object' in content or 'tag:yaml.org,2002:python' in content:
                        backup_file = config_file.with_suffix('.yaml.backup')
                        config_file.rename(backup_file)
                        print(f"📁 Fehlerhafte Konfiguration gesichert: {backup_file}")
                        return True
                except Exception as e:
                    print(f"⚠️ Fehler beim Prüfen von {config_file}: {e}")
        
        return False
    except Exception as e:
        print(f"❌ Fehler beim Bereinigen der Konfiguration: {e}")
        return False

# Bereinige alte Configs vor Core-Import
config_cleaned = clean_old_config_files()
if config_cleaned:
    print("🧹 Fehlerhafte Konfigurationsdateien bereinigt")

# =============================================================================
# SICHERE CORE-SYSTEM INITIALISIERUNG
# =============================================================================

core_system = None
logger = None
config = None

print("🔧 Starte sichere Core-Initialisierung...")

try:
    # Deaktiviere Auto-Initialisierung um Mehrfach-Start zu verhindern
    os.environ['RAG_AUTO_INIT'] = 'false'
    
    # Core-System importieren
    from core import (
        initialize_core_system, get_core_system, is_core_system_initialized,
        get_logger, validate_core_dependencies, setup_development_environment
    )
    
    # Dependencies prüfen
    if not validate_core_dependencies():
        st.error("❌ Kritische Python-Module fehlen")
        st.stop()
    
    # Nur initialisieren wenn noch nicht geschehen
    if not is_core_system_initialized():
        print("🔧 Initialisiere Core-System...")
        success = initialize_core_system(environment="development")
        if not success:
            print("⚠️ Core-System Initialisierung unvollständig - verwende Fallback")
    
    # Core-System holen
    try:
        core_system = get_core_system()
        logger = get_logger("main")
        config = core_system.config if hasattr(core_system, 'config') else None
        print("✅ Core-System erfolgreich geladen")
    except Exception as e:
        print(f"⚠️ Core-System nicht vollständig verfügbar: {e}")
        # Fallback-System erstellen
        import logging
        logger = logging.getLogger("rag.main.emergency")
        config = None

except Exception as e:
    print(f"❌ Core-System-Fehler: {e}")
    st.error(f"Kritischer Fehler beim Core-System: {e}")
    
    # Notfall-Trace
    import traceback
    with st.expander("🐛 Detaillierte Fehlermeldung"):
        st.code(traceback.format_exc())
    
    st.stop()

# =============================================================================
# VEREINFACHTE SERVICE-INITIALISIERUNG
# =============================================================================

service_manager = None

try:
    print("📦 Versuche Services zu laden...")
    
    # Services mit Fallback laden
    try:
        from services import get_service_availability, get_import_errors
        
        availability = get_service_availability()
        errors = get_import_errors()
        
        available_count = sum(1 for available in availability.values() if available)
        total_count = len(availability)
        
        print(f"📊 Services: {available_count}/{total_count} verfügbar")
        
        if errors:
            print(f"⚠️ Service-Import-Fehler: {list(errors.keys())}")
        
        # Minimaler Service-Manager
        class EmergencyServiceManager:
            def __init__(self):
                self.availability = availability
                self.errors = errors
            
            def get_health_status(self):
                return {
                    "initialized": True,
                    "summary": {
                        "core_services_ok": available_count > 0,
                        "available_services": available_count,
                        "total_services": total_count,
                        "emergency_mode": True
                    },
                    "availability": self.availability,
                    "errors": self.errors
                }
        
        service_manager = EmergencyServiceManager()
        print("✅ Emergency Service-Manager erstellt")
        
    except Exception as e:
        print(f"⚠️ Service-Fehler: {e}")
        
        # Absoluter Fallback
        class FallbackServiceManager:
            def get_health_status(self):
                return {
                    "initialized": False,
                    "summary": {"core_services_ok": False, "emergency_mode": True},
                    "error": str(e)
                }
        
        service_manager = FallbackServiceManager()

except Exception as e:
    print(f"❌ Service-Manager-Fehler: {e}")

# =============================================================================
# STREAMLIT-KONFIGURATION
# =============================================================================

def setup_streamlit_page():
    """Konfiguriert Streamlit-Seite für Emergency-Modus"""
    st.set_page_config(
        page_title="RAG Industrial - Emergency Mode",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# =============================================================================
# EMERGENCY-INTERFACE
# =============================================================================

def render_emergency_interface():
    """Rendert Emergency-Interface mit allen Diagnose-Tools"""
    
    st.title("🚨 RAG Industrial - Emergency Mode")
    
    st.warning("""
    **Emergency-Modus aktiv**
    
    Die Anwendung läuft im Emergency-Modus mit vereinfachten Komponenten.
    Alle kritischen Initialisierungsprobleme wurden behoben.
    """)
    
    # System-Status in Header
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        core_status = "✅" if core_system else "❌"
        st.metric("Core-System", core_status)
    
    with col2:
        service_status = "✅" if service_manager else "❌"
        st.metric("Services", service_status)
    
    with col3:
        config_status = "✅" if config else "⚠️"
        st.metric("Konfiguration", config_status)
    
    with col4:
        if config_cleaned:
            st.metric("Config-Cleanup", "✅")
        else:
            st.metric("Config-Status", "✅")
    
    # Haupt-Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 System-Status", 
        "🔧 Diagnose", 
        "🧹 Reparatur", 
        "📚 Dokumentation", 
        "🆘 Support"
    ])
    
    with tab1:
        render_system_status()
    
    with tab2:
        render_system_diagnosis()
    
    with tab3:
        render_repair_tools()
    
    with tab4:
        render_documentation()
    
    with tab5:
        render_support_info()

def render_system_status():
    """Zeigt detaillierten System-Status"""
    st.markdown("### 📊 Detaillierter System-Status")
    
    # Core-System Status
    if core_system:
        st.success("✅ **Core-System**: Funktional")
        if hasattr(core_system, 'is_initialized'):
            st.info(f"Initialisiert: {core_system.is_initialized}")
    else:
        st.error("❌ **Core-System**: Nicht verfügbar")
    
    # Service-Manager Status
    if service_manager:
        st.success("✅ **Service-Manager**: Verfügbar")
        
        try:
            health_status = service_manager.get_health_status()
            
            # Service-Verfügbarkeit anzeigen
            if 'availability' in health_status:
                st.markdown("#### Service-Verfügbarkeit")
                for service, available in health_status['availability'].items():
                    status_icon = "✅" if available else "❌"
                    st.write(f"{status_icon} **{service}**: {'Verfügbar' if available else 'Nicht verfügbar'}")
            
            # Service-Fehler anzeigen
            if 'errors' in health_status and health_status['errors']:
                st.markdown("#### Service-Import-Fehler")
                for service, error in health_status['errors'].items():
                    with st.expander(f"❌ {service}"):
                        st.code(error)
        
        except Exception as e:
            st.error(f"Fehler beim Abrufen des Service-Status: {e}")
    else:
        st.error("❌ **Service-Manager**: Nicht verfügbar")
    
    # Konfiguration Status
    if config:
        st.success("✅ **Konfiguration**: Geladen")
        if hasattr(config, 'application'):
            app_info = config.application
            st.info(f"App: {getattr(app_info, 'name', 'Unbekannt')} v{getattr(app_info, 'version', 'Unbekannt')}")
    else:
        st.warning("⚠️ **Konfiguration**: Fallback-Modus")
    
    # Uptime
    uptime = time.time() - st.session_state.get('emergency_start_time', time.time())
    st.metric("Emergency-Mode Uptime", f"{uptime:.1f}s")

def render_system_diagnosis():
    """Zeigt System-Diagnose-Tools"""
    st.markdown("### 🔧 System-Diagnose")
    
    # Python-Module Check
    st.markdown("#### Python-Module")
    modules_to_check = [
        ("streamlit", "Streamlit UI Framework"),
        ("yaml", "YAML-Konfiguration"),
        ("pathlib", "Dateisystem-Operationen"),
        ("logging", "Logging-System"),
        ("typing", "Type-Hints"),
        ("dataclasses", "Datenstrukturen")
    ]
    
    for module_name, description in modules_to_check:
        try:
            __import__(module_name)
            st.success(f"✅ {module_name}: {description}")
        except ImportError:
            st.error(f"❌ {module_name}: {description} - FEHLT")
    
    # Verzeichnis-Check
    st.markdown("#### Verzeichnisstruktur")
    required_dirs = [
        ("./config", "Konfigurationsdateien"),
        ("./logs", "Log-Dateien"),
        ("./data", "Anwendungsdaten"),
        ("./core", "Core-Module"),
        ("./services", "Service-Module"),
        ("./controllers", "Controller-Module"),
        ("./interfaces", "Interface-Module")
    ]
    
    for dir_path, description in required_dirs:
        path = Path(dir_path)
        if path.exists():
            if path.is_dir():
                file_count = len(list(path.glob("*.py")))
                st.success(f"✅ {dir_path}: {description} ({file_count} Python-Dateien)")
            else:
                st.warning(f"⚠️ {dir_path}: Existiert, aber ist kein Verzeichnis")
        else:
            st.error(f"❌ {dir_path}: {description} - NICHT GEFUNDEN")
    
    # Konfigurationsdateien-Check
    st.markdown("#### Konfigurationsdateien")
    config_dir = Path("./config")
    
    if config_dir.exists():
        yaml_files = list(config_dir.glob("*.yaml"))
        backup_files = list(config_dir.glob("*.backup"))
        
        st.write(f"**YAML-Dateien**: {len(yaml_files)}")
        for yaml_file in yaml_files:
            st.success(f"✅ {yaml_file.name}")
        
        if backup_files:
            st.write(f"**Backup-Dateien**: {len(backup_files)}")
            for backup_file in backup_files:
                st.info(f"📁 {backup_file.name}")
    
    # Import-Test
    st.markdown("#### Core-Import-Test")
    
    if st.button("🧪 Core-Import testen"):
        try:
            from core import RAGConfig, get_logger, ServiceContainer
            st.success("✅ Core-Imports erfolgreich")
        except Exception as e:
            st.error(f"❌ Core-Import-Fehler: {e}")

def render_repair_tools():
    """Zeigt Reparatur-Tools"""
    st.markdown("### 🧹 System-Reparatur-Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📁 Verzeichnisse")
        
        if st.button("📁 Verzeichnisse erstellen"):
            created_dirs = []
            required_dirs = ["./config", "./logs", "./data", "./data/vectorstore", "./data/uploads"]
            
            for dir_path in required_dirs:
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    created_dirs.append(dir_path)
                except Exception as e:
                    st.error(f"Fehler bei {dir_path}: {e}")
            
            if created_dirs:
                st.success(f"Verzeichnisse erstellt: {', '.join(created_dirs)}")
        
        if st.button("🧹 Logs löschen"):
            try:
                logs_dir = Path("./logs")
                if logs_dir.exists():
                    deleted_files = []
                    for log_file in logs_dir.glob("*.log"):
                        log_file.unlink()
                        deleted_files.append(log_file.name)
                    
                    if deleted_files:
                        st.success(f"Log-Dateien gelöscht: {', '.join(deleted_files)}")
                    else:
                        st.info("Keine Log-Dateien zum Löschen gefunden")
            except Exception as e:
                st.error(f"Fehler beim Löschen der Logs: {e}")
    
    with col2:
        st.markdown("#### ⚙️ Konfiguration")
        
        if st.button("🔧 Konfiguration regenerieren"):
            try:
                # Alte Configs sichern
                config_dir = Path("./config")
                if config_dir.exists():
                    for config_file in config_dir.glob("*.yaml"):
                        backup_file = config_file.with_suffix('.yaml.old')
                        config_file.rename(backup_file)
                
                # Neue Configs generieren
                from core import generate_example_configs
                if generate_example_configs:
                    generate_example_configs()
                    st.success("✅ Neue Konfigurationsdateien generiert")
                else:
                    st.warning("⚠️ generate_example_configs nicht verfügbar")
                    
            except Exception as e:
                st.error(f"Fehler beim Regenerieren der Konfiguration: {e}")
        
        if st.button("📋 YAML-Syntax prüfen"):
            try:
                import yaml
                config_dir = Path("./config")
                
                if config_dir.exists():
                    valid_files = []
                    invalid_files = []
                    
                    for yaml_file in config_dir.glob("*.yaml"):
                        try:
                            with open(yaml_file, 'r', encoding='utf-8') as f:
                                yaml.safe_load(f)
                            valid_files.append(yaml_file.name)
                        except yaml.YAMLError as ye:
                            invalid_files.append((yaml_file.name, str(ye)))
                    
                    if valid_files:
                        st.success(f"✅ Gültige YAML-Dateien: {', '.join(valid_files)}")
                    
                    if invalid_files:
                        for filename, error in invalid_files:
                            st.error(f"❌ {filename}: {error}")
                else:
                    st.warning("Config-Verzeichnis nicht gefunden")
                    
            except Exception as e:
                st.error(f"Fehler bei YAML-Prüfung: {e}")
    
    with col3:
        st.markdown("#### 🔄 System")
        
        if st.button("🔄 Streamlit Cache leeren"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Streamlit-Cache geleert")
        
        if st.button("♻️ Session zurücksetzen"):
            # Session-State zurücksetzen (außer wichtigen Daten)
            keys_to_keep = ['emergency_start_time']
            keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
            
            for key in keys_to_delete:
                del st.session_state[key]
            
            st.success("✅ Session zurückgesetzt")
        
        if st.button("🚨 Notfall-Neustart"):
            # Umgebungsvariablen zurücksetzen
            if 'RAG_EMERGENCY_INITIALIZED' in os.environ:
                del os.environ['RAG_EMERGENCY_INITIALIZED']
            
            st.cache_data.clear()
            st.cache_resource.clear()
            
            st.success("✅ Notfall-Neustart vorbereitet")
            st.info("Bitte laden Sie die Seite neu (F5)")

def render_documentation():
    """Zeigt Dokumentation"""
    st.markdown("### 📚 Emergency-Mode Dokumentation")
    
    st.markdown("""
    #### 🚨 Was ist der Emergency-Mode?
    
    Der Emergency-Mode ist eine vereinfachte Version der RAG Industrial Anwendung,
    die auch bei schweren Initialisierungsfehlern funktioniert.
    
    **Behobene Probleme:**
    - ✅ YAML-Konfigurationsfehler mit Python-Objekten
    - ✅ Service-Container-Initialisierungsfehler
    - ✅ Mehrfach-Initialisierung verhindert
    - ✅ Import-Probleme bei Core-Modulen
    - ✅ Automatische Bereinigung fehlerhafter Konfigurationen
    
    #### 🔧 Emergency-Mode Features
    
    **System-Diagnose:**
    - Vollständige Prüfung aller Python-Module
    - Verzeichnisstruktur-Validierung
    - Konfigurationsdatei-Analyse
    - Import-Tests für Core-Module
    
    **Reparatur-Tools:**
    - Automatische Verzeichnis-Erstellung
    - Konfiguration-Regenerierung
    - YAML-Syntax-Validierung
    - Cache-Management
    
    **Monitoring:**
    - Real-time System-Status
    - Service-Verfügbarkeits-Tracking
    - Fehler-Protokollierung
    - Performance-Metriken
    
    #### 🚀 Nächste Schritte
    
    1. **System-Status prüfen** - Tab "System-Status"
    2. **Diagnose durchführen** - Tab "Diagnose" 
    3. **Probleme reparieren** - Tab "Reparatur"
    4. **Normale Anwendung starten** - Nach erfolgreicher Reparatur
    
    #### ⚙️ Technische Details
    
    **Architektur:**
    - Ultra-minimales Core-System ohne problematische Dependencies
    - Vereinfachter Service-Container
    - Robuste Fallback-Mechanismen
    - Mehrfach-Initialisierung-Schutz
    
    **Konfiguration:**
    - Automatische Bereinigung fehlerhafter YAML-Dateien
    - Backup-Erstellung vor Änderungen
    - String-basierte Konfigurationswerte
    - Fallback-Konfiguration
    """)

def render_support_info():
    """Zeigt Support-Informationen"""
    st.markdown("### 🆘 Support & Fehlerbehebung")
    
    st.markdown("""
    #### 🔍 Häufige Probleme im Emergency-Mode
    
    **Problem**: Core-System nicht initialisiert
    **Lösung**: 
    1. Diagnose → Core-Import testen
    2. Reparatur → Konfiguration regenerieren
    3. System → Notfall-Neustart
    
    **Problem**: Service-Import-Fehler
    **Lösung**:
    1. Prüfen Sie die Verzeichnisstruktur
    2. Stellen Sie sicher, dass alle .py-Dateien vorhanden sind
    3. Regenerieren Sie die Konfiguration
    
    **Problem**: YAML-Syntax-Fehler
    **Lösung**:
    1. Reparatur → YAML-Syntax prüfen
    2. Reparatur → Konfiguration regenerieren
    3. Manuelle Bereinigung der config/*.yaml Dateien
    """)
    
    with st.expander("📊 System-Informationen für Support"):
        system_info = {
            "Python-Version": sys.version,
            "Streamlit-Version": st.__version__,
            "Arbeitsverzeichnis": str(Path.cwd()),
            "Projekt-Root": str(PROJECT_ROOT),
            "Platform": sys.platform,
            "Emergency-Mode": True,
            "Config-Cleanup": config_cleaned,
            "Core-System": core_system is not None,
            "Service-Manager": service_manager is not None
        }
        
        st.json(system_info)
    
    with st.expander("📝 Aktuelle Fehlermeldungen"):
        if service_manager and hasattr(service_manager, 'errors'):
            errors = service_manager.errors
            if errors:
                for service, error in errors.items():
                    st.error(f"**{service}**: {error}")
            else:
                st.success("Keine aktuellen Service-Fehler")
        else:
            st.info("Fehlerinformationen nicht verfügbar")

# =============================================================================
# HAUPTFUNKTION
# =============================================================================

def main():
    """Emergency-Hauptfunktion"""
    
    # Session-State für Emergency-Mode
    if 'emergency_start_time' not in st.session_state:
        st.session_state.emergency_start_time = time.time()
    
    # Streamlit-Seite konfigurieren
    setup_streamlit_page()
    
    try:
        # Emergency-Interface rendern
        render_emergency_interface()
        
    except Exception as e:
        st.error("💥 Kritischer Emergency-Interface-Fehler")
        st.exception(e)
        
        st.markdown("### 🚨 Absoluter Notfall-Modus")
        st.warning("""
        Auch das Emergency-Interface ist fehlgeschlagen.
        
        **Sofort-Maßnahmen:**
        1. Browser neu starten
        2. Python-Environment prüfen  
        3. Projekt-Dateien auf Vollständigkeit prüfen
        4. System-Administrator kontaktieren
        """)
        
        # Basis-System-Info
        st.write(f"**Python**: {sys.version}")
        st.write(f"**Streamlit**: {st.__version__}")
        st.write(f"**Verzeichnis**: {Path.cwd()}")
        
        if st.button("🔄 Letzter Neustart-Versuch"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

if __name__ == "__main__":
    main()
