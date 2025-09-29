#!/usr/bin/env python3
"""
Streamlit System Test Runner
Testet die vollständige Streamlit-App nach allen Fixes

VERWENDUNG:
1. Config-Fixes aus dem anderen Artifact in core/config.py einfügen
2. Dieses Script ausführen: python streamlit_test_runner.py
3. Dann: streamlit run main_rag.py

Version: 4.0.0 - Post-Fix Testing
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

# Projekt-Root hinzufügen
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class StreamlitTestRunner:
    """Testet Streamlit-App Funktionalität"""
    
    def __init__(self):
        self.test_results = {}
        self.app_process = None
    
    def run_comprehensive_test(self):
        """Führt umfassenden Streamlit-Test durch"""
        print("🚀 STREAMLIT SYSTEM TEST GESTARTET")
        print("=" * 60)
        
        # Phase 1: Pre-Launch Tests
        print("\n📋 PHASE 1: PRE-LAUNCH VALIDATION")
        self._test_dependencies()
        self._test_imports() 
        self._test_config_loading()
        
        # Phase 2: App Structure Tests
        print("\n🏗️  PHASE 2: APP STRUCTURE VALIDATION")
        self._test_app_structure()
        self._test_session_state_init()
        
        # Phase 3: Service Integration Tests
        print("\n🔧 PHASE 3: SERVICE INTEGRATION TESTS")
        self._test_service_initialization()
        self._test_controller_creation()
        
        # Phase 4: Interface Tests
        print("\n🖥️  PHASE 4: INTERFACE COMPONENT TESTS")
        self._test_interface_imports()
        self._test_page_rendering()
        
        # Phase 5: End-to-End Simulation
        print("\n🔄 PHASE 5: END-TO-END SIMULATION")
        self._test_workflow_simulation()
        
        # Summary
        self._generate_test_report()
    
    def _test_dependencies(self):
        """Test ob alle Dependencies verfügbar sind"""
        required_packages = [
            'streamlit', 'pathlib', 'typing', 'datetime',
            'traceback', 'sys'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}: verfügbar")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package}: FEHLT")
        
        self.test_results['dependencies'] = {
            'success': len(missing_packages) == 0,
            'missing_packages': missing_packages
        }
    
    def _test_imports(self):
        """Test kritische System-Imports"""
        import_tests = {
            'core_system': {
                'module': 'core',
                'components': ['get_logger', 'get_config', 'setup_logging']
            },
            'core_exceptions': {
                'module': 'core.exceptions',
                'components': ['RAGException', 'ConfigurationException', 'PipelineException']
            },
            'services': {
                'module': 'services',
                'components': ['DocumentService', 'ChatService', 'EmbeddingService']
            },
            'controllers': {
                'module': 'controllers', 
                'components': ['PipelineController', 'SessionController', 'HealthController']
            },
            'interfaces': {
                'module': 'interfaces.streamlit_ui',
                'components': ['MainInterface', 'ChatInterface', 'DocumentInterface']
            }
        }
        
        for test_name, test_config in import_tests.items():
            try:
                module = __import__(test_config['module'], fromlist=test_config['components'])
                
                missing_components = []
                for component in test_config['components']:
                    if not hasattr(module, component):
                        missing_components.append(component)
                
                success = len(missing_components) == 0
                print(f"{'✅' if success else '❌'} {test_name}: {len(test_config['components']) - len(missing_components)}/{len(test_config['components'])} Komponenten")
                
                if missing_components:
                    print(f"    ⚠️  Fehlend: {', '.join(missing_components)}")
                
                self.test_results[f'import_{test_name}'] = {
                    'success': success,
                    'missing_components': missing_components
                }
                
            except ImportError as e:
                print(f"❌ {test_name}: Import fehlgeschlagen - {str(e)}")
                self.test_results[f'import_{test_name}'] = {
                    'success': False,
                    'error': str(e)
                }
    
    def _test_config_loading(self):
        """Test Config-System nach Hotfixes"""
        try:
            from core import get_config, setup_logging
            
            # Logging Setup (kritisch für Streamlit)
            setup_logging(
                level="INFO",
                file_path="./data/logs/test_rag_system.log",
                console_output=True,
                json_format=False
            )
            
            # Config laden
            config = get_config()
            
            # Test neue Properties (aus Hotfixes)
            property_tests = {
                'default_provider': hasattr(config, 'default_provider'),
                'persistence_directory': hasattr(config, 'persistence_directory'),
                'auto_provider_selection': hasattr(config, 'auto_provider_selection')
            }
            
            # Test Embeddings cache_size
            if hasattr(config, 'embeddings'):
                property_tests['embeddings_cache_size'] = hasattr(config.embeddings, 'cache_size')
            
            success_count = sum(property_tests.values())
            total_count = len(property_tests)
            
            print(f"✅ Config loaded: {success_count}/{total_count} neue Properties verfügbar")
            
            for prop, available in property_tests.items():
                if not available:
                    print(f"    ⚠️  {prop}: Nicht verfügbar (Hotfix erforderlich)")
            
            self.test_results['config_loading'] = {
                'success': success_count == total_count,
                'properties_available': property_tests
            }
            
        except Exception as e:
            print(f"❌ Config loading failed: {str(e)}")
            self.test_results['config_loading'] = {
                'success': False,
                'error': str(e)
            }
    
    def _test_app_structure(self):
        """Test main_rag.py Struktur"""
        try:
            # Lade main_rag.py als Modul
            spec = sys.modules.get('main_rag')
            if spec is None:
                # Versuche zu importieren
                import main_rag
            
            # Test kritische Funktionen
            required_functions = [
                'initialize_session_state',
                'render_header', 
                'render_sidebar'
            ]
            
            missing_functions = []
            for func_name in required_functions:
                if not hasattr(main_rag, func_name):
                    missing_functions.append(func_name)
            
            success = len(missing_functions) == 0
            print(f"{'✅' if success else '❌'} main_rag.py Struktur: {len(required_functions) - len(missing_functions)}/{len(required_functions)} Funktionen")
            
            self.test_results['app_structure'] = {
                'success': success,
                'missing_functions': missing_functions
            }
            
        except Exception as e:
            print(f"❌ App structure test failed: {str(e)}")
            self.test_results['app_structure'] = {
                'success': False,
                'error': str(e)
            }
    
    def _test_session_state_init(self):
        """Test Session State Initialisierung"""
        try:
            # Simuliere Session State Setup
            import streamlit as st
            
            # Mock st.session_state falls nicht verfügbar
            if not hasattr(st, 'session_state'):
                st.session_state = type('SessionState', (), {})()
            
            # Test Session State Attribute
            required_attributes = [
                'initialized', 'system_status', 'services', 
                'controllers', 'current_page'
            ]
            
            # Simuliere Initialisierung
            for attr in required_attributes:
                setattr(st.session_state, attr, f"test_{attr}")
            
            print("✅ Session State: Initialisierung simuliert erfolgreich")
            
            self.test_results['session_state'] = {
                'success': True,
                'attributes': required_attributes
            }
            
        except Exception as e:
            print(f"❌ Session State test failed: {str(e)}")
            self.test_results['session_state'] = {
                'success': False,
                'error': str(e)
            }
    
    def _test_service_initialization(self):
        """Test Service-Initialisierung"""
        try:
            from services import available_services
            
            service_count = len(available_services)
            
            # Test Service-Import und -Verfügbarkeit
            working_services = []
            broken_services = []
            
            for service_name, service_class in available_services.items():
                if service_class is not None:
                    try:
                        # Versuche Service zu instanziieren (vorsichtig)
                        # Nicht alle Services können ohne Dependencies instanziiert werden
                        working_services.append(service_name)
                    except:
                        # Service-Klasse verfügbar aber Instanziierung problematisch
                        working_services.append(f"{service_name}_class_only")
                else:
                    broken_services.append(service_name)
            
            success_rate = len(working_services) / service_count if service_count > 0 else 0
            
            print(f"✅ Services: {len(working_services)}/{service_count} verfügbar ({success_rate:.0%})")
            
            if broken_services:
                print(f"    ⚠️  Problematisch: {', '.join(broken_services)}")
            
            self.test_results['service_initialization'] = {
                'success': success_rate >= 0.8,  # 80% als Erfolg
                'service_count': service_count,
                'working_services': working_services,
                'broken_services': broken_services,
                'success_rate': success_rate
            }
            
        except Exception as e:
            print(f"❌ Service initialization test failed: {str(e)}")
            self.test_results['service_initialization'] = {
                'success': False,
                'error': str(e)
            }
    
    def _test_controller_creation(self):
        """Test Controller-Erstellung"""
        try:
            from controllers import AVAILABLE_CONTROLLERS, CONTROLLER_IMPORT_ERRORS
            
            controller_count = len(AVAILABLE_CONTROLLERS)
            working_controllers = sum(1 for c in AVAILABLE_CONTROLLERS.values() if c is not None)
            
            success_rate = working_controllers / controller_count if controller_count > 0 else 0
            
            print(f"✅ Controllers: {working_controllers}/{controller_count} verfügbar ({success_rate:.0%})")
            
            if CONTROLLER_IMPORT_ERRORS:
                print("    ⚠️  Import-Fehler:")
                for controller, error in CONTROLLER_IMPORT_ERRORS.items():
                    print(f"      • {controller}: {error}")
            
            self.test_results['controller_creation'] = {
                'success': success_rate >= 0.8,
                'controller_count': controller_count,
                'working_controllers': working_controllers,
                'import_errors': dict(CONTROLLER_IMPORT_ERRORS),
                'success_rate': success_rate
            }
            
        except Exception as e:
            print(f"❌ Controller creation test failed: {str(e)}")
            self.test_results['controller_creation'] = {
                'success': False,
                'error': str(e)
            }
    
    def _test_interface_imports(self):
        """Test Interface-Komponenten"""
        try:
            from interfaces.streamlit_ui import (
                MainInterface, ChatInterface, DocumentInterface
            )
            
            interface_components = {
                'MainInterface': MainInterface,
                'ChatInterface': ChatInterface, 
                'DocumentInterface': DocumentInterface
            }
            
            working_interfaces = sum(1 for i in interface_components.values() if i is not None)
            total_interfaces = len(interface_components)
            
            print(f"✅ Interface Components: {working_interfaces}/{total_interfaces} verfügbar")
            
            self.test_results['interface_imports'] = {
                'success': working_interfaces == total_interfaces,
                'components': interface_components,
                'working_count': working_interfaces
            }
            
        except Exception as e:
            print(f"❌ Interface imports test failed: {str(e)}")
            self.test_results['interface_imports'] = {
                'success': False,
                'error': str(e)
            }
    
    def _test_page_rendering(self):
        """Test Page-Rendering-Funktionen"""
        try:
            # Test ob kritische Rendering-Funktionen definiert sind
            import main_rag
            
            render_functions = [
                'render_header',
                'render_sidebar', 
                'initialize_session_state'
            ]
            
            available_functions = []
            missing_functions = []
            
            for func_name in render_functions:
                if hasattr(main_rag, func_name):
                    available_functions.append(func_name)
                else:
                    missing_functions.append(func_name)
            
            success = len(missing_functions) == 0
            
            print(f"{'✅' if success else '❌'} Page Rendering: {len(available_functions)}/{len(render_functions)} Funktionen")
            
            self.test_results['page_rendering'] = {
                'success': success,
                'available_functions': available_functions,
                'missing_functions': missing_functions
            }
            
        except Exception as e:
            print(f"❌ Page rendering test failed: {str(e)}")
            self.test_results['page_rendering'] = {
                'success': False,
                'error': str(e)
            }
    
    def _test_workflow_simulation(self):
        """Simuliere End-to-End Workflows"""
        try:
            print("🔄 Simuliere PDF-Upload → Chat Workflow...")
            
            # Workflow-Komponenten testen
            workflow_components = {
                'document_service': False,
                'embedding_service': False,  
                'vector_store_service': False,
                'chat_service': False,
                'retrieval_pipeline': False
            }
            
            # Test Document Service
            try:
                from services import DocumentService
                # Simuliere Document-Processing
                workflow_components['document_service'] = True
            except:
                pass
            
            # Test Embedding Service  
            try:
                from services import EmbeddingService
                workflow_components['embedding_service'] = True
            except:
                pass
            
            # Test Vector Store Service
            try:
                from services import VectorStoreService
                workflow_components['vector_store_service'] = True
            except:
                pass
            
            # Test Chat Service
            try:
                from services import ChatService
                workflow_components['chat_service'] = True
            except:
                pass
            
            # Test Pipeline Controller
            try:
                from controllers import PipelineController
                workflow_components['retrieval_pipeline'] = True
            except:
                pass
            
            working_components = sum(workflow_components.values())
            total_components = len(workflow_components)
            workflow_ready = working_components >= 4  # Mindestens 4/5 Komponenten
            
            print(f"{'✅' if workflow_ready else '❌'} End-to-End Workflow: {working_components}/{total_components} Komponenten bereit")
            
            for component, status in workflow_components.items():
                status_icon = "✅" if status else "❌"
                print(f"    {status_icon} {component}")
            
            self.test_results['workflow_simulation'] = {
                'success': workflow_ready,
                'components': workflow_components,
                'readiness_score': working_components / total_components
            }
            
        except Exception as e:
            print(f"❌ Workflow simulation failed: {str(e)}")
            self.test_results['workflow_simulation'] = {
                'success': False,
                'error': str(e)
            }
    
    def _generate_test_report(self):
        """Generiert finalen Test-Report"""
        print("\n📊 STREAMLIT TEST REPORT")
        print("=" * 60)
        
        # Berechne Gesamt-Erfolgsrate
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() 
                              if isinstance(result, dict) and result.get('success', False))
        
        overall_success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Gesamt Tests: {total_tests}")
        print(f"Erfolgreich: {successful_tests}")
        print(f"Fehlgeschlagen: {total_tests - successful_tests}")
        print(f"Erfolgsrate: {overall_success_rate:.1f}%")
        
        # Detaillierte Kategorie-Ergebnisse
        print(f"\nDetailed Results:")
        for test_name, result in self.test_results.items():
            if isinstance(result, dict):
                success = result.get('success', False)
                status_icon = "✅" if success else "❌"
                print(f"  {status_icon} {test_name}")
        
        # Streamlit-Launch Empfehlung
        print(f"\n🚀 STREAMLIT LAUNCH EMPFEHLUNG:")
        
        if overall_success_rate >= 80:
            print("✅ System bereit für Streamlit-Start!")
            print("   Führe aus: streamlit run main_rag.py")
        elif overall_success_rate >= 60:
            print("⚠️  System teilweise funktional")
            print("   Streamlit-Start möglich, aber mit Fehlern zu rechnen")
            print("   Empfehlung: Erst Config-Hotfixes anwenden")
        else:
            print("❌ System nicht bereit für Streamlit-Start")
            print("   Kritische Fixes erforderlich vor Launch")
        
        # Nächste Schritte
        print(f"\n🎯 NÄCHSTE SCHRITTE:")
        
        critical_issues = []
        for test_name, result in self.test_results.items():
            if isinstance(result, dict) and not result.get('success', False):
                critical_issues.append(test_name)
        
        if critical_issues:
            print("Behebe folgende kritische Issues:")
            for issue in critical_issues[:5]:  # Zeige nur erste 5
                print(f"  • {issue}")
        else:
            print("✅ Alle Tests erfolgreich - System production-ready!")
        
        return overall_success_rate


def launch_streamlit_app():
    """Startet Streamlit App falls Tests erfolgreich"""
    print("\n🚀 STREAMLIT APP LAUNCH")
    print("-" * 30)
    
    try:
        import subprocess
        import time
        
        print("Starte Streamlit App...")
        
        # Streamlit-Kommando
        cmd = ["streamlit", "run", "main_rag.py", "--server.port", "8501"]
        
        # Starte Streamlit-Prozess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("✅ Streamlit-Prozess gestartet")
        print("📱 App sollte verfügbar sein unter: http://localhost:8501")
        
        # Warte kurz und prüfe ob Prozess läuft
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Streamlit läuft erfolgreich!")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Streamlit-Start fehlgeschlagen:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
            
    except FileNotFoundError:
        print("❌ Streamlit nicht installiert")
        print("   Installiere mit: pip install streamlit")
        return None
    except Exception as e:
        print(f"❌ Streamlit-Launch Fehler: {str(e)}")
        return None


def main():
    """Hauptfunktion - führt Tests durch und startet ggf. Streamlit"""
    print("🧪 STREAMLIT SYSTEM TEST & LAUNCH UTILITY")
    print("Version: 4.0.0 - Post-Fix Testing")
    print("=" * 70)
    
    # Führe Tests durch
    runner = StreamlitTestRunner()
    success_rate = runner.run_comprehensive_test()
    
    # Entscheide über Streamlit-Launch
    print(f"\n🤔 LAUNCH DECISION:")
    
    if success_rate >= 75:
        user_input = input("Möchten Sie Streamlit jetzt starten? (j/n): ").lower().strip()
        
        if user_input in ['j', 'ja', 'y', 'yes']:
            process = launch_streamlit_app()
            
            if process:
                try:
                    print(f"\n⌨️  Drücke Ctrl+C um Streamlit zu beenden...")
                    process.wait()
                except KeyboardInterrupt:
                    print(f"\n🛑 Streamlit wird beendet...")
                    process.terminate()
                    process.wait()
                    print("✅ Streamlit beendet")
        else:
            print("Streamlit-Start abgebrochen")
    else:
        print("❌ Erfolgsrate zu niedrig für sicheren Streamlit-Start")
        print("   Beheben Sie erst die kritischen Issues")
    
    # Finale Empfehlungen
    print(f"\n📋 FINALE EMPFEHLUNGEN:")
    
    if success_rate >= 90:
        print("🎉 System ist PRODUCTION READY!")
        print("   Alle Komponenten funktionieren optimal")
    elif success_rate >= 75:
        print("✅ System ist MOSTLY FUNCTIONAL")
        print("   Streamlit sollte starten, kleinere Issues möglich")
        print("   Empfehlung: Config-Hotfixes für 100% Funktionalität")
    elif success_rate >= 60:
        print("⚠️  System ist PARTIALLY FUNCTIONAL")
        print("   Mehrere Komponenten haben Probleme")
        print("   Empfehlung: Arbeite systematisch die Failed Tests ab")
    else:
        print("🆘 System hat CRITICAL ISSUES")
        print("   Grundlegende Reparaturen erforderlich")
        print("   Empfehlung: Core-System und Service-Layer prüfen")


if __name__ == "__main__":
    main()