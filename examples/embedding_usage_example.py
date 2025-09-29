# examples/embedding_usage_examples.py
"""
Verwendungsbeispiele für das neue Embedding-Module
Industrielle RAG-Architektur - Phase 2 Migration

Zeigt verschiedene Nutzungsszenarien und Migration-Strategien
"""

import time
from typing import List, Dict, Any
from pathlib import Path

# Core Imports
from core.logger import get_logger
from core.exceptions import EmbeddingException, ConfigurationException

# Embedding Module Imports
from modules.embeddings import (
    create_ollama_embeddings,
    create_auto_embeddings,
    EmbeddingFactory,
    get_available_providers,
    create_legacy_embedding_adapter,
    create_embedding_provider_from_yaml
)

logger = get_logger(__name__)


def example_1_basic_ollama_usage():
    """
    Beispiel 1: Basis-Nutzung mit Ollama Provider
    """
    print("=== Beispiel 1: Basis Ollama Embedding ===")
    
    try:
        # Einfache Provider-Erstellung
        embeddings = create_ollama_embeddings(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            max_batch_size=8,
            cache_embeddings=True
        )
        
        # Test-Texte
        documents = [
            "Industrielle Automatisierung mit KI-Systemen",
            "Maschinelle Lernverfahren in der Produktion", 
            "Qualitätssicherung durch Computer Vision",
            "Predictive Maintenance für Anlagen"
        ]
        
        # Embeddings erstellen
        print(f"Erstelle Embeddings für {len(documents)} Dokumente...")
        start_time = time.time()
        
        result = embeddings.create_embeddings(documents)
        
        processing_time = time.time() - start_time
        
        if result.success:
            print(f"✅ Erfolgreich! {len(result.embeddings)} Embeddings erstellt")
            print(f"⏱️ Verarbeitungszeit: {processing_time:.2f}s")
            print(f"📏 Embedding-Dimensionen: {len(result.embeddings[0]) if result.embeddings else 'N/A'}")
            print(f"🎯 Cache-Hits: {result.processing_stats.get('cache_hits', 0)}")
            print(f"🔍 Modell-Info: {result.model_info.get('name', 'Unknown')}")
            
            # Statistiken anzeigen
            stats = embeddings.get_statistics()
            print(f"📊 Cache-Hit-Rate: {stats['cache_hit_rate']:.1%}")
            print(f"📈 Durchschnittliche Verarbeitungszeit: {stats['avg_processing_time']:.3f}s")
        else:
            print(f"❌ Fehler: {result.error}")
            
    except Exception as e:
        print(f"❌ Fehler bei Beispiel 1: {str(e)}")
    
    print()


def example_2_automatic_provider_detection():
    """
    Beispiel 2: Automatische Provider-Erkennung
    """
    print("=== Beispiel 2: Automatische Provider-Erkennung ===")
    
    try:
        # Verfügbare Provider anzeigen
        providers = get_available_providers()
        print("Verfügbare Provider:")
        for name, info in providers.items():
            status_emoji = "✅" if info['available'] else "❌"
            print(f"  {status_emoji} {name}: {info['status']} ({len(info['models'])} Modelle)")
        
        print()
        
        # Automatische Provider-Erstellung
        print("Erstelle automatisch besten verfügbaren Provider...")
        embeddings = create_auto_embeddings(
            model_preferences=['nomic-embed-text', 'all-MiniLM-L6-v2']
        )
        
        # Test mit Query
        query = "Wie funktioniert Predictive Maintenance?"
        result = embeddings.create_embeddings([query])
        
        if result.success:
            model_info = result.model_info
            print(f"✅ Automatisch gewählter Provider: {model_info['provider']}")
            print(f"🤖 Modell: {model_info['name']}")
            print(f"📏 Dimensionen: {model_info.get('dimensions', 'Auto-Detect')}")
            
            # Query-Embedding Details
            query_embedding = result.embeddings[0]
            print(f"🔍 Query-Embedding Länge: {len(query_embedding)}")
            print(f"📊 Embedding-Bereich: [{min(query_embedding):.6f}, {max(query_embedding):.6f}]")
        
    except ConfigurationException as e:
        print(f"⚠️ Konfigurationsfehler: {str(e)}")
    except Exception as e:
        print(f"❌ Fehler bei Beispiel 2: {str(e)}")
    
    print()


def example_3_advanced_configuration():
    """
    Beispiel 3: Erweiterte Konfiguration mit Performance-Tuning
    """
    print("=== Beispiel 3: Erweiterte Konfiguration ===")
    
    try:
        # Erweiterte Ollama-Konfiguration
        config_dict = {
            'provider': 'ollama',
            'model_name': 'nomic-embed-text',
            'base_url': 'http://localhost:11434',
            'max_batch_size': 32,           # Höhere Batch-Größe für Durchsatz
            'timeout_seconds': 90,          # Längerer Timeout
            'cache_embeddings': True,       # Caching aktiviert
            'normalize_embeddings': True,   # L2-Normalisierung
            'retry_attempts': 5,           # Mehr Wiederholungen
            'retry_delay': 2.0,            # Längere Wartezeit
            'verify_ssl': False,           # Für Self-Signed Certificates
            'keep_alive': '10m',           # Modell länger im Speicher
            'temperature': 0.0,            # Deterministische Ergebnisse
            'custom_params': {             # Zusätzliche Ollama-Parameter
                'num_ctx': 2048,
                'num_batch': 512
            }
        }
        
        # Provider mit erweiteter Konfiguration erstellen
        embeddings = EmbeddingFactory.create_from_config(config_dict)
        
        # Performance-Optimierung durchführen
        print("Führe Performance-Optimierung durch...")
        optimization_results = embeddings.optimize_performance()
        
        print("Optimierungsempfehlungen:")
        print(f"  📊 Performance-Score: {optimization_results['performance_score']}/100")
        print(f"  🔧 Empfohlene Batch-Größe: {optimization_results['recommended_batch_size']}")
        print(f"  ⏰ Empfohlener Timeout: {optimization_results['recommended_timeout']}s")
        print(f"  ⚡ Durchschnittliche Zeit/Embedding: {optimization_results.get('avg_time_per_embedding', 0):.4f}s")
        
        if optimization_results['optimizations_applied']:
            print("  ✨ Angewendete Optimierungen:")
            for opt in optimization_results['optimizations_applied']:
                print(f"    • {opt}")
        
        # Server-Informationen anzeigen
        server_info = embeddings.get_server_info()
        print(f"\n🖥️ Server-Informationen:")
        print(f"  Status: {server_info['status']}")
        print(f"  Version: {server_info.get('version', 'Unknown')}")
        print(f"  Modell geladen: {'✅' if server_info.get('model_loaded') else '❌'}")
        print(f"  Installierte Modelle: {len(server_info.get('installed_models', []))}")
        
    except Exception as e:
        print(f"❌ Fehler bei Beispiel 3: {str(e)}")
    
    print()


def example_4_yaml_configuration():
    """
    Beispiel 4: YAML-basierte Konfiguration
    """
    print("=== Beispiel 4: YAML-Konfiguration ===")
    
    # YAML-Konfiguration erstellen (simuliert)
    yaml_config = """
embedding:
  provider: ollama
  model_name: nomic-embed-text
  base_url: http://localhost:11434
  max_batch_size: 16
  timeout_seconds: 60
  cache_embeddings: true
  normalize_embeddings: true
  verify_ssl: false
  keep_alive: 5m
  temperature: 0.0
  custom_params:
    num_ctx: 1024
    num_batch: 256
"""
    
    # Temporäre YAML-Datei erstellen
    config_path = Path("temp_embedding_config.yaml")
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(yaml_config)
        
        print(f"📄 YAML-Konfiguration erstellt: {config_path}")
        print("Inhalt:")
        print(yaml_config)
        
        # Provider aus YAML erstellen
        embeddings = create_embedding_provider_from_yaml(str(config_path))
        
        # Test-Embedding
        test_texts = ["YAML-Konfiguration erfolgreich geladen"]
        result = embeddings.create_embeddings(test_texts)
        
        if result.success:
            print("✅ YAML-Provider erfolgreich erstellt und getestet")
            print(f"🤖 Modell: {result.model_info['name']}")
            print(f"📏 Dimensionen: {result.model_info.get('dimensions')}")
        
    except Exception as e:
        print(f"❌ Fehler bei YAML-Konfiguration: {str(e)}")
    
    finally:
        # Cleanup
        if config_path.exists():
            config_path.unlink()
    
    print()


def example_5_legacy_migration():
    """
    Beispiel 5: Legacy-Migration mit Adapter-Pattern
    """
    print("=== Beispiel 5: Legacy-Migration ===")
    
    try:
        # Legacy-Adapter für bestehenden Code erstellen
        config_dict = {
            'provider': 'ollama',
            'model_name': 'nomic-embed-text',
            'max_batch_size': 8,
            'cache_embeddings': True
        }
        
        legacy_embeddings = create_legacy_embedding_adapter(config_dict)
        
        print("✅ Legacy-Adapter erstellt")
        
        # Simulation alter Schnittstelle
        print("Teste Legacy-Schnittstelle...")
        
        # embed_documents (alte Methode)
        documents = [
            "Legacy-Dokument 1: Maschinendaten",
            "Legacy-Dokument 2: Qualitätskontrolle"
        ]
        
        doc_embeddings = legacy_embeddings.embed_documents(documents)
        print(f"📄 Dokument-Embeddings: {len(doc_embeddings)} x {len(doc_embeddings[0]) if doc_embeddings else 0}")
        
        # embed_query (alte Methode)
        query = "Legacy-Query: Maschinenfehler analysieren"
        query_embedding = legacy_embeddings.embed_query(query)
        print(f"🔍 Query-Embedding: {len(query_embedding)} Dimensionen")
        
        # Ähnlichkeit berechnen (Beispiel für bestehenden Code)
        import numpy as np
        
        similarities = []
        for doc_emb in doc_embeddings:
            # Cosine-Ähnlichkeit
            dot_product = np.dot(query_embedding, doc_emb)
            norm_query = np.linalg.norm(query_embedding)
            norm_doc = np.linalg.norm(doc_emb)
            similarity = dot_product / (norm_query * norm_doc)
            similarities.append(similarity)
        
        print(f"📊 Ähnlichkeiten: {[f'{sim:.3f}' for sim in similarities]}")
        
        # Provider-Statistiken über Adapter
        stats = legacy_embeddings.provider.get_statistics()
        print(f"📈 Verarbeitete Anfragen: {stats['total_requests']}")
        print(f"🎯 Cache-Hit-Rate: {stats['cache_hit_rate']:.1%}")
        
        print("✅ Legacy-Migration erfolgreich - alter Code funktioniert weiterhin!")
        
    except Exception as e:
        print(f"❌ Fehler bei Legacy-Migration: {str(e)}")
    
    print()


def example_6_error_handling_and_monitoring():
    """
    Beispiel 6: Fehlerbehandlung und Monitoring
    """
    print("=== Beispiel 6: Fehlerbehandlung und Monitoring ===")
    
    try:
        # Provider mit bewusst problematischer Konfiguration
        embeddings = create_ollama_embeddings(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            max_batch_size=4,
            timeout_seconds=30,
            retry_attempts=3
        )
        
        # Health-Check durchführen
        print("🏥 Führe Health-Check durch...")
        is_healthy = embeddings.health_check()
        print(f"Gesundheitsstatus: {'✅ Gesund' if is_healthy else '❌ Problematisch'}")
        
        # Test mit verschiedenen Problem-Szenarien
        test_scenarios = [
            # Normal-Fall
            {
                'name': 'Normale Verarbeitung',
                'texts': ['Test-Text für normale Verarbeitung'],
                'expected': 'success'
            },
            # Leere Eingabe
            {
                'name': 'Leere Eingabe',
                'texts': [],
                'expected': 'error'
            },
            # Sehr lange Texte
            {
                'name': 'Sehr lange Texte',
                'texts': ['X' * 10000],  # 10k Zeichen
                'expected': 'success_or_timeout'
            },
            # Ungültige Zeichen
            {
                'name': 'Ungültige Zeichen',
                'texts': [None, '', '   '],
                'expected': 'error'
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n🧪 Test {i}: {scenario['name']}")
            
            try:
                start_time = time.time()
                result = embeddings.create_embeddings(scenario['texts'])
                processing_time = time.time() - start_time
                
                if result.success:
                    print(f"  ✅ Erfolgreich ({processing_time:.3f}s)")
                    print(f"     Embeddings: {len(result.embeddings)}")
                    print(f"     Cache-Hits: {result.processing_stats.get('cache_hits', 0)}")
                else:
                    print(f"  ⚠️ Verarbeitung fehlgeschlagen: {result.error}")
                
            except EmbeddingException as e:
                print(f"  ❌ Embedding-Fehler: {str(e)}")
            except ConfigurationException as e:
                print(f"  ⚙️ Konfigurationsfehler: {str(e)}")
            except Exception as e:
                print(f"  💥 Unerwarteter Fehler: {str(e)}")
        
        # Abschließende Statistiken
        print("\n📊 Abschließende Statistiken:")
        final_stats = embeddings.get_statistics()
        
        stats_display = [
            ('Gesamte Anfragen', final_stats['total_requests']),
            ('Cache-Hits', final_stats['cache_hits']),
            ('Cache-Misses', final_stats['cache_misses']),
            ('Fehleranzahl', final_stats['error_count']),
            ('Cache-Hit-Rate', f"{final_stats['cache_hit_rate']:.1%}"),
            ('Ø Verarbeitungszeit', f"{final_stats['avg_processing_time']:.3f}s"),
            ('Cache-Größe', final_stats['cache_size'])
        ]
        
        for label, value in stats_display:
            print(f"  {label}: {value}")
        
    except Exception as e:
        print(f"❌ Fehler bei Monitoring-Beispiel: {str(e)}")
    
    print()


def run_all_examples():
    """Führt alle Beispiele aus"""
    print("🚀 RAG Industrial - Embedding Module Examples")
    print("=" * 50)
    print()
    
    examples = [
        example_1_basic_ollama_usage,
        example_2_automatic_provider_detection,
        example_3_advanced_configuration,
        example_4_yaml_configuration,
        example_5_legacy_migration,
        example_6_error_handling_and_monitoring
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"❌ Beispiel {i} fehlgeschlagen: {str(e)}")
            print()
        
        # Kurze Pause zwischen Beispielen
        time.sleep(1)
    
    print("🏁 Alle Beispiele abgeschlossen!")
    print("\n💡 Tipps für die Produktionsnutzung:")
    print("  • Verwenden Sie YAML-Konfigurationen für bessere Wartbarkeit")
    print("  • Implementieren Sie Health-Checks in Ihrer Monitoring-Pipeline") 
    print("  • Nutzen Sie den Legacy-Adapter für schrittweise Migration")
    print("  • Konfigurieren Sie angemessene Timeouts für Ihre Umgebung")
    print("  • Aktivieren Sie Caching für bessere Performance")


if __name__ == '__main__':
    # Einzelne Beispiele zum Testen
    # example_1_basic_ollama_usage()
    # example_2_automatic_provider_detection()
    
    # Alle Beispiele ausführen
    run_all_examples()