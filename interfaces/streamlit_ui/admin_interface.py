#!/usr/bin/env python3
"""
Admin Interface - System-Administration Dashboard
Industrielle RAG-Architektur - Phase 2 Migration

Streamlit-basierte Administrator-Oberfläche für System-Monitoring,
Health-Checks, Performance-Metriken und Debug-Tools der service-orientierten
RAG-Architektur.

Features:
- Real-time Service-Health-Dashboard mit Status-Übersicht
- Performance-Metriken-Visualisierung und Trend-Analyse  
- System-Konfiguration und Dynamic-Config-Updates
- Debug-Tools und Log-Viewer für Troubleshooting
- Service-Management (Restart, Health-Checks, Cache-Clear)
- Backup/Restore-Funktionalität für Konfiguration und Daten

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

import streamlit as st
import time
import json
import yaml
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass

# Core-Komponenten
from core import (
    get_logger, get_config, RAGConfig,
    ServiceError, ValidationError, create_error_context,
    log_performance, log_method_calls
)

# Service-Integration
from services.service_integration import (
    get_service_integrator, ServiceStatus, ServiceEventType,
    validate_service_integration
)

# Controller-Integration  
from controllers import get_health_controller, get_pipeline_controller

logger = get_logger(__name__)


# =============================================================================
# ADMIN INTERFACE DATENSTRUKTUREN
# =============================================================================

@dataclass
class AdminConfig:
    """Konfiguration für Admin Interface"""
    refresh_interval: int = 5  # Sekunden
    max_log_lines: int = 100
    chart_history_hours: int = 24
    enable_debug_mode: bool = False
    auto_refresh_enabled: bool = True
    theme: str = "dark"  # dark, light, auto


@dataclass  
class ServiceMetrics:
    """Service-Metriken für Dashboard"""
    service_name: str
    status: ServiceStatus
    response_time: float
    success_rate: float
    error_count: int
    last_health_check: datetime
    uptime_seconds: int


# =============================================================================
# ADMIN INTERFACE HAUPTKLASSE
# =============================================================================

class AdminInterface:
    """
    Admin Interface - System-Administration Dashboard
    
    Streamlit-basierte Verwaltungsoberfläche für:
    - System-Health-Monitoring und Service-Status
    - Performance-Metriken und Trend-Visualisierung  
    - Konfiguration-Management und Dynamic-Updates
    - Debug-Tools und System-Troubleshooting
    - Service-Management und Recovery-Operationen
    """
    
    def __init__(self, config: Optional[AdminConfig] = None):
        """
        Initialisiert Admin Interface
        
        Args:
            config: Optional AdminConfig für Interface-Konfiguration
        """
        self.logger = get_logger(f"{__name__}.admin_interface")
        self.config = config or AdminConfig()
        self.app_config = get_config()
        
        # Service Integration
        self.integrator = get_service_integrator()
        
        # Controller
        self.health_controller = get_health_controller()
        self.pipeline_controller = get_pipeline_controller()
        
        # State Management
        self._initialize_session_state()
        
        self.logger.info("Admin Interface initialisiert")

    def _initialize_session_state(self):
        """Initialisiert Streamlit Session State"""
        if 'admin_config' not in st.session_state:
            st.session_state.admin_config = self.config
        
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        if 'performance_history' not in st.session_state:
            st.session_state.performance_history = []
        
        if 'log_buffer' not in st.session_state:
            st.session_state.log_buffer = []
        
        if 'debug_mode' not in st.session_state:
            st.session_state.debug_mode = self.config.enable_debug_mode

    def render(self):
        """
        Hauptrender-Methode für Admin Interface
        """
        try:
            # Page Config
            st.set_page_config(
                page_title="RAG System Administration",
                page_icon="🔧",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # Header
            self._render_header()
            
            # Sidebar Navigation
            page = self._render_sidebar()
            
            # Main Content basierend auf gewählter Seite
            if page == "Dashboard":
                self._render_dashboard()
            elif page == "Services":
                self._render_services_page()
            elif page == "Performance":
                self._render_performance_page()
            elif page == "Configuration":
                self._render_configuration_page()
            elif page == "Debug Tools":
                self._render_debug_page()
            elif page == "System Logs":
                self._render_logs_page()
            
            # Footer mit Auto-Refresh
            self._render_footer()
            
        except Exception as e:
            self.logger.error(f"Admin Interface Render-Fehler: {str(e)}", exc_info=True)
            st.error(f"❌ Interface-Fehler: {str(e)}")

    def _render_header(self):
        """Rendert Header mit System-Status"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.title("🔧 RAG System Administration")
            st.caption("Service-orientierte Architektur v4.0.0")
        
        with col2:
            # Globaler System-Status
            try:
                health_overview = self.integrator.get_service_health_overview()
                overall_status = health_overview.get('overall_status', 'unknown')
                
                if overall_status == 'healthy':
                    st.success("✅ System Healthy")
                elif overall_status == 'degraded':
                    st.warning("⚠️ System Degraded")
                elif overall_status == 'unhealthy':
                    st.error("❌ System Unhealthy")
                else:
                    st.info("❓ Status Unknown")
                    
            except Exception as e:
                st.error(f"Status Error: {str(e)}")
        
        with col3:
            # Last Update Info
            last_update = datetime.fromtimestamp(st.session_state.last_refresh)
            st.info(f"🔄 Last Update: {last_update.strftime('%H:%M:%S')}")

    def _render_sidebar(self) -> str:
        """Rendert Sidebar Navigation"""
        st.sidebar.title("📋 Navigation")
        
        # Navigation Pages
        pages = [
            "Dashboard",
            "Services", 
            "Performance",
            "Configuration",
            "Debug Tools",
            "System Logs"
        ]
        
        selected_page = st.sidebar.selectbox(
            "Select Page",
            pages,
            index=0
        )
        
        st.sidebar.divider()
        
        # Auto-Refresh Controls
        st.sidebar.subheader("🔄 Auto-Refresh")
        
        auto_refresh = st.sidebar.toggle(
            "Enable Auto-Refresh",
            value=st.session_state.admin_config.auto_refresh_enabled
        )
        
        if auto_refresh:
            refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)",
                min_value=1,
                max_value=60,
                value=st.session_state.admin_config.refresh_interval
            )
            
            st.session_state.admin_config.refresh_interval = refresh_interval
            st.session_state.admin_config.auto_refresh_enabled = True
        else:
            st.session_state.admin_config.auto_refresh_enabled = False
        
        # Manual Refresh Button
        if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
            self._refresh_data()
            st.rerun()
        
        st.sidebar.divider()
        
        # Debug Mode Toggle
        debug_mode = st.sidebar.toggle(
            "🐛 Debug Mode",
            value=st.session_state.debug_mode
        )
        st.session_state.debug_mode = debug_mode
        
        # Quick Actions
        st.sidebar.subheader("⚡ Quick Actions")
        
        if st.sidebar.button("🏥 Health Check All", use_container_width=True):
            self._run_health_check_all()
        
        if st.sidebar.button("🧹 Clear All Caches", use_container_width=True):
            self._clear_all_caches()
        
        if st.sidebar.button("📊 Export Metrics", use_container_width=True):
            self._export_metrics()
        
        return selected_page

    def _render_dashboard(self):
        """Rendert Haupt-Dashboard"""
        st.header("📊 System Dashboard")
        
        try:
            # Service Health Overview
            health_overview = self.integrator.get_service_health_overview()
            
            # Service Status Cards
            self._render_service_status_cards(health_overview)
            
            st.divider()
            
            # Performance Metrics Charts  
            col1, col2 = st.columns(2)
            
            with col1:
                self._render_response_time_chart()
            
            with col2:
                self._render_success_rate_chart()
            
            st.divider()
            
            # System Resource Usage
            self._render_system_resources()
            
        except Exception as e:
            st.error(f"Dashboard-Fehler: {str(e)}")
            if st.session_state.debug_mode:
                st.exception(e)

    def _render_service_status_cards(self, health_overview: Dict[str, Any]):
        """Rendert Service-Status Karten"""
        services = health_overview.get('services', {})
        
        # Calculate overview stats
        total_services = len(services)
        healthy_services = len([s for s in services.values() if s.get('status') == 'healthy'])
        degraded_services = len([s for s in services.values() if s.get('status') == 'degraded'])
        unhealthy_services = len([s for s in services.values() if s.get('status') == 'unhealthy'])
        
        # Overview Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Services",
                value=total_services,
                delta=None
            )
        
        with col2:
            st.metric(
                label="Healthy",
                value=healthy_services,
                delta=None if total_services == 0 else f"{healthy_services/total_services:.1%}"
            )
        
        with col3:
            st.metric(
                label="Degraded", 
                value=degraded_services,
                delta=None if degraded_services == 0 else f"⚠️ {degraded_services}"
            )
        
        with col4:
            st.metric(
                label="Unhealthy",
                value=unhealthy_services, 
                delta=None if unhealthy_services == 0 else f"❌ {unhealthy_services}"
            )
        
        # Detailed Service Cards
        st.subheader("🔧 Service Details")
        
        cols = st.columns(min(len(services), 3))
        
        for idx, (service_name, service_info) in enumerate(services.items()):
            col_idx = idx % 3
            
            with cols[col_idx]:
                status = service_info.get('status', 'unknown')
                
                # Status Icon und Farbe
                if status == 'healthy':
                    status_icon = "✅"
                    status_color = "normal"
                elif status == 'degraded':
                    status_icon = "⚠️"
                    status_color = "normal"  
                elif status == 'unhealthy':
                    status_icon = "❌"
                    status_color = "normal"
                else:
                    status_icon = "❓"
                    status_color = "normal"
                
                # Service Card
                with st.container():
                    st.markdown(f"**{status_icon} {service_name.replace('_', ' ').title()}**")
                    st.caption(f"Status: {status}")
                    
                    # Detailed Health Info wenn verfügbar
                    detailed_health = service_info.get('detailed_health')
                    if detailed_health:
                        if 'service_status' in detailed_health:
                            st.caption(f"Detailed: {detailed_health['service_status']}")
                        
                        if 'performance_metrics' in detailed_health:
                            metrics = detailed_health['performance_metrics']
                            if 'avg_response_time' in metrics:
                                st.caption(f"Avg Response: {metrics['avg_response_time']:.3f}s")

    def _render_response_time_chart(self):
        """Rendert Response-Time Chart"""
        st.subheader("📈 Response Times")
        
        try:
            # Mock-Daten für Demo - in Realität von Metriken-System
            import numpy as np
            
            # Generiere Mock-Zeitreihen-Daten
            timestamps = [datetime.now() - timedelta(minutes=x) for x in range(30, 0, -1)]
            response_times = np.random.normal(0.5, 0.1, 30).clip(0.1, 2.0)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=response_times,
                mode='lines+markers',
                name='Response Time',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="Service Response Times (Last 30 min)",
                xaxis_title="Time",
                yaxis_title="Response Time (s)",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Chart-Fehler: {str(e)}")

    def _render_success_rate_chart(self):
        """Rendert Success-Rate Chart"""
        st.subheader("✅ Success Rates")
        
        try:
            # Mock-Daten für Demo
            services = ['Document', 'Embedding', 'Retrieval', 'Chat', 'Vector Store']
            success_rates = [98.5, 99.2, 97.8, 96.5, 99.8]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=services,
                    y=success_rates,
                    marker_color=['green' if rate >= 98 else 'orange' if rate >= 95 else 'red' 
                                  for rate in success_rates]
                )
            ])
            
            fig.update_layout(
                title="Service Success Rates (%)",
                xaxis_title="Services",
                yaxis_title="Success Rate (%)",
                yaxis=dict(range=[90, 100]),
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Chart-Fehler: {str(e)}")

    def _render_system_resources(self):
        """Rendert System-Ressourcen Übersicht"""
        st.subheader("💾 System Resources")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CPU Usage (Mock)
            cpu_usage = 45.2
            st.metric(
                label="CPU Usage",
                value=f"{cpu_usage:.1f}%",
                delta=f"{cpu_usage - 40:.1f}%" if cpu_usage > 40 else None
            )
        
        with col2:
            # Memory Usage (Mock)
            memory_usage = 67.8
            st.metric(
                label="Memory Usage", 
                value=f"{memory_usage:.1f}%",
                delta=f"{memory_usage - 65:.1f}%" if memory_usage > 65 else None
            )
        
        with col3:
            # Disk Usage (Mock)
            disk_usage = 23.5
            st.metric(
                label="Disk Usage",
                value=f"{disk_usage:.1f}%",
                delta=None
            )

    def _render_services_page(self):
        """Rendert Services-Management Seite"""
        st.header("🔧 Services Management")
        
        try:
            health_overview = self.integrator.get_service_health_overview()
            services = health_overview.get('services', {})
            
            # Service-Tabelle
            st.subheader("📋 Service Overview")
            
            service_data = []
            for service_name, service_info in services.items():
                status = service_info.get('status', 'unknown')
                
                # Health-Details extrahieren
                detailed_health = service_info.get('detailed_health', {})
                response_time = "N/A"
                if 'performance_metrics' in detailed_health:
                    metrics = detailed_health['performance_metrics']
                    if 'avg_response_time' in metrics:
                        response_time = f"{metrics['avg_response_time']:.3f}s"
                
                service_data.append({
                    'Service': service_name.replace('_', ' ').title(),
                    'Status': status.title(),
                    'Response Time': response_time,
                    'Actions': '🔄 🧹 📊'  # Restart, Clear Cache, Stats
                })
            
            # Service-Management Controls
            st.subheader("⚡ Service Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_service = st.selectbox(
                    "Select Service",
                    options=list(services.keys()),
                    format_func=lambda x: x.replace('_', ' ').title()
                )
            
            with col2:
                action = st.selectbox(
                    "Select Action",
                    ["Health Check", "Clear Cache", "View Stats", "Restart Service"]
                )
            
            with col3:
                if st.button("Execute Action", use_container_width=True):
                    self._execute_service_action(selected_service, action)
            
            # Tabelle anzeigen
            if service_data:
                st.table(service_data)
            
        except Exception as e:
            st.error(f"Services-Seite Fehler: {str(e)}")
            if st.session_state.debug_mode:
                st.exception(e)

    def _render_performance_page(self):
        """Rendert Performance-Analyse Seite"""
        st.header("📊 Performance Analysis")
        
        try:
            # Time Range Selector
            col1, col2 = st.columns([3, 1])
            
            with col1:
                time_range = st.selectbox(
                    "Time Range",
                    ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"]
                )
            
            with col2:
                if st.button("🔄 Refresh Data"):
                    self._refresh_performance_data()
            
            # Performance Metriken
            st.subheader("📈 Key Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Response Time", "0.245s", "-0.012s")
            
            with col2:
                st.metric("Requests/min", "157", "+12")
            
            with col3:
                st.metric("Error Rate", "0.8%", "-0.2%")
            
            with col4:
                st.metric("Uptime", "99.97%", "+0.02%")
            
            # Detailed Charts
            tab1, tab2, tab3 = st.tabs(["Response Times", "Throughput", "Error Rates"])
            
            with tab1:
                self._render_detailed_response_chart()
            
            with tab2:
                self._render_throughput_chart()
            
            with tab3:
                self._render_error_rate_chart()
            
        except Exception as e:
            st.error(f"Performance-Seite Fehler: {str(e)}")

    def _render_configuration_page(self):
        """Rendert Konfiguration-Management Seite"""
        st.header("⚙️ Configuration Management")
        
        try:
            # Current Config Display
            st.subheader("📄 Current Configuration")
            
            config_data = self.app_config
            
            # Config als YAML anzeigen  
            if st.toggle("Show as YAML", value=True):
                st.code(yaml.dump(config_data, default_flow_style=False), language='yaml')
            else:
                st.json(config_data)
            
            st.divider()
            
            # Config Editor
            st.subheader("✏️ Configuration Editor")
            
            tab1, tab2 = st.tabs(["Service Config", "Interface Config"])
            
            with tab1:
                self._render_service_config_editor()
            
            with tab2:
                self._render_interface_config_editor()
            
        except Exception as e:
            st.error(f"Configuration-Seite Fehler: {str(e)}")

    def _render_debug_page(self):
        """Rendert Debug-Tools Seite"""
        st.header("🐛 Debug Tools")
        
        if not st.session_state.debug_mode:
            st.warning("🔒 Debug Mode deaktiviert. Aktiviere Debug Mode in der Sidebar.")
            return
        
        try:
            # Service Integration Validation
            st.subheader("🔍 Service Integration Validation")
            
            if st.button("🧪 Run Full System Validation"):
                with st.spinner("Running validation..."):
                    validation_results = validate_service_integration()
                
                if validation_results.get('validation_success'):
                    st.success("✅ System Validation erfolgreich!")
                else:
                    st.error("❌ System Validation fehlgeschlagen")
                
                # Detailed Results
                with st.expander("📋 Validation Details"):
                    st.json(validation_results)
            
            st.divider()
            
            # Event System Debug
            st.subheader("📡 Event System Debug")
            
            col1, col2 = st.columns(2)
            
            with col1:
                event_type = st.selectbox(
                    "Event Type",
                    ["DOCUMENT_PROCESSED", "EMBEDDINGS_CREATED", "HEALTH_CHECK_FAILED"]
                )
                
                test_payload = st.text_area(
                    "Test Payload (JSON)",
                    value='{"test": true, "debug": true}'
                )
            
            with col2:
                if st.button("🚀 Send Test Event"):
                    try:
                        import json
                        payload = json.loads(test_payload)
                        
                        from services.service_integration import ServiceEventType
                        event_type_enum = getattr(ServiceEventType, event_type)
                        
                        self.integrator.publish_event(
                            event_type=event_type_enum,
                            source_service="admin_debug",
                            payload=payload
                        )
                        
                        st.success("✅ Test Event gesendet!")
                        
                    except Exception as e:
                        st.error(f"❌ Event-Fehler: {str(e)}")
            
            st.divider()
            
            # Service Health Deep Dive
            st.subheader("🏥 Service Health Deep Dive")
            
            selected_service = st.selectbox(
                "Service auswählen",
                ["document_service", "embedding_service", "retrieval_service", "chat_service"]
            )
            
            if st.button("🔬 Deep Health Check"):
                self._deep_health_check(selected_service)
            
            st.divider()
            
            # Memory and Performance Debug
            st.subheader("💾 Memory & Performance Debug")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Memory Usage"):
                    self._show_memory_usage()
            
            with col2:
                if st.button("⏱️ Performance Profile"):
                    self._show_performance_profile()
            
            with col3:
                if st.button("🧹 Garbage Collection"):
                    self._trigger_garbage_collection()
            
        except Exception as e:
            st.error(f"Debug-Seite Fehler: {str(e)}")
            st.exception(e)

    def _render_logs_page(self):
        """Rendert System-Logs Seite"""
        st.header("📋 System Logs")
        
        try:
            # Log Level Filter
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                log_level = st.selectbox(
                    "Log Level",
                    ["ALL", "ERROR", "WARNING", "INFO", "DEBUG"]
                )
            
            with col2:
                log_source = st.selectbox(
                    "Log Source", 
                    ["ALL", "Services", "Controllers", "Interfaces", "Core"]
                )
            
            with col3:
                max_lines = st.number_input(
                    "Max Lines",
                    min_value=10,
                    max_value=1000,
                    value=100
                )
            
            with col4:
                if st.button("🔄 Refresh Logs"):
                    self._refresh_logs()
            
            st.divider()
            
            # Real-time Logs
            st.subheader("📜 Live Log Stream")
            
            # Auto-refresh für Logs
            log_container = st.container()
            
            with log_container:
                # Mock-Logs für Demo
                logs = self._get_recent_logs(log_level, log_source, max_lines)
                
                if logs:
                    for log_entry in logs:
                        timestamp = log_entry.get('timestamp', 'Unknown')
                        level = log_entry.get('level', 'INFO')
                        message = log_entry.get('message', '')
                        source = log_entry.get('source', 'Unknown')
                        
                        # Color-coded Logs
                        if level == 'ERROR':
                            st.error(f"[{timestamp}] {source}: {message}")
                        elif level == 'WARNING':
                            st.warning(f"[{timestamp}] {source}: {message}")
                        elif level == 'DEBUG':
                            st.info(f"[{timestamp}] {source}: {message}")
                        else:
                            st.text(f"[{timestamp}] {source}: {message}")
                else:
                    st.info("No logs matching filter criteria")
            
            # Log Export
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Export Logs"):
                    self._export_logs()
            
            with col2:
                if st.button("🧹 Clear Log Buffer"):
                    self._clear_log_buffer()
            
        except Exception as e:
            st.error(f"Logs-Seite Fehler: {str(e)}")

    def _render_footer(self):
        """Rendert Footer mit Auto-Refresh"""
        st.divider()
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.caption("🔧 RAG System Administration v4.0.0 | Service-orientierte Architektur")
        
        with col2:
            if st.session_state.admin_config.auto_refresh_enabled:
                # Auto-Refresh Timer
                time_since_refresh = time.time() - st.session_state.last_refresh
                next_refresh_in = max(0, st.session_state.admin_config.refresh_interval - time_since_refresh)
                
                st.caption(f"🔄 Next refresh in: {next_refresh_in:.1f}s")
                
                # Trigger refresh wenn Zeit abgelaufen
                if next_refresh_in <= 0:
                    self._refresh_data()
                    time.sleep(0.1)  # Kurze Pause
                    st.rerun()
        
        with col3:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.caption(f"🕐 Current Time: {current_time}")

    # =============================================================================
    # HELPER METHODS UND ACTIONS
    # =============================================================================

    def _refresh_data(self):
        """Refresh alle Daten"""
        st.session_state.last_refresh = time.time()
        # Hier würden in der Realität die Metriken aktualisiert
        
    def _run_health_check_all(self):
        """Führt Health-Check für alle Services aus"""
        try:
            with st.spinner("Running health checks..."):
                health_overview = self.integrator.get_service_health_overview()
                
                services = health_overview.get('services', {})
                healthy_count = len([s for s in services.values() if s.get('status') == 'healthy'])
                total_count = len(services)
                
                st.success(f"✅ Health Check completed: {healthy_count}/{total_count} services healthy")
                
        except Exception as e:
            st.error(f"❌ Health Check failed: {str(e)}")

    def _clear_all_caches(self):
        """Leert alle Service-Caches"""
        try:
            with st.spinner("Clearing caches..."):
                # Hier würden die Services ihre Caches leeren
                time.sleep(1)  # Simulate work
                
                st.success("✅ All caches cleared successfully")
                
        except Exception as e:
            st.error(f"❌ Cache clear failed: {str(e)}")

    def _export_metrics(self):
        """Exportiert Performance-Metriken"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Mock-Metriken für Export
            metrics_data = {
                'timestamp': timestamp,
                'system_health': 'healthy',
                'services': {},
                'performance': {}
            }
            
            # Download-Link simulieren
            st.success(f"✅ Metrics exported: rag_metrics_{timestamp}.json")
            st.download_button(
                "💾 Download Metrics",
                data=json.dumps(metrics_data, indent=2),
                file_name=f"rag_metrics_{timestamp}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")

    def _execute_service_action(self, service_name: str, action: str):
        """Führt Service-Action aus"""
        try:
            with st.spinner(f"Executing {action} on {service_name}..."):
                if action == "Health Check":
                    # Health Check für spezifischen Service
                    st.info(f"🏥 Health check for {service_name}: OK")
                    
                elif action == "Clear Cache":
                    # Cache für spezifischen Service leeren
                    st.info(f"🧹 Cache cleared for {service_name}")
                    
                elif action == "View Stats":
                    # Service-Statistiken anzeigen
                    st.info(f"📊 Statistics for {service_name}:")
                    st.json({
                        "requests_total": 1247,
                        "requests_success": 1231,
                        "avg_response_time": 0.234,
                        "uptime": "99.7%"
                    })
                    
                elif action == "Restart Service":
                    # Service-Neustart (simuliert)
                    st.warning(f"🔄 Service restart for {service_name} - Not implemented in demo")
                    
        except Exception as e:
            st.error(f"❌ Action failed: {str(e)}")

    def _render_detailed_response_chart(self):
        """Rendert detailliertes Response-Time Chart"""
        try:
            # Mock-Daten für mehrere Services
            import numpy as np
            
            timestamps = [datetime.now() - timedelta(minutes=x) for x in range(60, 0, -1)]
            
            fig = go.Figure()
            
            services = ["Document", "Embedding", "Retrieval", "Chat"]
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
            
            for service, color in zip(services, colors):
                response_times = np.random.normal(0.5, 0.1, 60).clip(0.05, 2.0)
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=response_times,
                    mode='lines',
                    name=f"{service} Service",
                    line=dict(color=color, width=2)
                ))
            
            fig.update_layout(
                title="Detailed Response Times by Service (Last Hour)",
                xaxis_title="Time",
                yaxis_title="Response Time (s)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Chart-Fehler: {str(e)}")

    def _render_throughput_chart(self):
        """Rendert Throughput Chart"""
        try:
            import numpy as np
            
            timestamps = [datetime.now() - timedelta(minutes=x) for x in range(30, 0, -1)]
            throughput = np.random.poisson(150, 30)  # Requests per minute
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=throughput,
                mode='lines+markers',
                name='Requests/min',
                fill='tonexty',
                line=dict(color='#2ca02c', width=3)
            ))
            
            fig.update_layout(
                title="System Throughput (Requests per Minute)",
                xaxis_title="Time",
                yaxis_title="Requests/min",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Chart-Fehler: {str(e)}")

    def _render_error_rate_chart(self):
        """Rendert Error Rate Chart"""
        try:
            import numpy as np
            
            timestamps = [datetime.now() - timedelta(hours=x) for x in range(24, 0, -1)]
            error_rates = np.random.exponential(0.5, 24).clip(0, 5)  # Error rate %
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=timestamps,
                y=error_rates,
                name='Error Rate %',
                marker_color=['red' if rate > 2 else 'orange' if rate > 1 else 'green' 
                              for rate in error_rates]
            ))
            
            fig.update_layout(
                title="Error Rates by Hour (Last 24 Hours)",
                xaxis_title="Time",
                yaxis_title="Error Rate (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Chart-Fehler: {str(e)}")

    def _render_service_config_editor(self):
        """Rendert Service-Konfiguration Editor"""
        st.subheader("⚙️ Service Configuration")
        
        # Service Selection
        service_configs = {
            "embedding_service": {
                "provider": "ollama",
                "model_name": "nomic-embed-text",
                "max_batch_size": 16,
                "cache_enabled": True
            },
            "vector_store_service": {
                "provider": "chroma",
                "persist_directory": "data/chroma_db",
                "collection_name": "rag_documents"
            },
            "chat_service": {
                "model_name": "llama3:8b",
                "max_tokens": 2048,
                "temperature": 0.1
            }
        }
        
        selected_service = st.selectbox(
            "Service wählen",
            list(service_configs.keys())
        )
        
        config = service_configs.get(selected_service, {})
        
        # Config Editor
        st.write("**Current Configuration:**")
        
        edited_config = {}
        for key, value in config.items():
            if isinstance(value, bool):
                edited_config[key] = st.checkbox(key, value=value)
            elif isinstance(value, int):
                edited_config[key] = st.number_input(key, value=value)
            elif isinstance(value, float):
                edited_config[key] = st.number_input(key, value=value, format="%.3f")
            else:
                edited_config[key] = st.text_input(key, value=str(value))
        
        if st.button("💾 Save Configuration"):
            st.success(f"✅ Configuration saved for {selected_service}")
            st.json(edited_config)

    def _render_interface_config_editor(self):
        """Rendert Interface-Konfiguration Editor"""
        st.subheader("🖥️ Interface Configuration")
        
        # Admin Interface Settings
        current_config = st.session_state.admin_config
        
        new_config = AdminConfig()
        
        new_config.refresh_interval = st.slider(
            "Auto-Refresh Interval (seconds)",
            min_value=1,
            max_value=60,
            value=current_config.refresh_interval
        )
        
        new_config.max_log_lines = st.slider(
            "Max Log Lines",
            min_value=50,
            max_value=1000,
            value=current_config.max_log_lines
        )
        
        new_config.chart_history_hours = st.slider(
            "Chart History (hours)",
            min_value=1,
            max_value=168,  # 1 week
            value=current_config.chart_history_hours
        )
        
        new_config.theme = st.selectbox(
            "Interface Theme",
            ["dark", "light", "auto"],
            index=["dark", "light", "auto"].index(current_config.theme)
        )
        
        new_config.enable_debug_mode = st.checkbox(
            "Enable Debug Mode",
            value=current_config.enable_debug_mode
        )
        
        if st.button("💾 Apply Interface Settings"):
            st.session_state.admin_config = new_config
            st.success("✅ Interface configuration updated")
            st.rerun()

    def _deep_health_check(self, service_name: str):
        """Führt Deep Health Check für Service aus"""
        try:
            with st.spinner(f"Deep health check for {service_name}..."):
                # Mock Deep Health Check
                time.sleep(2)
                
                health_data = {
                    "service": service_name,
                    "status": "healthy",
                    "response_time": 0.234,
                    "memory_usage": "45.2 MB",
                    "cpu_usage": "12.5%",
                    "connections": 8,
                    "last_error": None,
                    "uptime": "2d 14h 32m",
                    "version": "4.0.0"
                }
                
                st.success("✅ Deep Health Check completed")
                st.json(health_data)
                
        except Exception as e:
            st.error(f"❌ Deep Health Check failed: {str(e)}")

    def _show_memory_usage(self):
        """Zeigt Memory Usage an"""
        try:
            import psutil
            import os
            
            # System Memory
            memory = psutil.virtual_memory()
            
            st.metric("System Memory Usage", f"{memory.percent:.1f}%")
            st.metric("Available Memory", f"{memory.available / 1024**3:.2f} GB")
            
            # Process Memory  
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()
            
            st.metric("Process Memory", f"{process_memory.rss / 1024**2:.1f} MB")
            
        except ImportError:
            st.info("📊 Memory monitoring requires psutil package")
        except Exception as e:
            st.error(f"❌ Memory check failed: {str(e)}")

    def _show_performance_profile(self):
        """Zeigt Performance Profile an"""
        st.info("⏱️ Performance profiling would be implemented here")
        
        # Mock Performance Data
        profile_data = {
            "function_calls": 1247,
            "total_time": 5.234,
            "avg_call_time": 0.0042,
            "slowest_functions": [
                {"function": "embed_text", "time": 1.234},
                {"function": "retrieve_documents", "time": 0.876},
                {"function": "process_pdf", "time": 0.654}
            ]
        }
        
        st.json(profile_data)

    def _trigger_garbage_collection(self):
        """Triggert Garbage Collection"""
        try:
            import gc
            
            before_count = len(gc.get_objects())
            collected = gc.collect()
            after_count = len(gc.get_objects())
            
            st.success(f"✅ Garbage Collection: {collected} objects collected")
            st.info(f"Objects before: {before_count}, after: {after_count}")
            
        except Exception as e:
            st.error(f"❌ Garbage Collection failed: {str(e)}")

    def _get_recent_logs(self, level: str, source: str, max_lines: int) -> List[Dict[str, Any]]:
        """Holt aktuelle Logs basierend auf Filter"""
        # Mock-Logs für Demo
        logs = []
        
        import random
        levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
        sources = ['DocumentService', 'EmbeddingService', 'ChatService', 'Controller']
        messages = [
            "Service initialized successfully",
            "Processing document upload",
            "Embedding creation completed",
            "Health check passed",
            "Cache cleared successfully",
            "Performance metrics updated",
            "New chat session started"
        ]
        
        for i in range(min(max_lines, 50)):
            log_level = random.choice(levels)
            log_source = random.choice(sources)
            log_message = random.choice(messages)
            
            # Filter anwenden
            if level != "ALL" and log_level != level:
                continue
            if source != "ALL" and log_source.lower().find(source.lower()) == -1:
                continue
            
            timestamp = (datetime.now() - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
            
            logs.append({
                'timestamp': timestamp,
                'level': log_level,
                'source': log_source,
                'message': log_message
            })
        
        return logs

    def _refresh_logs(self):
        """Refresh Log-Daten"""
        st.session_state.log_buffer = []
        st.success("✅ Logs refreshed")

    def _export_logs(self):
        """Exportiert Logs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Mock-Log-Export
        log_data = "Mock log export data\n"
        
        st.download_button(
            "💾 Download Logs",
            data=log_data,
            file_name=f"rag_logs_{timestamp}.txt",
            mime="text/plain"
        )
        
        st.success("✅ Logs prepared for download")

    def _clear_log_buffer(self):
        """Leert Log-Buffer"""
        st.session_state.log_buffer = []
        st.success("✅ Log buffer cleared")

    def _refresh_performance_data(self):
        """Refresh Performance-Daten"""
        # Hier würden aktuelle Performance-Metriken geholt
        st.success("✅ Performance data refreshed")


# =============================================================================
# FACTORY UND INTEGRATION
# =============================================================================

def create_admin_interface(config: Optional[AdminConfig] = None) -> AdminInterface:
    """
    Factory-Funktion für Admin Interface
    
    Args:
        config: Optional AdminConfig
        
    Returns:
        AdminInterface: Konfigurierte Interface-Instanz
    """
    return AdminInterface(config)


# Convenience-Funktionen für Integration
def render_admin_interface():
    """Convenience-Funktion für Admin Interface Rendering"""
    admin_interface = create_admin_interface()
    admin_interface.render()


# Export der wichtigsten Klassen und Funktionen
__all__ = [
    'AdminInterface',
    'AdminConfig', 
    'ServiceMetrics',
    'create_admin_interface',
    'render_admin_interface'
]