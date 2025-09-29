# interfaces/streamlit_ui/components.py
"""
UI Components - Wiederverwendbare Streamlit-Komponenten
Industrielle RAG-Architektur - Phase 4 Migration

Modulare, wiederverwendbare UI-Komponenten für konsistente
Benutzererfahrung und wartbaren Code.
"""

import streamlit as st
import time
from typing import Dict, Any, Optional, List, Callable
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from core.logger import get_logger

logger = get_logger(__name__)


def setup_page_config():
    """Konfiguriert Streamlit-Seite mit industriellem Theme"""
    st.set_page_config(
        page_title="RAG Industrial - Intelligente Dokumentenanalyse",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': """
            **RAG Industrial v4.0**
            
            Service-orientierte RAG-Architektur für industrielle Anwendungen.
            
            Features:
            • Modulare Controller-Architektur
            • Robustes Session-Management
            • Health-Monitoring & Performance-Tracking
            • Industrial-Grade Reliability
            """
        }
    )
    
    # Custom CSS für industrielles Theme
    st.markdown("""
    <style>
    /* Haupt-Theme */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Sidebar-Styling */
    .css-1d391kg {
        background-color: #2c3e50;
    }
    
    /* Header-Styling */
    .css-1y4p8pa {
        background-color: #34495e;
        color: white;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #3498db;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Metriken */
    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e1e8ed;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Status-Badges */
    .status-healthy {
        background-color: #27ae60;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: bold;
    }
    
    .status-warning {
        background-color: #f39c12;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: bold;
    }
    
    .status-error {
        background-color: #e74c3c;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: bold;
    }
    
    /* Chat-Message Styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chat-message.user {
        background-color: #3498db;
        color: white;
        margin-left: 20%;
    }
    
    .chat-message.assistant {
        background-color: #ecf0f1;
        border-left: 4px solid #3498db;
    }
    
    /* Upload-Area */
    .upload-area {
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #2980b9;
        background-color: #e8f4fd;
    }
    
    /* Progress-Bars */
    .stProgress > div > div > div {
        background-color: #3498db;
    }
    
    /* Alerts */
    .alert-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .alert-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar_navigation(pages: Dict[str, Dict[str, Any]], current_page: str) -> str:
    """
    Rendert Sidebar-Navigation
    
    Args:
        pages: Dictionary mit Seiten-Definitionen
        current_page: Aktuell aktive Seite
        
    Returns:
        str: Ausgewählte Seite
    """
    st.markdown("### 🧭 Navigation")
    
    selected_page = current_page
    
    for page_key, page_info in pages.items():
        icon = page_info.get('icon', '📄')
        label = page_info.get('label', page_key.title())
        description = page_info.get('description', '')
        
        # Button mit aktiver Kennzeichnung
        is_active = page_key == current_page
        button_type = "primary" if is_active else "secondary"
        
        if st.button(
            f"{icon} {label}",
            key=f"nav_{page_key}",
            type=button_type,
            help=description,
            use_container_width=True
        ):
            selected_page = page_key
    
    return selected_page


def render_system_status(health_data: Dict[str, Any], compact: bool = True):
    """
    Rendert System-Status Widget
    
    Args:
        health_data: Gesundheitsdaten vom Health Controller
        compact: Kompakte Anzeige für Sidebar
    """
    overall_status = health_data.get('overall_status', 'unknown')
    
    # Status-Mapping
    status_config = {
        'healthy': {'color': 'green', 'icon': '✅', 'text': 'Gesund'},
        'degraded': {'color': 'orange', 'icon': '⚠️', 'text': 'Beeinträchtigt'},
        'unhealthy': {'color': 'red', 'icon': '❌', 'text': 'Probleme'},
        'critical': {'color': 'red', 'icon': '🚨', 'text': 'Kritisch'},
        'unknown': {'color': 'gray', 'icon': '❓', 'text': 'Unbekannt'}
    }
    
    config = status_config.get(overall_status, status_config['unknown'])
    
    if compact:
        # Kompakte Sidebar-Anzeige
        st.markdown(f":{config['color']}[{config['icon']} **{config['text']}**]")
        
        # Komponenten-Status
        components = health_data.get('components', {})
        if components:
            healthy_count = sum(1 for comp in components.values() if comp.get('status') == 'healthy')
            total_count = len(components)
            
            st.caption(f"🔧 Komponenten: {healthy_count}/{total_count}")
        
        # Alerts
        alerts = health_data.get('alerts', {})
        alert_count = alerts.get('total', 0)
        if alert_count > 0:
            st.caption(f"⚠️ {alert_count} Alerts")
        
    else:
        # Detaillierte Anzeige
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "System-Status",
                config['text'],
                delta=None,
                delta_color="normal"
            )
        
        with col2:
            components = health_data.get('components', {})
            healthy_count = sum(1 for comp in components.values() if comp.get('status') == 'healthy')
            total_count = len(components)
            st.metric("Komponenten", f"{healthy_count}/{total_count}")
        
        with col3:
            alert_count = alerts.get('total', 0)
            st.metric("Aktive Alerts", alert_count)


def render_loading_spinner(message: str = "Verarbeitung läuft..."):
    """
    Rendert Loading-Spinner mit Nachricht
    
    Args:
        message: Anzuzeigende Nachricht
    """
    return st.spinner(message)


def render_progress_bar(
    progress: float, 
    message: str = "", 
    show_percentage: bool = True
) -> st.delta_generator.DeltaGenerator:
    """
    Rendert Progress-Bar mit optionaler Nachricht
    
    Args:
        progress: Fortschritt zwischen 0.0 und 1.0
        message: Optionale Nachricht
        show_percentage: Ob Prozentsatz angezeigt werden soll
        
    Returns:
        Progress-Bar Element für Updates
    """
    if message:
        st.text(message)
    
    progress_bar = st.progress(progress)
    
    if show_percentage:
        st.caption(f"Fortschritt: {int(progress * 100)}%")
    
    return progress_bar


def render_error_message(
    title: str, 
    message: str, 
    details: Optional[Dict[str, Any]] = None,
    show_details: bool = True
):
    """
    Rendert strukturierte Fehlermeldung
    
    Args:
        title: Fehler-Titel
        message: Haupt-Fehlermeldung
        details: Zusätzliche Fehler-Details
        show_details: Ob Details standardmäßig angezeigt werden
    """
    st.error(f"❌ **{title}**")
    st.markdown(message)
    
    if details and show_details:
        with st.expander("🔍 Technische Details", expanded=False):
            if isinstance(details, dict):
                for key, value in details.items():
                    st.write(f"**{key}:** {value}")
            else:
                st.code(str(details))


def render_success_message(
    title: str, 
    message: str, 
    details: Optional[Dict[str, Any]] = None
):
    """
    Rendert Erfolgsmeldung
    
    Args:
        title: Erfolgs-Titel
        message: Haupt-Nachricht
        details: Zusätzliche Details
    """
    st.success(f"✅ **{title}**")
    st.markdown(message)
    
    if details:
        with st.expander("📊 Details", expanded=False):
            for key, value in details.items():
                st.write(f"**{key}:** {value}")


def render_info_message(
    title: str, 
    message: str, 
    icon: str = "ℹ️"
):
    """
    Rendert Info-Nachricht
    
    Args:
        title: Info-Titel
        message: Haupt-Nachricht
        icon: Icon für die Nachricht
    """
    st.info(f"{icon} **{title}**\n\n{message}")


def render_warning_message(
    title: str, 
    message: str, 
    action_button: Optional[Dict[str, Any]] = None
):
    """
    Rendert Warnmeldung mit optionalem Action-Button
    
    Args:
        title: Warn-Titel
        message: Haupt-Nachricht
        action_button: Optional {'label': str, 'callback': callable, 'key': str}
    """
    st.warning(f"⚠️ **{title}**\n\n{message}")
    
    if action_button:
        if st.button(
            action_button['label'], 
            key=action_button.get('key', 'action_btn'),
            type="secondary"
        ):
            action_button['callback']()


def render_metric_card(
    title: str,
    value: Any,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None,
    format_func: Optional[Callable] = None
):
    """
    Rendert Metrik-Karte mit erweiterten Optionen
    
    Args:
        title: Metrik-Titel
        value: Metrik-Wert
        delta: Änderungs-Indikator
        delta_color: Farbe des Deltas ('normal', 'inverse')
        help_text: Hilfetext
        format_func: Funktion zur Wert-Formatierung
    """
    if format_func:
        formatted_value = format_func(value)
    else:
        formatted_value = value
    
    st.metric(
        label=title,
        value=formatted_value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )


def render_data_table(
    data: List[Dict[str, Any]],
    columns: List[str],
    sortable: bool = True,
    searchable: bool = True,
    paginated: bool = True,
    page_size: int = 10
):
    """
    Rendert erweiterte Daten-Tabelle
    
    Args:
        data: Liste von Zeilen-Dictionaries
        columns: Anzuzeigende Spalten
        sortable: Ob sortierbar
        searchable: Ob durchsuchbar
        paginated: Ob paginiert
        page_size: Zeilen pro Seite
    """
    if not data:
        st.info("📭 Keine Daten verfügbar")
        return
    
    # Such-Filter
    filtered_data = data
    if searchable:
        search_term = st.text_input(
            "🔍 Suchen",
            placeholder="Durchsuchen Sie die Tabelle...",
            key="table_search"
        )
        
        if search_term:
            search_lower = search_term.lower()
            filtered_data = [
                row for row in data
                if any(search_lower in str(row.get(col, '')).lower() for col in columns)
            ]
    
    # Sortierung
    if sortable:
        sort_col = st.selectbox(
            "📊 Sortieren nach",
            options=columns,
            key="table_sort"
        )
        
        sort_desc = st.checkbox("🔄 Absteigend", key="table_sort_desc")
        
        try:
            filtered_data = sorted(
                filtered_data,
                key=lambda x: x.get(sort_col, ''),
                reverse=sort_desc
            )
        except Exception:
            st.warning("⚠️ Sortierung für diese Spalte nicht möglich")
    
    # Paginierung
    if paginated and len(filtered_data) > page_size:
        total_pages = (len(filtered_data) - 1) // page_size + 1
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            current_page = st.selectbox(
                f"Seite (1-{total_pages})",
                options=list(range(1, total_pages + 1)),
                key="table_page"
            )
        
        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        page_data = filtered_data[start_idx:end_idx]
        
        st.caption(f"Zeige {start_idx + 1}-{min(end_idx, len(filtered_data))} von {len(filtered_data)} Einträgen")
    else:
        page_data = filtered_data
    
    # Tabelle rendern
    if page_data:
        # Header
        header_cols = st.columns(len(columns))
        for i, col in enumerate(columns):
            with header_cols[i]:
                st.markdown(f"**{col}**")
        
        st.divider()
        
        # Daten-Zeilen
        for row in page_data:
            data_cols = st.columns(len(columns))
            for i, col in enumerate(columns):
                with data_cols[i]:
                    value = row.get(col, '')
                    st.write(str(value))
        
    else:
        st.info("🔍 Keine Daten entsprechen den Filterkriterien")


def render_status_badge(status: str, custom_config: Optional[Dict[str, Dict[str, str]]] = None):
    """
    Rendert Status-Badge
    
    Args:
        status: Status-String
        custom_config: Custom Status-Konfiguration
    """
    default_config = {
        'healthy': {'color': 'green', 'icon': '✅'},
        'degraded': {'color': 'orange', 'icon': '⚠️'},
        'unhealthy': {'color': 'red', 'icon': '❌'},
        'critical': {'color': 'red', 'icon': '🚨'},
        'active': {'color': 'blue', 'icon': '🔵'},
        'inactive': {'color': 'gray', 'icon': '⚪'},
        'pending': {'color': 'yellow', 'icon': '🟡'},
        'success': {'color': 'green', 'icon': '✅'},
        'error': {'color': 'red', 'icon': '❌'},
        'warning': {'color': 'orange', 'icon': '⚠️'},
        'info': {'color': 'blue', 'icon': 'ℹ️'}
    }
    
    config = custom_config or default_config
    status_config = config.get(status.lower(), {'color': 'gray', 'icon': '❓'})
    
    st.markdown(
        f":{status_config['color']}[{status_config['icon']} **{status.title()}**]"
    )


def render_file_upload(
    label: str,
    file_types: List[str],
    multiple: bool = False,
    max_size_mb: int = 50,
    help_text: Optional[str] = None
):
    """
    Rendert erweiterten File-Upload mit Validierung
    
    Args:
        label: Upload-Label
        file_types: Erlaubte Dateitypen
        multiple: Mehrfach-Upload
        max_size_mb: Maximale Dateigröße
        help_text: Hilfetext
    
    Returns:
        Uploaded files oder None
    """
    help_text = help_text or f"Unterstützte Formate: {', '.join(file_types.upper())}"
    
    uploaded_files = st.file_uploader(
        label,
        type=file_types,
        accept_multiple_files=multiple,
        help=help_text
    )
    
    # Validierung
    if uploaded_files:
        files_to_validate = uploaded_files if multiple else [uploaded_files]
        
        for file in files_to_validate:
            # Größe prüfen
            size_mb = file.size / (1024 * 1024)
            if size_mb > max_size_mb:
                st.error(f"❌ Datei '{file.name}' ist zu groß: {size_mb:.1f}MB (Max: {max_size_mb}MB)")
                return None
        
        # Upload-Zusammenfassung
        if multiple and len(files_to_validate) > 1:
            total_size = sum(f.size for f in files_to_validate)
            st.info(f"📤 {len(files_to_validate)} Dateien ausgewählt ({total_size / (1024*1024):.1f}MB)")
    
    return uploaded_files


def render_confirmation_dialog(
    title: str,
    message: str,
    confirm_text: str = "Bestätigen",
    cancel_text: str = "Abbrechen",
    danger: bool = False
) -> Optional[bool]:
    """
    Rendert Bestätigungs-Dialog
    
    Args:
        title: Dialog-Titel
        message: Dialog-Nachricht
        confirm_text: Text für Bestätigen-Button
        cancel_text: Text für Abbrechen-Button
        danger: Ob es sich um eine gefährliche Aktion handelt
    
    Returns:
        True wenn bestätigt, False wenn abgebrochen, None wenn nicht interagiert
    """
    st.markdown(f"### {title}")
    st.markdown(message)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            confirm_text,
            type="primary" if not danger else "secondary",
            key="confirm_dialog_yes"
        ):
            return True
    
    with col2:
        if st.button(
            cancel_text,
            key="confirm_dialog_no"
        ):
            return False
    
    return None


def render_chat_input(
    placeholder: str = "Nachricht eingeben...",
    key: str = "chat_input",
    height: int = 100
):
    """
    Rendert Chat-Input mit erweiterten Features
    
    Args:
        placeholder: Placeholder-Text
        key: Unique key für Input
        height: Höhe des Input-Felds
    
    Returns:
        Eingabe-Text oder None
    """
    with st.form(key=f"{key}_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_area(
                "Nachricht",
                placeholder=placeholder,
                height=height,
                key=f"{key}_field",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacing
            send_button = st.form_submit_button("📤 Senden", type="primary")
            
            # Shortcuts-Info
            st.caption("💡 Ctrl+Enter")
        
        if send_button and user_input.strip():
            return user_input.strip()
    
    return None


def render_chat_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rendert Chat-Einstellungen Panel
    
    Args:
        config: Aktuelle Konfiguration
        
    Returns:
        Neue Konfiguration
    """
    with st.expander("⚙️ Chat-Einstellungen", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🤖 LLM-Parameter**")
            
            temperature = st.slider(
                "Kreativität",
                min_value=0.0,
                max_value=1.0,
                value=config.get('temperature', 0.7),
                step=0.1,
                help="Höhere Werte = kreativer"
            )
            
            max_tokens = st.slider(
                "Max. Tokens",
                min_value=100,
                max_value=2000,
                value=config.get('max_tokens', 1000),
                step=100
            )
        
        with col2:
            st.markdown("**🔍 Retrieval**")
            
            top_k = st.slider(
                "Anzahl Quellen",
                min_value=1,
                max_value=20,
                value=config.get('top_k', 5)
            )
            
            include_sources = st.checkbox(
                "Quellen anzeigen",
                value=config.get('include_sources', True)
            )
    
    return {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_k': top_k,
        'include_sources': include_sources
    }


def render_performance_chart(
    data: List[Dict[str, Any]],
    x_field: str,
    y_field: str,
    title: str = "Performance Chart",
    chart_type: str = "line"
):
    """
    Rendert Performance-Chart mit Plotly
    
    Args:
        data: Chart-Daten
        x_field: X-Achsen-Feld
        y_field: Y-Achsen-Feld
        title: Chart-Titel
        chart_type: Chart-Typ ('line', 'bar', 'scatter')
    """
    if not data:
        st.info("📊 Keine Chart-Daten verfügbar")
        return
    
    try:
        if chart_type == "line":
            fig = px.line(
                data, 
                x=x_field, 
                y=y_field,
                title=title,
                template="plotly_white"
            )
        elif chart_type == "bar":
            fig = px.bar(
                data,
                x=x_field,
                y=y_field,
                title=title,
                template="plotly_white"
            )
        elif chart_type == "scatter":
            fig = px.scatter(
                data,
                x=x_field,
                y=y_field,
                title=title,
                template="plotly_white"
            )
        else:
            st.error(f"❌ Unbekannter Chart-Typ: {chart_type}")
            return
        
        # Chart-Styling
        fig.update_layout(
            showlegend=True,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Chart-Rendering Fehler: {str(e)}")


def render_system_metrics_dashboard(metrics: Dict[str, Any]):
    """
    Rendert System-Metriken Dashboard
    
    Args:
        metrics: System-Metriken Dictionary
    """
    st.markdown("### 📊 System-Metriken")
    
    # Haupt-Metriken
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cpu_usage = metrics.get('cpu_usage', 0)
        cpu_color = "normal" if cpu_usage < 70 else "inverse"
        render_metric_card(
            "CPU-Auslastung",
            f"{cpu_usage:.1f}%",
            help_text="Aktuelle CPU-Nutzung"
        )
    
    with col2:
        memory_usage = metrics.get('memory_usage', 0)
        render_metric_card(
            "RAM-Auslastung", 
            f"{memory_usage:.1f}%",
            help_text="Aktueller RAM-Verbrauch"
        )
    
    with col3:
        disk_usage = metrics.get('disk_usage', 0)
        render_metric_card(
            "Festplatte",
            f"{disk_usage:.1f}%",
            help_text="Festplatten-Belegung"
        )
    
    with col4:
        uptime = metrics.get('uptime_hours', 0)
        render_metric_card(
            "Uptime",
            f"{uptime:.1f}h",
            help_text="Systemlaufzeit"
        )
    
    # Performance-Trends (falls verfügbar)
    if 'performance_history' in metrics:
        st.markdown("#### 📈 Performance-Verlauf")
        render_performance_chart(
            data=metrics['performance_history'],
            x_field='timestamp',
            y_field='cpu_usage',
            title="CPU-Auslastung über Zeit",
            chart_type="line"
        )


def render_document_preview(
    document: Dict[str, Any],
    max_length: int = 500
):
    """
    Rendert Dokument-Vorschau
    
    Args:
        document: Dokument-Dictionary
        max_length: Maximale Vorschau-Länge
    """
    title = document.get('title', 'Unbenanntes Dokument')
    content = document.get('content', '')
    doc_type = document.get('type', 'Unbekannt')
    
    st.markdown(f"**📄 {title}**")
    
    # Metadaten
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption(f"Typ: {doc_type}")
        
    with col2:
        size = document.get('size', 0)
        if size > 0:
            st.caption(f"Größe: {size / 1024:.1f} KB")
    
    # Content-Preview
    if content:
        if len(content) > max_length:
            preview = content[:max_length] + "..."
        else:
            preview = content
        
        st.code(preview, language='text')
    else:
        st.info("Keine Vorschau verfügbar")


def render_metadata_editor(
    metadata: Dict[str, Any],
    editable_fields: List[str],
    key_prefix: str = "metadata"
) -> Dict[str, Any]:
    """
    Rendert Metadaten-Editor
    
    Args:
        metadata: Aktuelle Metadaten
        editable_fields: Editierbare Felder
        key_prefix: Prefix für Form-Keys
        
    Returns:
        Neue Metadaten
    """
    st.markdown("**📝 Metadaten bearbeiten**")
    
    new_metadata = metadata.copy()
    
    with st.form(key=f"{key_prefix}_form"):
        for field in editable_fields:
            current_value = metadata.get(field, '')
            
            if field in ['description', 'notes']:
                new_value = st.text_area(
                    field.title(),
                    value=str(current_value),
                    key=f"{key_prefix}_{field}"
                )
            elif field == 'tags':
                if isinstance(current_value, list):
                    tag_string = ', '.join(current_value)
                else:
                    tag_string = str(current_value)
                
                new_tags = st.text_input(
                    "Tags (komma-getrennt)",
                    value=tag_string,
                    key=f"{key_prefix}_tags"
                )
                new_value = [tag.strip() for tag in new_tags.split(',') if tag.strip()]
            else:
                new_value = st.text_input(
                    field.title(),
                    value=str(current_value),
                    key=f"{key_prefix}_{field}"
                )
            
            new_metadata[field] = new_value
        
        if st.form_submit_button("💾 Speichern"):
            return new_metadata
    
    return metadata


def render_source_citations(sources: List[Dict[str, Any]], max_sources: int = 5):
    """
    Rendert Quellen-Citations
    
    Args:
        sources: Liste von Quellen-Dictionaries
        max_sources: Maximale Anzahl anzuzeigender Quellen
    """
    if not sources:
        return
    
    st.markdown("**📚 Quellen:**")
    
    for i, source in enumerate(sources[:max_sources]):
        with st.expander(f"📄 Quelle {i+1}: {source.get('title', 'Unbekannt')}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Content-Excerpt
                content = source.get('content', '')
                if content:
                    excerpt = content[:300] + "..." if len(content) > 300 else content
                    st.markdown(f"> {excerpt}")
            
            with col2:
                # Relevanz-Score
                score = source.get('score', 0.0)
                st.progress(score)
                st.caption(f"Relevanz: {int(score * 100)}%")
            
            # Metadaten
            metadata = source.get('metadata', {})
            if metadata:
                st.caption("**Metadaten:**")
                for key, value in metadata.items():
                    st.caption(f"• {key}: {value}")


def render_message_bubble(
    content: str,
    role: str,
    timestamp: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Rendert Chat-Message-Bubble
    
    Args:
        content: Message-Inhalt
        role: Rolle ('user', 'assistant', 'system')
        timestamp: Nachricht-Timestamp
        metadata: Zusätzliche Metadaten
    """
    # Timestamp
    if timestamp:
        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
    else:
        time_str = datetime.now().strftime("%H:%M:%S")
    
    if role == 'user':
        # User-Message (rechts)
        st.markdown(
            f'<div class="chat-message user">'
            f'<strong>👤 Sie:</strong><br>{content}'
            f'<div style="text-align: right; font-size: 0.8em; margin-top: 0.5rem;">{time_str}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    elif role == 'assistant':
        # Assistant-Message (links)
        st.markdown(
            f'<div class="chat-message assistant">'
            f'<strong>🤖 Assistent:</strong><br>{content}'
            f'<div style="font-size: 0.8em; margin-top: 0.5rem;">{time_str}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Quellen anzeigen
        if metadata and 'sources' in metadata:
            render_source_citations(metadata['sources'])
    
    elif role == 'system':
        # System-Message
        st.info(f"ℹ️ **System ({time_str}):** {content}")


# Utility-Funktionen
def format_file_size(size_bytes: int) -> str:
    """Formatiert Dateigröße human-readable"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.1f} MB"
    else:
        return f"{size_bytes / (1024**3):.1f} GB"


def format_duration(seconds: float) -> str:
    """Formatiert Zeitdauer human-readable"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def format_timestamp(timestamp: float, format_str: str = "%d.%m.%Y %H:%M") -> str:
    """Formatiert Timestamp human-readable"""
    return datetime.fromtimestamp(timestamp).strftime(format_str)