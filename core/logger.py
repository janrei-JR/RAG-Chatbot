#!/usr/bin/env python3
"""
Core Logger - NoneType getMessage Fehler behoben
Industrielle RAG-Architektur - Kritischer Logging Bugfix

PROBLEM BEHOBEN:
- 'NoneType' object has no attribute 'getMessage'
- Fehlerhafte Log-Datei Erstellung
- Missing Directory Creation
"""

import logging
import logging.handlers
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from functools import wraps


# =============================================================================
# CUSTOM LOG RECORD - BUGFIX für getMessage
# =============================================================================

class SafeLogRecord(logging.LogRecord):
    """
    Sichere LogRecord-Implementierung die NoneType getMessage Fehler verhindert
    
    BUGFIX: Behandelt None-Werte sicher
    """
    
    def getMessage(self) -> str:
        """
        Sichere getMessage Implementierung
        
        BUGFIX: Verhindert NoneType Attribute Error
        """
        try:
            # Standard getMessage Logic mit Sicherheitsprüfungen
            if self.msg is None:
                return "None"
            
            if isinstance(self.msg, str):
                if self.args:
                    try:
                        return self.msg % self.args
                    except (TypeError, ValueError):
                        return f"{self.msg} (args: {self.args})"
                else:
                    return self.msg
            else:
                return str(self.msg)
                
        except Exception as e:
            # Fallback bei jedem Fehler
            return f"LOG_ERROR: {e} (original_msg: {self.msg}, args: {self.args})"


# =============================================================================
# CUSTOM FORMATTER MIT JSON SUPPORT
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON-Format für strukturierte Logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatiere LogRecord als JSON - mit Fehlerbehandlung"""
        try:
            # Sichere Message-Extraktion
            if hasattr(record, 'getMessage'):
                message = record.getMessage()
            else:
                message = str(record.msg) if record.msg is not None else "None"
            
            log_obj = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": message,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Exception-Informationen hinzufügen
            if record.exc_info:
                log_obj["exception"] = {
                    "type": record.exc_info[0].__name__ if record.exc_info[0] else "Unknown",
                    "message": str(record.exc_info[1]) if record.exc_info[1] else "None",
                    "traceback": traceback.format_exception(*record.exc_info)
                }
            
            # Zusätzliche Felder aus LogRecord
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                              'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process', 'getMessage']:
                    try:
                        # Nur JSON-serialisierbare Werte
                        json.dumps(value)
                        log_obj[key] = value
                    except (TypeError, ValueError):
                        log_obj[key] = str(value)
            
            return json.dumps(log_obj, ensure_ascii=False)
            
        except Exception as e:
            # Fallback bei JSON-Formatierungs-Fehlern
            return f"JSON_FORMAT_ERROR: {e} - Original: {record}"


class SafeFormatter(logging.Formatter):
    """
    Sichere Standard-Formatter-Implementierung
    
    BUGFIX: Verhindert NoneType Fehler bei der Formatierung
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Sichere Formatierung mit Fehlerbehandlung"""
        try:
            # Ersetze None-Werte durch sichere Alternativen
            if record.msg is None:
                record.msg = "None"
            
            # Standard-Formatierung
            formatted = super().format(record)
            return formatted
            
        except Exception as e:
            # Fallback-Formatierung
            timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
            return f"{timestamp} - {record.name} - {record.levelname} - FORMAT_ERROR: {e}"


# =============================================================================
# LOGGER CONFIGURATION
# =============================================================================

class RAGLogger:
    """
    Zentrale Logger-Konfiguration für das RAG-System
    
    BUGFIXES:
    - Sichere Directory-Erstellung
    - NoneType getMessage Fehler behoben
    - Robuste Datei-Handler
    """
    
    def __init__(
        self,
        name: str = "RAG",
        level: str = "INFO",
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        file_path: Optional[Union[str, Path]] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        json_format: bool = False,
        console_output: bool = True
    ):
        self.name = name
        self.level = level
        self.log_format = log_format
        self.file_path = Path(file_path) if file_path else None
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.json_format = json_format
        self.console_output = console_output
        
        # Logger erstellen
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Bestehende Handler entfernen
        self.logger.handlers.clear()
        
        # Handler konfigurieren
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Konfiguriere Logger-Handler - MIT BUGFIXES"""
        try:
            # Console Handler
            if self.console_output:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(getattr(logging, self.level.upper()))
                
                if self.json_format:
                    console_handler.setFormatter(JSONFormatter())
                else:
                    console_handler.setFormatter(SafeFormatter(self.log_format))
                
                self.logger.addHandler(console_handler)
            
            # File Handler - MIT DIRECTORY CREATION BUGFIX
            if self.file_path:
                try:
                    # BUGFIX: Erstelle Verzeichnis falls es nicht existiert
                    self.file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Rotating File Handler
                    file_handler = logging.handlers.RotatingFileHandler(
                        filename=str(self.file_path),
                        maxBytes=self.max_file_size,
                        backupCount=self.backup_count,
                        encoding='utf-8'
                    )
                    file_handler.setLevel(getattr(logging, self.level.upper()))
                    
                    if self.json_format:
                        file_handler.setFormatter(JSONFormatter())
                    else:
                        file_handler.setFormatter(SafeFormatter(self.log_format))
                    
                    self.logger.addHandler(file_handler)
                    
                except Exception as e:
                    # Fallback: Console-Warnung bei File-Handler Fehler
                    print(f"WARNUNG: Log-Datei konnte nicht erstellt werden: {e}")
                    
        except Exception as e:
            print(f"KRITISCHER FEHLER: Logger-Setup fehlgeschlagen: {e}")
            # Minimaler Fallback-Logger
            fallback_handler = logging.StreamHandler(sys.stdout)
            fallback_handler.setFormatter(SafeFormatter("%(levelname)s: %(message)s"))
            self.logger.addHandler(fallback_handler)
    
    def get_logger(name: str = "RAG", *args, **kwargs):
        """
        Holt Logger-Instanz mit flexiblen Parametern

        Args:
            name: Logger-Name
            *args: Zusätzliche Name-Teile
            **kwargs: Weitere Parameter (ignoriert)

        Returns:
            Logger-Instanz
        """
        if args:
            full_name = ".".join([name] + [str(arg) for arg in args])
        else:
            full_name = name

        return logging.getLogger(full_name)
# =============================================================================
# GLOBAL LOGGER SETUP - MIT BUGFIXES
# =============================================================================

# Globale Logger-Instanz
_global_logger: Optional[RAGLogger] = None
_initialized = False

def setup_logging(
    level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    file_path: Optional[Union[str, Path]] = None,
    max_file_size: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    json_format: bool = False,
    console_output: bool = True
) -> None:
    """
    Globale Logger-Konfiguration - MIT BUGFIXES
    
    BUGFIXES:
    - Sichere Directory-Erstellung
    - NoneType Fehlerbehandlung
    - Robust gegen mehrfache Initialisierung
    """
    global _global_logger, _initialized
    
    if _initialized:
        return  # Bereits initialisiert
    
    try:
        # Standard-Pfad falls keiner angegeben
        if file_path is None:
            default_log_dir = Path("./data/logs")
            file_path = default_log_dir / "rag_system.log"
        
        # RAGLogger erstellen
        _global_logger = RAGLogger(
            name="RAG",
            level=level,
            log_format=log_format,
            file_path=file_path,
            max_file_size=max_file_size,
            backup_count=backup_count,
            json_format=json_format,
            console_output=console_output
        )
        
        _initialized = True
        
        # Test-Log um sicherzustellen dass alles funktioniert
        logger = _global_logger.get_logger()
        logger.info("Logging-System erfolgreich initialisiert")
        
    except Exception as e:
        print(f"KRITISCHER FEHLER: Logging-Setup fehlgeschlagen: {e}")
        # Minimaler Fallback
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        _initialized = True


def get_logger(name: str = "RAG", *args, **kwargs):
    """
    Holt Logger-Instanz mit flexiblen Parametern

    Args:
        name: Logger-Name
        *args: Zusätzliche Name-Teile
        **kwargs: Weitere Parameter (ignoriert)

    Returns:
        Logger-Instanz
    """
    if args:
        full_name = ".".join([name] + [str(arg) for arg in args])
    else:
        full_name = name

    return logging.getLogger(full_name)
# =============================================================================
# PERFORMANCE MONITORING DECORATOR - BUGFIX
# =============================================================================

def log_performance(operation_name: Optional[str] = None):
    """
    Decorator für Performance-Monitoring - MIT BUGFIXES
    
    BUGFIX: Sichere Zeiterfassung und Fehlerbehandlung
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger = get_logger(f"performance.{func.__module__}")
            
            op_name = operation_name or f"{func.__name__}"
            
            try:
                logger.debug(f"[PERFORMANCE] {op_name} gestartet")
                result = func(*args, **kwargs)
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"[PERFORMANCE] {op_name} abgeschlossen in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"[PERFORMANCE] {op_name} fehlgeschlagen nach {duration:.3f}s: {e}")
                raise
                
        return wrapper
    return decorator


def log_method_calls(include_args: bool = False, include_result: bool = False):
    """
    Decorator für Method-Call-Logging - MIT BUGFIXES
    
    BUGFIX: Sichere Argument-Serialisierung
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(f"calls.{func.__module__}")
            
            try:
                # Sichere Argument-Darstellung
                if include_args:
                    safe_args = []
                    for arg in args:
                        try:
                            safe_args.append(str(arg)[:100])  # Limitiere Länge
                        except:
                            safe_args.append("<unrepresentable>")
                    
                    safe_kwargs = {}
                    for k, v in kwargs.items():
                        try:
                            safe_kwargs[k] = str(v)[:100]  # Limitiere Länge
                        except:
                            safe_kwargs[k] = "<unrepresentable>"
                    
                    logger.debug(f"[CALL] {func.__name__}(args={safe_args}, kwargs={safe_kwargs})")
                else:
                    logger.debug(f"[CALL] {func.__name__}()")
                
                result = func(*args, **kwargs)
                
                if include_result:
                    try:
                        safe_result = str(result)[:200]  # Limitiere Länge
                        logger.debug(f"[RESULT] {func.__name__} -> {safe_result}")
                    except:
                        logger.debug(f"[RESULT] {func.__name__} -> <unrepresentable>")
                
                return result
                
            except Exception as e:
                logger.error(f"[ERROR] {func.__name__} failed: {e}")
                raise
                
        return wrapper
    return decorator


# =============================================================================
# EMERGENCY LOGGING - FALLBACK SYSTEM
# =============================================================================

def emergency_log(message: str, level: str = "ERROR") -> None:
    """
    Notfall-Logging falls das Haupt-System fehlschlägt
    
    BUGFIX: Immer verfügbare Logging-Funktion
    """
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        emergency_msg = f"{timestamp} - EMERGENCY - {level} - {message}"
        
        # Versuche verschiedene Ausgabekanäle
        try:
            print(emergency_msg, file=sys.stderr)
        except:
            pass
        
        try:
            # Versuche in Datei zu schreiben
            emergency_log_file = Path("./emergency.log")
            with open(emergency_log_file, "a", encoding="utf-8") as f:
                f.write(emergency_msg + "\n")
        except:
            pass
            
    except:
        # Absoluter Fallback
        try:
            print(f"EMERGENCY: {message}", file=sys.stderr)
        except:
            pass


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    "RAGLogger",
    "setup_logging",
    "get_logger",
    "log_performance",
    "log_method_calls",
    "emergency_log",
    "JSONFormatter",
    "SafeFormatter",
    "SafeLogRecord"
]