# 📋 Iterative Bugfix ToDo-Liste - Schritt für Schritt

**Ansatz:** Sicherheitsorientierte, schrittweise Reparatur ohne Breaking Changes  
**Priorität:** Ein Service nach dem anderen funktionsfähig machen  
**Validierung:** Nach jedem Schritt testen, ob bestehende Funktionalität erhalten bleibt

---

## 🎯 PHASE 1: FOUNDATION STABILISIEREN

### ✅ ABGESCHLOSSEN
- [x] **SCHRITT 0:** BUG-001, BUG-002, BUG-003 (InterfaceException, RAGSystemException, get_current_config)

---

## 🔴 PHASE 2: EINZELNE SERVICES REPARIEREN

### **SCHRITT 1: ConfigurationError für EmbeddingService** 
**Ziel:** EmbeddingService funktionsfähig machen  
**Fehler:** `cannot import name 'ConfigurationError' from 'core.exceptions'`

- [ ] **1.1** `ConfigurationError` Klasse zu `core/exceptions.py` hinzufügen
- [ ] **1.2** `ConfigurationError` zu `__all__` Liste in `exceptions.py` hinzufügen  
- [ ] **1.3** `ConfigurationError` zu `core/__init__.py` Export hinzufügen
- [ ] **1.4** **TEST:** `python -c "from core.exceptions import ConfigurationError; print('✅ ConfigurationError verfügbar')"`
- [ ] **1.5** **VALIDATION:** Streamlit starten und EmbeddingService-Status prüfen

**Erwartetes Resultat:** EmbeddingService lädt ohne ImportError

---

### **SCHRITT 2: ChatServiceError für ChatService**
**Ziel:** ChatService funktionsfähig machen  
**Fehler:** `cannot import name 'ChatServiceError' from 'core'`

- [ ] **2.1** `ChatServiceError` Klasse zu `core/exceptions.py` hinzufügen
- [ ] **2.2** `ChatServiceError` zu `__all__` Listen hinzufügen
- [ ] **2.3** `ChatServiceError` zu `core/__init__.py` Export hinzufügen
- [ ] **2.4** **TEST:** `python -c "from core import ChatServiceError; print('✅ ChatServiceError verfügbar')"`
- [ ] **2.5** **VALIDATION:** ChatService-Import testen

**Erwartetes Resultat:** ChatService lädt ohne ImportError

---

### **SCHRITT 3: ResourceError für SessionService**
**Ziel:** SessionService funktionsfähig machen  
**Fehler:** `cannot import name 'ResourceError' from 'core'`

- [ ] **3.1** `ResourceError` Klasse zu `core/exceptions.py` hinzufügen
- [ ] **3.2** `ResourceError` zu Export-Listen hinzufügen
- [ ] **3.3** **TEST:** `python -c "from core import ResourceError; print('✅ ResourceError verfügbar')"`
- [ ] **3.4** **VALIDATION:** SessionService-Import testen

**Erwartetes Resultat:** SessionService lädt ohne ImportError

---

### **SCHRITT 4: RetrievalService + SearchService (ConfigurationError)**
**Ziel:** RetrievalService und SearchService funktionsfähig machen  
**Fehler:** Gleicher ConfigurationError wie EmbeddingService

- [ ] **4.1** Prüfen ob ConfigurationError aus Schritt 1 bereits beide Services repariert
- [ ] **4.2** **TEST:** RetrievalService Import einzeln testen
- [ ] **4.3** **TEST:** SearchService Import einzeln testen
- [ ] **4.4** **VALIDATION:** Beide Services in Streamlit-Log prüfen

**Erwartetes Resultat:** Beide Services laden ohne ImportError (sollte durch Schritt 1 gelöst sein)

---

## 🟡 PHASE 3: CONTROLLER REPARIEREN

### **SCHRITT 5: ServiceException für Controller**
**Ziel:** PipelineController und HealthController funktionsfähig machen  
**Fehler:** `cannot import name 'ServiceException' from 'core.exceptions'`

- [ ] **5.1** `ServiceException` Klasse zu `core/exceptions.py` hinzufügen
- [ ] **5.2** `ServiceException` zu Export-Listen hinzufügen
- [ ] **5.3** **TEST:** `python -c "from core.exceptions import ServiceException; print('✅ ServiceException verfügbar')"`
- [ ] **5.4** **VALIDATION:** PipelineController und HealthController Import testen

**Erwartetes Resultat:** Beide Controller laden ohne ImportError

---

### **SCHRITT 6: SessionException für SessionController**
**Ziel:** SessionController funktionsfähig machen  
**Fehler:** `cannot import name 'SessionException' from 'core.exceptions'`

- [ ] **6.1** `SessionException` Klasse zu `core/exceptions.py` hinzufügen
- [ ] **6.2** `SessionException` zu Export-Listen hinzufügen
- [ ] **6.3** **TEST:** `python -c "from core.exceptions import SessionException; print('✅ SessionException verfügbar')"`
- [ ] **6.4** **VALIDATION:** SessionController Import testen

**Erwartetes Resultat:** SessionController lädt ohne ImportError

---

## 🟠 PHASE 4: KONFIGURATIONSFEHLER BEHEBEN

### **SCHRITT 7: RAGConfig.default_provider Property**
**Ziel:** VectorStoreService Initialisierung reparieren  
**Fehler:** `'RAGConfig' object has no attribute 'default_provider'`

- [ ] **7.1** `default_provider` Property zu `RAGConfig` Klasse in `core/config.py` hinzufügen
- [ ] **7.2** Property-Logik implementieren (Vector Store Provider ermitteln)
- [ ] **7.3** **TEST:** `python -c "from core import get_config; print(get_config().default_provider)"`
- [ ] **7.4** **VALIDATION:** VectorStoreService Initialisierung testen

**Erwartetes Resultat:** VectorStoreService initialisiert ohne AttributeError

---

### **SCHRITT 8: Logger-Parameter-Problem beheben**
**Ziel:** DocumentService Logger-Fehler beheben  
**Fehler:** `get_logger() takes from 0 to 1 positional arguments but 2 were given`

**Option A: Logger-Funktion erweitern**
- [ ] **8.1** `get_logger()` in `core/logger.py` um zweiten Parameter erweitern
- [ ] **8.2** Backward-Kompatibilität sicherstellen
- [ ] **8.3** **TEST:** Beide Aufruf-Varianten testen

**Option B: Service-Aufrufe korrigieren**
- [ ] **8.1** DocumentService Logger-Aufruf korrigieren
- [ ] **8.2** Andere Services mit 2-Parameter-Aufrufen identifizieren und korrigieren

- [ ] **8.4** **VALIDATION:** DocumentService Initialisierung testen

**Erwartetes Resultat:** DocumentService initialisiert ohne Logger-Fehler

---

## 🔵 PHASE 5: CONTROLLER-INITIALISIERUNG REPARIEREN

### **SCHRITT 9: Controller NoneType-Fehler beheben**
**Ziel:** Controller-Instanzen erfolgreich erstellen  
**Fehler:** `'NoneType' object is not callable`

- [ ] **9.1** Controller-Factory-Code analysieren
- [ ] **9.2** Controller-Instanziierung debuggen  
- [ ] **9.3** Service-Dependencies für Controller prüfen
- [ ] **9.4** **VALIDATION:** Controller-Erstellung testen

**Erwartetes Resultat:** Alle drei Controller erfolgreich instanziiert

---

## 📊 ZWISCHEN-VALIDIERUNGEN

### **Nach jedem Schritt ausführen:**

```bash
# 1. Import-Test
python -c "from core.exceptions import [NEUE_EXCEPTION]; print('✅ Import erfolgreich')"

# 2. Service-Status prüfen  
python -c "
from services import initialize_services
services = initialize_services()
print('Verfügbare Services:', list(services.keys()))
"

# 3. Vollständiger Anwendungsstart
streamlit run main_rag.py
# Logfile auf neue Fehler prüfen
```

### **Rollback-Strategie bei Problemen:**
- Letzte Änderung rückgängig machen
- Isolation testen: Nur die neue Exception einzeln importieren
- Konflikt-Analyse: Prüfen ob neue Exception bestehende überschreibt

---

## 🎯 SUCCESS METRICS

### **Nach Phase 2 (Services):**
- [ ] Mind. 5/7 Services laden ohne ImportError
- [ ] EmbeddingService, ChatService, SessionService funktionsfähig
- [ ] Keine neuen Breaking Changes in bestehenden Services

### **Nach Phase 3 (Controller):**
- [ ] Alle 3 Controller laden ohne ImportError  
- [ ] Controller-Factory erstellt Instanzen erfolgreich

### **Nach Phase 4 (Config):**
- [ ] VectorStoreService initialisiert ohne Fehler
- [ ] Alle Logger-Aufrufe funktionieren
- [ ] Anwendung startet vollständig durch

### **Finale Validierung:**
```bash
# Sollte keine kritischen ImportErrors mehr zeigen:
streamlit run main_rag.py 2>&1 | grep -i "importerror\|cannot import"
```

---

## ⚠️ VORSICHTSMASSNAHMEN

1. **Vor jeder Änderung:** Git commit/branch erstellen
2. **Nach jeder Änderung:** Sofortiger Test der spezifischen Komponente  
3. **Reihenfolge einhalten:** Nicht vorgreifen auf spätere Schritte
4. **Bei Problemen:** Schritt isoliert debuggen, nicht mehrere gleichzeitig ändern
5. **Dokumentation:** Jede Änderung im Code kommentieren mit Schritt-Nummer

---

**Start mit SCHRITT 1 wenn bereit! 🚀**