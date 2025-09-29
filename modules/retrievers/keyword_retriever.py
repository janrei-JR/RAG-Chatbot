        if not self.keyword_config.case_sensitive:
            text = text.lower()
        
        # Akronyme expandieren
        if self.keyword_config.acronym_expansion:
            text = self._expand_acronyms(text)
        
        # Zahlen normalisieren
        if self.keyword_config.number_normalization:
            text = self._normalize_numbers(text)
        
        # Preprocessing-Pipeline anwenden
        if self.keyword_config.preprocessing == TextPreprocessing.BASIC:
            text = self._basic_preprocessing(text)
        elif self.keyword_config.preprocessing == TextPreprocessing.STEMMING:
            text = self._stemming_preprocessing(text)
        elif self.keyword_config.preprocessing == TextPreprocessing.INDUSTRIAL:
            text = self._industrial_preprocessing(text)
        elif self.keyword_config.preprocessing == TextPreprocessing.MULTILINGUAL:
            text = self._multilingual_preprocessing(text)
        
        return text

    def _extract_query_terms(self, query_text: str) -> List[str]:
        """
        Extrahiert Suchterme aus preprocessed Query
        
        Args:
            query_text: Preprocessed Query-Text
            
        Returns:
            List[str]: Liste der Suchterme
        """
        # Basis-Tokenisierung
        terms = re.findall(r'\b\w+\b', query_text)
        
        # Längen-Filter
        terms = [term for term in terms if len(term) >= self.keyword_config.min_term_length]
        
        # N-Grams hinzufügen
        if self.keyword_config.ngram_range[1] > 1:
            ngrams = self._generate_ngrams(terms, self.keyword_config.ngram_range)
            terms.extend(ngrams)
        
        # Synonym-Expansion
        if self.keyword_config.enable_synonyms:
            expanded_terms = []
            for term in terms:
                expanded_terms.append(term)
                expanded_terms.extend(self._get_synonyms(term))
            terms = expanded_terms
        
        # Duplikate entfernen und limitieren
        unique_terms = list(set(terms))
        return unique_terms[:self.keyword_config.max_query_terms]

    def _bm25_search(self, query_terms: List[str], query: RetrievalQuery) -> List[Tuple[Document, float]]:
        """
        BM25-Algorithmus für Keyword-Suche
        
        Args:
            query_terms: Liste der Suchterme
            query: Original Query
            
        Returns:
            List[Tuple[Document, float]]: BM25-bewertete Dokumente
        """
        if not query_terms:
            return []
        
        scored_documents = []
        
        # Durchschnittliche Dokumentenlänge berechnen
        if self._document_lengths:
            avg_doc_length = sum(self._document_lengths.values()) / len(self._document_lengths)
        else:
            avg_doc_length = 100  # Fallback
        
        # Dokumente durchsuchen
        candidate_docs = self._get_candidate_documents(query_terms, query)
        
        for doc in candidate_docs:
            doc_id = doc.doc_id or str(hash(doc.content[:100]))
            
            # Dokument-Terme extrahieren falls nicht gecacht
            if doc_id not in self._document_terms:
                doc_terms = self._extract_document_terms(doc.content)
                self._document_terms[doc_id] = doc_terms
                self._document_lengths[doc_id] = len(doc_terms)
                self._update_term_frequencies(doc_id, doc_terms)
            
            doc_terms = self._document_terms[doc_id]
            doc_length = self._document_lengths[doc_id]
            
            # BM25-Score berechnen
            bm25_score = 0.0
            
            for term in query_terms:
                if term in doc_terms:
                    # Term-Frequency im Dokument
                    tf = self._term_frequencies[doc_id][term]
                    
                    # Document-Frequency (für IDF)
                    df = self._document_frequencies[term]
                    if df == 0:
                        continue
                    
                    # IDF-Komponente
                    idf = math.log((self._total_documents - df + 0.5) / (df + 0.5))
                    
                    # BM25-Formel
                    numerator = tf * (self.keyword_config.bm25_k1 + 1)
                    denominator = tf + self.keyword_config.bm25_k1 * (
                        1 - self.keyword_config.bm25_b + 
                        self.keyword_config.bm25_b * (doc_length / avg_doc_length)
                    )
                    
                    bm25_score += idf * (numerator / denominator)
                    self._term_lookups += 1
            
            # Technische Terme boosten
            if self._contains_technical_terms(doc.content):
                bm25_score *= self.keyword_config.technical_terms_boost
            
            if bm25_score > 0:
                # Normalisierung auf [0,1]
                normalized_score = min(1.0, bm25_score / 10.0)  # Empirische Normalisierung
                scored_documents.append((doc, normalized_score))
        
        # Nach Score sortieren
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        return scored_documents

    def _tfidf_search(self, query_terms: List[str], query: RetrievalQuery) -> List[Tuple[Document, float]]:
        """
        TF-IDF-Algorithmus für Keyword-Suche
        
        Args:
            query_terms: Liste der Suchterme
            query: Original Query
            
        Returns:
            List[Tuple[Document, float]]: TF-IDF-bewertete Dokumente
        """
        if not query_terms:
            return []
        
        scored_documents = []
        candidate_docs = self._get_candidate_documents(query_terms, query)
        
        for doc in candidate_docs:
            doc_id = doc.doc_id or str(hash(doc.content[:100]))
            
            # Dokument-Terme extrahieren falls nicht gecacht
            if doc_id not in self._document_terms:
                doc_terms = self._extract_document_terms(doc.content)
                self._document_terms[doc_id] = doc_terms
                self._update_term_frequencies(doc_id, doc_terms)
            
            tfidf_score = 0.0
            
            for term in query_terms:
                if term in self._term_frequencies[doc_id]:
                    # Term-Frequency
                    tf = self._term_frequencies[doc_id][term]
                    if self.keyword_config.tfidf_use_log:
                        tf = 1 + math.log(tf)
                    
                    # Inverse Document-Frequency
                    df = self._document_frequencies[term]
                    if df > 0:
                        if self.keyword_config.tfidf_smooth_idf:
                            idf = math.log(self._total_documents / df) + 1
                        else:
                            idf = math.log(self._total_documents / df)
                        
                        tfidf_score += tf * idf
                        self._term_lookups += 1
            
            # Technische Terme boosten
            if self._contains_technical_terms(doc.content):
                tfidf_score *= self.keyword_config.technical_terms_boost
            
            if tfidf_score > 0:
                # Normalisierung
                normalized_score = min(1.0, tfidf_score / 20.0)  # Empirische Normalisierung
                scored_documents.append((doc, normalized_score))
        
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        return scored_documents

    def _boolean_search(self, query_text: str, query: RetrievalQuery) -> List[Tuple[Document, float]]:
        """
        Boolean-Suche mit AND, OR, NOT Operatoren
        
        Args:
            query_text: Query mit Boolean-Operatoren
            query: Original Query
            
        Returns:
            List[Tuple[Document, float]]: Boolean-gefilterte Dokumente
        """
        # Einfache Boolean-Parsing (kann erweitert werden)
        and_terms = []
        or_terms = []
        not_terms = []
        
        # AND-Terme extrahieren (Standard)
        if ' AND ' in query_text:
            and_terms = [term.strip() for term in query_text.split(' AND ')]
        elif ' OR ' not in query_text and ' NOT ' not in query_text:
            and_terms = [term.strip() for term in query_text.split()]
        
        # OR-Terme extrahieren
        if ' OR ' in query_text:
            or_parts = query_text.split(' OR ')
            for part in or_parts:
                or_terms.extend([term.strip() for term in part.split() if 'NOT' not in term])
        
        # NOT-Terme extrahieren
        if ' NOT ' in query_text:
            not_parts = re.findall(r'NOT\s+(\w+)', query_text, re.IGNORECASE)
            not_terms.extend(not_parts)
        
        scored_documents = []
        candidate_docs = self._get_all_documents(query)
        
        for doc in candidate_docs:
            content_lower = doc.content.lower()
            matches = True
            score = 0.0
            
            # AND-Bedingungen prüfen
            if and_terms:
                and_matches = all(term.lower() in content_lower for term in and_terms if term)
                if not and_matches:
                    matches = False
                else:
                    score += 0.5  # Basis-Score für AND-Match
            
            # OR-Bedingungen prüfen
            if or_terms and matches:
                or_matches = any(term.lower() in content_lower for term in or_terms if term)
                if or_matches:
                    score += 0.3
            
            # NOT-Bedingungen prüfen
            if not_terms:
                not_matches = any(term.lower() in content_lower for term in not_terms)
                if not_matches:
                    matches = False
            
            if matches and score > 0:
                # Häufigkeit der Treffer zählen
                total_occurrences = 0
                for term in and_terms + or_terms:
                    if term:
                        total_occurrences += content_lower.count(term.lower())
                
                # Score basierend auf Häufigkeit anpassen
                frequency_bonus = min(0.5, total_occurrences * 0.05)
                final_score = min(1.0, score + frequency_bonus)
                
                scored_documents.append((doc, final_score))
        
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        return scored_documents

    def _fuzzy_search(self, query_terms: List[str], query: RetrievalQuery) -> List[Tuple[Document, float]]:
        """
        Fuzzy-Suche für Tippfehler-Toleranz
        
        Args:
            query_terms: Liste der Suchterme
            query: Original Query
            
        Returns:
            List[Tuple[Document, float]]: Fuzzy-bewertete Dokumente
        """
        scored_documents = []
        candidate_docs = self._get_all_documents(query)
        
        for doc in candidate_docs:
            doc_words = set(self._extract_document_terms(doc.content))
            fuzzy_score = 0.0
            matches_found = 0
            
            for query_term in query_terms:
                best_match_score = 0.0
                
                for doc_word in doc_words:
                    # Fuzzy-Similarity berechnen
                    similarity = self._fuzzy_similarity(query_term, doc_word)
                    
                    if similarity >= self.keyword_config.fuzzy_threshold:
                        best_match_score = max(best_match_score, similarity)
                        self._fuzzy_matches += 1
                
                if best_match_score > 0:
                    fuzzy_score += best_match_score
                    matches_found += 1
            
            # Score normalisieren
            if matches_found > 0:
                normalized_score = fuzzy_score / len(query_terms)
                scored_documents.append((doc, normalized_score))
        
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        return scored_documents

    def _phrase_search(self, query_text: str, query: RetrievalQuery) -> List[Tuple[Document, float]]:
        """
        Exakte Phrase-Suche
        
        Args:
            query_text: Query-Text als Phrase
            query: Original Query
            
        Returns:
            List[Tuple[Document, float]]: Phrase-bewertete Dokumente
        """
        scored_documents = []
        candidate_docs = self._get_all_documents(query)
        
        # Query in Anführungszeichen für exakte Phrases
        phrases = re.findall(r'"([^"]*)"', query_text)
        if not phrases:
            phrases = [query_text.strip()]
        
        for doc in candidate_docs:
            content = doc.content.lower() if not self.keyword_config.case_sensitive else doc.content
            phrase_score = 0.0
            
            for phrase in phrases:
                phrase = phrase.lower() if not self.keyword_config.case_sensitive else phrase
                
                if phrase in content:
                    # Phrase-Häufigkeit zählen
                    occurrences = content.count(phrase)
                    phrase_score += occurrences * (1.0 / len(phrase.split()))
            
            if phrase_score > 0:
                # Normalisierung basierend auf Content-Länge
                normalized_score = min(1.0, phrase_score / (len(content.split()) / 100))
                scored_documents.append((doc, normalized_score))
        
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        return scored_documents

    def _get_candidate_documents(self, query_terms: List[str], query: RetrievalQuery) -> List[Document]:
        """
        Holt Kandidaten-Dokumente basierend auf Query-Termen
        
        Args:
            query_terms: Suchterme
            query: Original Query
            
        Returns:
            List[Document]: Kandidaten-Dokumente
        """
        if self.keyword_config.build_inverted_index and self._inverted_index:
            # Inverted Index verwenden
            candidate_doc_ids = set()
            
            for term in query_terms:
                if term in self._inverted_index:
                    candidate_doc_ids.update(self._inverted_index[term])
            
            # Dokumente aus Store holen
            candidates = []
            for doc_id in candidate_doc_ids:
                doc = self._get_document_by_id(doc_id)
                if doc:
                    candidates.append(doc)
            
            return candidates
        else:
            # Fallback: Alle Dokumente durchsuchen
            return self._get_all_documents(query)

    def _get_all_documents(self, query: RetrievalQuery) -> List[Document]:
        """
        Holt alle verfügbaren Dokumente
        
        Args:
            query: Query für Filterung
            
        Returns:
            List[Document]: Alle Dokumente
        """
        if self.document_store:
            try:
                return self.document_store.get_all_documents(filters=query.filters)
            except Exception as e:
                self.logger.warning(f"Document Store Zugriff fehlgeschlagen: {e}")
                return []
        else:
            self.logger.warning("Kein Document Store verfügbar")
            return []

    def _get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Holt Dokument nach ID"""
        if self.document_store:
            try:
                return self.document_store.get_document(doc_id)
            except Exception:
                return None
        return None

    def _build_inverted_index(self):
        """
        Baut Inverted Index für schnelle Term-Lookups
        """
        if not self.document_store:
            return
        
        self.logger.info("Baue Inverted Index...")
        self._inverted_index.clear()
        self._term_frequencies.clear()
        self._document_frequencies.clear()
        
        # Alle Dokumente verarbeiten
        documents = self.document_store.get_all_documents()
        self._total_documents = len(documents)
        
        for doc in documents:
            doc_id = doc.doc_id or str(hash(doc.content[:100]))
            doc_terms = self._extract_document_terms(doc.content)
            
            # Inverted Index aktualisieren
            unique_terms = set(doc_terms)
            for term in unique_terms:
                self._inverted_index[term].add(doc_id)
                self._document_frequencies[term] += 1
            
            # Term-Frequencies speichern
            self._update_term_frequencies(doc_id, doc_terms)
            self._document_terms[doc_id] = doc_terms
            self._document_lengths[doc_id] = len(doc_terms)
        
        self._index_builds += 1
        self.logger.info(f"Inverted Index erstellt: {len(self._inverted_index)} Terms, {self._total_documents} Dokumente")

    def _extract_document_terms(self, content: str) -> List[str]:
        """
        Extrahiert Terme aus Dokument-Content
        
        Args:
            content: Dokument-Content
            
        Returns:
            List[str]: Extrahierte Terme
        """
        # Preprocessing anwenden
        processed_content = self._preprocess_query(content)
        
        # Tokenisierung
        terms = re.findall(r'\b\w+\b', processed_content)
        
        # Längen-Filter
        terms = [term for term in terms if len(term) >= self.keyword_config.min_term_length]
        
        return terms

    def _update_term_frequencies(self, doc_id: str, terms: List[str]):
        """Aktualisiert Term-Frequencies für Dokument"""
        term_counts = Counter(terms)
        for term, count in term_counts.items():
            self._term_frequencies[doc_id][term] = count

    def _generate_ngrams(self, terms: List[str], ngram_range: Tuple[int, int]) -> List[str]:
        """
        Generiert N-Grams aus Term-Liste
        
        Args:
            terms: Liste der Basis-Terme
            ngram_range: (min_n, max_n) für N-Gram-Größen
            
        Returns:
            List[str]: N-Gram-Liste
        """
        ngrams = []
        min_n, max_n = ngram_range
        
        for n in range(min_n, max_n + 1):
            if n == 1:
                continue  # Unigrams sind bereits in terms enthalten
            
            for i in range(len(terms) - n + 1):
                ngram = ' '.join(terms[i:i + n])
                ngrams.append(ngram)
        
        return ngrams

    def _fuzzy_similarity(self, str1: str, str2: str) -> float:
        """
        Berechnet Fuzzy-Similarity zwischen zwei Strings
        
        Args:
            str1: Erster String
            str2: Zweiter String
            
        Returns:
            float: Similarity-Score zwischen 0.0 und 1.0
        """
        # Einfache Edit-Distance basierte Similarity
        edit_dist = self._edit_distance(str1, str2)
        max_len = max(len(str1), len(str2))
        
        if max_len == 0:
            return 1.0
        
        similarity = 1.0 - (edit_dist / max_len)
        return max(0.0, similarity)

    def _edit_distance(self, str1: str, str2: str) -> int:
        """Berechnet Edit-Distance (Levenshtein) zwischen zwei Strings"""
        if len(str1) < len(str2):
            return self._edit_distance(str2, str1)
        
        if len(str2) == 0:
            return len(str1)
        
        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    def _expand_acronyms(self, text: str) -> str:
        """Expandiert Akronyme in Text"""
        for acronym, expansion in self._acronym_expansions.items():
            pattern = r'\b' + re.escape(acronym) + r'\b'
            text = re.sub(pattern, f"{acronym} {expansion}", text, flags=re.IGNORECASE)
        return text

    def _normalize_numbers(self, text: str) -> str:
        """Normalisiert Zahlen in Text"""
        # Dezimalzahlen normalisieren (Komma zu Punkt)
        text = re.sub(r'(\d+),(\d+)', r'\1.\2', text)
        
        # Maßeinheiten standardisieren
        unit_mappings = {
            r'\bmm\b': 'millimeter',
            r'\bcm\b': 'zentimeter', 
            r'\bm\b(?!\w)': 'meter',
            r'\bkg\b': 'kilogramm',
            r'\bg\b(?!\w)': 'gramm',
            r'\bbar\b': 'bar druck',
            r'\b°c\b': 'grad celsius'
        }
        
        for pattern, replacement in unit_mappings.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _basic_preprocessing(self, text: str) -> str:
        """Basis Text-Preprocessing"""
        # Satzzeichen entfernen (außer Bindestrichen)
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Mehrfache Leerzeichen reduzieren
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _stemming_preprocessing(self, text: str) -> str:
        """Preprocessing mit Stemming"""
        text = self._basic_preprocessing(text)
        
        # Einfaches Deutsch-Stemming (kann durch NLTK ersetzt werden)
        words = text.split()
        stemmed_words = []
        
        for word in words:
            stemmed = self._simple_german_stem(word)
            stemmed_words.append(stemmed)
        
        return ' '.join(stemmed_words)

    def _industrial_preprocessing(self, text: str) -> str:
        """Industrielles Preprocessing mit Fachbegriffen"""
        text = self._stemming_preprocessing(text)
        
        # Technische Begriffe normalisieren
        tech_normalizations = {
            r'\bspeicher\w*steuer\w*': 'sps',
            r'\bhuman\s*machine\s*interface\b': 'hmi',
            r'\bprogrammable\s*logic\s*controller\b': 'plc',
            r'\bvariable\s*frequency\s*drive\b': 'frequenzumrichter'
        }
        
        for pattern, replacement in tech_normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _multilingual_preprocessing(self, text: str) -> str:
        """Mehrsprachiges Preprocessing"""
        text = self._industrial_preprocessing(text)
        
        # Deutsch-Englisch Übersetzungen hinzufügen
        de_en_terms = {
            'motor': 'motor engine',
            'pumpe': 'pump',
            'ventil': 'valve',
            'sensor': 'sensor detector',
            'steuerung': 'control controller',
            'fehler': 'error fault'
        }
        
        words = text.split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            if word in de_en_terms:
                expanded_words.append(de_en_terms[word])
        
        return ' '.join(expanded_words)

    def _simple_german_stem(self, word: str) -> str:
        """
        Einfaches Deutsch-Stemming (Suffix-Entfernung)
        
        Args:
            word: Zu stemmendes Wort
            
        Returns:
            str: Gestemmtes Wort
        """
        if len(word) <= 3:
            return word
        
        # Deutsche Suffixe entfernen
        suffixes = ['ung', 'heit', 'keit', 'lich', 'isch', 'bar', 'los', 'voll', 'er', 'en', 'es', 'e', 's']
        
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        
        return word

    def _get_synonyms(self, term: str) -> List[str]:
        """
        Holt Synonyme für einen Begriff
        
        Args:
            term: Suchbegriff
            
        Returns:
            List[str]: Liste der Synonyme
        """
        return self._german_synonyms.get(term.lower(), [])

    def _contains_technical_terms(self, content: str) -> bool:
        """Prüft ob Content technische Begriffe enthält"""
        content_lower = content.lower()
        return any(term in content_lower for term in self._technical_terms)

    def _post_process_keyword_results(self, 
                                    results: List[Tuple[Document, float]], 
                                    query: RetrievalQuery) -> List[Tuple[Document, float]]:
        """
        Post-Processing der Keyword-Ergebnisse
        
        Args:
            results: Bewertete Ergebnisse
            query: Original Query
            
        Returns:
            List[Tuple[Document, float]]: Finalisierte Ergebnisse
        """
        processed_results = results
        
        # Score-Threshold anwenden
        threshold = getattr(query, 'score_threshold', 0.0)
        if threshold > 0.0:
            processed_results = [
                (doc, score) for doc, score in processed_results
                if score >= threshold
            ]
        
        return processed_results

    def _setup_preprocessing_pipeline(self):
        """Setup der Text-Preprocessing-Pipeline"""
        # Kann erweitert werden für komplexere Preprocessing-Schritte
        pass

    def _load_technical_terms(self) -> Set[str]:
        """Lädt technische Fachbegriffe"""
        return {
            'sps', 'plc', 'hmi', 'scada', 'profibus', 'profinet', 'modbus',
            'motor', 'antrieb', 'servo', 'frequenzumrichter', 'encoder',
            'sensor', 'aktor', 'ventil', 'pumpe', 'kompressor', 
            'temperatur', 'druck', 'durchfluss', 'füllstand',
            'pneumatik', 'hydraulik', 'safety', 'sicherheit', 'notaus',
            'wartung', 'maintenance', 'diagnose', 'kalibrierung'
        }

    def _load_acronym_expansions(self) -> Dict[str, str]:
        """Lädt Akronym-Expansionen"""
        return {
            'SPS': 'Speicherprogrammierbare Steuerung',
            'HMI': 'Human Machine Interface', 
            'SCADA': 'Supervisory Control And Data Acquisition',
            'PLC': 'Programmable Logic Controller',
            'VFD': 'Variable Frequency Drive',
            'I/O': 'Input Output',
            'AI': 'Analog Input',
            'DI': 'Digital Input',
            'CPU': 'Central Processing Unit',
            'RAM': 'Random Access Memory',
            'ROM': 'Read Only Memory'
        }

    def _load_german_synonyms(self) -> Dict[str, List[str]]:
        """Lädt deutsche Synonyme"""
        return {
            'fehler': ['störung', 'defekt', 'problem', 'alarm'],
            'motor': ['antrieb', 'maschine', 'aggregat'],
            'pumpe': ['verdichter', 'gebläse', 'kompressor'],
            'ventil': ['absperrorgan', 'regelventil', 'stellglied'], 
            'sensor': ['messfühler', 'detektor', 'aufnehmer'],
            'temperatur': ['wärme', 'hitze', 'grad'],
            'druck': ['kraft', 'belastung', 'spannung'],
            'sicherheit': ['safety', 'schutz', 'protection'],
            'wartung': ['maintenance', 'instandhaltung', 'service']
        }

    def _custom_health_check(self) -> Optional[Dict[str, Any]]:
        """
        Keyword-spezifische Health-Checks
        
        Returns:
            Dict[str, Any]: Health-Status der Keyword-Komponenten
        """
        health_data = {
            'document_store_available': self.document_store is not None,
            'algorithm': self.keyword_config.algorithm.value,
            'preprocessing': self.keyword_config.preprocessing.value,
            'inverted_index_built': bool(self._inverted_index)
        }
        
        # Document Store Health-Check
        if self.document_store:
            try:
                store_health = self.document_store.health_check()
                health_data['document_store_status'] = store_health.get('status', 'unknown')
                health_data['total_documents'] = self._total_documents
            except Exception as e:
                health_data['document_store_status'] = 'error'
                health_data['document_store_error'] = str(e)
        
        # Index-Statistiken
        if self._inverted_index:
            health_data.update({
                'index_terms': len(self._inverted_index),
                'index_builds': self._index_builds,
                'avg_terms_per_doc': len(self._inverted_index) / max(1, self._total_documents)
            })
        
        # Performance-Metriken
        health_data.update({
            'term_lookups': self._term_lookups,
            'fuzzy_matches': self._fuzzy_matches,
            'phrase_matches': self._phrase_matches,
            'boolean_queries': self._boolean_queries
        })
        
        return health_data

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Erweiterte Performance-Statistiken für Keyword Retriever
        
        Returns:
            Dict[str, Any]: Detaillierte Performance-Metriken
        """
        base_stats = super().get_performance_stats()
        
        keyword_stats = {
            'algorithm': self.keyword_config.algorithm.value,
            'preprocessing': self.keyword_config.preprocessing.value,
            'term_lookups': self._term_lookups,
            'fuzzy_matches': self._fuzzy_matches,
            'phrase_matches': self._phrase_matches,
            'boolean_queries': self._boolean_queries,
            'index_builds': self._index_builds,
            'total_indexed_documents': self._total_documents,
            'index_terms': len(self._inverted_index)
        }
        
        # Algorithmus-spezifische Stats
        if self.keyword_config.algorithm == KeywordAlgorithm.BM25:
            keyword_stats['bm25_k1'] = self.keyword_config.bm25_k1
            keyword_stats['bm25_b'] = self.keyword_config.bm25_b
        elif self.keyword_config.algorithm == KeywordAlgorithm.FUZZY:
            keyword_stats['fuzzy_threshold'] = self.keyword_config.fuzzy_threshold
            if self._total_queries > 0:
                keyword_stats['fuzzy_match_rate'] = self._fuzzy_matches / self._total_queries
        
        # Index-Effizienz
        if self._total_queries > 0 and self._term_lookups > 0:
            keyword_stats['avg_lookups_per_query'] = self._term_lookups / self._total_queries
        
        # Stats kombinieren
        base_stats.update(keyword_stats)
        return base_stats

    def rebuild_index(self):
        """Rebuild des Inverted Index"""
        if self.keyword_config.build_inverted_index:
            self._build_inverted_index()
            self.logger.info(f"Index für {self.keyword_config.name} neu aufgebaut")

    def update_algorithm(self, algorithm: KeywordAlgorithm):
        """
        Ändert Keyword-Algorithmus zur Laufzeit
        
        Args:
            algorithm: Neuer Keyword-Algorithmus
        """
        old_algorithm = self.keyword_config.algorithm
        self.keyword_config.algorithm = algorithm
        
        self.logger.info(f"Keyword-Algorithmus geändert: {old_algorithm.value} -> {algorithm.value}")

    def set_preprocessing(self, preprocessing: TextPreprocessing):
        """
        Ändert Text-Preprocessing-Strategie zur Laufzeit
        
        Args:
            preprocessing: Neue Preprocessing-Strategie
        """
        old_preprocessing = self.keyword_config.preprocessing
        self.keyword_config.preprocessing = preprocessing
        
        # Index neu aufbauen da sich Preprocessing ändert
        if self.keyword_config.build_inverted_index:
            self.rebuild_index()
        
        self.logger.info(f"Text-Preprocessing geändert: {old_preprocessing.value} -> {preprocessing.value}")

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Detaillierte Index-Statistiken
        
        Returns:
            Dict[str, Any]: Index-Informationen
        """
        if not self._inverted_index:
            return {'index_built': False}
        
        # Top-Terme nach Document-Frequency
        top_terms = sorted(
            self._document_frequencies.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Document-Length-Statistiken
        doc_lengths = list(self._document_lengths.values())
        
        return {
            'index_built': True,
            'total_terms': len(self._inverted_index),
            'total_documents': self._total_documents,
            'avg_doc_length': sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0,
            'min_doc_length': min(doc_lengths) if doc_lengths else 0,
            'max_doc_length': max(doc_lengths) if doc_lengths else 0,
            'top_terms': [{'term': term, 'doc_frequency': freq} for term, freq in top_terms],
            'index_builds': self._index_builds
        }


# =============================================================================
# ADVANCED KEYWORD RETRIEVER
# =============================================================================

class AdvancedKeywordRetriever(KeywordRetriever):
    """
    Erweiterte Keyword Retriever-Implementierung mit zusätzlichen Features
    
    Erweitert den Standard Keyword Retriever um:
    - Query-Auto-Completion
    - Term-Highlighting
    - Query-Suggestion basierend auf Index
    - Advanced Boolean-Parsing
    - Performance-optimierte Batch-Suche
    """
    
    def __init__(self, 
                 config: KeywordRetrieverConfig,
                 document_store=None):
        super().__init__(config, document_store)
        
        # Erweiterte Features
        self._query_suggestions: Dict[str, List[str]] = {}
        self._term_completions: Dict[str, Set[str]] = defaultdict(set)
        self._popular_queries: Counter = Counter()
        
        # Performance-Metriken für erweiterte Features
        self._suggestion_requests = 0
        self._completion_requests = 0
        self._batch_queries = 0
        
        self.logger.info(f"Advanced Keyword Retriever initialisiert: {config.name}")

    def get_query_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """
        Generiert Query-Vorschläge basierend auf partieller Eingabe
        
        Args:
            partial_query: Unvollständige Query
            max_suggestions: Maximum Anzahl Vorschläge
            
        Returns:
            List[str]: Liste der Query-Vorschläge
        """
        self._suggestion_requests += 1
        
        suggestions = []
        partial_lower = partial_query.lower().strip()
        
        # Aus Index-Termen Vorschläge generieren
        for term in self._inverted_index.keys():
            if term.startswith(partial_lower):
                suggestions.append(term)
        
        # Aus populären Queries Vorschläge hinzufügen
        for query in self._popular_queries.keys():
            if partial_lower in query.lower():
                suggestions.append(query)
        
        # Nach Popularität/Document-Frequency sortieren
        scored_suggestions = []
        for suggestion in set(suggestions):
            score = self._document_frequencies.get(suggestion, 0)
            scored_suggestions.append((suggestion, score))
        
        scored_suggestions.sort(key=lambda x: x[1], reverse=True)
        return [suggestion for suggestion, _ in scored_suggestions[:max_suggestions]]

    def get_term_completions(self, prefix: str, max_completions: int = 10) -> List[str]:
        """
        Auto-Completion für Term-Eingaben
        
        Args:
            prefix: Term-Präfix
            max_completions: Maximum Anzahl Completions
            
        Returns:
            List[str]: Liste der Term-Completions
        """
        self._completion_requests += 1
        
        completions = []
        prefix_lower = prefix.lower()
        
        # Aus Index-Termen filtern
        for term in self._inverted_index.keys():
            if term.startswith(prefix_lower):
                doc_freq = self._document_frequencies.get(term, 0)
                completions.append((term, doc_freq))
        
        # Nach Document-Frequency sortieren
        completions.sort(key=lambda x: x[1], reverse=True)
        return [term for term, _ in completions[:max_completions]]

    def batch_search(self, queries: List[str], k: int = 5) -> List[RetrievalResult]:
        """
        Batch-Suche für mehrere Queries gleichzeitig
        
        Args:
            queries: Liste der Query-Strings
            k: Anzahl Ergebnisse pro Query
            
        Returns:
            List[RetrievalResult]: Ergebnisse für alle Queries
        """
        self._batch_queries += 1
        
        results = []
        
        for query_text in queries:
            # Standard-Query-Objekt erstellen
            query = RetrievalQuery(text=query_text, k=k)
            
            try:
                # Standard-Retrieval ausführen
                result = self._retrieve_impl(query)
                results.append(result)
                
                # Query-Popularität tracken
                self._popular_queries[query_text] += 1
                
            except Exception as e:
                # Leeres Ergebnis bei Fehler
                self.logger.warning(f"Batch-Query fehlgeschlagen: {query_text}, Fehler: {e}")
                empty_result = RetrievalResult(
                    documents=[], 
                    query=query, 
                    total_found=0,
                    metadata={'error': str(e)}
                )
                results.append(empty_result)
        
        return results

    def highlight_terms(self, content: str, query_terms: List[str], 
                       highlight_tag: str = '<mark>') -> str:
        """
        Highlightet Query-Terme in Content
        
        Args:
            content: Zu highlightender Content
            query_terms: Suchterme zum Highlighten
            highlight_tag: HTML-Tag für Highlighting
            
        Returns:
            str: Content mit gehighlighteten Termen
        """
        highlighted = content
        close_tag = highlight_tag.replace('<', '</')
        
        for term in query_terms:
            if len(term) >= self.keyword_config.min_term_length:
                pattern = r'\b' + re.escape(term) + r'\b'
                replacement = f"{highlight_tag}{term}{close_tag}"
                highlighted = re.sub(pattern, replacement, highlighted, flags=re.IGNORECASE)
        
        return highlighted

    def get_term_context(self, term: str, doc_id: str, context_size: int = 50) -> List[str]:
        """
        Holt Kontext um einen Term in einem Dokument
        
        Args:
            term: Suchterm
            doc_id: Dokument-ID
            context_size: Anzahl Zeichen um den Term
            
        Returns:
            List[str]: Liste der Kontext-Snippets
        """
        doc = self._get_document_by_id(doc_id)
        if not doc:
            return []
        
        content = doc.content
        contexts = []
        
        # Alle Vorkommen des Terms finden
        pattern = r'\b' + re.escape(term) + r'\b'
        for match in re.finditer(pattern, content, re.IGNORECASE):
            start = max(0, match.start() - context_size)
            end = min(len(content), match.end() + context_size)
            context = content[start:end].strip()
            contexts.append(context)
        
        return contexts

    def analyze_query_performance(self, query_text: str) -> Dict[str, Any]:
        """
        Analysiert Performance-Eigenschaften einer Query
        
        Args:
            query_text: Zu analysierende Query
            
        Returns:
            Dict[str, Any]: Performance-Analyse
        """
        processed_query = self._preprocess_query(query_text)
        query_terms = self._extract_query_terms(processed_query)
        
        analysis = {
            'original_query': query_text,
            'processed_query': processed_query,
            'term_count': len(query_terms),
            'estimated_candidates': 0,
            'rare_terms': [],
            'common_terms': [],
            'unknown_terms': []
        }
        
        # Term-Analyse
        total_candidates = set()
        for term in query_terms:
            if term in self._inverted_index:
                candidates = self._inverted_index[term]
                total_candidates.update(candidates)
                
                doc_freq = self._document_frequencies.get(term, 0)
                if doc_freq < 5:
                    analysis['rare_terms'].append(term)
                elif doc_freq > self._total_documents * 0.1:
                    analysis['common_terms'].append(term)
            else:
                analysis['unknown_terms'].append(term)
        
        analysis['estimated_candidates'] = len(total_candidates)
        
        # Performance-Schätzung
        if len(total_candidates) > 1000:
            analysis['performance_warning'] = 'Viele Kandidaten-Dokumente - langsame Suche möglich'
        elif len(analysis['unknown_terms']) > len(query_terms) / 2:
            analysis['performance_warning'] = 'Viele unbekannte Terme - wenige Ergebnisse erwartet'
        
        return analysis

    def get_advanced_stats(self) -> Dict[str, Any]:
        """Erweiterte Statistiken für Advanced Features"""
        base_stats = self.get_performance_stats()
        
        advanced_stats = {
            'suggestion_requests': self._suggestion_requests,
            'completion_requests': self._completion_requests,
            'batch_queries': self._batch_queries,
            'popular_queries_count': len(self._popular_queries),
            'top_queries': self._popular_queries.most_common(5)
        }
        
        if self._suggestion_requests > 0:
            advanced_stats['avg_suggestions_per_request'] = len(self._query_suggestions) / self._suggestion_requests
        
        base_stats.update(advanced_stats)
        return base_stats


# =============================================================================
# FACTORY UND REGISTRY INTEGRATION
# =============================================================================

def create_keyword_retriever(config: Dict[str, Any], 
                            document_store=None) -> KeywordRetriever:
    """
    Factory-Funktion für Keyword Retriever-Erstellung
    
    Args:
        config: Konfiguration als Dictionary
        document_store: Document Store Instanz
        
    Returns:
        KeywordRetriever: Konfigurierte Keyword-Retriever Instanz
    """
    # Config-Objekt erstellen
    keyword_config = KeywordRetrieverConfig(
        name=config.get('name', 'keyword_retriever'),
        description=config.get('description', 'Traditional Keyword-based Text Retriever'),
        algorithm=KeywordAlgorithm(config.get('algorithm', 'bm25')),
        preprocessing=TextPreprocessing(config.get('preprocessing', 'industrial')),
        **{k: v for k, v in config.items() if k not in ['name', 'description', 'algorithm', 'preprocessing']}
    )
    
    # Advanced oder Standard Keyword Retriever
    if config.get('advanced_features', False):
        return AdvancedKeywordRetriever(keyword_config, document_store)
    else:
        return KeywordRetriever(keyword_config, document_store)


# Registrierung im Retriever-Registry (wird von __init__.py aufgerufen)
def register_keyword_retrievers():
    """Registriert Keyword Retriever im globalen Registry"""
    from .base_retriever import RetrieverRegistry
    
    RetrieverRegistry.register('keyword', KeywordRetriever)
    RetrieverRegistry.register('advanced_keyword', AdvancedKeywordRetriever)#!/usr/bin/env python3
"""
Keyword Retriever - Traditional Text-based Search
Industrielle RAG-Architektur - Module Layer

Spezialisierter Retriever für keyword-basierte, traditionelle Textsuche
mit erweiterten Algorithmen wie TF-IDF und BM25 für präzise 
Begriffssuche in industriellen RAG-Anwendungen.

Features:
- Multiple Keyword-Algorithmen (TF-IDF, BM25, Boolean, Fuzzy)
- Erweiterte Query-Processing mit Stemming und Synonyme
- N-Gram-Analyse und Phrase-Matching
- Industrielle Fachterminologie-Optimierung
- Production-Features: Inverted Index, Caching, Batch-Processing

Autor: KI-Consultant für industrielle Automatisierung
Version: 4.0.0 - Service-orientierte Architektur
"""

import re
import math
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum

# Core-Komponenten  
from core import get_logger, ValidationError, create_error_context
from core.config import get_config

# Base Retriever-Komponenten
from .base_retriever import (
    BaseRetriever, 
    RetrieverConfig, 
    RetrievalQuery, 
    RetrievalResult, 
    RetrievalMode,
    Document
)


# =============================================================================
# KEYWORD RETRIEVER KONFIGURATION
# =============================================================================

class KeywordAlgorithm(str, Enum):
    """Keyword-basierte Algorithmen für Text-Retrieval"""
    TF_IDF = "tf_idf"                    # Term Frequency-Inverse Document Frequency
    BM25 = "bm25"                        # Best Matching 25 (Standard)
    BOOLEAN = "boolean"                  # Boolean Search (AND, OR, NOT)
    FUZZY = "fuzzy"                      # Fuzzy String Matching
    PHRASE = "phrase"                    # Exact Phrase Matching
    WILDCARD = "wildcard"                # Wildcard Search (*, ?)


class TextPreprocessing(str, Enum):
    """Text-Preprocessing-Strategien"""
    BASIC = "basic"                      # Lowercase, Punctuation removal
    STEMMING = "stemming"                # Stemming + Basic
    LEMMATIZATION = "lemmatization"      # Lemmatization + Basic
    INDUSTRIAL = "industrial"            # Industrielle Fachbegriffe + Stemming
    MULTILINGUAL = "multilingual"        # Mehrsprachige Behandlung


@dataclass
class KeywordRetrieverConfig(RetrieverConfig):
    """Erweiterte Konfiguration für Keyword Retriever"""
    
    # Algorithmus-Parameter
    algorithm: KeywordAlgorithm = KeywordAlgorithm.BM25
    preprocessing: TextPreprocessing = TextPreprocessing.INDUSTRIAL
    case_sensitive: bool = False         # Case-sensitive Suche
    
    # BM25-spezifische Parameter
    bm25_k1: float = 1.5                # Term frequency saturation parameter
    bm25_b: float = 0.75                # Length normalization parameter
    
    # TF-IDF-Parameter
    tfidf_use_log: bool = True           # Logarithmic term frequency
    tfidf_smooth_idf: bool = True        # Smooth inverse document frequency
    
    # Fuzzy-Search Parameter
    fuzzy_threshold: float = 0.8         # Minimum similarity für Fuzzy Match
    max_edit_distance: int = 2           # Maximum edit distance
    
    # Query-Processing
    min_term_length: int = 2             # Minimale Term-Länge
    max_query_terms: int = 50            # Maximum Query-Begriffe
    enable_stemming: bool = True         # Stemming aktivieren
    enable_synonyms: bool = True         # Synonym-Expansion
    
    # N-Gram-Parameter
    ngram_range: Tuple[int, int] = (1, 2)  # (min_n, max_n) für N-Grams
    enable_phrase_matching: bool = True  # Phrase-Matching aktivieren
    
    # Industrielle Features
    technical_terms_boost: float = 1.3   # Boost für Fachbegriffe
    acronym_expansion: bool = True       # Akronyme expandieren
    number_normalization: bool = True    # Zahlen normalisieren
    
    # Performance-Parameter
    build_inverted_index: bool = True    # Inverted Index für Performance
    index_cache_size: int = 50000        # Cache-Größe für Index
    batch_processing: bool = True        # Batch-Verarbeitung aktivieren

    def __post_init__(self):
        """Validierung der Keyword Retriever-Konfiguration"""
        super().__post_init__()
        
        # BM25-Parameter Validierung
        if self.bm25_k1 < 0.0:
            self.bm25_k1 = 1.5
        if not (0.0 <= self.bm25_b <= 1.0):
            self.bm25_b = 0.75
        
        # Fuzzy-Parameter Validierung
        if not (0.0 <= self.fuzzy_threshold <= 1.0):
            self.fuzzy_threshold = 0.8
        if self.max_edit_distance < 0:
            self.max_edit_distance = 2
        
        # N-Gram-Parameter Validierung
        if len(self.ngram_range) != 2 or self.ngram_range[0] > self.ngram_range[1]:
            self.ngram_range = (1, 2)


# =============================================================================
# KEYWORD RETRIEVER IMPLEMENTIERUNG
# =============================================================================

class KeywordRetriever(BaseRetriever):
    """
    Keyword Retriever für traditionelle textbasierte Suche
    
    Implementiert verschiedene Keyword-basierte Algorithmen für präzise
    Begriffssuche ohne semantische Komponenten. Optimiert für industrielle
    Fachterminologie und technische Dokumentation.
    
    Features:
    - BM25 als State-of-the-Art Keyword-Algorithmus
    - TF-IDF für klassische Relevanz-Bewertung
    - Boolean-Suche für präzise Filterung
    - Fuzzy-Matching für Tippfehler-Toleranz
    - Industrielle Text-Preprocessing-Pipeline
    """
    
    def __init__(self, 
                 config: KeywordRetrieverConfig,
                 document_store=None):
        """
        Initialisiert Keyword Retriever
        
        Args:
            config: Keyword Retriever-Konfiguration
            document_store: Document Store für Volltext-Suche (Injection)
        """
        super().__init__(config)
        self.keyword_config = config
        self.document_store = document_store
        
        # Inverted Index für Performance
        self._inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self._document_terms: Dict[str, List[str]] = {}
        self._document_lengths: Dict[str, int] = {}
        self._term_frequencies: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._document_frequencies: Dict[str, int] = defaultdict(int)
        self._total_documents = 0
        
        # Industrielle Begriffe und Synonyme
        self._technical_terms = self._load_technical_terms()
        self._acronym_expansions = self._load_acronym_expansions()
        self._german_synonyms = self._load_german_synonyms()
        
        # Performance-Metriken
        self._index_builds = 0
        self._term_lookups = 0
        self._fuzzy_matches = 0
        self._phrase_matches = 0
        self._boolean_queries = 0
        
        # Text-Preprocessing-Pipeline
        self._setup_preprocessing_pipeline()
        
        self.logger.info(f"Keyword Retriever initialisiert: {config.name}")
        self.logger.info(f"Algorithmus: {config.algorithm.value}, Preprocessing: {config.preprocessing.value}")

    def _retrieve_impl(self, query: RetrievalQuery) -> RetrievalResult:
        """
        Implementiert Keyword-basierte Retrieval-Logik
        
        Args:
            query: Retrieval-Query mit Parametern
            
        Returns:
            RetrievalResult: Keyword-relevante Dokumente mit Scores
        """
        try:
            # Query-Preprocessing
            processed_query = self._preprocess_query(query.text)
            query_terms = self._extract_query_terms(processed_query)
            
            # Inverted Index aktualisieren falls nötig
            if self.keyword_config.build_inverted_index and not self._inverted_index:
                self._build_inverted_index()
            
            # Algorithmus-spezifische Suche
            if self.keyword_config.algorithm == KeywordAlgorithm.BM25:
                scored_documents = self._bm25_search(query_terms, query)
            elif self.keyword_config.algorithm == KeywordAlgorithm.TF_IDF:
                scored_documents = self._tfidf_search(query_terms, query)
            elif self.keyword_config.algorithm == KeywordAlgorithm.BOOLEAN:
                scored_documents = self._boolean_search(processed_query, query)
                self._boolean_queries += 1
            elif self.keyword_config.algorithm == KeywordAlgorithm.FUZZY:
                scored_documents = self._fuzzy_search(query_terms, query)
            elif self.keyword_config.algorithm == KeywordAlgorithm.PHRASE:
                scored_documents = self._phrase_search(processed_query, query)
                self._phrase_matches += 1
            else:
                # Default: BM25
                scored_documents = self._bm25_search(query_terms, query)
            
            # Post-Processing
            final_results = self._post_process_keyword_results(scored_documents, query)
            
            return RetrievalResult(
                documents=final_results[:query.k],
                query=query,
                total_found=len(final_results),
                processing_time_ms=0.0,  # Wird von BaseRetriever gesetzt
                metadata={
                    'algorithm': self.keyword_config.algorithm.value,
                    'query_terms': len(query_terms),
                    'candidates_found': len(scored_documents),
                    'preprocessing': self.keyword_config.preprocessing.value,
                    'index_used': bool(self._inverted_index)
                }
            )
            
        except Exception as e:
            error_context = create_error_context(
                operation="keyword_retrieve", 
                query=query.text,
                config=self.keyword_config.name
            )
            self.logger.error(f"Keyword Retrieval Fehler: {e}", extra=error_context)
            raise

    def _preprocess_query(self, query_text: str) -> str:
        """
        Preprocesses Query-Text für Keyword-Suche
        
        Args:
            query_text: Original Query-Text
            
        Returns:
            str: Preprocessed Query-Text
        """
        text = query_text.strip()
        
        # Case-handling
        if not self.keyword_config.case_sensitive: