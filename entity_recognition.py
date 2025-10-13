import re
import spacy
from typing import Dict, List, Any
from collections import Counter

class MedicalEntityRecognizer:
    def __init__(self):
        try:
            # Try to load spaCy medical model, fallback to English
            self.nlp = spacy.load("en_core_sci_sm")
            self.use_spacy = True
        except OSError:
            try:
                # Fallback to standard English model
                self.nlp = spacy.load("en_core_web_sm")
                self.use_spacy = True
                print("Using standard spaCy model (en_core_web_sm)")
            except OSError:
                print("spaCy model not found, using regex-based NER")
                self.nlp = None
                self.use_spacy = False
        
        # Enhanced medical entity patterns
        self.patterns = {
            'diseases': [
                r'\b(?:diabetes|hypertension|asthma|pneumonia|migraine|cancer|heart disease|'
                r'arthritis|alzheimer|parkinson|stroke|copd|epilepsy|depression|anxiety|'
                r'covid[-\s]?19|coronavirus|influenza|flu|tuberculosis|hiv|aids|hepatitis|'
                r'osteoporosis|osteoarthritis|multiple sclerosis|lupus|fibromyalgia|'
                r'crohn\'?s disease|ulcerative colitis|celiac disease)\b'
            ],
            'symptoms': [
                r'\b(?:fever|cough|headache|fatigue|nausea|vomiting|pain|dizziness|'
                r'weakness|shortness of breath|chest pain|abdominal pain|'
                r'blurred vision|weight loss|weight gain|rash|swelling|'
                r'numbness|tingling|palpitations|constipation|diarrhea|'
                r'bleeding|bruising|inflammation|stiffness)\b'
            ],
            'treatments': [
                r'\b(?:chemotherapy|surgery|medication|therapy|insulin|'
                r'ace inhibitors|beta blockers|diuretics|vaccine|antibiotics|'
                r'physical therapy|radiation|antiviral|antidepressant|'
                r'analgesic|anti-inflammatory|statin|vaccination|'
                r'immunotherapy|biologics|transplant)\b'
            ],
            'body_parts': [
                r'\b(?:heart|liver|kidney|lungs|brain|stomach|intestines|'
                r'arteries|veins|nerves|muscles|bones|joints|skin|eyes|'
                r'ears|nose|throat|pancreas|spleen|bladder|prostate|ovaries)\b'
            ],
            'medical_tests': [
                r'\b(?:x-ray|mri|ct scan|blood test|urine test|biopsy|'
                r'ekg|ecg|ultrasound|endoscopy|colonoscopy|mammogram|'
                r'pap smear|blood pressure|cholesterol test|glucose test)\b'
            ],
            'medications': [
                r'\b(?:aspirin|ibuprofen|metformin|lisinopril|atorvastatin|'
                r'levothyroxine|metoprolol|amlodipine|omeprazole|'
                r'simvastatin|losartan|albuterol|warfarin|insulin)\b'
            ]
        }
        
        # Compile regex patterns
        self.compiled_patterns = {}
        for entity_type, pattern_list in self.patterns.items():
            combined_pattern = '|'.join(pattern_list)
            self.compiled_patterns[entity_type] = re.compile(combined_pattern, re.IGNORECASE)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text using multiple methods"""
        entities = {}
        
        if self.use_spacy and self.nlp:
            # Use spaCy for more accurate entity recognition
            spacy_entities = self._extract_entities_spacy(text)
            entities.update(spacy_entities)
        
        # Fallback to regex patterns
        regex_entities = self._extract_entities_regex(text)
        for entity_type, entity_list in regex_entities.items():
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].extend(entity_list)
        
        # Remove duplicates
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
        
        return entities
    
    def _extract_entities_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy"""
        entities = {}
        doc = self.nlp(text)
        
        for ent in doc.ents:
            entity_type = self._map_spacy_label(ent.label_)
            if entity_type:
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append(ent.text.lower())
        
        return entities
    
    def _map_spacy_label(self, label: str) -> str:
        """Map spaCy entity labels to our categories"""
        mapping = {
            'DISEASE': 'diseases',
            'SYMPTOM': 'symptoms',
            'TREATMENT': 'treatments',
            'ANATOMY': 'body_parts',
            'CHEMICAL': 'medications'
        }
        return mapping.get(label, None)
    
    def _extract_entities_regex(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using regex patterns"""
        entities = {}
        
        for entity_type, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text)
            # Remove duplicates and convert to lowercase
            unique_matches = list(set([match.lower().strip() for match in matches if len(match.strip()) > 2]))
            if unique_matches:
                entities[entity_type] = unique_matches
        
        return entities
    
    def get_entity_summary(self, entities: Dict[str, List[str]]) -> str:
        """Create a formatted summary of extracted entities"""
        if not any(entities.values()):
            return "No medical entities detected."
        
        summary_parts = []
        for entity_type, entity_list in entities.items():
            if entity_list:
                # Limit to top 5 entities per type for readability
                display_entities = entity_list[:5]
                summary_parts.append(f"**{entity_type.replace('_', ' ').title()}:** {', '.join(display_entities)}")
        
        return " | ".join(summary_parts)
    
    def get_entity_statistics(self, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Get statistics about extracted entities"""
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        entity_counts = {entity_type: len(entity_list) for entity_type, entity_list in entities.items()}
        
        return {
            'total_entities': total_entities,
            'entity_counts': entity_counts,
            'entity_types_present': list(entities.keys())
        }
    
    def analyze_entity_distribution(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze entity distribution across multiple texts"""
        all_entities = []
        entity_frequencies = Counter()
        
        for text in texts:
            entities = self.extract_entities(text)
            for entity_type, entity_list in entities.items():
                all_entities.extend([(entity_type, entity) for entity in entity_list])
                entity_frequencies.update(entity_list)
        
        # Most common entities
        most_common = entity_frequencies.most_common(10)
        
        # Entity type distribution
        type_distribution = Counter(entity_type for entity_type, _ in all_entities)
        
        return {
            'total_entities_found': len(all_entities),
            'most_common_entities': most_common,
            'entity_type_distribution': dict(type_distribution),
            'unique_entities': len(entity_frequencies)
        }