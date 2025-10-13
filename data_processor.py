import pandas as pd
import os
from datasets import load_dataset
import pickle
from typing import List, Dict, Any
import re

class MedQuADProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def load_all_data(self) -> pd.DataFrame:
        """Load MedQuAD dataset from Hugging Face with enhanced processing"""
        try:
            # Load MedQuAD dataset from Hugging Face
            print("Loading MedQuAD dataset from Hugging Face...")
            dataset = load_dataset("abachaa/MedQuAD", trust_remote_code=True)
            
            # Convert to pandas DataFrame
            all_data = []
            
            # Process each split
            for split_name, split_data in dataset.items():
                print(f"Processing {split_name} split with {len(split_data)} examples")
                
                for item in split_data:
                    processed_item = self._process_item(item)
                    if processed_item:
                        all_data.append(processed_item)
            
            df = pd.DataFrame(all_data)
            print(f"Loaded {len(df)} total Q&A pairs")
            
            # Enhanced data cleaning
            df = self._clean_data(df)
            
            # Add metadata
            df = self._add_metadata(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading from Hugging Face: {e}")
            print("Falling back to enhanced sample data...")
            return self._create_enhanced_sample_data()
    
    def _process_item(self, item: Dict) -> Dict:
        """Process individual dataset item"""
        try:
            return {
                'question': item.get('question', '').strip(),
                'answer': item.get('answer', '').strip(),
                'doc_type': item.get('doc_type', 'medical'),
                'question_type': self._classify_question_type(item.get('question', '')),
                'source': item.get('source', 'MedQuAD'),
                'answer_length': len(item.get('answer', '')),
                'question_length': len(item.get('question', ''))
            }
        except Exception as e:
            print(f"Error processing item: {e}")
            return None
    
    def _classify_question_type(self, question: str) -> str:
        """Enhanced question type classification"""
        question_lower = question.lower()
        
        type_patterns = {
            'symptoms': r'symptom|sign|feel|experience|manifestation',
            'treatment': r'treat|cure|medication|therapy|drug|management|how to treat',
            'causes': r'cause|reason|why|develop|etiology|pathogenesis',
            'diagnosis': r'diagnos|test|detect|identif|confirm|screen',
            'prevention': r'prevent|avoid|risk|reduce.*risk|preventive',
            'definition': r'what is|define|meaning|what does|explain',
            'prognosis': r'prognosis|outcome|survival|recovery|progression',
            'complications': r'complication|side effect|adverse|risk factor'
        }
        
        for q_type, pattern in type_patterns.items():
            if re.search(pattern, question_lower):
                return q_type
        
        return 'general'
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data cleaning"""
        initial_count = len(df)
        
        # Remove empty questions or answers
        df = df.dropna(subset=['question', 'answer'])
        df = df[(df['question'].str.len() > 5) & (df['answer'].str.len() > 10)]
        
        # Remove duplicates based on question similarity
        df = self._remove_similar_questions(df)
        
        # Remove very short or very long answers (potential data issues)
        df = df[(df['answer_length'] > 20) & (df['answer_length'] < 5000)]
        
        # Clean text
        df['question'] = df['question'].apply(self._clean_text)
        df['answer'] = df['answer'].apply(self._clean_text)
        
        final_count = len(df)
        print(f"Data cleaning: {initial_count} -> {final_count} records ({initial_count - final_count} removed)")
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common encoding issues
        text = text.replace('â€™', "'").replace('â€"', '-')
        
        return text.strip()
    
    def _remove_similar_questions(self, df: pd.DataFrame, similarity_threshold: float = 0.9) -> pd.DataFrame:
        """Remove very similar questions using simple heuristic"""
        # Simple approach: remove exact duplicates first
        df = df.drop_duplicates(subset=['question'])
        
        # More sophisticated similarity-based deduplication could be added here
        # For now, we'll use exact matching on normalized questions
        df['question_normalized'] = df['question'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        df = df.drop_duplicates(subset=['question_normalized'])
        df = df.drop('question_normalized', axis=1)
        
        return df
    
    def _add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional metadata to the dataset"""
        # Add complexity score based on answer length and medical terms
        df['complexity_score'] = df.apply(
            lambda x: min(1.0, (x['answer_length'] / 1000) + (len(re.findall(r'\b[A-Z][a-z]+ disease|\b[A-Z][a-z]+ syndrome', x['answer'])) * 0.1)),
            axis=1
        )
        
        # Categorize by complexity
        df['complexity_level'] = pd.cut(
            df['complexity_score'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['basic', 'intermediate', 'advanced']
        )
        
        return df
    
    def _create_enhanced_sample_data(self) -> pd.DataFrame:
        """Create enhanced sample medical Q&A data"""
        sample_data = [
            {
                "question": "What are the symptoms of type 2 diabetes?",
                "answer": "Common symptoms of type 2 diabetes include increased thirst, frequent urination, increased hunger, unintended weight loss, fatigue, blurred vision, slow-healing sores, frequent infections, and areas of darkened skin.",
                "doc_type": "disease_info",
                "question_type": "symptoms",
                "source": "sample",
                "answer_length": 250,
                "question_length": 40
            },
            {
                "question": "How is hypertension treated in elderly patients?",
                "answer": "Hypertension in elderly patients is typically treated with lifestyle modifications and antihypertensive medications. Treatment goals may be less aggressive than in younger patients. Common medications include thiazide diuretics, ACE inhibitors, ARBs, and calcium channel blockers. Lifestyle changes include sodium restriction, weight management, and regular physical activity.",
                "doc_type": "treatment_info", 
                "question_type": "treatment",
                "source": "sample",
                "answer_length": 350,
                "question_length": 50
            },
            {
                "question": "What causes high blood pressure?",
                "answer": "High blood pressure (hypertension) can be caused by various factors including genetics, age, obesity, high sodium intake, lack of physical activity, excessive alcohol consumption, stress, and certain chronic conditions like kidney disease and sleep apnea. In many cases, the exact cause is unknown (essential hypertension).",
                "doc_type": "disease_info",
                "question_type": "causes",
                "source": "sample",
                "answer_length": 300,
                "question_length": 35
            },
            {
                "question": "How is asthma diagnosed?",
                "answer": "Asthma is diagnosed through medical history, physical examination, and lung function tests. Spirometry measures how much air you can breathe out and how quickly. Peak flow monitoring, allergy testing, and methacholine challenge tests may also be used. Imaging tests like chest X-rays may be done to rule out other conditions.",
                "doc_type": "diagnosis_info",
                "question_type": "diagnosis",
                "source": "sample",
                "answer_length": 320,
                "question_length": 30
            },
            {
                "question": "What are the risk factors for heart disease?",
                "answer": "Major risk factors for heart disease include high blood pressure, high cholesterol, smoking, diabetes, obesity, physical inactivity, unhealthy diet, excessive alcohol use, family history of heart disease, age (risk increases with age), and chronic stress. Some risk factors can be modified through lifestyle changes.",
                "doc_type": "prevention_info",
                "question_type": "prevention",
                "source": "sample",
                "answer_length": 310,
                "question_length": 45
            }
        ]
        
        df = pd.DataFrame(sample_data)
        return self._add_metadata(df)
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        stats = {
            'total_records': len(df),
            'question_types': df['question_type'].value_counts().to_dict() if 'question_type' in df.columns else {},
            'doc_types': df['doc_type'].value_counts().to_dict() if 'doc_type' in df.columns else {},
            'avg_answer_length': df['answer_length'].mean() if 'answer_length' in df.columns else 0,
            'avg_question_length': df['question_length'].mean() if 'question_length' in df.columns else 0
        }
        
        # Add complexity distribution if available
        if 'complexity_level' in df.columns:
            stats['complexity_distribution'] = df['complexity_level'].value_counts().to_dict()
        
        return stats
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_medquad.pkl"):
        """Save processed data to pickle file"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(df, f)
        print(f"Data saved to {filepath}")
    
    def load_processed_data(self, filename: str = "processed_medquad.pkl") -> pd.DataFrame:
        """Load processed data from pickle file"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                df = pickle.load(f)
            print(f"Data loaded from {filepath}")
            return df
        else:
            print(f"File {filepath} not found. Loading fresh data...")
            return self.load_all_data()