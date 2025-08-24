"""
Data Preprocessing Pipeline for Pharmaceutical Supply Chain ML Models

This module handles data cleaning, transformation, and preparation for training
domain-specific models for document classification, risk assessment, and compliance checking.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from transformers import AutoTokenizer
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Text preprocessing utilities for pharmaceutical supply chain documents
    """
    
    def __init__(self):
        # Pharmaceutical-specific stop words and terms
        self.pharma_stopwords = {
            'common_terms': ['mg', 'ml', 'tablet', 'capsule', 'solution', 'injection'],
            'regulatory_terms': ['fda', 'gmp', 'gdp', 'ich', 'usp', 'ep'],
            'quality_terms': ['assay', 'purity', 'dissolution', 'stability', 'impurity']
        }
        
        # Regex patterns for pharmaceutical data
        self.patterns = {
            'batch_number': r'LOT[-\s]?\d+|BATCH[-\s]?\d+',
            'drug_concentration': r'\d+\.?\d*\s*mg|\d+\.?\d*\s*ml',
            'dates': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            'regulatory_codes': r'[A-Z]{2,4}[-\s]?\d{3,6}',
            'monetary_amounts': r'\$[\d,]+\.?\d*',
            'percentages': r'\d+\.?\d*%'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and standardize text data"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Standardize pharmaceutical terms
        text = re.sub(r'\bmg\b', 'milligrams', text)
        text = re.sub(r'\bml\b', 'milliliters', text)
        text = re.sub(r'\bgmp\b', 'good manufacturing practice', text)
        text = re.sub(r'\bgdp\b', 'good distribution practice', text)
        
        # Remove special characters but preserve pharmaceutical notation
        text = re.sub(r'[^\w\s\-\.\%\$]', ' ', text)
        
        return text
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract pharmaceutical-specific entities from text"""
        entities = {}
        
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = matches
        
        return entities
    
    def create_features(self, text: str) -> Dict[str, float]:
        """Create domain-specific text features"""
        if not text:
            return {}
        
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
        }
        
        # Pharmaceutical-specific features
        features['has_batch_number'] = 1 if re.search(self.patterns['batch_number'], text, re.IGNORECASE) else 0
        features['has_drug_concentration'] = 1 if re.search(self.patterns['drug_concentration'], text, re.IGNORECASE) else 0
        features['has_regulatory_code'] = 1 if re.search(self.patterns['regulatory_codes'], text, re.IGNORECASE) else 0
        features['has_monetary_amount'] = 1 if re.search(self.patterns['monetary_amounts'], text, re.IGNORECASE) else 0
        
        # Term frequency features
        text_lower = text.lower()
        for category, terms in self.pharma_stopwords.items():
            feature_name = f'{category}_term_count'
            features[feature_name] = sum(1 for term in terms if term in text_lower)
        
        return features

class DatasetPreprocessor:
    """
    Main preprocessing pipeline for pharmaceutical supply chain datasets
    """
    
    def __init__(self, 
                 tokenizer_name: str = "microsoft/deberta-v3-base",
                 max_length: int = 512,
                 data_dir: str = "data"):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data_dir = Path(data_dir)
        self.text_processor = TextPreprocessor()
        
        # Label encoders for different tasks
        self.label_encoders = {
            'document_classification': LabelEncoder(),
            'risk_assessment': LabelEncoder(), 
            'compliance_checking': LabelEncoder()
        }
        
        # Feature scalers
        self.scalers = {
            'numerical': StandardScaler(),
            'text_features': StandardScaler()
        }
        
    def load_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """Load all synthetic datasets"""
        datasets = {}
        synthetic_dir = self.data_dir / "synthetic"
        
        file_mapping = {
            'supply_contracts': 'supply_contracts.csv',
            'quality_reports': 'quality_reports.csv',
            'risk_scenarios': 'risk_scenarios.csv',
            'compliance_documents': 'compliance_documents.csv'
        }
        
        for dataset_name, filename in file_mapping.items():
            filepath = synthetic_dir / filename
            if filepath.exists():
                datasets[dataset_name] = pd.read_csv(filepath)
                logger.info(f"Loaded {dataset_name}: {len(datasets[dataset_name])} records")
            else:
                logger.warning(f"Dataset file not found: {filepath}")
        
        return datasets
    
    def prepare_document_classification_data(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare data for document classification task
        
        Classes: Supply Contract, Quality Report, Risk Assessment, Compliance Report
        """
        
        all_documents = []
        
        # Process each dataset type
        for dataset_name, df in datasets.items():
            if df.empty:
                continue
                
            if dataset_name == 'supply_contracts':
                # Create text content from contract fields
                df['text_content'] = df.apply(lambda row: 
                    f"Supply Contract for {row.get('drug_name', '')} from {row.get('supplier_name', '')}. "
                    f"Quantity: {row.get('quantity', '')} units at ${row.get('unit_price', '')} per unit. "
                    f"Quality requirements: {row.get('quality_requirements', '')}. "
                    f"Storage: {row.get('storage_conditions', '')}. "
                    f"Regulatory: {row.get('regulatory_requirements', '')}. "
                    f"Risk category: {row.get('risk_category', '')}.",
                    axis=1
                )
                df['document_class'] = 'Supply Contract'
                
            elif dataset_name == 'quality_reports':
                df['text_content'] = df.apply(lambda row:
                    f"Quality Control Report for batch {row.get('batch_number', '')} of {row.get('drug_name', '')}. "
                    f"Assay result: {row.get('assay_result', '')}% (spec: {row.get('assay_specification', '')}). "
                    f"Moisture: {row.get('moisture_content', '')}% (limit: {row.get('moisture_limit', '')}). "
                    f"Dissolution: {row.get('dissolution_rate', '')}% (spec: {row.get('dissolution_spec', '')}). "
                    f"Microbial count: {row.get('microbial_count', '')} CFU/g. "
                    f"Overall result: {row.get('overall_result', '')}.",
                    axis=1
                )
                df['document_class'] = 'Quality Report'
                
            elif dataset_name == 'risk_scenarios':
                df['text_content'] = df.apply(lambda row:
                    f"Risk Assessment for {row.get('supplier_name', '')} regarding {row.get('drug_name', '')}. "
                    f"Risk type: {row.get('risk_type', '')} in {row.get('risk_category', '')} category. "
                    f"Probability: {row.get('probability', '')}, Impact: {row.get('impact', '')}. "
                    f"Risk score: {row.get('risk_score', '')}/25. "
                    f"Estimated cost impact: ${row.get('estimated_cost_impact', '')}. "
                    f"Mitigation: {row.get('mitigation_strategy', '')}.",
                    axis=1
                )
                df['document_class'] = 'Risk Assessment'
                
            elif dataset_name == 'compliance_documents':
                df['text_content'] = df.apply(lambda row:
                    f"Compliance Report for {row.get('supplier_name', '')} facility. "
                    f"Category: {row.get('compliance_category', '')} audited by {row.get('regulatory_agency', '')}. "
                    f"Audit type: {row.get('audit_type', '')}, Score: {row.get('compliance_score', '')}/100. "
                    f"Findings: {row.get('critical_findings', '')} critical, {row.get('major_findings', '')} major, {row.get('minor_findings', '')} minor. "
                    f"Status: {row.get('overall_status', '')}, Risk rating: {row.get('risk_rating', '')}.",
                    axis=1
                )
                df['document_class'] = 'Compliance Report'
            
            # Add to combined dataset
            document_data = df[['text_content', 'document_class']].copy()
            all_documents.append(document_data)
        
        # Combine all documents
        combined_df = pd.concat(all_documents, ignore_index=True)
        
        # Clean text content
        combined_df['text_content'] = combined_df['text_content'].apply(self.text_processor.clean_text)
        
        # Encode labels
        combined_df['label'] = self.label_encoders['document_classification'].fit_transform(combined_df['document_class'])
        
        logger.info(f"Prepared document classification data: {len(combined_df)} samples")
        logger.info(f"Classes: {combined_df['document_class'].value_counts().to_dict()}")
        
        return combined_df
    
    def prepare_risk_assessment_data(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare data for risk assessment task
        
        Classes: Low, Medium, High risk levels
        """
        
        risk_df = datasets.get('risk_scenarios', pd.DataFrame())
        if risk_df.empty:
            logger.warning("No risk scenarios data available")
            return pd.DataFrame()
        
        # Create comprehensive text features
        risk_df['text_content'] = risk_df.apply(lambda row:
            f"Risk assessment for supplier {row.get('supplier_name', '')} and product {row.get('drug_name', '')}. "
            f"Risk type: {row.get('risk_type', '')} in {row.get('risk_category', '')} category. "
            f"Current status: {row.get('current_status', '')}. "
            f"Estimated cost impact: ${row.get('estimated_cost_impact', '')}. "
            f"Time to resolution: {row.get('time_to_resolution', '')} days. "
            f"Mitigation strategy: {row.get('mitigation_strategy', '')}. "
            f"Regulatory impact: {row.get('regulatory_impact', '')}, Customer impact: {row.get('customer_impact', '')}.",
            axis=1
        )
        
        # Map risk scores to risk levels
        def risk_score_to_level(score):
            if score <= 8:
                return 'Low'
            elif score <= 16:
                return 'Medium' 
            else:
                return 'High'
        
        risk_df['risk_level'] = risk_df['risk_score'].apply(risk_score_to_level)
        
        # Clean text
        risk_df['text_content'] = risk_df['text_content'].apply(self.text_processor.clean_text)
        
        # Encode labels
        risk_df['label'] = self.label_encoders['risk_assessment'].fit_transform(risk_df['risk_level'])
        
        # Prepare final dataset
        final_df = risk_df[['text_content', 'risk_level', 'label', 'risk_score']].copy()
        
        logger.info(f"Prepared risk assessment data: {len(final_df)} samples")
        logger.info(f"Risk levels: {final_df['risk_level'].value_counts().to_dict()}")
        
        return final_df
    
    def prepare_compliance_data(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare data for compliance checking task
        
        Classes: Compliant, Non-Compliant
        """
        
        compliance_df = datasets.get('compliance_documents', pd.DataFrame())
        if compliance_df.empty:
            logger.warning("No compliance documents data available")
            return pd.DataFrame()
        
        # Create text content
        compliance_df['text_content'] = compliance_df.apply(lambda row:
            f"Compliance audit for {row.get('supplier_name', '')} facility. "
            f"Compliance category: {row.get('compliance_category', '')}. "
            f"Regulatory agency: {row.get('regulatory_agency', '')}. "
            f"Audit type: {row.get('audit_type', '')}, Score: {row.get('compliance_score', '')}/100. "
            f"Total findings: {row.get('findings_count', '')} "
            f"({row.get('critical_findings', '')} critical, {row.get('major_findings', '')} major, {row.get('minor_findings', '')} minor). "
            f"Corrective actions required: {row.get('corrective_actions_required', '')}. "
            f"Risk rating: {row.get('risk_rating', '')}, Certification status: {row.get('certification_status', '')}.",
            axis=1
        )
        
        # Map overall status to binary classification
        compliance_df['compliant'] = compliance_df['overall_status'].map({
            'Compliant': 1,
            'Non-Compliant': 0
        })
        
        # Remove rows with missing labels
        compliance_df = compliance_df.dropna(subset=['compliant'])
        
        # Clean text
        compliance_df['text_content'] = compliance_df['text_content'].apply(self.text_processor.clean_text)
        
        # Encode labels (binary classification)
        compliance_df['label'] = compliance_df['compliant']
        
        # Prepare final dataset
        final_df = compliance_df[['text_content', 'overall_status', 'label', 'compliance_score']].copy()
        
        logger.info(f"Prepared compliance checking data: {len(final_df)} samples")
        logger.info(f"Compliance status: {final_df['overall_status'].value_counts().to_dict()}")
        
        return final_df
    
    def tokenize_data(self, df: pd.DataFrame, text_column: str = 'text_content') -> Dict:
        """
        Tokenize text data for transformer models
        """
        
        texts = df[text_column].tolist()
        labels = df['label'].tolist()
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors=None
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }
    
    def create_train_test_splits(self, 
                                df: pd.DataFrame, 
                                test_size: float = 0.2, 
                                val_size: float = 0.1,
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits with stratification
        """
        
        # First split: train + val / test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['label'] if 'label' in df.columns else None,
            random_state=random_state
        )
        
        # Second split: train / val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            stratify=train_val_df['label'] if 'label' in train_val_df.columns else None,
            random_state=random_state
        )
        
        logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, 
                           datasets: Dict[str, pd.DataFrame], 
                           task_name: str,
                           output_dir: str = "data/processed"):
        """
        Save processed datasets to disk
        """
        
        output_path = Path(output_dir) / task_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, df in datasets.items():
            file_path = output_path / f"{split_name}.csv"
            df.to_csv(file_path, index=False)
            
            # Save tokenized data as JSON
            if not df.empty:
                tokenized = self.tokenize_data(df)
                json_path = output_path / f"{split_name}_tokenized.json"
                with open(json_path, 'w') as f:
                    json.dump(tokenized, f)
        
        # Save label encoders
        encoders_path = output_path / "label_encoders.json"
        encoder_classes = {}
        if task_name in self.label_encoders:
            encoder = self.label_encoders[task_name]
            encoder_classes[task_name] = {
                'classes': encoder.classes_.tolist() if hasattr(encoder, 'classes_') else []
            }
        
        with open(encoders_path, 'w') as f:
            json.dump(encoder_classes, f, indent=2)
        
        logger.info(f"Saved processed data for {task_name} to {output_path}")

def main():
    """
    Main preprocessing pipeline
    """
    
    print("Pharmaceutical Supply Chain Data Preprocessing")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor()
    
    # Load synthetic data
    print("\n1. Loading synthetic datasets...")
    datasets = preprocessor.load_synthetic_data()
    
    if not datasets:
        print("No datasets found! Please run data_sources.py first to generate synthetic data.")
        return
    
    # Process each task
    tasks = {
        'document_classification': preprocessor.prepare_document_classification_data,
        'risk_assessment': preprocessor.prepare_risk_assessment_data,
        'compliance_checking': preprocessor.prepare_compliance_data
    }
    
    for task_name, prepare_func in tasks.items():
        print(f"\n2. Processing {task_name} data...")
        
        # Prepare task-specific data
        task_df = prepare_func(datasets)
        
        if task_df.empty:
            print(f"   Warning: No data available for {task_name}")
            continue
        
        # Create train/test splits
        train_df, val_df, test_df = preprocessor.create_train_test_splits(task_df)
        
        # Save processed data
        split_datasets = {
            'train': train_df,
            'val': val_df, 
            'test': test_df
        }
        
        preprocessor.save_processed_data(split_datasets, task_name)
        
        print(f"   ✓ {task_name} data processed and saved")
        print(f"     Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    print(f"\n3. Preprocessing completed successfully!")
    print("Ready for model training.")

if __name__ == "__main__":
    main()