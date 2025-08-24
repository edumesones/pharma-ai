"""
Pharmaceutical Supply Chain Data Sources and Collection Pipeline

This module handles data collection from public sources and synthetic data generation
for training domain-specific models for pharmaceutical supply chain applications.
"""

import pandas as pd
import requests
import json
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from faker import Faker
import random
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuration for data sources"""
    name: str
    url: str
    data_type: str
    description: str
    rate_limit: float = 1.0  # seconds between requests
    
class PharmaDataCollector:
    """
    Collects and processes pharmaceutical supply chain data from public sources
    
    Data Sources:
    1. FDA Orange Book (Drug Approvals)
    2. USPTO Patent Database (Pharmaceutical Patents) 
    3. SEC EDGAR (Public Company Filings)
    4. EU Medicines Agency (EMA) Database
    5. WHO Global Health Observatory
    6. OpenFDA APIs
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.fake = Faker()
        
        # Public API endpoints (no authentication required)
        self.data_sources = {
            'fda_orange_book': DataSource(
                name="FDA Orange Book",
                url="https://www.fda.gov/media/76860/download",
                data_type="drug_approvals",
                description="FDA approved drug products with therapeutic equivalence evaluations"
            ),
            'openfda_adverse_events': DataSource(
                name="OpenFDA Adverse Events",
                url="https://api.fda.gov/drug/event.json",
                data_type="adverse_events", 
                description="FDA adverse event reporting system data",
                rate_limit=1.0
            ),
            'openfda_drug_labels': DataSource(
                name="OpenFDA Drug Labels",
                url="https://api.fda.gov/drug/label.json",
                data_type="drug_labels",
                description="FDA drug labeling information",
                rate_limit=1.0
            ),
            'sec_edgar': DataSource(
                name="SEC EDGAR",
                url="https://www.sec.gov/Archives/edgar/daily-index/",
                data_type="financial_filings",
                description="Public company financial filings"
            )
        }
        
    def collect_fda_orange_book(self) -> pd.DataFrame:
        """
        Collect FDA Orange Book data containing approved drugs
        Returns: DataFrame with drug approval information
        """
        try:
            logger.info("Downloading FDA Orange Book data...")
            
            # Download the Orange Book data file
            url = self.data_sources['fda_orange_book'].url
            response = requests.get(url)
            response.raise_for_status()
            
            # Save raw data
            orange_book_file = self.data_dir / "fda_orange_book.zip"
            with open(orange_book_file, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"FDA Orange Book data saved to {orange_book_file}")
            return pd.DataFrame()  # Would process the actual file in production
            
        except Exception as e:
            logger.error(f"Error collecting FDA Orange Book data: {e}")
            return pd.DataFrame()
    
    def collect_openfda_data(self, endpoint: str, limit: int = 1000) -> List[Dict]:
        """
        Collect data from OpenFDA APIs
        
        Args:
            endpoint: API endpoint (adverse_events, drug_labels, etc.)
            limit: Maximum number of records to collect
        """
        try:
            source = self.data_sources[f'openfda_{endpoint}']
            url = source.url
            
            all_records = []
            skip = 0
            batch_size = 100
            
            while len(all_records) < limit:
                params = {
                    'limit': batch_size,
                    'skip': skip
                }
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                records = data.get('results', [])
                
                if not records:
                    break
                    
                all_records.extend(records)
                skip += batch_size
                
                # Rate limiting
                time.sleep(source.rate_limit)
                
                logger.info(f"Collected {len(all_records)} {endpoint} records")
            
            # Save collected data
            output_file = self.data_dir / f"openfda_{endpoint}.json"
            with open(output_file, 'w') as f:
                json.dump(all_records[:limit], f, indent=2)
                
            logger.info(f"OpenFDA {endpoint} data saved to {output_file}")
            return all_records[:limit]
            
        except Exception as e:
            logger.error(f"Error collecting OpenFDA {endpoint} data: {e}")
            return []

class SyntheticDataGenerator:
    """
    Generates synthetic pharmaceutical supply chain data for training
    
    Generates realistic but artificial data for:
    1. Supply contracts and purchase orders
    2. Quality control reports and certificates  
    3. Shipping manifests and customs documents
    4. Regulatory submissions and compliance reports
    5. Risk assessment scenarios
    """
    
    def __init__(self, output_dir: str = "data/synthetic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fake = Faker()
        
        # Pharmaceutical-specific data
        self.drug_names = [
            "Aspirin", "Ibuprofen", "Metformin", "Lisinopril", "Atorvastatin",
            "Amlodipine", "Metoprolol", "Omeprazole", "Simvastatin", "Losartan",
            "Gabapentin", "Hydrochlorothiazide", "Sertraline", "Montelukast", "Fluticasone"
        ]
        
        self.suppliers = [
            "Pfizer Manufacturing", "Novartis Pharma", "Roche Diagnostics",
            "Merck & Co", "Johnson & Johnson", "Bristol-Myers Squibb",
            "AbbVie Inc", "Amgen Inc", "Gilead Sciences", "Biogen Inc"
        ]
        
        self.regulatory_agencies = ["FDA", "EMA", "Health Canada", "PMDA", "TGA"]
        
        self.compliance_categories = [
            "Good Manufacturing Practice (GMP)",
            "Good Distribution Practice (GDP)", 
            "Pharmacovigilance Compliance",
            "Serialization Requirements",
            "Cold Chain Management",
            "Import/Export Regulations"
        ]
        
        self.risk_types = [
            "Supplier Financial Distress",
            "Quality Control Failure", 
            "Transportation Delay",
            "Regulatory Non-Compliance",
            "Geopolitical Risk",
            "Natural Disaster",
            "Cyber Security Incident",
            "Raw Material Shortage"
        ]
        
    def generate_supply_contracts(self, num_contracts: int = 1000) -> pd.DataFrame:
        """Generate synthetic supply chain contracts"""
        contracts = []
        
        for i in range(num_contracts):
            contract = {
                'contract_id': f"SC-{i+1:06d}",
                'supplier_name': random.choice(self.suppliers),
                'drug_name': random.choice(self.drug_names),
                'quantity': random.randint(1000, 100000),
                'unit_price': round(random.uniform(0.10, 50.00), 2),
                'total_value': 0,  # Will calculate
                'contract_date': self.fake.date_between(start_date='-2y', end_date='today'),
                'delivery_date': self.fake.date_between(start_date='today', end_date='+6m'),
                'quality_requirements': random.choice([
                    "USP Grade", "EP Grade", "API Grade", "Excipient Grade"
                ]),
                'storage_conditions': random.choice([
                    "Room Temperature", "Refrigerated (2-8°C)", "Frozen (-20°C)", "Controlled"
                ]),
                'regulatory_requirements': random.choice([
                    "FDA cGMP", "EMA GMP", "ICH Guidelines", "USP Standards"
                ]),
                'risk_category': random.choice(['Low', 'Medium', 'High']),
                'payment_terms': random.choice(['Net 30', 'Net 60', 'Net 90', '2/10 Net 30']),
                'document_type': 'Supply Contract'
            }
            
            contract['total_value'] = contract['quantity'] * contract['unit_price']
            contracts.append(contract)
        
        df = pd.DataFrame(contracts)
        
        # Save to file
        output_file = self.output_dir / "supply_contracts.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Generated {num_contracts} supply contracts saved to {output_file}")
        
        return df
    
    def generate_quality_reports(self, num_reports: int = 500) -> pd.DataFrame:
        """Generate synthetic quality control reports"""
        reports = []
        
        for i in range(num_reports):
            report = {
                'report_id': f"QC-{i+1:06d}",
                'batch_number': f"LOT-{random.randint(100000, 999999)}",
                'drug_name': random.choice(self.drug_names),
                'supplier_name': random.choice(self.suppliers),
                'test_date': self.fake.date_between(start_date='-1y', end_date='today'),
                'assay_result': round(random.uniform(95.0, 105.0), 2),
                'assay_specification': '95.0 - 105.0%',
                'moisture_content': round(random.uniform(0.1, 5.0), 2),
                'moisture_limit': '≤ 5.0%',
                'dissolution_rate': round(random.uniform(80.0, 100.0), 1),
                'dissolution_spec': '≥ 80% in 30 minutes',
                'microbial_count': random.randint(0, 100),
                'microbial_limit': '≤ 100 CFU/g',
                'heavy_metals': round(random.uniform(0.1, 10.0), 2),
                'heavy_metals_limit': '≤ 20 ppm',
                'overall_result': random.choice(['Pass', 'Pass', 'Pass', 'Fail']),  # 75% pass rate
                'reviewer': self.fake.name(),
                'approval_date': self.fake.date_between(start_date='-1y', end_date='today'),
                'document_type': 'Quality Control Report'
            }
            
            reports.append(report)
        
        df = pd.DataFrame(reports)
        
        # Save to file  
        output_file = self.output_dir / "quality_reports.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Generated {num_reports} quality reports saved to {output_file}")
        
        return df
    
    def generate_risk_scenarios(self, num_scenarios: int = 300) -> pd.DataFrame:
        """Generate synthetic risk assessment scenarios"""
        scenarios = []
        
        for i in range(num_scenarios):
            scenario = {
                'risk_id': f"RISK-{i+1:06d}",
                'supplier_name': random.choice(self.suppliers),
                'drug_name': random.choice(self.drug_names),
                'risk_type': random.choice(self.risk_types),
                'risk_category': random.choice(['Operational', 'Financial', 'Regulatory', 'Strategic']),
                'probability': random.choice(['Low', 'Medium', 'High']),
                'impact': random.choice(['Low', 'Medium', 'High', 'Critical']),
                'risk_score': random.randint(1, 25),  # 1-25 scale
                'detection_date': self.fake.date_between(start_date='-6m', end_date='today'),
                'estimated_cost_impact': random.randint(10000, 10000000),
                'time_to_resolution': random.randint(1, 90),  # days
                'mitigation_strategy': random.choice([
                    'Diversify suppliers', 'Increase inventory', 'Alternative sourcing',
                    'Enhanced monitoring', 'Contract renegotiation', 'Quality audit'
                ]),
                'current_status': random.choice(['Open', 'In Progress', 'Resolved', 'Closed']),
                'assigned_to': self.fake.name(),
                'regulatory_impact': random.choice(['None', 'Low', 'Medium', 'High']),
                'customer_impact': random.choice(['None', 'Low', 'Medium', 'High']),
                'document_type': 'Risk Assessment'
            }
            
            scenarios.append(scenario)
        
        df = pd.DataFrame(scenarios)
        
        # Save to file
        output_file = self.output_dir / "risk_scenarios.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Generated {num_scenarios} risk scenarios saved to {output_file}")
        
        return df
    
    def generate_compliance_documents(self, num_documents: int = 400) -> pd.DataFrame:
        """Generate synthetic compliance checking documents"""
        documents = []
        
        for i in range(num_documents):
            document = {
                'document_id': f"COMP-{i+1:06d}",
                'supplier_name': random.choice(self.suppliers),
                'facility_name': f"{random.choice(self.suppliers)} Manufacturing Facility",
                'compliance_category': random.choice(self.compliance_categories),
                'regulatory_agency': random.choice(self.regulatory_agencies),
                'audit_date': self.fake.date_between(start_date='-2y', end_date='today'),
                'audit_type': random.choice(['Internal', 'External', 'Regulatory', 'Customer']),
                'compliance_score': random.randint(70, 100),
                'findings_count': random.randint(0, 15),
                'critical_findings': random.randint(0, 3),
                'major_findings': random.randint(0, 8),
                'minor_findings': random.randint(0, 10),
                'overall_status': random.choice(['Compliant', 'Compliant', 'Non-Compliant']),  # 67% compliant
                'corrective_actions_required': random.randint(0, 12),
                'target_closure_date': self.fake.date_between(start_date='today', end_date='+6m'),
                'risk_rating': random.choice(['Low', 'Medium', 'High']),
                'certification_status': random.choice(['Valid', 'Expired', 'Pending', 'Suspended']),
                'next_audit_date': self.fake.date_between(start_date='+6m', end_date='+2y'),
                'auditor': self.fake.name(),
                'document_type': 'Compliance Report'
            }
            
            documents.append(document)
        
        df = pd.DataFrame(documents)
        
        # Save to file
        output_file = self.output_dir / "compliance_documents.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Generated {num_documents} compliance documents saved to {output_file}")
        
        return df

    def generate_all_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """Generate all synthetic datasets"""
        logger.info("Starting synthetic data generation...")
        
        datasets = {
            'supply_contracts': self.generate_supply_contracts(1000),
            'quality_reports': self.generate_quality_reports(500), 
            'risk_scenarios': self.generate_risk_scenarios(300),
            'compliance_documents': self.generate_compliance_documents(400)
        }
        
        logger.info("Synthetic data generation completed!")
        return datasets

def main():
    """Main function to demonstrate data collection and generation"""
    
    # Initialize collectors
    collector = PharmaDataCollector()
    generator = SyntheticDataGenerator()
    
    print("Pharmaceutical Supply Chain Data Pipeline")
    print("=" * 50)
    
    # Collect public data
    print("\n1. Collecting public pharmaceutical data...")
    collector.collect_fda_orange_book()
    collector.collect_openfda_data('adverse_events', limit=100)
    collector.collect_openfda_data('drug_labels', limit=100)
    
    # Generate synthetic data
    print("\n2. Generating synthetic supply chain data...")
    synthetic_datasets = generator.generate_all_synthetic_data()
    
    # Summary statistics
    print("\n3. Dataset Summary:")
    for name, df in synthetic_datasets.items():
        print(f"   {name}: {len(df)} records")
    
    print("\nData collection and generation completed successfully!")
    print("Ready for preprocessing and model training.")

if __name__ == "__main__":
    main()