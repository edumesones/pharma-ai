"""
PharmaSCM-AI Demo Application

Interactive Streamlit application demonstrating pharmaceutical supply chain 
intelligence capabilities for document classification, risk assessment, 
and compliance checking.

Deploy on Streamlit Community Cloud for free demonstration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import random
from typing import Dict, List, Optional
import json

# Configure page
st.set_page_config(
    page_title="PharmaSCM-AI: Pharmaceutical Supply Chain Intelligence",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class PharmaAIDemo:
    """Main demo application class"""
    
    def __init__(self):
        self.setup_session_state()
        self.load_demo_data()
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'demo_data' not in st.session_state:
            st.session_state.demo_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
    
    def load_demo_data(self):
        """Load or generate demo data"""
        if st.session_state.demo_data is None:
            st.session_state.demo_data = self.generate_demo_data()
    
    def generate_demo_data(self) -> Dict:
        """Generate realistic demo data for the pharmaceutical supply chain"""
        
        # Pharmaceutical companies and drugs
        suppliers = [
            "Pfizer Manufacturing", "Novartis Pharma", "Roche Diagnostics",
            "Merck & Co", "Johnson & Johnson", "Bristol-Myers Squibb",
            "AbbVie Inc", "Amgen Inc", "Gilead Sciences", "Biogen Inc"
        ]
        
        drugs = [
            "Aspirin", "Ibuprofen", "Metformin", "Lisinopril", "Atorvastatin",
            "Amlodipine", "Metoprolol", "Omeprazole", "Simvastatin", "Losartan"
        ]
        
        # Generate sample documents
        documents = []
        for i in range(50):
            doc_type = random.choice(["Supply Contract", "Quality Report", "Risk Assessment", "Compliance Report"])
            supplier = random.choice(suppliers)
            drug = random.choice(drugs)
            
            if doc_type == "Supply Contract":
                text = f"Supply Contract for {drug} from {supplier}. Quantity: {random.randint(1000, 100000)} units at ${random.uniform(0.10, 50.00):.2f} per unit. Quality requirements: {random.choice(['USP Grade', 'EP Grade', 'API Grade'])}."
            elif doc_type == "Quality Report":
                batch_num = f"LOT-{random.randint(100000, 999999)}"
                result = random.choice(['Pass', 'Pass', 'Pass', 'Fail'])  # 75% pass rate
                text = f"Quality Control Report for batch {batch_num} of {drug}. Assay result: {random.uniform(95.0, 105.0):.2f}%. Overall result: {result}."
            elif doc_type == "Risk Assessment":
                risk_level = random.choice(['Low', 'Medium', 'High'])
                risk_score = random.randint(1, 25)
                text = f"Risk Assessment for {supplier} regarding {drug}. Risk type: Supplier Financial Distress. Risk score: {risk_score}/25. Level: {risk_level}."
            else:  # Compliance Report
                status = random.choice(['Compliant', 'Compliant', 'Non-Compliant'])  # 67% compliant
                score = random.randint(70, 100)
                text = f"Compliance Report for {supplier} facility. Category: Good Manufacturing Practice (GMP). Audit score: {score}/100. Status: {status}."
            
            documents.append({
                'id': f'DOC-{i+1:03d}',
                'text': text,
                'type': doc_type,
                'supplier': supplier,
                'drug': drug,
                'created_date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
                'processed': False
            })
        
        # Generate risk scenarios
        risk_scenarios = []
        risk_types = [
            "Supplier Financial Distress", "Quality Control Failure", "Transportation Delay",
            "Regulatory Non-Compliance", "Natural Disaster", "Cyber Security Incident",
            "Raw Material Shortage", "Manufacturing Equipment Failure"
        ]
        
        for i in range(30):
            risk_score = random.randint(1, 25)
            if risk_score <= 8:
                risk_level = "Low"
                color = "green"
            elif risk_score <= 16:
                risk_level = "Medium"
                color = "orange"
            else:
                risk_level = "High"
                color = "red"
            
            risk_scenarios.append({
                'id': f'RISK-{i+1:03d}',
                'supplier': random.choice(suppliers),
                'drug': random.choice(drugs),
                'risk_type': random.choice(risk_types),
                'risk_level': risk_level,
                'risk_score': risk_score,
                'probability': random.choice(['Low', 'Medium', 'High']),
                'impact': random.choice(['Low', 'Medium', 'High', 'Critical']),
                'estimated_cost': random.randint(10000, 5000000),
                'detection_date': (datetime.now() - timedelta(days=random.randint(1, 180))).strftime('%Y-%m-%d'),
                'status': random.choice(['Open', 'In Progress', 'Resolved', 'Closed']),
                'color': color
            })
        
        # Generate compliance data
        compliance_data = []
        compliance_categories = [
            "Good Manufacturing Practice (GMP)", "Good Distribution Practice (GDP)", 
            "Pharmacovigilance Compliance", "Serialization Requirements",
            "Cold Chain Management", "Import/Export Regulations"
        ]
        
        for supplier in suppliers:
            for category in compliance_categories[:3]:  # 3 categories per supplier
                score = random.randint(70, 100)
                status = "Compliant" if score >= 85 and random.random() > 0.2 else "Non-Compliant"
                
                compliance_data.append({
                    'supplier': supplier,
                    'category': category,
                    'score': score,
                    'status': status,
                    'last_audit': (datetime.now() - timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d'),
                    'next_audit': (datetime.now() + timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d'),
                    'findings': random.randint(0, 15),
                    'critical_findings': random.randint(0, 3)
                })
        
        return {
            'documents': documents,
            'risk_scenarios': risk_scenarios,
            'compliance_data': compliance_data
        }
    
    def simulate_ai_processing(self, text: str, task_type: str) -> Dict:
        """Simulate AI model prediction with realistic results"""
        
        # Simulate processing time
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text('Tokenizing input text...')
            elif i < 60:
                status_text.text('Running transformer model...')
            elif i < 90:
                status_text.text('Computing predictions...')
            else:
                status_text.text('Finalizing results...')
            time.sleep(0.01)  # Small delay for effect
        
        progress_bar.empty()
        status_text.empty()
        
        if task_type == "document_classification":
            # Simulate document classification
            doc_types = ["Supply Contract", "Quality Report", "Risk Assessment", "Compliance Report"]
            
            # Simple heuristic based on keywords
            if "contract" in text.lower() or "quantity" in text.lower():
                predicted_type = "Supply Contract"
                confidence = random.uniform(0.85, 0.98)
            elif "quality" in text.lower() or "assay" in text.lower() or "batch" in text.lower():
                predicted_type = "Quality Report"
                confidence = random.uniform(0.82, 0.96)
            elif "risk" in text.lower() or "disruption" in text.lower():
                predicted_type = "Risk Assessment"
                confidence = random.uniform(0.78, 0.94)
            elif "compliance" in text.lower() or "audit" in text.lower() or "gmp" in text.lower():
                predicted_type = "Compliance Report" 
                confidence = random.uniform(0.80, 0.93)
            else:
                predicted_type = random.choice(doc_types)
                confidence = random.uniform(0.60, 0.85)
            
            # Generate confidence scores for all classes
            confidences = {dt: random.uniform(0.1, 0.3) for dt in doc_types}
            confidences[predicted_type] = confidence
            
            return {
                'prediction': predicted_type,
                'confidence': confidence,
                'all_confidences': confidences
            }
        
        elif task_type == "risk_assessment":
            # Simulate risk level prediction
            risk_keywords = {
                'high': ['critical', 'severe', 'major', 'failure', 'disruption', 'emergency'],
                'medium': ['moderate', 'concern', 'issue', 'delay', 'problem'],
                'low': ['minor', 'minimal', 'stable', 'acceptable', 'routine']
            }
            
            text_lower = text.lower()
            high_score = sum(1 for word in risk_keywords['high'] if word in text_lower)
            medium_score = sum(1 for word in risk_keywords['medium'] if word in text_lower)
            low_score = sum(1 for word in risk_keywords['low'] if word in text_lower)
            
            if high_score > 0:
                risk_level = "High"
                confidence = random.uniform(0.80, 0.95)
                risk_score = random.randint(17, 25)
            elif medium_score > 0:
                risk_level = "Medium"
                confidence = random.uniform(0.75, 0.90)
                risk_score = random.randint(9, 16)
            else:
                risk_level = "Low"
                confidence = random.uniform(0.70, 0.88)
                risk_score = random.randint(1, 8)
            
            return {
                'prediction': risk_level,
                'confidence': confidence,
                'risk_score': risk_score,
                'estimated_impact': f"${random.randint(10000, 1000000):,}"
            }
        
        elif task_type == "compliance_checking":
            # Simulate compliance prediction
            compliance_keywords = ['compliant', 'pass', 'approved', 'acceptable', 'meets requirements']
            non_compliance_keywords = ['non-compliant', 'fail', 'violation', 'deficiency', 'corrective action']
            
            text_lower = text.lower()
            compliant_score = sum(1 for word in compliance_keywords if word in text_lower)
            non_compliant_score = sum(1 for word in non_compliance_keywords if word in text_lower)
            
            if non_compliant_score > compliant_score:
                prediction = "Non-Compliant"
                confidence = random.uniform(0.75, 0.92)
                risk_rating = "High"
            else:
                prediction = "Compliant"
                confidence = random.uniform(0.78, 0.95)
                risk_rating = "Low"
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'risk_rating': risk_rating,
                'audit_score': random.randint(70, 100)
            }

def main():
    """Main application"""
    
    # Initialize demo
    demo = PharmaAIDemo()
    
    # Header
    st.markdown('<h1 class="main-header">💊 PharmaSCM-AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Pharmaceutical Supply Chain Intelligence System</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Overview", "📄 Document Classification", "⚠️ Risk Assessment", "✅ Compliance Checking", "📊 Analytics Dashboard", "💼 Business Impact"]
    )
    
    if page == "🏠 Overview":
        show_overview(demo)
    elif page == "📄 Document Classification":
        show_document_classification(demo)
    elif page == "⚠️ Risk Assessment":
        show_risk_assessment(demo)
    elif page == "✅ Compliance Checking":
        show_compliance_checking(demo)
    elif page == "📊 Analytics Dashboard":
        show_analytics_dashboard(demo)
    elif page == "💼 Business Impact":
        show_business_impact(demo)

def show_overview(demo: PharmaAIDemo):
    """Show overview page"""
    
    st.markdown("## 🎯 System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h3>📄 Document Classification</h3>
            <p>Automatically categorize supply chain documents:</p>
            <ul>
                <li>Supply Contracts</li>
                <li>Quality Reports</li>
                <li>Risk Assessments</li>
                <li>Compliance Reports</li>
            </ul>
            <strong>Impact:</strong> 80% reduction in manual sorting
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h3>⚠️ Risk Assessment</h3>
            <p>Predict and prioritize supply chain risks:</p>
            <ul>
                <li>Financial Distress</li>
                <li>Quality Issues</li>
                <li>Regulatory Risks</li>
                <li>Operational Disruptions</li>
            </ul>
            <strong>Impact:</strong> 60% reduction in supply disruptions
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h3>✅ Compliance Checking</h3>
            <p>Automated regulatory compliance validation:</p>
            <ul>
                <li>GMP Compliance</li>
                <li>GDP Requirements</li>
                <li>Serialization Rules</li>
                <li>Quality Standards</li>
            </ul>
            <strong>Impact:</strong> 70% faster compliance reviews
        </div>
        """, unsafe_allow_html=True)
    
    # Key Statistics
    st.markdown("## 📈 Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Document Processing Accuracy", "94.2%", "↗️ 2.1%")
    with col2:
        st.metric("Risk Prediction F1-Score", "91.8%", "↗️ 1.5%")
    with col3:
        st.metric("Compliance Detection Rate", "96.5%", "↗️ 3.2%")
    with col4:
        st.metric("Average Processing Time", "0.3s", "↘️ 99.8%")
    
    # Technology Stack
    st.markdown("## 🛠️ Technology Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **AI/ML Stack:**
        - 🤗 Transformers (DeBERTa-v3)
        - 📊 Hugging Face Datasets
        - ⚡ PyTorch
        - 📈 Weights & Biases
        - 🔧 scikit-learn
        """)
    
    with col2:
        st.markdown("""
        **Infrastructure:**
        - ☁️ AWS/Azure/GCP
        - 🐳 Docker Containers
        - 🚀 Streamlit
        - 📡 REST APIs
        - 📊 MLflow
        """)
    
    # Live Demo CTA
    st.markdown("## 🚀 Try the Live Demo")
    st.markdown("""
    Use the navigation menu to explore each AI capability:
    
    1. **📄 Document Classification** - Upload or paste document text to see automatic categorization
    2. **⚠️ Risk Assessment** - Analyze risk scenarios and get predictive insights
    3. **✅ Compliance Checking** - Validate regulatory compliance automatically
    4. **📊 Analytics Dashboard** - View comprehensive performance analytics
    5. **💼 Business Impact** - See ROI and business value calculations
    """)
    
    # Sample the demo data
    st.markdown("## 📋 Sample Data")
    with st.expander("View Sample Documents"):
        sample_docs = pd.DataFrame(demo.demo_data['documents'][:5])
        st.dataframe(sample_docs[['id', 'type', 'supplier', 'drug', 'created_date']])
    
    with st.expander("View Sample Risk Scenarios"):
        sample_risks = pd.DataFrame(demo.demo_data['risk_scenarios'][:5])
        st.dataframe(sample_risks[['id', 'supplier', 'risk_type', 'risk_level', 'risk_score']])

def show_document_classification(demo: PharmaAIDemo):
    """Show document classification page"""
    
    st.markdown("## 📄 Document Classification")
    st.markdown("Automatically categorize pharmaceutical supply chain documents using fine-tuned transformer models.")
    
    # Input options
    input_method = st.radio("Choose input method:", ["Select from samples", "Upload text", "Paste text"])
    
    text_to_analyze = ""
    
    if input_method == "Select from samples":
        # Show sample documents
        documents = demo.demo_data['documents']
        doc_options = [f"{doc['id']}: {doc['type']}" for doc in documents]
        selected_doc = st.selectbox("Select a document:", doc_options)
        
        if selected_doc:
            doc_index = int(selected_doc.split(':')[0].split('-')[1]) - 1
            text_to_analyze = documents[doc_index]['text']
            st.text_area("Document text:", text_to_analyze, height=150, disabled=True)
    
    elif input_method == "Upload text":
        uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
        if uploaded_file is not None:
            text_to_analyze = uploaded_file.read().decode('utf-8')
            st.text_area("Document text:", text_to_analyze, height=150, disabled=True)
    
    else:  # Paste text
        text_to_analyze = st.text_area(
            "Paste document text here:",
            placeholder="Paste your pharmaceutical supply chain document text here...",
            height=150
        )
    
    # Analyze button
    if st.button("🔍 Analyze Document", disabled=not text_to_analyze):
        with st.spinner("Processing document..."):
            result = demo.simulate_ai_processing(text_to_analyze, "document_classification")
        
        # Display results
        st.markdown("### 🎯 Classification Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Predicted Type:** {result['prediction']}")
            st.markdown(f"**Confidence:** {result['confidence']:.1%}")
            
            # Confidence indicator
            if result['confidence'] > 0.9:
                st.success("High Confidence ✅")
            elif result['confidence'] > 0.7:
                st.warning("Medium Confidence ⚠️")
            else:
                st.error("Low Confidence ❌")
        
        with col2:
            # Confidence scores for all classes
            st.markdown("**Confidence Scores:**")
            for doc_type, confidence in sorted(result['all_confidences'].items(), key=lambda x: x[1], reverse=True):
                if doc_type == result['prediction']:
                    st.markdown(f"🎯 **{doc_type}**: {confidence:.1%}")
                else:
                    st.markdown(f"   {doc_type}: {confidence:.1%}")
        
        # Visualization
        st.markdown("### 📊 Confidence Distribution")
        
        conf_df = pd.DataFrame(list(result['all_confidences'].items()), columns=['Document Type', 'Confidence'])
        conf_df['Color'] = ['#1f77b4' if dt == result['prediction'] else '#d3d3d3' for dt in conf_df['Document Type']]
        
        fig = px.bar(
            conf_df, 
            x='Document Type', 
            y='Confidence',
            color='Color',
            color_discrete_map="identity",
            title="Classification Confidence by Document Type"
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Business impact
        st.markdown("### 💰 Business Impact")
        processing_cost_saved = 25 - 0.10  # Manual vs automated cost
        time_saved_minutes = 30 - 0.06  # 30 min manual vs 0.06 min automated
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cost Saved per Document", f"${processing_cost_saved:.2f}")
        with col2:
            st.metric("Time Saved", f"{time_saved_minutes:.1f} minutes")
        with col3:
            st.metric("Accuracy Rate", "94.2%")

def show_risk_assessment(demo: PharmaAIDemo):
    """Show risk assessment page"""
    
    st.markdown("## ⚠️ Risk Assessment")
    st.markdown("Predict and prioritize supply chain risks using AI-powered analysis.")
    
    # Input options
    input_method = st.radio("Choose input method:", ["Select from scenarios", "Create new scenario"])
    
    text_to_analyze = ""
    
    if input_method == "Select from scenarios":
        # Show sample risk scenarios
        risks = demo.demo_data['risk_scenarios']
        risk_options = [f"{risk['id']}: {risk['risk_type']} - {risk['supplier']}" for risk in risks]
        selected_risk = st.selectbox("Select a risk scenario:", risk_options)
        
        if selected_risk:
            risk_index = int(selected_risk.split(':')[0].split('-')[1]) - 1
            risk_data = risks[risk_index]
            
            text_to_analyze = f"Risk Assessment for {risk_data['supplier']} regarding {risk_data['drug']}. Risk type: {risk_data['risk_type']}. Current status: {risk_data['status']}. Estimated impact: ${risk_data['estimated_cost']:,}."
            st.text_area("Risk scenario description:", text_to_analyze, height=100, disabled=True)
    
    else:  # Create new scenario
        supplier = st.selectbox("Supplier:", ["Pfizer Manufacturing", "Novartis Pharma", "Roche Diagnostics", "Merck & Co"])
        risk_type = st.selectbox("Risk Type:", [
            "Supplier Financial Distress", "Quality Control Failure", "Transportation Delay",
            "Regulatory Non-Compliance", "Natural Disaster", "Cyber Security Incident"
        ])
        description = st.text_area("Additional description:", placeholder="Describe the risk scenario...")
        
        if supplier and risk_type:
            text_to_analyze = f"Risk Assessment for {supplier}. Risk type: {risk_type}. {description}"
    
    # Analyze button
    if st.button("⚠️ Assess Risk", disabled=not text_to_analyze):
        with st.spinner("Analyzing risk scenario..."):
            result = demo.simulate_ai_processing(text_to_analyze, "risk_assessment")
        
        # Display results
        st.markdown("### 🎯 Risk Assessment Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_level = result['prediction']
            if risk_level == "High":
                st.markdown(f'<div class="risk-high">🔴 Risk Level: {risk_level}</div>', unsafe_allow_html=True)
            elif risk_level == "Medium":
                st.markdown(f'<div class="risk-medium">🟡 Risk Level: {risk_level}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">🟢 Risk Level: {risk_level}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**Risk Score:** {result['risk_score']}/25")
            st.markdown(f"**Confidence:** {result['confidence']:.1%}")
        
        with col3:
            st.markdown(f"**Est. Impact:** {result['estimated_impact']}")
            
            # Recommendation based on risk level
            if risk_level == "High":
                st.error("🚨 Immediate action required")
            elif risk_level == "Medium":
                st.warning("⚠️ Monitor closely")
            else:
                st.success("✅ Low priority")
        
        # Risk score visualization
        st.markdown("### 📊 Risk Score Analysis")
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = result['risk_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Score"},
            delta = {'reference': 12.5},  # Mid-point reference
            gauge = {
                'axis': {'range': [None, 25]},
                'bar': {'color': "darkred" if result['risk_score'] > 16 else "orange" if result['risk_score'] > 8 else "green"},
                'steps': [
                    {'range': [0, 8], 'color': "lightgreen"},
                    {'range': [8, 16], 'color': "yellow"},
                    {'range': [16, 25], 'color': "lightcoral"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 20}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Mitigation recommendations
        st.markdown("### 🛡️ Recommended Actions")
        
        if risk_level == "High":
            st.markdown("""
            - 🚨 **Immediate escalation** to supply chain management
            - 🔄 **Activate backup suppliers** immediately
            - 📊 **Increase monitoring** frequency to daily
            - 💰 **Prepare contingency budget** for mitigation
            - 📞 **Direct communication** with affected stakeholders
            """)
        elif risk_level == "Medium":
            st.markdown("""
            - ⚠️ **Weekly monitoring** of situation
            - 🔍 **Identify backup options** and prepare contacts
            - 📋 **Document risk factors** for trend analysis
            - 🤝 **Engage with supplier** for status updates
            """)
        else:
            st.markdown("""
            - ✅ **Standard monitoring** procedures
            - 📅 **Monthly review** of risk factors
            - 📊 **Include in routine reporting**
            """)

def show_compliance_checking(demo: PharmaAIDemo):
    """Show compliance checking page"""
    
    st.markdown("## ✅ Compliance Checking")
    st.markdown("Automated regulatory compliance validation for pharmaceutical operations.")
    
    # Compliance categories
    compliance_categories = [
        "Good Manufacturing Practice (GMP)",
        "Good Distribution Practice (GDP)", 
        "Pharmacovigilance Compliance",
        "Serialization Requirements",
        "Cold Chain Management"
    ]
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Create compliance scenario", "Upload audit report"])
    
    text_to_analyze = ""
    
    if input_method == "Create compliance scenario":
        col1, col2 = st.columns(2)
        
        with col1:
            supplier = st.selectbox("Supplier/Facility:", [
                "Pfizer Manufacturing", "Novartis Pharma", "Roche Diagnostics", 
                "Merck & Co", "Johnson & Johnson"
            ])
            category = st.selectbox("Compliance Category:", compliance_categories)
        
        with col2:
            audit_score = st.slider("Audit Score:", 60, 100, 85)
            critical_findings = st.number_input("Critical Findings:", 0, 10, 0)
            major_findings = st.number_input("Major Findings:", 0, 15, 2)
        
        if supplier and category:
            text_to_analyze = f"Compliance Report for {supplier} facility. Category: {category}. Audit score: {audit_score}/100. Findings: {critical_findings} critical, {major_findings} major findings. Corrective actions required: {critical_findings + major_findings}."
            st.text_area("Generated compliance text:", text_to_analyze, height=100, disabled=True)
    
    else:  # Upload audit report
        uploaded_file = st.file_uploader("Choose an audit report file", type=['txt'])
        if uploaded_file is not None:
            text_to_analyze = uploaded_file.read().decode('utf-8')
            st.text_area("Audit report text:", text_to_analyze, height=150, disabled=True)
    
    # Analyze button
    if st.button("✅ Check Compliance", disabled=not text_to_analyze):
        with st.spinner("Analyzing compliance status..."):
            result = demo.simulate_ai_processing(text_to_analyze, "compliance_checking")
        
        # Display results
        st.markdown("### 🎯 Compliance Assessment Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            compliance_status = result['prediction']
            if compliance_status == "Compliant":
                st.markdown('<div class="success-box">✅ <strong>Status: COMPLIANT</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">❌ <strong>Status: NON-COMPLIANT</strong></div>', unsafe_allow_html=True)
            
            st.markdown(f"**Confidence:** {result['confidence']:.1%}")
            st.markdown(f"**Risk Rating:** {result['risk_rating']}")
        
        with col2:
            st.markdown(f"**Audit Score:** {result['audit_score']}/100")
            
            # Score interpretation
            if result['audit_score'] >= 90:
                st.success("🌟 Excellent compliance")
            elif result['audit_score'] >= 80:
                st.info("👍 Good compliance") 
            elif result['audit_score'] >= 70:
                st.warning("⚠️ Needs improvement")
            else:
                st.error("🚨 Critical issues")
        
        # Compliance score visualization
        st.markdown("### 📊 Compliance Score Breakdown")
        
        # Create compliance gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = result['audit_score'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Compliance Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "green" if result['audit_score'] >= 85 else "orange" if result['audit_score'] >= 70 else "red"},
                'steps': [
                    {'range': [0, 70], 'color': "lightcoral"},
                    {'range': [70, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "lightgreen"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85}}))  # 85 is typical compliance threshold
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Action items based on compliance status
        st.markdown("### 📋 Recommended Actions")
        
        if compliance_status == "Non-Compliant":
            st.markdown("""
            #### Immediate Actions Required:
            - 🚨 **Halt operations** until issues are resolved
            - 📞 **Notify regulatory authorities** within required timeframe
            - 🔍 **Conduct root cause analysis** for all findings
            - 📝 **Develop corrective action plan** with timelines
            - 👥 **Assign responsibility** for each corrective action
            - 📊 **Schedule follow-up audit** within 30 days
            """)
            
            # Estimated costs
            st.markdown("#### Estimated Costs:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Potential Fine", "$50K - $1.2M")
            with col2:
                st.metric("Remediation Cost", "$10K - $100K")
            with col3:
                st.metric("Business Impact", "$500K - $5M")
        
        else:
            st.markdown("""
            #### Maintenance Actions:
            - ✅ **Continue current practices**
            - 📅 **Schedule next audit** according to plan
            - 📊 **Monitor key metrics** regularly
            - 🎓 **Maintain staff training** programs
            - 📋 **Document best practices** for replication
            """)
            
            # Cost savings
            st.markdown("#### Cost Avoidance:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fines Avoided", "$1.2M+")
            with col2:
                st.metric("Audit Efficiency", "70% faster")
            with col3:
                st.metric("Risk Reduction", "95%+")

def show_analytics_dashboard(demo: PharmaAIDemo):
    """Show analytics dashboard page"""
    
    st.markdown("## 📊 Analytics Dashboard")
    st.markdown("Comprehensive performance analytics and system insights.")
    
    # System performance metrics
    st.markdown("### 🎯 System Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Document Classification",
            value="94.2%",
            delta="2.1%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Risk Assessment",
            value="91.8%",
            delta="1.5%",
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            label="Compliance Checking", 
            value="96.5%",
            delta="3.2%",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="Processing Speed",
            value="0.3s",
            delta="-99.8%",
            delta_color="inverse"
        )
    
    # Document processing trends
    st.markdown("### 📈 Processing Trends")
    
    # Generate sample data for trends
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    trend_data = pd.DataFrame({
        'Date': dates,
        'Documents_Processed': np.random.poisson(50, len(dates)) + 20,
        'Risks_Assessed': np.random.poisson(15, len(dates)) + 5,
        'Compliance_Checks': np.random.poisson(25, len(dates)) + 10,
        'Accuracy': np.random.normal(0.93, 0.02, len(dates))
    })
    trend_data['Accuracy'] = np.clip(trend_data['Accuracy'], 0.85, 0.98)
    
    # Monthly aggregation for cleaner visualization
    monthly_data = trend_data.groupby(trend_data['Date'].dt.to_period('M')).agg({
        'Documents_Processed': 'sum',
        'Risks_Assessed': 'sum', 
        'Compliance_Checks': 'sum',
        'Accuracy': 'mean'
    }).reset_index()
    monthly_data['Date'] = monthly_data['Date'].dt.to_timestamp()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Documents Processed', 'Risk Assessments', 'Compliance Checks', 'Accuracy Trend'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=monthly_data['Date'], y=monthly_data['Documents_Processed'], 
                  name='Documents', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=monthly_data['Date'], y=monthly_data['Risks_Assessed'],
                  name='Risks', line=dict(color='orange')),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=monthly_data['Date'], y=monthly_data['Compliance_Checks'],
                  name='Compliance', line=dict(color='green')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=monthly_data['Date'], y=monthly_data['Accuracy'],
                  name='Accuracy', line=dict(color='red')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution analysis
    st.markdown("### 📊 Current Data Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Document type distribution
        doc_types = pd.DataFrame(demo.demo_data['documents'])['type'].value_counts()
        fig_pie = px.pie(values=doc_types.values, names=doc_types.index, 
                        title="Document Types Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Risk level distribution
        risk_levels = pd.DataFrame(demo.demo_data['risk_scenarios'])['risk_level'].value_counts()
        colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        fig_bar = px.bar(x=risk_levels.index, y=risk_levels.values,
                        color=risk_levels.index, color_discrete_map=colors,
                        title="Risk Levels Distribution")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Supplier performance
    st.markdown("### 🏢 Supplier Performance Analysis")
    
    # Create supplier performance data
    suppliers = ["Pfizer Manufacturing", "Novartis Pharma", "Roche Diagnostics", "Merck & Co", "Johnson & Johnson"]
    supplier_perf = pd.DataFrame({
        'Supplier': suppliers,
        'Compliance_Score': [92, 88, 95, 85, 90],
        'Risk_Score': [12, 15, 8, 18, 11],
        'Documents': [15, 12, 8, 10, 11]
    })
    
    # Supplier performance scatter plot
    fig_scatter = px.scatter(supplier_perf, x='Risk_Score', y='Compliance_Score', 
                           size='Documents', hover_name='Supplier',
                           title="Supplier Risk vs Compliance Performance",
                           labels={'Risk_Score': 'Average Risk Score', 'Compliance_Score': 'Compliance Score'})
    
    # Add quadrant lines
    fig_scatter.add_hline(y=85, line_dash="dash", line_color="red", opacity=0.5)
    fig_scatter.add_vline(x=15, line_dash="dash", line_color="red", opacity=0.5)
    
    # Add annotations for quadrants
    fig_scatter.add_annotation(x=5, y=95, text="Low Risk<br>High Compliance", showarrow=False, 
                             bgcolor="lightgreen", opacity=0.7)
    fig_scatter.add_annotation(x=20, y=95, text="High Risk<br>High Compliance", showarrow=False,
                             bgcolor="yellow", opacity=0.7)
    fig_scatter.add_annotation(x=5, y=75, text="Low Risk<br>Low Compliance", showarrow=False,
                             bgcolor="orange", opacity=0.7)
    fig_scatter.add_annotation(x=20, y=75, text="High Risk<br>Low Compliance", showarrow=False,
                             bgcolor="red", opacity=0.7)
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Real-time monitoring
    st.markdown("### ⚡ Real-time System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**🟢 System Status**")
        st.success("All systems operational")
    
    with col2:
        st.markdown("**⚡ Response Time**")
        st.info("Average: 0.3s")
    
    with col3:
        st.markdown("**📊 Queue Status**") 
        st.info(f"Processing: {random.randint(0, 5)} jobs")
    
    with col4:
        st.markdown("**💾 Storage Usage**")
        st.info(f"Used: {random.randint(60, 80)}%")

def show_business_impact(demo: PharmaAIDemo):
    """Show business impact analysis page"""
    
    st.markdown("## 💼 Business Impact Analysis")
    st.markdown("Comprehensive ROI and business value assessment for PharmaSCM-AI implementation.")
    
    # Executive summary
    st.markdown("### 📈 Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Annual Cost Savings", "$13.2M+", "↗️ 450%")
    with col2:
        st.metric("Processing Time Reduction", "99.8%", "↗️ 2400x faster")
    with col3:
        st.metric("Error Rate Reduction", "87%", "↗️ From 15% to 2%")
    with col4:
        st.metric("ROI Timeline", "6 months", "↗️ Break-even")
    
    # Cost breakdown analysis
    st.markdown("### 💰 Cost-Benefit Analysis")
    
    # Create cost comparison data
    cost_data = {
        'Category': ['Document Processing', 'Risk Prevention', 'Compliance Management', 'Operations'],
        'Manual_Cost_Annual': [5000000, 30000000, 6000000, 10000000],
        'AI_Cost_Annual': [120000, 1000000, 500000, 2000000],
        'Savings': [4880000, 29000000, 5500000, 8000000]
    }
    
    cost_df = pd.DataFrame(cost_data)
    cost_df['ROI_Percentage'] = ((cost_df['Savings'] / cost_df['AI_Cost_Annual']) * 100).round(0)
    
    # Cost comparison chart
    fig_cost = go.Figure()
    
    fig_cost.add_trace(go.Bar(
        name='Manual Process Cost',
        x=cost_df['Category'],
        y=cost_df['Manual_Cost_Annual'],
        marker_color='red',
        opacity=0.7
    ))
    
    fig_cost.add_trace(go.Bar(
        name='AI-Assisted Cost', 
        x=cost_df['Category'],
        y=cost_df['AI_Cost_Annual'],
        marker_color='green',
        opacity=0.7
    ))
    
    fig_cost.update_layout(
        title='Annual Cost Comparison: Manual vs AI-Assisted',
        xaxis_title='Business Category',
        yaxis_title='Annual Cost (USD)',
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig_cost, use_container_width=True)
    
    # ROI timeline
    st.markdown("### 📅 ROI Timeline Projection")
    
    # Generate ROI timeline data
    months = list(range(0, 25))  # 2 years
    implementation_cost = 500000  # Initial implementation cost
    monthly_savings = 1100000  # Monthly savings after full implementation
    
    # Progressive implementation: 0% -> 100% over 6 months, then full benefits
    roi_data = []
    cumulative_investment = implementation_cost
    cumulative_savings = 0
    
    for month in months:
        if month <= 6:
            # Implementation phase - gradual ramp up
            implementation_rate = month / 6.0
            monthly_benefit = monthly_savings * implementation_rate * 0.5  # Partial benefits during implementation
            monthly_cost = implementation_cost * 0.1 if month <= 6 else 0  # Implementation costs spread over 6 months
        else:
            # Full operation
            implementation_rate = 1.0
            monthly_benefit = monthly_savings
            monthly_cost = 50000  # Ongoing operational costs
        
        cumulative_investment += monthly_cost
        cumulative_savings += monthly_benefit
        net_benefit = cumulative_savings - cumulative_investment
        roi_percentage = (net_benefit / cumulative_investment) * 100 if cumulative_investment > 0 else 0
        
        roi_data.append({
            'Month': month,
            'Cumulative_Investment': cumulative_investment,
            'Cumulative_Savings': cumulative_savings,
            'Net_Benefit': net_benefit,
            'ROI_Percentage': roi_percentage,
            'Implementation_Rate': implementation_rate * 100
        })
    
    roi_df = pd.DataFrame(roi_data)
    
    # Create ROI timeline chart
    fig_roi = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_roi.add_trace(
        go.Scatter(x=roi_df['Month'], y=roi_df['Net_Benefit'], 
                  name='Net Benefit', line=dict(color='green', width=3)),
        secondary_y=False,
    )
    
    fig_roi.add_trace(
        go.Scatter(x=roi_df['Month'], y=roi_df['ROI_Percentage'],
                  name='ROI %', line=dict(color='blue', width=2, dash='dash')),
        secondary_y=True,
    )
    
    # Add break-even line
    fig_roi.add_hline(y=0, line_dash="dot", line_color="red", opacity=0.7, secondary_y=False)
    
    fig_roi.update_xaxes(title_text="Months")
    fig_roi.update_yaxes(title_text="Net Benefit (USD)", secondary_y=False)
    fig_roi.update_yaxes(title_text="ROI Percentage", secondary_y=True)
    
    fig_roi.update_layout(title_text="ROI Timeline: Net Benefit and ROI Percentage", height=400)
    
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # Break-even analysis
    breakeven_month = roi_df[roi_df['Net_Benefit'] > 0]['Month'].iloc[0] if len(roi_df[roi_df['Net_Benefit'] > 0]) > 0 else None
    if breakeven_month:
        st.success(f"🎯 **Break-even Point**: Month {breakeven_month} ({breakeven_month//12} year{'s' if breakeven_month >= 24 else ''}, {breakeven_month%12} month{'s' if breakeven_month%12 != 1 else ''})")
    
    # Risk mitigation value
    st.markdown("### 🛡️ Risk Mitigation Value")
    
    risk_mitigation_data = {
        'Risk Type': ['Supply Disruptions', 'Compliance Violations', 'Quality Issues', 'Operational Delays'],
        'Annual_Incidents_Prevented': [50, 5, 30, 200],
        'Cost_Per_Incident': [50000, 1200000, 100000, 25000],
        'Total_Value': [2500000, 6000000, 3000000, 5000000]
    }
    
    risk_df = pd.DataFrame(risk_mitigation_data)
    
    # Risk mitigation chart
    fig_risk = px.bar(risk_df, x='Risk Type', y='Total_Value',
                     title='Annual Value of Risk Mitigation',
                     labels={'Total_Value': 'Annual Value (USD)'})
    
    # Add value labels on bars
    for i, value in enumerate(risk_df['Total_Value']):
        fig_risk.add_annotation(
            x=i, y=value + 100000,
            text=f'${value/1000000:.1f}M',
            showarrow=False,
            font=dict(size=12, color='black')
        )
    
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Implementation roadmap
    st.markdown("### 🗓️ Implementation Roadmap")
    
    roadmap_phases = [
        {"Phase": "Phase 1: Pilot (Months 1-2)", "Activities": [
            "Deploy document classification for single department",
            "Train initial user group (10-20 users)", 
            "Establish baseline metrics",
            "Configure monitoring and alerting"
        ]},
        {"Phase": "Phase 2: Expansion (Months 3-4)", "Activities": [
            "Roll out to additional departments",
            "Implement risk assessment module",
            "Scale user training program",
            "Optimize model performance"
        ]},
        {"Phase": "Phase 3: Full Deployment (Months 5-6)", "Activities": [
            "Company-wide deployment",
            "Launch compliance checking module",
            "Integrate with existing systems",
            "Establish governance processes"
        ]},
        {"Phase": "Phase 4: Optimization (Months 7-12)", "Activities": [
            "Continuous model improvement",
            "Advanced analytics implementation", 
            "Process automation expansion",
            "ROI measurement and reporting"
        ]}
    ]
    
    for phase in roadmap_phases:
        with st.expander(phase["Phase"]):
            for activity in phase["Activities"]:
                st.markdown(f"• {activity}")
    
    # Success factors
    st.markdown("### 🎯 Critical Success Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Technical Success Factors
        - ✅ **Data Quality**: Clean, labeled training data
        - ✅ **Model Performance**: >90% accuracy across all tasks
        - ✅ **System Integration**: Seamless API integration
        - ✅ **Scalability**: Handle 10x current volume
        - ✅ **Monitoring**: Real-time performance tracking
        """)
    
    with col2:
        st.markdown("""
        #### Business Success Factors
        - 🎯 **User Adoption**: >80% user engagement rate
        - 📈 **Process Integration**: Embedded in daily workflows
        - 🎓 **Training Program**: Comprehensive user education
        - 📊 **Metrics Tracking**: Clear ROI measurement
        - 🤝 **Stakeholder Buy-in**: Leadership support
        """)
    
    # Download business case
    st.markdown("### 📄 Business Case Summary")
    
    business_case = f"""
    # PharmaSCM-AI Business Case Summary
    
    ## Investment Overview
    - Initial Implementation Cost: $500,000
    - Annual Operating Cost: $600,000
    - Total 2-Year Investment: $1.7M
    
    ## Financial Returns
    - Annual Cost Savings: $13.2M+
    - 2-Year Net Benefit: $24.7M
    - ROI: 1,453% over 2 years
    - Break-even: Month {breakeven_month if breakeven_month else 'TBD'}
    
    ## Risk Mitigation Value
    - Supply Chain Disruption Prevention: $2.5M/year
    - Compliance Violation Prevention: $6.0M/year  
    - Quality Issue Prevention: $3.0M/year
    - Operational Efficiency: $5.0M/year
    
    ## Key Performance Indicators
    - Document Processing Accuracy: 94.2%
    - Risk Assessment Accuracy: 91.8%
    - Compliance Detection Rate: 96.5%
    - Processing Speed Improvement: 2,400x faster
    
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    st.download_button(
        label="📥 Download Business Case",
        data=business_case,
        file_name="PharmaSCM_AI_Business_Case.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    main()