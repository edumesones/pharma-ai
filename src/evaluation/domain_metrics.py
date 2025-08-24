"""
Domain-Specific Evaluation Metrics for Pharmaceutical Supply Chain Models

This module implements specialized metrics and evaluation frameworks that capture
business value and domain-specific requirements for pharmaceutical supply chain AI.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class BusinessImpactConfig:
    """Configuration for business impact calculations"""
    
    # Cost per error (in USD)
    false_positive_cost: float = 100  # Manual review cost
    false_negative_cost: float = 10000  # Missed critical issue
    
    # Processing costs
    manual_processing_cost_per_doc: float = 25  # Manual document processing
    automated_processing_cost_per_doc: float = 0.10  # AI processing
    
    # Time savings
    manual_processing_time_hours: float = 0.5  # Hours per document
    automated_processing_time_hours: float = 0.001  # AI processing time
    
    # Risk costs
    supply_disruption_cost: float = 50000  # Per disruption event
    compliance_violation_fine: float = 1200000  # Average regulatory fine
    
    # Volume assumptions
    monthly_document_volume: int = 10000
    annual_risk_events_prevented: int = 50
    compliance_violations_prevented: int = 5

class PharmaEvaluationMetrics:
    """
    Comprehensive evaluation framework for pharmaceutical supply chain models
    
    Provides both technical metrics and business impact analysis
    """
    
    def __init__(self, config: BusinessImpactConfig = None):
        self.config = config or BusinessImpactConfig()
        
    def compute_technical_metrics(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 y_prob: Optional[np.ndarray] = None,
                                 class_names: List[str] = None) -> Dict:
        """
        Compute comprehensive technical metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_prob: Prediction probabilities (optional)
            class_names: Names of classes (optional)
            
        Returns:
            Dictionary containing all technical metrics
        """
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        if class_names:
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            for i, class_name in enumerate(class_names):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
        
        # ROC AUC for binary/multiclass
        if y_prob is not None:
            unique_classes = np.unique(y_true)
            if len(unique_classes) == 2:
                # Binary classification
                if y_prob.shape[1] == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            else:
                # Multiclass
                try:
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                    metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
                except ValueError:
                    logger.warning("Could not compute ROC AUC for multiclass")
        
        return metrics
    
    def compute_business_impact(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               task_type: str = 'general') -> Dict:
        """
        Compute business impact metrics specific to pharmaceutical supply chain
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            task_type: Type of task ('document_classification', 'risk_assessment', 'compliance_checking')
        """
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else self._extract_binary_metrics(cm)
        
        # Base error costs
        fp_cost = fp * self.config.false_positive_cost
        fn_cost = fn * self.config.false_negative_cost
        
        # Task-specific business impact
        if task_type == 'document_classification':
            return self._compute_document_classification_impact(cm, fp, fn)
        elif task_type == 'risk_assessment':
            return self._compute_risk_assessment_impact(cm, fp, fn)
        elif task_type == 'compliance_checking':
            return self._compute_compliance_impact(cm, fp, fn)
        else:
            return self._compute_general_impact(cm, fp, fn)
    
    def _compute_document_classification_impact(self, cm: np.ndarray, fp: int, fn: int) -> Dict:
        """Business impact for document classification task"""
        
        total_docs = np.sum(cm)
        correctly_classified = np.trace(cm)
        
        # Processing cost savings
        manual_cost = total_docs * self.config.manual_processing_cost_per_doc
        automated_cost = total_docs * self.config.automated_processing_cost_per_doc
        processing_savings = manual_cost - automated_cost
        
        # Error costs
        error_cost = fp * self.config.false_positive_cost + fn * self.config.false_negative_cost
        
        # Time savings
        manual_time = total_docs * self.config.manual_processing_time_hours
        automated_time = total_docs * self.config.automated_processing_time_hours
        time_saved_hours = manual_time - automated_time
        
        # Net benefit
        net_benefit = processing_savings - error_cost
        
        return {
            'total_documents': int(total_docs),
            'correctly_classified': int(correctly_classified),
            'classification_accuracy': correctly_classified / total_docs,
            'processing_cost_savings': processing_savings,
            'error_cost': error_cost,
            'net_benefit': net_benefit,
            'time_saved_hours': time_saved_hours,
            'monthly_savings': net_benefit * (self.config.monthly_document_volume / total_docs),
            'annual_savings': net_benefit * (self.config.monthly_document_volume * 12 / total_docs),
            'roi_percentage': (net_benefit / automated_cost) * 100 if automated_cost > 0 else 0
        }
    
    def _compute_risk_assessment_impact(self, cm: np.ndarray, fp: int, fn: int) -> Dict:
        """Business impact for risk assessment task"""
        
        total_assessments = np.sum(cm)
        correct_assessments = np.trace(cm)
        
        # Risk prevention value
        # Assume model prevents supply disruptions by early warning
        prevented_disruptions = correct_assessments * 0.6  # 60% prevention rate
        disruption_savings = prevented_disruptions * self.config.supply_disruption_cost
        
        # False negative cost (missed high-risk events)
        missed_risk_cost = fn * self.config.supply_disruption_cost * 0.8
        
        # False positive cost (unnecessary risk mitigation)
        false_alarm_cost = fp * 5000  # Cost of unnecessary risk mitigation
        
        net_benefit = disruption_savings - missed_risk_cost - false_alarm_cost
        
        return {
            'total_risk_assessments': int(total_assessments),
            'correct_assessments': int(correct_assessments),
            'assessment_accuracy': correct_assessments / total_assessments,
            'prevented_disruptions': prevented_disruptions,
            'disruption_savings': disruption_savings,
            'missed_risk_cost': missed_risk_cost,
            'false_alarm_cost': false_alarm_cost,
            'net_benefit': net_benefit,
            'annual_risk_prevention_value': self.config.annual_risk_events_prevented * self.config.supply_disruption_cost,
            'risk_prevention_efficiency': (prevented_disruptions / total_assessments) * 100
        }
    
    def _compute_compliance_impact(self, cm: np.ndarray, fp: int, fn: int) -> Dict:
        """Business impact for compliance checking task"""
        
        total_checks = np.sum(cm)
        correct_checks = np.trace(cm)
        
        # Compliance violation prevention
        violations_prevented = correct_checks * 0.95  # 95% prevention rate for correct classifications
        violation_savings = violations_prevented * self.config.compliance_violation_fine
        
        # False negative cost (missed non-compliance)
        missed_violation_cost = fn * self.config.compliance_violation_fine
        
        # False positive cost (unnecessary compliance review)
        false_positive_review_cost = fp * 2000  # Cost of unnecessary deep compliance review
        
        # Audit preparation time savings
        audit_time_savings = total_checks * 2  # 2 hours saved per automated check
        
        net_benefit = violation_savings - missed_violation_cost - false_positive_review_cost
        
        return {
            'total_compliance_checks': int(total_checks),
            'correct_checks': int(correct_checks),
            'compliance_accuracy': correct_checks / total_checks,
            'violations_prevented': violations_prevented,
            'violation_savings': violation_savings,
            'missed_violation_cost': missed_violation_cost,
            'false_positive_cost': false_positive_review_cost,
            'net_benefit': net_benefit,
            'audit_time_saved_hours': audit_time_savings,
            'annual_compliance_value': self.config.compliance_violations_prevented * self.config.compliance_violation_fine,
            'compliance_efficiency': (violations_prevented / total_checks) * 100
        }
    
    def _compute_general_impact(self, cm: np.ndarray, fp: int, fn: int) -> Dict:
        """General business impact calculation"""
        
        total_predictions = np.sum(cm)
        correct_predictions = np.trace(cm)
        
        error_cost = fp * self.config.false_positive_cost + fn * self.config.false_negative_cost
        
        return {
            'total_predictions': int(total_predictions),
            'correct_predictions': int(correct_predictions),
            'accuracy': correct_predictions / total_predictions,
            'error_cost': error_cost,
            'false_positive_cost': fp * self.config.false_positive_cost,
            'false_negative_cost': fn * self.config.false_negative_cost
        }
    
    def _extract_binary_metrics(self, cm: np.ndarray) -> Tuple[int, int, int, int]:
        """Extract binary metrics from multiclass confusion matrix"""
        # For multiclass, approximate as macro average
        n_classes = cm.shape[0]
        tn = fp = fn = tp = 0
        
        for i in range(n_classes):
            tp_i = cm[i, i]
            fp_i = cm[:, i].sum() - tp_i
            fn_i = cm[i, :].sum() - tp_i
            tn_i = cm.sum() - tp_i - fp_i - fn_i
            
            tp += tp_i
            fp += fp_i
            fn += fn_i
            tn += tn_i
        
        return tn // n_classes, fp // n_classes, fn // n_classes, tp // n_classes
    
    def create_evaluation_report(self, 
                                results: Dict[str, Dict],
                                model_name: str = "PharmaSCM-AI") -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            results: Dictionary containing results for each task
            model_name: Name of the model system
            
        Returns:
            Formatted evaluation report as string
        """
        
        report = f"# {model_name} - Comprehensive Evaluation Report\n\n"
        report += f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        
        total_accuracy = np.mean([r['technical']['accuracy'] for r in results.values()])
        total_f1 = np.mean([r['technical']['f1_weighted'] for r in results.values()])
        
        report += f"- **Overall System Accuracy**: {total_accuracy:.3f}\n"
        report += f"- **Overall System F1-Score**: {total_f1:.3f}\n"
        report += f"- **Tasks Evaluated**: {len(results)}\n\n"
        
        # Business Impact Summary
        total_annual_savings = 0
        for task_name, result in results.items():
            if 'business' in result:
                annual_savings = result['business'].get('annual_savings', 0)
                if annual_savings == 0:
                    annual_savings = result['business'].get('annual_compliance_value', 0)
                if annual_savings == 0:
                    annual_savings = result['business'].get('annual_risk_prevention_value', 0)
                total_annual_savings += annual_savings
        
        report += f"- **Estimated Annual Business Value**: ${total_annual_savings:,.0f}\n\n"
        
        # Detailed Task Analysis
        for task_name, result in results.items():
            report += f"## {task_name.replace('_', ' ').title()}\n\n"
            
            # Technical Metrics
            tech = result['technical']
            report += "### Technical Performance\n\n"
            report += f"- **Accuracy**: {tech['accuracy']:.4f}\n"
            report += f"- **Precision (Weighted)**: {tech['precision_weighted']:.4f}\n"
            report += f"- **Recall (Weighted)**: {tech['recall_weighted']:.4f}\n"
            report += f"- **F1-Score (Weighted)**: {tech['f1_weighted']:.4f}\n\n"
            
            if 'roc_auc' in tech:
                report += f"- **ROC AUC**: {tech['roc_auc']:.4f}\n\n"
            
            # Business Impact
            if 'business' in result:
                business = result['business']
                report += "### Business Impact\n\n"
                
                if 'net_benefit' in business:
                    report += f"- **Net Business Benefit**: ${business['net_benefit']:,.0f}\n"
                if 'annual_savings' in business:
                    report += f"- **Annual Savings**: ${business['annual_savings']:,.0f}\n"
                if 'roi_percentage' in business:
                    report += f"- **ROI**: {business['roi_percentage']:.1f}%\n"
                if 'time_saved_hours' in business:
                    report += f"- **Time Saved**: {business['time_saved_hours']:.0f} hours\n"
                
                report += "\n"
            
        # Recommendations
        report += "## Recommendations\n\n"
        report += "### Technical Improvements\n"
        
        lowest_f1_task = min(results.keys(), key=lambda k: results[k]['technical']['f1_weighted'])
        lowest_f1_score = results[lowest_f1_task]['technical']['f1_weighted']
        
        if lowest_f1_score < 0.85:
            report += f"- **Priority**: Improve {lowest_f1_task.replace('_', ' ')} model (F1: {lowest_f1_score:.3f})\n"
            report += "- Consider data augmentation and hyperparameter tuning\n"
        
        report += "- Implement model monitoring and drift detection\n"
        report += "- Set up A/B testing framework for model updates\n\n"
        
        report += "### Business Implementation\n"
        report += "- Begin with pilot deployment in controlled environment\n"
        report += "- Establish baseline metrics for ROI measurement\n"
        report += "- Train end users on AI-assisted workflows\n"
        report += "- Implement feedback loops for continuous improvement\n\n"
        
        report += "### Production Scaling\n"
        report += "- Deploy on cloud infrastructure with auto-scaling\n"
        report += "- Implement comprehensive logging and monitoring\n"
        report += "- Establish model governance and compliance processes\n"
        report += "- Plan for regular model retraining cycles\n\n"
        
        return report
    
    def visualize_performance(self, 
                             results: Dict[str, Dict],
                             save_path: Optional[str] = None) -> None:
        """
        Create comprehensive performance visualizations
        
        Args:
            results: Dictionary containing results for each task
            save_path: Optional path to save the plot
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Task names and metrics
        tasks = list(results.keys())
        task_labels = [t.replace('_', '\n').title() for t in tasks]
        
        # Technical metrics comparison
        accuracies = [results[t]['technical']['accuracy'] for t in tasks]
        f1_scores = [results[t]['technical']['f1_weighted'] for t in tasks]
        precisions = [results[t]['technical']['precision_weighted'] for t in tasks]
        recalls = [results[t]['technical']['recall_weighted'] for t in tasks]
        
        x = np.arange(len(tasks))
        width = 0.2
        
        axes[0, 0].bar(x - width*1.5, accuracies, width, label='Accuracy', alpha=0.8)
        axes[0, 0].bar(x - width*0.5, f1_scores, width, label='F1-Score', alpha=0.8)
        axes[0, 0].bar(x + width*0.5, precisions, width, label='Precision', alpha=0.8)
        axes[0, 0].bar(x + width*1.5, recalls, width, label='Recall', alpha=0.8)
        
        axes[0, 0].set_xlabel('Tasks')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Technical Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(task_labels)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Business impact visualization
        business_values = []
        business_labels = []
        
        for task in tasks:
            if 'business' in results[task]:
                business = results[task]['business']
                value = business.get('annual_savings', 
                                   business.get('annual_compliance_value',
                                              business.get('annual_risk_prevention_value', 0)))
                business_values.append(value / 1000000)  # Convert to millions
                business_labels.append(task.replace('_', '\n').title())
        
        if business_values:
            colors = plt.cm.Set3(np.linspace(0, 1, len(business_values)))
            bars = axes[0, 1].bar(business_labels, business_values, color=colors, alpha=0.8)
            axes[0, 1].set_ylabel('Annual Value (Million USD)')
            axes[0, 1].set_title('Estimated Annual Business Impact')
            
            # Add value labels on bars
            for bar, value in zip(bars, business_values):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'${value:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # Confusion matrices (for first task as example)
        first_task = tasks[0]
        cm = results[first_task]['technical']['confusion_matrix']
        
        # Get class names if available
        class_names = None
        if 'classification_report' in results[first_task]['technical']:
            report = results[first_task]['technical']['classification_report']
            class_names = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], 
                   xticklabels=class_names, yticklabels=class_names, cmap='Blues')
        axes[1, 0].set_title(f'Confusion Matrix - {first_task.replace("_", " ").title()}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # ROI timeline projection
        months = ['Month 1', 'Month 6', 'Month 12', 'Month 18', 'Month 24']
        total_annual_value = sum(business_values) if business_values else 50  # Default assumption
        
        # Assume gradual implementation: 0%, 30%, 70%, 85%, 100%
        implementation_rates = [0, 0.3, 0.7, 0.85, 1.0]
        cumulative_savings = [rate * total_annual_value * (i+1)/12 for i, rate in enumerate(implementation_rates)]
        
        axes[1, 1].plot(months, cumulative_savings, marker='o', linewidth=3, markersize=8, 
                       color='green', markerfacecolor='lightgreen')
        axes[1, 1].fill_between(months, cumulative_savings, alpha=0.3, color='green')
        axes[1, 1].set_ylabel('Cumulative Savings (Million USD)')
        axes[1, 1].set_title('ROI Timeline Projection')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xticklabels(months, rotation=45)
        
        # Add annotations for key milestones
        axes[1, 1].annotate('Pilot Launch', xy=(0, 0), xytext=(0.5, max(cumulative_savings)*0.8),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        axes[1, 1].annotate('Full Production', xy=(4, cumulative_savings[-1]), 
                           xytext=(3, max(cumulative_savings)*0.9),
                           arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance visualization saved to {save_path}")
        
        plt.show()

def evaluate_pharma_model(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_prob: Optional[np.ndarray] = None,
                         task_type: str = 'general',
                         class_names: List[str] = None,
                         config: BusinessImpactConfig = None) -> Dict:
    """
    Convenience function for complete model evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)
        task_type: Type of task for business impact calculation
        class_names: Names of classes
        config: Business impact configuration
        
    Returns:
        Dictionary containing technical and business metrics
    """
    
    evaluator = PharmaEvaluationMetrics(config)
    
    # Compute technical metrics
    technical_metrics = evaluator.compute_technical_metrics(
        y_true, y_pred, y_prob, class_names
    )
    
    # Compute business impact
    business_metrics = evaluator.compute_business_impact(
        y_true, y_pred, task_type
    )
    
    return {
        'technical': technical_metrics,
        'business': business_metrics
    }

# Example usage and testing
def main():
    """Demonstrate evaluation framework"""
    
    # Generate sample predictions for demonstration
    np.random.seed(42)
    
    # Document classification example (4 classes)
    doc_y_true = np.random.choice([0, 1, 2, 3], size=1000, p=[0.3, 0.25, 0.25, 0.2])
    doc_y_pred = doc_y_true.copy()
    doc_y_pred[np.random.choice(1000, 100, replace=False)] = np.random.choice([0, 1, 2, 3], 100)  # Add some errors
    
    doc_class_names = ['Supply Contract', 'Quality Report', 'Risk Assessment', 'Compliance Report']
    
    # Risk assessment example (3 classes)
    risk_y_true = np.random.choice([0, 1, 2], size=500, p=[0.4, 0.4, 0.2])
    risk_y_pred = risk_y_true.copy()
    risk_y_pred[np.random.choice(500, 50, replace=False)] = np.random.choice([0, 1, 2], 50)
    
    risk_class_names = ['Low', 'Medium', 'High']
    
    # Compliance checking example (2 classes)
    comp_y_true = np.random.choice([0, 1], size=400, p=[0.3, 0.7])
    comp_y_pred = comp_y_true.copy()
    comp_y_pred[np.random.choice(400, 30, replace=False)] = 1 - comp_y_pred[np.random.choice(400, 30, replace=False)]
    
    comp_class_names = ['Non-Compliant', 'Compliant']
    
    # Evaluate each task
    doc_results = evaluate_pharma_model(
        doc_y_true, doc_y_pred, 
        task_type='document_classification',
        class_names=doc_class_names
    )
    
    risk_results = evaluate_pharma_model(
        risk_y_true, risk_y_pred,
        task_type='risk_assessment', 
        class_names=risk_class_names
    )
    
    comp_results = evaluate_pharma_model(
        comp_y_true, comp_y_pred,
        task_type='compliance_checking',
        class_names=comp_class_names
    )
    
    # Combine results
    all_results = {
        'document_classification': doc_results,
        'risk_assessment': risk_results,
        'compliance_checking': comp_results
    }
    
    # Create evaluation report
    evaluator = PharmaEvaluationMetrics()
    report = evaluator.create_evaluation_report(all_results)
    print(report)
    
    # Create visualizations
    evaluator.visualize_performance(all_results, 'pharma_evaluation_demo.png')
    
    print("\nEvaluation framework demonstration completed!")

if __name__ == "__main__":
    main()