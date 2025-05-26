"""
Evaluation Pipeline for Intent Detection
=======================================

This module provides comprehensive evaluation capabilities including
metrics calculation, visualization, model comparison, and results analysis.

Author: Intent Detection System
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support, accuracy_score
)
from sklearn.metrics import precision_score, recall_score

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class MetricsCalculator:
    """
    Calculates comprehensive evaluation metrics for classification tasks.
    """
    
    def __init__(self):
        """Initialize MetricsCalculator."""
        pass
    
    def calculate_f1_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate macro, micro, and weighted F1-scores.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with F1-scores
        """
        return {
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'micro_f1': f1_score(y_true, y_pred, average='micro'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   labels: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Calculate per-class precision, recall, and F1-score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels (optional)
            
        Returns:
            Dictionary with per-class metrics
        """
        if labels is None:
            labels = sorted(list(set(y_true)))
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        
        metrics = {}
        for i, label in enumerate(labels):
            metrics[label] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
        
        return metrics
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  labels: Optional[List[str]] = None) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels (optional)
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred, labels=labels)
    
    def calculate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      labels: Optional[List[str]] = None) -> str:
        """
        Generate classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels (optional)
            
        Returns:
            Classification report string
        """
        return classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      labels: Optional[List[str]] = None) -> Dict:
        """
        Calculate all metrics in one go.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels (optional)
            
        Returns:
            Dictionary with all metrics
        """
        if labels is None:
            labels = sorted(list(set(y_true)))
        
        return {
            'f1_scores': self.calculate_f1_scores(y_true, y_pred),
            'per_class_metrics': self.calculate_per_class_metrics(y_true, y_pred, labels),
            'confusion_matrix': self.calculate_confusion_matrix(y_true, y_pred, labels),
            'classification_report': self.calculate_classification_report(y_true, y_pred, labels),
            'accuracy': accuracy_score(y_true, y_pred),
            'class_labels': labels
        }


class Visualizer:
    """
    Creates visualizations for model evaluation results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize Visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str],
                             title: str = "Confusion Matrix",
                             normalize: bool = False,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            title: Plot title
            normalize: Whether to normalize the matrix
            save_path: Path to save the plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=self.figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_per_class_metrics(self, per_class_metrics: Dict[str, Dict],
                              metric: str = 'f1_score',
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot per-class metrics as bar chart.
        
        Args:
            per_class_metrics: Per-class metrics dictionary
            metric: Metric to plot ('precision', 'recall', 'f1_score')
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        classes = list(per_class_metrics.keys())
        values = [per_class_metrics[cls][metric] for cls in classes]
        
        plt.figure(figsize=(max(12, len(classes) * 0.6), 8))
        
        bars = plt.bar(classes, values, color='skyblue', alpha=0.7, edgecolor='navy')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        if title is None:
            title = f'Per-Class {metric.replace("_", " ").title()}'
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Intent Classes', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_f1_scores_comparison(self, f1_scores: Dict[str, float],
                                 title: str = "F1-Score Comparison",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of different F1-scores.
        
        Args:
            f1_scores: Dictionary with F1-scores
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        scores = ['macro_f1', 'micro_f1', 'weighted_f1']
        values = [f1_scores.get(score, 0) for score in scores]
        labels = ['Macro F1', 'Micro F1', 'Weighted F1']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('F1-Score', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        # Highlight macro F1 as primary metric
        plt.text(0, values[0] + 0.05, 'PRIMARY\nMETRIC', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_model_comparison(self, comparison_data: pd.DataFrame,
                             metric: str = 'macro_f1',
                             title: Optional[str] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison between multiple models.
        
        Args:
            comparison_data: DataFrame with model comparison data
            metric: Metric to compare
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            matplotlib Figure object
        """
        plt.figure(figsize=(max(12, len(comparison_data) * 0.8), 8))
        
        # Sort by metric value
        comparison_data = comparison_data.sort_values(metric, ascending=True)
        
        bars = plt.barh(comparison_data['name'], comparison_data[metric],
                       color='lightcoral', alpha=0.7, edgecolor='darkred')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, comparison_data[metric])):
            plt.text(value + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center', ha='left', fontweight='bold')
        
        if title is None:
            title = f'Model Comparison - {metric.replace("_", " ").title()}'
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.xlim(0, 1.1)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()


class ResultsComparator:
    """
    Compares results from multiple experiments and generates comparison reports.
    """
    
    def __init__(self, results_dir: str = 'results/experiment_logs'):
        """
        Initialize ResultsComparator.
        
        Args:
            results_dir: Directory containing experiment logs
        """
        self.results_dir = results_dir
        self.experiments = {}
    
    def load_experiments(self, experiment_ids: Optional[List[str]] = None) -> Dict:
        """
        Load experiment data from JSON files.
        
        Args:
            experiment_ids: Specific experiment IDs to load (optional)
            
        Returns:
            Dictionary with experiment data
        """
        if experiment_ids is None:
            # Load all experiments
            if not os.path.exists(self.results_dir):
                print(f"Results directory {self.results_dir} not found!")
                return {}
                
            experiment_files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
            experiment_ids = [f.replace('.json', '') for f in experiment_files]
        
        for exp_id in experiment_ids:
            log_file = os.path.join(self.results_dir, f"{exp_id}.json")
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r') as f:
                        self.experiments[exp_id] = json.load(f)
                except Exception as e:
                    print(f"Error loading experiment {exp_id}: {e}")
        
        print(f"Loaded {len(self.experiments)} experiments")
        return self.experiments
    
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """
        Create DataFrame for easy comparison of experiments.
        
        Returns:
            DataFrame with experiment comparison data
        """
        if not self.experiments:
            self.load_experiments()
        
        comparison_data = []
        
        for exp_id, exp_data in self.experiments.items():
            row = {
                'experiment_id': exp_id,
                'name': exp_data.get('name', 'Unknown'),
                'model': exp_data.get('config', {}).get('model_name', 'Unknown'),
                'vectorizer': exp_data.get('config', {}).get('vectorizer', 'Unknown'),
                'imbalance_method': exp_data.get('config', {}).get('imbalance_method', 'none'),
                'tuning_method': exp_data.get('config', {}).get('tuning_method', 'none'),
                'training_time': exp_data.get('training_time', 0),
                'timestamp': exp_data.get('timestamp', '')
            }
            
            # Add test results if available
            test_results = exp_data.get('results', {}).get('test_results', {})
            row.update({
                'macro_f1': test_results.get('macro_f1', 0),
                'micro_f1': test_results.get('micro_f1', 0),
                'weighted_f1': test_results.get('weighted_f1', 0)
            })
            
            # Add CV score if available
            row['cv_score'] = exp_data.get('results', {}).get('cv_score', 0)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def show_leaderboard(self, metric: str = 'macro_f1', top_n: int = 10) -> pd.DataFrame:
        """
        Show top performing experiments.
        
        Args:
            metric: Metric to rank by
            top_n: Number of top experiments to show
            
        Returns:
            DataFrame with top experiments
        """
        df = self.create_comparison_dataframe()
        
        if df.empty:
            print("No experiments found!")
            return df
        
        # Sort by metric (descending)
        df_sorted = df.sort_values(metric, ascending=False).head(top_n)
        
        # Select relevant columns for display
        display_cols = ['name', 'model', 'imbalance_method', metric, 'training_time']
        if 'vectorizer' in df.columns:
            display_cols.insert(2, 'vectorizer')
        
        leaderboard = df_sorted[display_cols].copy()
        leaderboard[metric] = leaderboard[metric].round(4)
        leaderboard['training_time'] = leaderboard['training_time'].round(1)
        
        print(f"\n🏆 TOP {top_n} EXPERIMENTS (by {metric.upper()})")
        print("=" * 80)
        print(leaderboard.to_string(index=False))
        
        return leaderboard
    
    def compare_experiments(self, experiment_ids: List[str],
                           metrics: List[str] = ['macro_f1', 'micro_f1', 'weighted_f1']) -> pd.DataFrame:
        """
        Compare specific experiments across multiple metrics.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            Comparison DataFrame
        """
        df = self.create_comparison_dataframe()
        
        # Filter for specific experiments
        comparison_df = df[df['experiment_id'].isin(experiment_ids)].copy()
        
        if comparison_df.empty:
            print("No matching experiments found!")
            return comparison_df
        
        # Select columns for comparison
        display_cols = ['name', 'model'] + metrics + ['training_time']
        comparison_result = comparison_df[display_cols]
        
        # Round numeric columns
        for metric in metrics:
            if metric in comparison_result.columns:
                comparison_result[metric] = comparison_result[metric].round(4)
        comparison_result['training_time'] = comparison_result['training_time'].round(1)
        
        print(f"\n📊 EXPERIMENT COMPARISON")
        print("=" * 60)
        print(comparison_result.to_string(index=False))
        
        return comparison_result
    
    def generate_comparison_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive comparison report.
        
        Args:
            save_path: Path to save HTML report (optional)
            
        Returns:
            HTML report string
        """
        df = self.create_comparison_dataframe()
        
        if df.empty:
            return "<p>No experiments found!</p>"
        
        # Generate HTML report
        report_html = f"""
        <html>
        <head>
            <title>Intent Detection - Experiment Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #ffeb3b; }}
                .best-score {{ font-weight: bold; color: #4caf50; }}
            </style>
        </head>
        <body>
            <h1>Intent Detection - Experiment Comparison Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary Statistics</h2>
            <ul>
                <li>Total Experiments: {len(df)}</li>
                <li>Best Macro F1: {df['macro_f1'].max():.4f}</li>
                <li>Average Macro F1: {df['macro_f1'].mean():.4f}</li>
                <li>Best Model: {df.loc[df['macro_f1'].idxmax(), 'model']}</li>
            </ul>
            
            <h2>Detailed Results</h2>
            {df.sort_values('macro_f1', ascending=False).to_html(index=False, classes='table')}
            
            <h2>Model Performance Analysis</h2>
            <h3>By Model Type:</h3>
            {df.groupby('model').agg({
                'macro_f1': ['mean', 'std', 'count'],
                'training_time': 'mean'
            }).round(4).to_html()}
            
            <h3>By Imbalance Handling Method:</h3>
            {df.groupby('imbalance_method').agg({
                'macro_f1': ['mean', 'std', 'count']
            }).round(4).to_html()}
        </body>
        </html>
        """
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_html)
            print(f"Report saved to: {save_path}")
        
        return report_html


class EvaluationPipeline:
    """
    Main evaluation pipeline that orchestrates all evaluation tasks.
    """
    
    def __init__(self, results_dir: str = 'results'):
        """
        Initialize EvaluationPipeline.
        
        Args:
            results_dir: Directory for storing results
        """
        self.results_dir = results_dir
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
        self.comparator = ResultsComparator(os.path.join(results_dir, 'experiment_logs'))
        
        # Create directories
        os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'comparison_reports'), exist_ok=True)
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      class_names: Optional[List[str]] = None,
                      experiment_name: str = "Model Evaluation") -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            class_names: List of class names
            experiment_name: Name for the evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n🔍 EVALUATING: {experiment_name}")
        print("=" * 50)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate all metrics
        metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            y_test, y_pred, class_names
        )
        
        # Print key results
        f1_scores = metrics['f1_scores']
        print(f"📊 F1-SCORES:")
        print(f"   Macro F1 (Primary):    {f1_scores['macro_f1']:.4f}")
        print(f"   Micro F1:              {f1_scores['micro_f1']:.4f}")
        print(f"   Weighted F1:           {f1_scores['weighted_f1']:.4f}")
        print(f"   Accuracy:              {metrics['accuracy']:.4f}")
        
        # Show per-class summary
        per_class = metrics['per_class_metrics']
        best_class = max(per_class.keys(), key=lambda x: per_class[x]['f1_score'])
        worst_class = min(per_class.keys(), key=lambda x: per_class[x]['f1_score'])
        
        print(f"\n📈 PER-CLASS PERFORMANCE:")
        print(f"   Best performing:  {best_class} (F1: {per_class[best_class]['f1_score']:.4f})")
        print(f"   Worst performing: {worst_class} (F1: {per_class[worst_class]['f1_score']:.4f})")
        
        # Generate visualizations
        plots_dir = os.path.join(self.results_dir, 'plots')
        experiment_safe_name = experiment_name.replace(' ', '_').replace('/', '_')
        
        # 1. Confusion Matrix
        cm_path = os.path.join(plots_dir, f"{experiment_safe_name}_confusion_matrix.png")
        self.visualizer.plot_confusion_matrix(
            metrics['confusion_matrix'], 
            metrics['class_labels'],
            title=f"{experiment_name} - Confusion Matrix",
            save_path=cm_path
        )
        plt.show()
        
        # 2. F1-Scores Comparison
        f1_path = os.path.join(plots_dir, f"{experiment_safe_name}_f1_comparison.png")
        self.visualizer.plot_f1_scores_comparison(
            f1_scores,
            title=f"{experiment_name} - F1-Score Comparison",
            save_path=f1_path
        )
        plt.show()
        
        # 3. Per-class F1-scores
        per_class_path = os.path.join(plots_dir, f"{experiment_safe_name}_per_class_f1.png")
        self.visualizer.plot_per_class_metrics(
            per_class,
            metric='f1_score',
            title=f"{experiment_name} - Per-Class F1-Scores",
            save_path=per_class_path
        )
        plt.show()
        
        # Print classification report
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print("-" * 50)
        print(metrics['classification_report'])
        
        # Add file paths to results
        metrics['visualization_paths'] = {
            'confusion_matrix': cm_path,
            'f1_comparison': f1_path,
            'per_class_f1': per_class_path
        }
        
        return metrics
    
    def quick_evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      class_names: Optional[List[str]] = None) -> Dict:
        """
        Quick evaluation with essential metrics only.
        
        Args:
            model: Trained model
            X_test: Test features  
            y_test: Test labels
            class_names: List of class names
            
        Returns:
            Dictionary with key metrics
        """
        y_pred = model.predict(X_test)
        f1_scores = self.metrics_calculator.calculate_f1_scores(y_test, y_pred)
        
        print(f"Quick Evaluation Results:")
        print(f"  Macro F1:    {f1_scores['macro_f1']:.4f}")
        print(f"  Micro F1:    {f1_scores['micro_f1']:.4f}")
        print(f"  Weighted F1: {f1_scores['weighted_f1']:.4f}")
        
        return f1_scores
    
    def compare_models(self, models_data: List[Dict],
                      X_test: np.ndarray, y_test: np.ndarray,
                      class_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple models on the same test set.
        
        Args:
            models_data: List of dictionaries with 'name' and 'model' keys
            X_test: Test features
            y_test: Test labels
            class_names: List of class names
            
        Returns:
            Comparison DataFrame
        """
        comparison_results = []
        
        print(f"\n🔄 COMPARING {len(models_data)} MODELS")
        print("=" * 50)
        
        for model_info in models_data:
            model_name = model_info['name']
            model = model_info['model']
            
            print(f"Evaluating {model_name}...")
            
            # Quick evaluation
            y_pred = model.predict(X_test)
            f1_scores = self.metrics_calculator.calculate_f1_scores(y_test, y_pred)
            
            result = {
                'name': model_name,
                'macro_f1': f1_scores['macro_f1'],
                'micro_f1': f1_scores['micro_f1'],
                'weighted_f1': f1_scores['weighted_f1'],
                'accuracy': accuracy_score(y_test, y_pred)
            }
            
            comparison_results.append(result)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('macro_f1', ascending=False)
        
        # Display results
        print(f"\n📊 MODEL COMPARISON RESULTS:")
        print("-" * 60)
        print(comparison_df.round(4).to_string(index=False))
        
        # Plot comparison
        plots_dir = os.path.join(self.results_dir, 'plots')
        comparison_path = os.path.join(plots_dir, "model_comparison.png")
        self.visualizer.plot_model_comparison(
            comparison_df,
            title="Model Performance Comparison",
            save_path=comparison_path
        )
        plt.show()
        
        return comparison_df
    
    def show_experiment_leaderboard(self, metric: str = 'macro_f1', top_n: int = 10):
        """Show leaderboard of all experiments."""
        return self.comparator.show_leaderboard(metric, top_n)
    
    def generate_final_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive final report."""
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'comparison_reports', 'final_report.html')
        
        return self.comparator.generate_comparison_report(save_path)


# Utility functions
def quick_evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                        class_names: Optional[List[str]] = None) -> Dict:
    """
    Quick function to evaluate a model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
        
    Returns:
        Evaluation results
    """
    evaluator = EvaluationPipeline()
    return evaluator.quick_evaluate(model, X_test, y_test, class_names)


def create_evaluation_pipeline(**kwargs) -> EvaluationPipeline:
    """Create an evaluation pipeline with custom settings."""
    return EvaluationPipeline(**kwargs)


def show_leaderboard(results_dir: str = 'results', metric: str = 'macro_f1', top_n: int = 10):
    """Quick function to show experiment leaderboard."""
    comparator = ResultsComparator(os.path.join(results_dir, 'experiment_logs'))
    return comparator.show_leaderboard(metric, top_n)


if __name__ == "__main__":
    # Example usage
    print("Evaluation Pipeline - Example Usage")
    print("=" * 50)
    
    # Create pipeline
    evaluator = EvaluationPipeline()
    
    print("Evaluation pipeline ready for use!")
    print("\nExample usage:")
    print("results = evaluator.evaluate_model(model, X_test, y_test, class_names)")
    print("evaluator.show_experiment_leaderboard()")
    print("evaluator.generate_final_report()")