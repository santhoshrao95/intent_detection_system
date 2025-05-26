"""
Intent Detection Utils Package
=============================

This package provides comprehensive utilities for intent detection tasks including
data processing, model training, and evaluation.

Modules:
- data_processing: Text preprocessing and vectorization
- training: Model training, hyperparameter tuning, and imbalance handling  
- evaluation: Metrics calculation, visualization, and model comparison

Author: Intent Detection System
Date: 2025
"""

from .data_processing import (
    DataPipeline,
    TextCleaner, 
    VectorizerFactory,
    create_data_pipeline,
    quick_data_prep
)

from .training import (
    TrainingPipeline,
    ModelManager,
    ImbalanceHandler,
    HyperparameterTuner,
    ExperimentLogger,
    quick_train_model,
    create_training_pipeline
)

from .evaluation import (
    EvaluationPipeline,
    MetricsCalculator,
    Visualizer,
    ResultsComparator,
    quick_evaluate_model,
    create_evaluation_pipeline,
    show_leaderboard
)

__version__ = "1.0.0"
__author__ = "Intent Detection System"

# Quick access functions
def run_complete_experiment(data_path: str, 
                          experiment_name: str = "Complete Experiment",
                          vectorizer_type: str = 'tfidf',
                          model_name: str = 'logistic_regression',
                          imbalance_method: str = 'smote',
                          test_size: float = 0.2):
    """
    Run a complete experiment from data loading to evaluation.
    
    Args:
        data_path: Path to CSV data file
        experiment_name: Name for the experiment
        vectorizer_type: Type of vectorizer ('tfidf', 'word2vec', 'glove')  
        model_name: Name of model to use
        imbalance_method: Method for handling class imbalance
        test_size: Proportion of test set
        
    Returns:
        Dictionary with all results
    """
    print(f"🚀 RUNNING COMPLETE EXPERIMENT: {experiment_name}")
    print("=" * 60)
    
    # 1. Data Processing
    print("📊 Step 1: Data Processing...")
    data_pipeline = create_data_pipeline(vectorizer_type)
    X_train, X_test, y_train, y_test, _, _ = data_pipeline.prepare_data(
        data_path, test_size=test_size
    )
    
    # 2. Model Training  
    print("🔧 Step 2: Model Training...")
    trainer = create_training_pipeline()
    trainer.setup_experiment(
        experiment_name=experiment_name,
        model_name=model_name,
        imbalance_method=imbalance_method,
        tuning_method='grid'
    )
    training_results = trainer.train_model(X_train, y_train, X_test, y_test)
    
    # 3. Model Evaluation
    print("🔍 Step 3: Model Evaluation...")
    evaluator = create_evaluation_pipeline()
    
    # Get unique class names
    class_names = sorted(list(set(y_test)))
    
    evaluation_results = evaluator.evaluate_model(
        training_results['model'], 
        X_test, y_test, 
        class_names=class_names,
        experiment_name=experiment_name
    )
    
    # Combine results
    complete_results = {
        'experiment_name': experiment_name,
        'data_pipeline': data_pipeline,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'test_data': {
            'X_test': X_test,
            'y_test': y_test,
            'class_names': class_names
        }
    }
    
    print(f"✅ EXPERIMENT COMPLETED: {experiment_name}")
    print(f"🎯 Primary Metric (Macro F1): {evaluation_results['f1_scores']['macro_f1']:.4f}")
    
    return complete_results


# Package information
__all__ = [
    # Data Processing
    'DataPipeline', 'TextCleaner', 'VectorizerFactory',
    'create_data_pipeline', 'quick_data_prep',
    
    # Training
    'TrainingPipeline', 'ModelManager', 'ImbalanceHandler', 
    'HyperparameterTuner', 'ExperimentLogger',
    'quick_train_model', 'create_training_pipeline',
    
    # Evaluation
    'EvaluationPipeline', 'MetricsCalculator', 'Visualizer', 
    'ResultsComparator', 'quick_evaluate_model', 
    'create_evaluation_pipeline', 'show_leaderboard',
    
    # Complete workflow
    'run_complete_experiment'
]