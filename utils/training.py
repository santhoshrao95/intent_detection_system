"""
Training Pipeline for Intent Detection
=====================================

This module provides comprehensive training capabilities including
model management, hyperparameter tuning, class imbalance handling,
and experiment logging.

Author: Intent Detection System
Date: 2025
"""

import numpy as np
import pandas as pd
import json
import os
import pickle
import joblib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import time
import logging

# Core ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, f1_score

# Imbalance handling
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    logging.warning("imbalanced-learn not available. Some imbalance handling methods won't work.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImbalanceHandler:
    """
    Handles various strategies for dealing with class imbalance
    in the training data.
    """
    
    def __init__(self, method: str = 'smote', random_state: int = 42):
        """
        Initialize ImbalanceHandler.
        
        Args:
            method: Imbalance handling method ('smote', 'random_oversample', 
                   'random_undersample', 'smoteenn', 'class_weight', 'none')
            random_state: Random seed for reproducibility
        """
        self.method = method.lower()
        self.random_state = random_state
        self.sampler = None
        
        if self.method != 'none' and self.method != 'class_weight':
            if not IMBALANCED_LEARN_AVAILABLE:
                raise ImportError("imbalanced-learn is required for sampling methods. "
                                "Install with: pip install imbalanced-learn")
        
        self._initialize_sampler()
    
    def _initialize_sampler(self):
        """Initialize the appropriate sampler based on method."""
        if self.method == 'smote':
            self.sampler = SMOTE(random_state=self.random_state)
        elif self.method == 'random_oversample':
            self.sampler = RandomOverSampler(random_state=self.random_state)
        elif self.method == 'random_undersample':
            self.sampler = RandomUnderSampler(random_state=self.random_state)
        elif self.method == 'smoteenn':
            self.sampler = SMOTEENN(random_state=self.random_state)
        elif self.method in ['class_weight', 'none']:
            self.sampler = None
        else:
            raise ValueError(f"Unknown imbalance handling method: {self.method}")
    
    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply imbalance handling to the data.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Resampled X and y
        """
        if self.method == 'none':
            return X, y
        elif self.method == 'class_weight':
            # Class weights are handled in the model, return original data
            return X, y
        else:
            logger.info(f"Applying {self.method} for imbalance handling")
            original_shape = X.shape[0]
            X_resampled, y_resampled = self.sampler.fit_resample(X, y)
            new_shape = X_resampled.shape[0]
            logger.info(f"Data resampled: {original_shape} -> {new_shape} samples")
            return X_resampled, y_resampled


class HyperparameterTuner:
    """
    Handles hyperparameter tuning using GridSearch and RandomizedSearch.
    """
    
    def __init__(self, cv_folds: int = 5, scoring: str = 'f1_macro', 
                 random_state: int = 42, n_jobs: int = -1):
        """
        Initialize HyperparameterTuner.
        
        Args:
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for hyperparameter tuning
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 for all processors)
        """
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Define parameter grids for different models
        self.param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'class_weight': ['balanced', None],
                'max_iter': [1000]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'class_weight': ['balanced', None],
                'gamma': ['scale', 'auto']
            },
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced', None]
            },
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            },
            'mlp': {
                'hidden_layer_sizes': [(100,), (50, 50), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500]
            }
        }
    
    def get_param_grid(self, model_name: str) -> Dict:
        """Get parameter grid for a specific model."""
        return self.param_grids.get(model_name.lower(), {})
    
    def grid_search(self, model, X: np.ndarray, y: np.ndarray, 
                    param_grid: Optional[Dict] = None) -> GridSearchCV:
        """
        Perform grid search hyperparameter tuning.
        
        Args:
            model: Sklearn model instance
            X: Feature matrix
            y: Target labels
            param_grid: Custom parameter grid (optional)
            
        Returns:
            Fitted GridSearchCV object
        """
        if param_grid is None:
            model_name = model.__class__.__name__.lower()
            if 'logistic' in model_name:
                model_name = 'logistic_regression'
            elif 'svc' in model_name or 'svm' in model_name:
                model_name = 'svm'
            elif 'random' in model_name and 'forest' in model_name:
                model_name = 'random_forest'
            elif 'naive' in model_name or 'bayes' in model_name:
                model_name = 'naive_bayes'
            elif 'mlp' in model_name:
                model_name = 'mlp'
            
            param_grid = self.get_param_grid(model_name)
        
        if not param_grid:
            logger.warning(f"No parameter grid found for {model.__class__.__name__}")
            return model.fit(X, y)
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                            random_state=self.random_state)
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        logger.info(f"Starting grid search with {len(param_grid)} parameters")
        grid_search.fit(X, y)
        
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best params: {grid_search.best_params_}")
        
        return grid_search
    
    def random_search(self, model, X: np.ndarray, y: np.ndarray,
                      param_grid: Optional[Dict] = None, 
                      n_iter: int = 20) -> RandomizedSearchCV:
        """
        Perform randomized search hyperparameter tuning.
        
        Args:
            model: Sklearn model instance
            X: Feature matrix
            y: Target labels
            param_grid: Custom parameter grid (optional)
            n_iter: Number of parameter settings sampled
            
        Returns:
            Fitted RandomizedSearchCV object
        """
        if param_grid is None:
            model_name = model.__class__.__name__.lower()
            if 'logistic' in model_name:
                model_name = 'logistic_regression'
            elif 'svc' in model_name or 'svm' in model_name:
                model_name = 'svm'
            elif 'random' in model_name and 'forest' in model_name:
                model_name = 'random_forest'
            elif 'naive' in model_name or 'bayes' in model_name:
                model_name = 'naive_bayes'
            elif 'mlp' in model_name:
                model_name = 'mlp'
                
            param_grid = self.get_param_grid(model_name)
        
        if not param_grid:
            logger.warning(f"No parameter grid found for {model.__class__.__name__}")
            return model.fit(X, y)
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                            random_state=self.random_state)
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1
        )
        
        logger.info(f"Starting random search with {n_iter} iterations")
        random_search.fit(X, y)
        
        logger.info(f"Best score: {random_search.best_score_:.4f}")
        logger.info(f"Best params: {random_search.best_params_}")
        
        return random_search


class ModelManager:
    """
    Manages different machine learning models with a unified interface.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ModelManager.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default models with standard parameters."""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            'svm': SVC(
                random_state=self.random_state, probability=True
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state, n_estimators=100
            ),
            'naive_bayes': MultinomialNB(),
            'mlp': MLPClassifier(
                random_state=self.random_state, max_iter=500
            )
        }
    
    def add_model(self, name: str, model):
        """Add a custom model to the manager."""
        self.models[name] = model
        logger.info(f"Added model: {name}")
    
    def get_model(self, name: str):
        """Get a model by name."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self.models.keys())}")
        
        # Return a fresh copy of the model
        model = self.models[name]
        return type(model)(**model.get_params())
    
    def list_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.models.keys())


class ExperimentLogger:
    """
    Handles experiment logging and result storage for easy comparison.
    """
    
    def __init__(self, results_dir: str = 'results/experiment_logs'):
        """
        Initialize ExperimentLogger.
        
        Args:
            results_dir: Directory to store experiment logs
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}"
    
    def log_experiment(self, experiment_id: str, experiment_name: str,
                      config: Dict, results: Dict, training_time: float,
                      notes: str = "") -> str:
        """
        Log experiment results to JSON file.
        
        Args:
            experiment_id: Unique experiment identifier
            experiment_name: Human-readable experiment name
            config: Experiment configuration
            results: Experiment results
            training_time: Training time in seconds
            notes: Additional notes
            
        Returns:
            Path to saved log file
        """
        log_data = {
            'experiment_id': experiment_id,
            'name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': results,
            'training_time': training_time,
            'notes': notes
        }
        
        log_file = os.path.join(self.results_dir, f"{experiment_id}.json")
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        logger.info(f"Experiment logged: {log_file}")
        return log_file
    
    def load_experiment(self, experiment_id: str) -> Dict:
        """Load experiment data by ID."""
        log_file = os.path.join(self.results_dir, f"{experiment_id}.json")
        
        if not os.path.exists(log_file):
            raise FileNotFoundError(f"Experiment {experiment_id} not found")
        
        with open(log_file, 'r') as f:
            return json.load(f)
    
    def list_experiments(self) -> List[str]:
        """Get list of all experiment IDs."""
        experiments = []
        for file in os.listdir(self.results_dir):
            if file.endswith('.json'):
                experiments.append(file.replace('.json', ''))
        return sorted(experiments)


class TrainingPipeline:
    """
    Main training pipeline that orchestrates model training, hyperparameter tuning,
    imbalance handling, and experiment logging.
    """
    
    def __init__(self, random_state: int = 42, results_dir: str = 'results'):
        """
        Initialize TrainingPipeline.
        
        Args:
            random_state: Random seed for reproducibility
            results_dir: Directory for storing results
        """
        self.random_state = random_state
        self.results_dir = results_dir
        
        # Initialize components
        self.model_manager = ModelManager(random_state)
        self.tuner = HyperparameterTuner(random_state=random_state)
        self.logger = ExperimentLogger(os.path.join(results_dir, 'experiment_logs'))
        
        # Create directories
        os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
        
        # Current experiment settings
        self.current_experiment = {
            'name': None,
            'model_name': None,
            'imbalance_method': 'none',
            'tuning_method': 'grid',
            'custom_params': {}
        }
    
    def setup_experiment(self, experiment_name: str, model_name: str,
                        imbalance_method: str = 'none', 
                        tuning_method: str = 'grid',
                        custom_params: Optional[Dict] = None):
        """
        Setup experiment configuration.
        
        Args:
            experiment_name: Name for the experiment
            model_name: Name of the model to use
            imbalance_method: Method for handling class imbalance
            tuning_method: Hyperparameter tuning method ('grid', 'random', 'none')
            custom_params: Custom parameters for model or tuning
        """
        self.current_experiment = {
            'name': experiment_name,
            'model_name': model_name,
            'imbalance_method': imbalance_method,
            'tuning_method': tuning_method,
            'custom_params': custom_params or {}
        }
        
        logger.info(f"Experiment setup: {experiment_name}")
        logger.info(f"Model: {model_name}, Imbalance: {imbalance_method}, Tuning: {tuning_method}")
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: Optional[np.ndarray] = None, 
                   y_test: Optional[np.ndarray] = None) -> Dict:
        """
        Train model with current experiment configuration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional, for evaluation)
            y_test: Test labels (optional, for evaluation)
            
        Returns:
            Dictionary containing trained model and results
        """
        if not self.current_experiment['name']:
            raise ValueError("Must setup experiment before training")
        
        start_time = time.time()
        
        # Get model
        model = self.model_manager.get_model(self.current_experiment['model_name'])
        
        # Handle class imbalance
        imbalance_handler = ImbalanceHandler(
            method=self.current_experiment['imbalance_method'],
            random_state=self.random_state
        )
        X_train_balanced, y_train_balanced = imbalance_handler.apply(X_train, y_train)
        
        # Apply hyperparameter tuning
        tuning_method = self.current_experiment['tuning_method']
        
        if tuning_method == 'grid':
            trained_model = self.tuner.grid_search(
                model, X_train_balanced, y_train_balanced,
                param_grid=self.current_experiment['custom_params'].get('param_grid')
            )
            if hasattr(trained_model, 'best_estimator_'):
                best_model = trained_model.best_estimator_
                best_params = trained_model.best_params_
                cv_score = trained_model.best_score_
            else:
                best_model = trained_model
                best_params = {}
                cv_score = None
                
        elif tuning_method == 'random':
            n_iter = self.current_experiment['custom_params'].get('n_iter', 20)
            trained_model = self.tuner.random_search(
                model, X_train_balanced, y_train_balanced,
                param_grid=self.current_experiment['custom_params'].get('param_grid'),
                n_iter=n_iter
            )
            if hasattr(trained_model, 'best_estimator_'):
                best_model = trained_model.best_estimator_
                best_params = trained_model.best_params_
                cv_score = trained_model.best_score_
            else:
                best_model = trained_model
                best_params = {}
                cv_score = None
                
        else:  # no tuning
            best_model = model.fit(X_train_balanced, y_train_balanced)
            best_params = {}
            cv_score = None
        
        training_time = time.time() - start_time
        
        # Evaluate on test set if provided
        test_results = {}
        if X_test is not None and y_test is not None:
            y_pred = best_model.predict(X_test)
            test_results = {
                'macro_f1': f1_score(y_test, y_pred, average='macro'),
                'micro_f1': f1_score(y_test, y_pred, average='micro'),
                'weighted_f1': f1_score(y_test, y_pred, average='weighted')
            }
        
        # Prepare results
        results = {
            'model': best_model,
            'best_params': best_params,
            'cv_score': cv_score,
            'test_results': test_results,
            'training_time': training_time,
            'data_shape': {
                'original_train': X_train.shape,
                'balanced_train': X_train_balanced.shape
            }
        }
        
        # Log experiment
        experiment_id = self.logger.generate_experiment_id()
        config = {
            'model_name': self.current_experiment['model_name'],
            'imbalance_method': self.current_experiment['imbalance_method'],
            'tuning_method': self.current_experiment['tuning_method'],
            'best_params': best_params,
            'data_info': results['data_shape']
        }
        
        log_results = {
            'cv_score': cv_score,
            'test_results': test_results,
            'training_time': training_time
        }
        
        self.logger.log_experiment(
            experiment_id=experiment_id,
            experiment_name=self.current_experiment['name'],
            config=config,
            results=log_results,
            training_time=training_time
        )
        
        # Save model
        model_path = os.path.join(self.results_dir, 'models', f"{experiment_id}_model.pkl")
        joblib.dump(best_model, model_path)
        
        results['experiment_id'] = experiment_id
        results['model_path'] = model_path
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        if cv_score:
            logger.info(f"CV Score: {cv_score:.4f}")
        if test_results:
            logger.info(f"Test Macro F1: {test_results['macro_f1']:.4f}")
        
        return results
    
    def quick_train(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: Optional[np.ndarray] = None, 
                   y_test: Optional[np.ndarray] = None,
                   model_name: str = 'logistic_regression',
                   experiment_name: Optional[str] = None) -> Dict:
        """
        Quick training with default settings.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            model_name: Name of model to use
            experiment_name: Name for experiment
            
        Returns:
            Training results
        """
        if experiment_name is None:
            experiment_name = f"Quick_{model_name}_{datetime.now().strftime('%H%M%S')}"
        
        self.setup_experiment(
            experiment_name=experiment_name,
            model_name=model_name,
            imbalance_method='smote',
            tuning_method='grid'
        )
        
        return self.train_model(X_train, y_train, X_test, y_test)
    
    def load_model(self, model_path: str):
        """Load a saved model."""
        return joblib.load(model_path)
    
    def load_experiment_model(self, experiment_id: str):
        """Load model from a specific experiment."""
        model_path = os.path.join(self.results_dir, 'models', f"{experiment_id}_model.pkl")
        return self.load_model(model_path)


# Utility functions
def quick_train_model(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: Optional[np.ndarray] = None, 
                     y_test: Optional[np.ndarray] = None,
                     model_name: str = 'logistic_regression',
                     experiment_name: Optional[str] = None) -> Dict:
    """
    Quick function to train a model with default settings.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features (optional)
        y_test: Test labels (optional)
        model_name: Name of model to use
        experiment_name: Name for experiment
        
    Returns:
        Training results
    """
    pipeline = TrainingPipeline()
    return pipeline.quick_train(X_train, y_train, X_test, y_test, 
                               model_name, experiment_name)


def create_training_pipeline(**kwargs) -> TrainingPipeline:
    """Create a training pipeline with custom settings."""
    return TrainingPipeline(**kwargs)


if __name__ == "__main__":
    # Example usage
    print("Training Pipeline - Example Usage")
    print("=" * 50)
    
    # Create pipeline
    trainer = TrainingPipeline()
    
    # Show available models
    print("Available models:")
    for model in trainer.model_manager.list_models():
        print(f"  - {model}")
    
    print("\nTraining pipeline ready for use!")
    print("Example usage:")
    print("trainer.setup_experiment('My Experiment', 'logistic_regression')")
    print("results = trainer.train_model(X_train, y_train, X_test, y_test)")