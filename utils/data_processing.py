"""
Data Processing Pipeline for Intent Detection
============================================

This module provides comprehensive text preprocessing and vectorization capabilities
for intent detection tasks. It includes automatic text cleaning, stop word removal,
and support for multiple vectorization methods (TF-IDF, Word2Vec, GloVe).

Author: Intent Detection System
Date: 2025
"""

import pandas as pd
import numpy as np
import re
import string
import pickle
import os
from typing import List, Union, Dict, Any, Optional, Tuple
from collections import Counter
import logging

# Core ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Word2Vec
try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logging.warning("Gensim not available. Word2Vec vectorizer will not work.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define custom stop words (excluding important words like "not")
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
    'of', 'with', 'by', 'from', 'up', 'about', 'above', 'after', 'again', 
    'against', 'all', 'am', 'are', 'as', 'be', 'been', 'being', 
    'below', 'between', 'both', 'could', 'did', 'do', 'does', 
    'doing', 'down', 'during', 'each', 'few', 'further', 
    'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 
    'him', 'himself', 'his', 'if', 'into', 'is', 'it', 'its', 
    'itself', 'me', 'more', 'most', 'my', 'myself', 'off', 
    'once', 'only', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 
    'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that', 
    'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 
    'they', 'this', 'those', 'through', 'too', 'under', 'until', 'very', 
    'was', 'we', 'were', 'while', 
    'will', 'would', 'you', 'your', 'yours', 
    'yourself', 'yourselves','s'
}


class TextCleaner:
    """
    Handles all text preprocessing operations including cleaning,
    normalization, and stop word removal.
    """
    
    def __init__(self, remove_stopwords: bool = True, keep_not: bool = True):
        """
        Initialize TextCleaner.
        
        Args:
            remove_stopwords: Whether to remove stop words
            keep_not: Whether to keep "not" even if removing stop words
        """
        self.remove_stopwords = remove_stopwords
        self.keep_not = keep_not
        self.stop_words = STOP_WORDS.copy()
        if keep_not and 'not' in self.stop_words:
            self.stop_words.remove('not')
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace and normalize spacing."""
        return re.sub(r'\s+', ' ', text).strip()
    
    def remove_stop_words(self, text: str) -> str:
        """Remove stop words from text."""
        if not self.remove_stopwords:
            return text
            
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def clean_text(self, text: str) -> str:
        """
        Apply all cleaning operations to text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Apply cleaning steps in order
        text = self.remove_punctuation(text)
        text = self.to_lowercase(text)
        text = self.remove_extra_whitespace(text)
        text = self.remove_stop_words(text)
        
        return text
    
    def clean_texts(self, texts: List[str]) -> List[str]:
        """Clean a list of texts."""
        return [self.clean_text(text) for text in texts]


class Word2VecVectorizer:
    """
    Word2Vec vectorizer that trains on the provided corpus
    and converts texts to averaged word vectors.
    """
    
    def __init__(self, vector_size: int = 100, window: int = 5, 
                 min_count: int = 1, workers: int = 4):
        """
        Initialize Word2Vec vectorizer.
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is required for Word2Vec. Install with: pip install gensim")
        
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """Train Word2Vec model on the provided texts."""
        # Tokenize texts
        sentences = [simple_preprocess(text) for text in texts]
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=10
        )
        self.is_fitted = True
        logger.info(f"Word2Vec model trained with vocabulary size: {len(self.model.wv.key_to_index)}")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert texts to averaged word vectors."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        vectors = []
        for text in texts:
            words = simple_preprocess(text)
            word_vectors = []
            
            for word in words:
                if word in self.model.wv.key_to_index:
                    word_vectors.append(self.model.wv[word])
            
            if word_vectors:
                # Average word vectors
                avg_vector = np.mean(word_vectors, axis=0)
            else:
                # Zero vector if no words found
                avg_vector = np.zeros(self.vector_size)
            
            vectors.append(avg_vector)
        
        return np.array(vectors)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit model and transform texts."""
        self.fit(texts)
        return self.transform(texts)


class GloVeVectorizer:
    """
    GloVe vectorizer that uses pre-trained GloVe embeddings
    to convert texts to averaged word vectors.
    """
    
    def __init__(self, glove_path: Optional[str] = None, vector_size: int = 100):
        """
        Initialize GloVe vectorizer.
        
        Args:
            glove_path: Path to GloVe embeddings file
            vector_size: Size of word vectors
        """
        self.glove_path = glove_path
        self.vector_size = vector_size
        self.word_vectors = {}
        self.is_fitted = False
    
    def load_glove_embeddings(self, glove_path: str):
        """Load pre-trained GloVe embeddings."""
        logger.info(f"Loading GloVe embeddings from {glove_path}")
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                self.word_vectors[word] = vector
        
        self.vector_size = len(next(iter(self.word_vectors.values())))
        logger.info(f"Loaded {len(self.word_vectors)} GloVe vectors of size {self.vector_size}")
    
    def fit(self, texts: List[str]):
        """
        Fit GloVe vectorizer (loads embeddings if not already loaded).
        
        Args:
            texts: List of texts (not used for GloVe, but kept for API consistency)
        """
        if self.glove_path and not self.word_vectors:
            self.load_glove_embeddings(self.glove_path)
        elif not self.word_vectors:
            logger.warning("No GloVe path provided and no vectors loaded. Using zero vectors.")
        
        self.is_fitted = True
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert texts to averaged GloVe vectors."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        vectors = []
        for text in texts:
            words = text.lower().split()
            word_vectors = []
            
            for word in words:
                if word in self.word_vectors:
                    word_vectors.append(self.word_vectors[word])
            
            if word_vectors:
                # Average word vectors
                avg_vector = np.mean(word_vectors, axis=0)
            else:
                # Zero vector if no words found
                avg_vector = np.zeros(self.vector_size)
            
            vectors.append(avg_vector)
        
        return np.array(vectors)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer and transform texts."""
        self.fit(texts)
        return self.transform(texts)


class TFIDFWeightedWord2VecVectorizer:
    """
    TF-IDF weighted Word2Vec vectorizer that weights word vectors
    by their TF-IDF scores for better sentence representation.
    """
    
    def __init__(self, vector_size: int = 100, window: int = 5, 
                 min_count: int = 1, workers: int = 4,
                 tfidf_max_features: int = 5000):
        """
        Initialize TF-IDF weighted Word2Vec vectorizer.
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads
            tfidf_max_features: Maximum features for TF-IDF
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is required for Word2Vec. Install with: pip install gensim")
        
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.tfidf_max_features = tfidf_max_features
        
        self.w2v_model = None
        self.tfidf_vectorizer = None
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """Train both Word2Vec and TF-IDF models."""
        # Train Word2Vec model
        sentences = [simple_preprocess(text) for text in texts]
        self.w2v_model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=10
        )
        
        # Train TF-IDF model
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            lowercase=True,
            stop_words=None  # Already handled in preprocessing
        )
        self.tfidf_vectorizer.fit(texts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        self.is_fitted = True
        logger.info(f"TF-IDF Weighted Word2Vec trained with {len(self.w2v_model.wv.key_to_index)} word vectors")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert texts to TF-IDF weighted word vectors."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        # Get TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        vectors = []
        for i, text in enumerate(texts):
            words = simple_preprocess(text)
            weighted_vectors = []
            total_weight = 0
            
            for word in words:
                if word in self.w2v_model.wv.key_to_index:
                    # Get TF-IDF weight for this word
                    if word in self.feature_names:
                        feature_idx = np.where(self.feature_names == word)[0]
                        if len(feature_idx) > 0:
                            tfidf_weight = tfidf_matrix[i, feature_idx[0]]
                        else:
                            tfidf_weight = 0.1  # Small weight for unknown words
                    else:
                        tfidf_weight = 0.1
                    
                    # Weight the word vector by TF-IDF score
                    word_vector = self.w2v_model.wv[word] * tfidf_weight
                    weighted_vectors.append(word_vector)
                    total_weight += tfidf_weight
            
            if weighted_vectors and total_weight > 0:
                # Normalize by total weight
                sentence_vector = np.sum(weighted_vectors, axis=0) / total_weight
            else:
                # Zero vector if no words found
                sentence_vector = np.zeros(self.vector_size)
            
            vectors.append(sentence_vector)
        
        return np.array(vectors)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit model and transform texts."""
        self.fit(texts)
        return self.transform(texts)


class Doc2VecVectorizer:
    """
    Doc2Vec vectorizer for sentence-level embeddings.
    """
    
    def __init__(self, vector_size: int = 100, window: int = 5,
                 min_count: int = 1, workers: int = 4, epochs: int = 20):
        """
        Initialize Doc2Vec vectorizer.
        
        Args:
            vector_size: Dimensionality of document vectors
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads
            epochs: Number of training epochs
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is required for Doc2Vec. Install with: pip install gensim")
        
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None
        self.TaggedDocument = TaggedDocument
        self.Doc2Vec = Doc2Vec
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """Train Doc2Vec model."""
        # Prepare tagged documents
        tagged_docs = []
        for i, text in enumerate(texts):
            words = simple_preprocess(text)
            tagged_docs.append(self.TaggedDocument(words, [f'doc_{i}']))
        
        # Train Doc2Vec model
        self.model = self.Doc2Vec(
            documents=tagged_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs
        )
        
        self.is_fitted = True
        logger.info(f"Doc2Vec model trained with {len(tagged_docs)} documents")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert texts to document vectors."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        vectors = []
        for text in texts:
            words = simple_preprocess(text)
            # Infer vector for new document
            vector = self.model.infer_vector(words)
            vectors.append(vector)
        
        return np.array(vectors)
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit model and transform texts."""
        self.fit(texts)
        return self.transform(texts)


class SentenceBERTVectorizer:
    """
    Sentence-BERT vectorizer using pre-trained sentence transformers.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize Sentence-BERT vectorizer.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.is_fitted = True  # Pre-trained model
            logger.info(f"Loaded Sentence-BERT model: {model_name}")
        except ImportError:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to load Sentence-BERT model: {e}")
            raise
    
    def fit(self, texts: List[str]):
        """Fit method (no-op for pre-trained model)."""
        pass  # Pre-trained model doesn't need fitting
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert texts to sentence embeddings."""
        if not self.is_fitted:
            raise ValueError("Model not properly initialized")
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error during Sentence-BERT transformation: {e}")
            # Fallback to zero vectors
            return np.zeros((len(texts), 384))  # Default MiniLM dimension
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit model and transform texts."""
        self.fit(texts)
        return self.transform(texts)


class UniversalSentenceEncoderVectorizer:
    """
    Universal Sentence Encoder vectorizer using TensorFlow Hub.
    """
    
    def __init__(self, model_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4"):
        """
        Initialize Universal Sentence Encoder.
        
        Args:
            model_url: URL of the Universal Sentence Encoder model
        """
        try:
            import tensorflow_hub as hub
            import tensorflow as tf
            
            # Suppress TensorFlow warnings
            tf.get_logger().setLevel('ERROR')
            
            self.model = hub.load(model_url)
            self.model_url = model_url
            self.is_fitted = True
            logger.info(f"Loaded Universal Sentence Encoder from: {model_url}")
        except ImportError:
            raise ImportError("tensorflow and tensorflow-hub are required. Install with: pip install tensorflow tensorflow-hub")
        except Exception as e:
            logger.error(f"Failed to load Universal Sentence Encoder: {e}")
            raise
    
    def fit(self, texts: List[str]):
        """Fit method (no-op for pre-trained model)."""
        pass
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Convert texts to sentence embeddings."""
        if not self.is_fitted:
            raise ValueError("Model not properly initialized")
        
        try:
            embeddings = self.model(texts)
            return embeddings.numpy()
        except Exception as e:
            logger.error(f"Error during USE transformation: {e}")
            # Fallback to zero vectors
            return np.zeros((len(texts), 512))  # USE dimension
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit model and transform texts."""
        self.fit(texts)
        return self.transform(texts)


class HybridTFIDFWord2VecVectorizer:
    """
    Hybrid vectorizer that concatenates TF-IDF and Word2Vec features.
    """
    
    def __init__(self, tfidf_max_features: int = 5000, w2v_vector_size: int = 100,
                 w2v_window: int = 5, w2v_min_count: int = 1):
        """
        Initialize hybrid vectorizer.
        
        Args:
            tfidf_max_features: Maximum features for TF-IDF
            w2v_vector_size: Word2Vec vector size
            w2v_window: Word2Vec window size
            w2v_min_count: Word2Vec minimum count
        """
        # TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95
        )
        
        # Word2Vec vectorizer
        self.w2v_vectorizer = Word2VecVectorizer(
            vector_size=w2v_vector_size,
            window=w2v_window,
            min_count=w2v_min_count
        )
        
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """Fit both vectorizers."""
        self.tfidf_vectorizer.fit(texts)
        self.w2v_vectorizer.fit(texts)
        self.is_fitted = True
        logger.info("Hybrid TF-IDF + Word2Vec vectorizer fitted")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using both vectorizers and concatenate."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        # Get TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
        
        # Get Word2Vec features
        w2v_features = self.w2v_vectorizer.transform(texts)
        
        # Concatenate features
        combined_features = np.concatenate([tfidf_features, w2v_features], axis=1)
        
        logger.info(f"Combined features shape: {combined_features.shape}")
        return combined_features
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizers and transform texts."""
        self.fit(texts)
        return self.transform(texts)


class CharacterNgramTFIDFVectorizer:
    """
    Advanced TF-IDF vectorizer with character-level n-grams.
    """
    
    def __init__(self, word_ngram_range: Tuple[int, int] = (1, 3),
                 char_ngram_range: Tuple[int, int] = (2, 4),
                 max_features: int = 10000, min_df: int = 2, max_df: float = 0.95):
        """
        Initialize character n-gram TF-IDF vectorizer.
        
        Args:
            word_ngram_range: Range for word n-grams
            char_ngram_range: Range for character n-grams
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        
        # Word-level TF-IDF
        self.word_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=word_ngram_range,
            max_features=max_features // 2,
            min_df=min_df,
            max_df=max_df
        )
        
        # Character-level TF-IDF
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=char_ngram_range,
            max_features=max_features // 2,
            min_df=min_df,
            max_df=max_df
        )
        
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """Fit both word and character vectorizers."""
        self.word_vectorizer.fit(texts)
        self.char_vectorizer.fit(texts)
        self.is_fitted = True
        logger.info("Character n-gram TF-IDF vectorizer fitted")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts using both word and character n-grams."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        # Get word-level features
        word_features = self.word_vectorizer.transform(texts).toarray()
        
        # Get character-level features
        char_features = self.char_vectorizer.transform(texts).toarray()
        
        # Concatenate features
        combined_features = np.concatenate([word_features, char_features], axis=1)
        
        logger.info(f"Character n-gram features shape: {combined_features.shape}")
        return combined_features
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizers and transform texts."""
        self.fit(texts)
        return self.transform(texts)


class VectorizerFactory:
    """
    Factory class for creating different types of vectorizers
    with standardized parameters.
    """
    
    @staticmethod
    def create_tfidf(**params) -> TfidfVectorizer:
        """
        Create TF-IDF vectorizer with default parameters.
        
        Args:
            **params: Additional parameters for TfidfVectorizer
            
        Returns:
            Configured TfidfVectorizer
        """
        default_params = {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95,
            'stop_words': None  # We handle stop words in preprocessing
        }
        default_params.update(params)
        
        return TfidfVectorizer(**default_params)
    
    @staticmethod
    def create_word2vec(**params) -> Word2VecVectorizer:
        """
        Create Word2Vec vectorizer with default parameters.
        
        Args:
            **params: Additional parameters for Word2VecVectorizer
            
        Returns:
            Configured Word2VecVectorizer
        """
        default_params = {
            'vector_size': 100,
            'window': 5,
            'min_count': 1,
            'workers': 4
        }
        default_params.update(params)
        
        return Word2VecVectorizer(**default_params)
    
    @staticmethod
    def create_glove(glove_path: Optional[str] = None, **params) -> GloVeVectorizer:
        """
        Create GloVe vectorizer with default parameters.
        
        Args:
            glove_path: Path to GloVe embeddings file
            **params: Additional parameters for GloVeVectorizer
            
        Returns:
            Configured GloVeVectorizer
        """
        default_params = {
            'vector_size': 100
        }
        default_params.update(params)
        
        return GloVeVectorizer(glove_path=glove_path, **default_params)
    
    @staticmethod
    def create_tfidf_weighted_word2vec(**params) -> TFIDFWeightedWord2VecVectorizer:
        """
        Create TF-IDF weighted Word2Vec vectorizer.
        
        Args:
            **params: Additional parameters
            
        Returns:
            Configured TFIDFWeightedWord2VecVectorizer
        """
        default_params = {
            'vector_size': 100,
            'window': 5,
            'min_count': 1,
            'workers': 4,
            'tfidf_max_features': 5000
        }
        default_params.update(params)
        
        return TFIDFWeightedWord2VecVectorizer(**default_params)
    
    @staticmethod
    def create_doc2vec(**params) -> Doc2VecVectorizer:
        """
        Create Doc2Vec vectorizer.
        
        Args:
            **params: Additional parameters
            
        Returns:
            Configured Doc2VecVectorizer
        """
        default_params = {
            'vector_size': 100,
            'window': 5,
            'min_count': 1,
            'workers': 4,
            'epochs': 20
        }
        default_params.update(params)
        
        return Doc2VecVectorizer(**default_params)
    
    @staticmethod
    def create_sentence_bert(**params) -> SentenceBERTVectorizer:
        """
        Create Sentence-BERT vectorizer.
        
        Args:
            **params: Additional parameters
            
        Returns:
            Configured SentenceBERTVectorizer
        """
        default_params = {
            'model_name': 'all-MiniLM-L6-v2'
        }
        default_params.update(params)
        
        return SentenceBERTVectorizer(**default_params)
    
    @staticmethod
    def create_universal_sentence_encoder(**params) -> UniversalSentenceEncoderVectorizer:
        """
        Create Universal Sentence Encoder vectorizer.
        
        Args:
            **params: Additional parameters
            
        Returns:
            Configured UniversalSentenceEncoderVectorizer
        """
        default_params = {
            'model_url': "https://tfhub.dev/google/universal-sentence-encoder/4"
        }
        default_params.update(params)
        
        return UniversalSentenceEncoderVectorizer(**default_params)
    
    @staticmethod
    def create_hybrid_tfidf_word2vec(**params) -> HybridTFIDFWord2VecVectorizer:
        """
        Create hybrid TF-IDF + Word2Vec vectorizer.
        
        Args:
            **params: Additional parameters
            
        Returns:
            Configured HybridTFIDFWord2VecVectorizer
        """
        default_params = {
            'tfidf_max_features': 5000,
            'w2v_vector_size': 100,
            'w2v_window': 5,
            'w2v_min_count': 1
        }
        default_params.update(params)
        
        return HybridTFIDFWord2VecVectorizer(**default_params)
    
    @staticmethod
    def create_char_ngram_tfidf(**params) -> CharacterNgramTFIDFVectorizer:
        """
        Create character n-gram TF-IDF vectorizer.
        
        Args:
            **params: Additional parameters
            
        Returns:
            Configured CharacterNgramTFIDFVectorizer
        """
        default_params = {
            'word_ngram_range': (1, 2),
            'char_ngram_range': (2, 4),
            'max_features': 10000,
            'min_df': 2,
            'max_df': 0.95
        }
        default_params.update(params)
        
        return CharacterNgramTFIDFVectorizer(**default_params)
    
    @staticmethod
    def get_vectorizer(vectorizer_type: str = 'tfidf', **params):
        """
        Get vectorizer by type.
        
        Args:
            vectorizer_type: Type of vectorizer
            **params: Parameters for the vectorizer
            
        Returns:
            Configured vectorizer
        """
        vectorizer_type = vectorizer_type.lower()
        
        if vectorizer_type == 'tfidf':
            return VectorizerFactory.create_tfidf(**params)
        elif vectorizer_type == 'word2vec':
            return VectorizerFactory.create_word2vec(**params)
        elif vectorizer_type == 'glove':
            return VectorizerFactory.create_glove(**params)
        elif vectorizer_type == 'tfidf_weighted_word2vec':
            return VectorizerFactory.create_tfidf_weighted_word2vec(**params)
        elif vectorizer_type == 'doc2vec':
            return VectorizerFactory.create_doc2vec(**params)
        elif vectorizer_type == 'sentence_bert':
            return VectorizerFactory.create_sentence_bert(**params)
        elif vectorizer_type == 'universal_sentence_encoder':
            return VectorizerFactory.create_universal_sentence_encoder(**params)
        elif vectorizer_type == 'hybrid_tfidf_word2vec':
            return VectorizerFactory.create_hybrid_tfidf_word2vec(**params)
        elif vectorizer_type == 'char_ngram_tfidf':
            return VectorizerFactory.create_char_ngram_tfidf(**params)
        else:
            raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")
    
    @staticmethod
    def list_available_vectorizers() -> List[str]:
        """Get list of available vectorizer types."""
        return [
            'tfidf',
            'word2vec', 
            'glove',
            'tfidf_weighted_word2vec',
            'doc2vec',
            'sentence_bert',
            'universal_sentence_encoder',
            'hybrid_tfidf_word2vec',
            'char_ngram_tfidf'
        ]
    
    @staticmethod
    def create_word2vec(**params) -> Word2VecVectorizer:
        """
        Create Word2Vec vectorizer with default parameters.
        
        Args:
            **params: Additional parameters for Word2VecVectorizer
            
        Returns:
            Configured Word2VecVectorizer
        """
        default_params = {
            'vector_size': 100,
            'window': 5,
            'min_count': 1,
            'workers': 4
        }
        default_params.update(params)
        
        return Word2VecVectorizer(**default_params)
    
    @staticmethod
    def create_glove(glove_path: Optional[str] = None, **params) -> GloVeVectorizer:
        """
        Create GloVe vectorizer with default parameters.
        
        Args:
            glove_path: Path to GloVe embeddings file
            **params: Additional parameters for GloVeVectorizer
            
        Returns:
            Configured GloVeVectorizer
        """
        default_params = {
            'vector_size': 100
        }
        default_params.update(params)
        
        return GloVeVectorizer(glove_path=glove_path, **default_params)
    
    @staticmethod
    def get_vectorizer(vectorizer_type: str = 'tfidf', **params):
        """
        Get vectorizer by type.
        
        Args:
            vectorizer_type: Type of vectorizer
            **params: Parameters for the vectorizer
            
        Returns:
            Configured vectorizer
        """
        vectorizer_type = vectorizer_type.lower()
        
        if vectorizer_type == 'tfidf':
            return VectorizerFactory.create_tfidf(**params)
        elif vectorizer_type == 'word2vec':
            return VectorizerFactory.create_word2vec(**params)
        elif vectorizer_type == 'glove':
            return VectorizerFactory.create_glove(**params)
        elif vectorizer_type == 'tfidf_weighted_word2vec':
            return VectorizerFactory.create_tfidf_weighted_word2vec(**params)
        elif vectorizer_type == 'doc2vec':
            return VectorizerFactory.create_doc2vec(**params)
        elif vectorizer_type == 'sentence_bert':
            return VectorizerFactory.create_sentence_bert(**params)
        elif vectorizer_type == 'universal_sentence_encoder':
            return VectorizerFactory.create_universal_sentence_encoder(**params)
        elif vectorizer_type == 'hybrid_tfidf_word2vec':
            return VectorizerFactory.create_hybrid_tfidf_word2vec(**params)
        elif vectorizer_type == 'char_ngram_tfidf':
            return VectorizerFactory.create_char_ngram_tfidf(**params)
        else:
            raise ValueError(f"Unknown vectorizer type: {vectorizer_type}")



class DataPipeline:
    """
    Complete data processing pipeline that handles text cleaning,
    vectorization, and train/test splitting for intent detection.
    """
    
    def __init__(self, vectorizer_type: str = 'tfidf', 
                 text_cleaner_params: Optional[Dict] = None,
                 vectorizer_params: Optional[Dict] = None):
        """
        Initialize DataPipeline.
        
        Args:
            vectorizer_type: Type of vectorizer to use
            text_cleaner_params: Parameters for TextCleaner
            vectorizer_params: Parameters for vectorizer
        """
        self.vectorizer_type = vectorizer_type
        
        # Initialize text cleaner
        text_cleaner_params = text_cleaner_params or {}
        self.text_cleaner = TextCleaner(**text_cleaner_params)
        
        # Initialize vectorizer
        vectorizer_params = vectorizer_params or {}
        self.vectorizer = VectorizerFactory.get_vectorizer(
            vectorizer_type, **vectorizer_params
        )
        
        self.is_fitted = False
        
        logger.info(f"DataPipeline initialized with {vectorizer_type} vectorizer")
    
    def load_data(self, file_path: str, text_column: str = 'sentence', 
                  label_column: str = 'label') -> Tuple[List[str], List[str]]:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Tuple of (texts, labels)
        """
        df = pd.read_csv(file_path)
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].astype(str).tolist()
        
        logger.info(f"Loaded {len(texts)} samples from {file_path}")
        logger.info(f"Found {len(set(labels))} unique classes")
        
        return texts, labels
    
    def fit(self, texts: List[str]):
        """
        Fit the pipeline on training texts.
        
        Args:
            texts: List of raw text strings
        """
        # Clean texts
        cleaned_texts = self.text_cleaner.clean_texts(texts)
        
        # Fit vectorizer
        self.vectorizer.fit(cleaned_texts)
        self.is_fitted = True
        
        logger.info("DataPipeline fitted successfully")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        # Clean texts
        cleaned_texts = self.text_cleaner.clean_texts(texts)
        
        # Transform to vectors
        X = self.vectorizer.transform(cleaned_texts)
        
        return X
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit pipeline and transform texts.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            Feature matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    def prepare_data(self, file_path: str, test_size: float = 0.2, 
                     random_state: int = 42, stratify: bool = True,
                     text_column: str = 'sentence', 
                     label_column: str = 'label') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Load data, split, and prepare features.
        
        Args:
            file_path: Path to CSV file
            test_size: Proportion of test set
            random_state: Random seed
            stratify: Whether to stratify split by labels
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            X_train, X_test, y_train, y_test, train_texts, test_texts
        """
        # Load data
        texts, labels = self.load_data(file_path, text_column, label_column)
        
        # Split data
        stratify_param = labels if stratify else None
        train_texts, test_texts, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state,
            stratify=stratify_param
        )
        
        # Fit on training data and transform both sets
        X_train = self.fit_transform(train_texts)
        X_test = self.transform(test_texts)
        
        logger.info(f"Data prepared: Train {X_train.shape}, Test {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, train_texts, test_texts
    
    def save_pipeline(self, file_path: str):
        """Save the fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        pipeline_data = {
            'vectorizer_type': self.vectorizer_type,
            'text_cleaner': self.text_cleaner,
            'vectorizer': self.vectorizer,
            'is_fitted': self.is_fitted
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        logger.info(f"Pipeline saved to {file_path}")
    
    @staticmethod
    def load_pipeline(file_path: str) -> 'DataPipeline':
        """Load a saved pipeline."""
        with open(file_path, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        # Create new pipeline instance
        pipeline = DataPipeline.__new__(DataPipeline)
        pipeline.vectorizer_type = pipeline_data['vectorizer_type']
        pipeline.text_cleaner = pipeline_data['text_cleaner']
        pipeline.vectorizer = pipeline_data['vectorizer']
        pipeline.is_fitted = pipeline_data['is_fitted']
        
        logger.info(f"Pipeline loaded from {file_path}")
        return pipeline


# Utility functions for quick access
def create_data_pipeline(vectorizer_type: str = 'tfidf', **kwargs) -> DataPipeline:
    """
    Quick function to create a data pipeline.
    
    Args:
        vectorizer_type: Type of vectorizer
        **kwargs: Additional parameters
        
    Returns:
        Configured DataPipeline
    """
    return DataPipeline(vectorizer_type=vectorizer_type, **kwargs)


def quick_data_prep(file_path: str, vectorizer_type: str = 'tfidf', 
                    test_size: float = 0.2, **kwargs):
    """
    Quick function to prepare data with default settings.
    
    Args:
        file_path: Path to CSV file
        vectorizer_type: Type of vectorizer
        test_size: Test set proportion
        **kwargs: Additional parameters
        
    Returns:
        X_train, X_test, y_train, y_test, pipeline
    """
    pipeline = create_data_pipeline(vectorizer_type)
    X_train, X_test, y_train, y_test, _, _ = pipeline.prepare_data(
        file_path, test_size=test_size, **kwargs
    )
    
    return X_train, X_test, y_train, y_test, pipeline


def list_available_vectorizers() -> List[str]:
    """Get list of all available vectorizer types."""
    return VectorizerFactory.list_available_vectorizers()


if __name__ == "__main__":
    # Example usage
    print("Data Processing Pipeline - Example Usage")
    print("=" * 50)
    
    # List available vectorizers
    print("Available vectorizers:")
    for vectorizer in list_available_vectorizers():
        print(f"  - {vectorizer}")
    
    # Create pipeline
    pipeline = create_data_pipeline('tfidf')
    
    # Example texts
    sample_texts = [
        "Do you provide EMI options?",
        "What is the cost of mattress?",
        "I want to cancel my order",
        "Tell me about warranty details"
    ]
    
    # Clean and process
    cleaned = pipeline.text_cleaner.clean_texts(sample_texts)
    print("\nOriginal texts:")
    for text in sample_texts:
        print(f"  {text}")
    
    print("\nCleaned texts:")
    for text in cleaned:
        print(f"  {text}")
    
    print("\nPipeline ready for use!")
    print("\nNew vectorizer types available:")
    print("  - tfidf_weighted_word2vec: TF-IDF weighted Word2Vec")
    print("  - doc2vec: Document-level embeddings")
    print("  - sentence_bert: Sentence-BERT embeddings")
    print("  - universal_sentence_encoder: Universal Sentence Encoder")
    print("  - hybrid_tfidf_word2vec: TF-IDF + Word2Vec concatenated")
    print("  - char_ngram_tfidf: Character + Word n-gram TF-IDF")
    
    def load_data(self, file_path: str, text_column: str = 'sentence', 
                  label_column: str = 'label') -> Tuple[List[str], List[str]]:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            Tuple of (texts, labels)
        """
        df = pd.read_csv(file_path)
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].astype(str).tolist()
        
        logger.info(f"Loaded {len(texts)} samples from {file_path}")
        logger.info(f"Found {len(set(labels))} unique classes")
        
        return texts, labels
    
    def fit(self, texts: List[str]):
        """
        Fit the pipeline on training texts.
        
        Args:
            texts: List of raw text strings
        """
        # Clean texts
        cleaned_texts = self.text_cleaner.clean_texts(texts)
        
        # Fit vectorizer
        self.vectorizer.fit(cleaned_texts)
        self.is_fitted = True
        
        logger.info("DataPipeline fitted successfully")
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        # Clean texts
        cleaned_texts = self.text_cleaner.clean_texts(texts)
        
        # Transform to vectors
        X = self.vectorizer.transform(cleaned_texts)
        
        return X
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit pipeline and transform texts.
        
        Args:
            texts: List of raw text strings
            
        Returns:
            Feature matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    def prepare_data(self, file_path: str, test_size: float = 0.2, 
                     random_state: int = 42, stratify: bool = True,
                     text_column: str = 'sentence', 
                     label_column: str = 'label') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Load data, split, and prepare features.
        
        Args:
            file_path: Path to CSV file
            test_size: Proportion of test set
            random_state: Random seed
            stratify: Whether to stratify split by labels
            text_column: Name of text column
            label_column: Name of label column
            
        Returns:
            X_train, X_test, y_train, y_test, train_texts, test_texts
        """
        # Load data
        texts, labels = self.load_data(file_path, text_column, label_column)
        
        # Split data
        stratify_param = labels if stratify else None
        train_texts, test_texts, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state,
            stratify=stratify_param
        )
        
        # Fit on training data and transform both sets
        X_train = self.fit_transform(train_texts)
        X_test = self.transform(test_texts)
        
        logger.info(f"Data prepared: Train {X_train.shape}, Test {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, train_texts, test_texts
    
    def save_pipeline(self, file_path: str):
        """Save the fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        pipeline_data = {
            'vectorizer_type': self.vectorizer_type,
            'text_cleaner': self.text_cleaner,
            'vectorizer': self.vectorizer,
            'is_fitted': self.is_fitted
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        logger.info(f"Pipeline saved to {file_path}")
    
    @staticmethod
    def load_pipeline(file_path: str) -> 'DataPipeline':
        """Load a saved pipeline."""
        with open(file_path, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        # Create new pipeline instance
        pipeline = DataPipeline.__new__(DataPipeline)
        pipeline.vectorizer_type = pipeline_data['vectorizer_type']
        pipeline.text_cleaner = pipeline_data['text_cleaner']
        pipeline.vectorizer = pipeline_data['vectorizer']
        pipeline.is_fitted = pipeline_data['is_fitted']
        
        logger.info(f"Pipeline loaded from {file_path}")
        return pipeline


# Utility functions for quick access
def create_data_pipeline(vectorizer_type: str = 'tfidf', **kwargs) -> DataPipeline:
    """
    Quick function to create a data pipeline.
    
    Args:
        vectorizer_type: Type of vectorizer ('tfidf', 'word2vec', 'glove')
        **kwargs: Additional parameters
        
    Returns:
        Configured DataPipeline
    """
    return DataPipeline(vectorizer_type=vectorizer_type, **kwargs)


def quick_data_prep(file_path: str, vectorizer_type: str = 'tfidf', 
                    test_size: float = 0.2, **kwargs):
    """
    Quick function to prepare data with default settings.
    
    Args:
        file_path: Path to CSV file
        vectorizer_type: Type of vectorizer
        test_size: Test set proportion
        **kwargs: Additional parameters
        
    Returns:
        X_train, X_test, y_train, y_test, pipeline
    """
    pipeline = create_data_pipeline(vectorizer_type)
    X_train, X_test, y_train, y_test, _, _ = pipeline.prepare_data(
        file_path, test_size=test_size, **kwargs
    )
    
    return X_train, X_test, y_train, y_test, pipeline


if __name__ == "__main__":
    # Example usage
    print("Data Processing Pipeline - Example Usage")
    print("=" * 50)
    
    # Create pipeline
    pipeline = create_data_pipeline('tfidf')
    
    # Example texts
    sample_texts = [
        "Do you provide EMI options?",
        "What is the cost of mattress?",
        "I want to cancel my order",
        "Tell me about warranty details"
    ]
    
    # Clean and process
    cleaned = pipeline.text_cleaner.clean_texts(sample_texts)
    print("Original texts:")
    for text in sample_texts:
        print(f"  {text}")
    
    print("\nCleaned texts:")
    for text in cleaned:
        print(f"  {text}")
    
    print("\nPipeline ready for use!")