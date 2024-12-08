import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class MetadataExtractor(BaseEstimator, TransformerMixin):
    """Extract metadata features from commit messages."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features_list = []
        for message in X:
            features = {}
            # Message length
            features['msg_length'] = len(message)
            # Number of words
            features['word_count'] = len(message.split())
            # Contains issue reference (e.g., #123)
            features['has_issue_ref'] = int(bool(re.search(r'#\d+', message)))
            # Contains version number
            features['has_version'] = int(bool(re.search(r'v\d+\.?\d*\.?\d*', message)))
            # Contains file extensions
            features['has_file_extension'] = int(bool(re.search(r'\.[a-zA-Z0-9]{1,4}\b', message)))
            # Contains URLs
            features['has_url'] = int(bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message)))
            # Contains code snippets
            features['has_code'] = int(bool(re.search(r'`.*?`|\[.*?\]', message)))
            # Contains common commit keywords
            keywords = ['fix', 'bug', 'feature', 'implement', 'update', 'add', 'remove', 'refactor', 'test']
            for keyword in keywords:
                features[f'has_{keyword}'] = int(bool(re.search(rf'\b{keyword}\b', message.lower())))
            features_list.append(list(features.values()))
        return np.array(features_list) 

class CommitTypeEncoder(LabelEncoder):
    """Custom label encoder that preserves commit type labels"""
    def __init__(self):
        super().__init__()
        self.commit_types_ = None
    
    def fit(self, y):
        super().fit(y)
        self.commit_types_ = self.classes_
        return self
    
    def fit_transform(self, y):
        self.fit(y)
        return super().transform(y)
    
    def inverse_transform(self, y):
        return super().inverse_transform(y) 