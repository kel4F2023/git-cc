import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import torch
from collections import Counter

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

class TextCNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(in_channels=1,
                     out_channels=n_filters,
                     kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        
        self.fc = torch.nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [torch.nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

class VocabularyBuilder:
    def __init__(self, min_freq=2, max_vocab_size=10000):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.word_freq = Counter()
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

    def build(self, texts):
        # Count words
        for text in texts:
            words = self._tokenize(text)
            self.word_freq.update(words)
        
        # Filter by frequency and vocab size
        valid_words = [word for word, count in self.word_freq.most_common(self.max_vocab_size) 
                      if count >= self.min_freq]
        
        # Create vocabulary
        for word in valid_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
    
    def _tokenize(self, text):
        # Simple tokenization
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def text_to_indices(self, text, max_length=100):
        words = self._tokenize(text)
        indices = [self.word2idx.get(word, 1) for word in words[:max_length]]  # 1 is <unk>
        # Pad if necessary
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))  # 0 is <pad>
        return indices

class CNNCommitClassifier:
    """PyTorch CNN model wrapper for commit classification"""
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab = None
        self.label_encoder = None
        self.load_model(model_path)
    
    def load_model(self, model_path):
        # First load with weights_only=False but in a controlled way
        from torch.serialization import add_safe_globals
        add_safe_globals([VocabularyBuilder])
        
        try:
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False  # Changed to False since we've added safe globals
            )
            
            # Load vocabulary and label encoder
            self.vocab = checkpoint['vocab']
            self.label_encoder = checkpoint['label_encoder']
            
            # Initialize model with saved parameters
            params = checkpoint['model_params']
            self.model = TextCNN(
                vocab_size=len(self.vocab.word2idx),
                **params
            ).to(self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, text):
        # Prepare input
        indices = self.vocab.text_to_indices(text)
        tensor = torch.tensor(indices).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(tensor)
            _, predicted = torch.max(output, 1)
            return self.label_encoder.inverse_transform(predicted.cpu().numpy())[0] 