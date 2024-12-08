import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re
from tqdm import tqdm
import torch.nn.functional as F

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class CommitDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts.reset_index(drop=True)  # Reset index to avoid missing keys
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        indices = self.vocab.text_to_indices(text, self.max_length)
        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                     out_channels=n_filters,
                     kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        # text shape: [batch size, sent len]
        embedded = self.embedding(text)
        # embedded shape: [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        # embedded shape: [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n shape: [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n shape: [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat shape: [batch size, n_filters * len(filter_sizes)]
        
        return self.fc(cat)

def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        text = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(predictions, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += len(labels)
        total_loss += loss.item()
    
    return total_loss / len(data_loader), correct_predictions.double() / total_predictions

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in data_loader:
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            predictions = model(text)
            loss = criterion(predictions, labels)
            
            _, preds = torch.max(predictions, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += len(labels)
            total_loss += loss.item()
    
    return total_loss / len(data_loader), correct_predictions.double() / total_predictions

def main():
    # Load dataset
    data = pd.read_csv('data/raw.csv')
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data['type'])
    num_classes = len(label_encoder.classes_)
    
    # Build vocabulary
    vocab = VocabularyBuilder(min_freq=2, max_vocab_size=10000)
    vocab.build(data['commit_message'])
    
    # Split data and convert to pandas Series
    X_train, X_test, y_train, y_test = train_test_split(
        pd.Series(data['commit_message']),  # Convert to Series
        encoded_labels,
        test_size=0.2,
        random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = CommitDataset(X_train, y_train, vocab)
    test_dataset = CommitDataset(X_test, y_test, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Model parameters
    EMBEDDING_DIM = 100
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    OUTPUT_DIM = num_classes
    DROPOUT = 0.5
    PAD_IDX = 0
    
    # Initialize model and training components
    model = TextCNN(
        vocab_size=len(vocab.word2idx),
        embedding_dim=EMBEDDING_DIM,
        n_filters=N_FILTERS,
        filter_sizes=FILTER_SIZES,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
        pad_idx=PAD_IDX
    )
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    epochs = 10
    best_accuracy = 0
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, test_loader, criterion)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            # Save the best model
            model_save = {
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'label_encoder': label_encoder,
                'model_params': {
                    'embedding_dim': EMBEDDING_DIM,
                    'n_filters': N_FILTERS,
                    'filter_sizes': FILTER_SIZES,
                    'output_dim': OUTPUT_DIM,
                    'dropout': DROPOUT,
                    'pad_idx': PAD_IDX
                }
            }
            torch.save(model_save, '../src/git_cc/model/cnn_classifier.pt')

if __name__ == "__main__":
    main()