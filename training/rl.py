import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from git_cc.transformers import VocabularyBuilder

class CommitClassifierRL(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        hidden = lstm_out[:, -1, :]  # Use last hidden state
        hidden = torch.relu(self.fc(hidden))
        return self.output(hidden)

class RLCommitClassifier:
    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=128):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab = VocabularyBuilder(min_freq=1, max_vocab_size=vocab_size)
        self.label_encoder = LabelEncoder()
        
        # Initialize with common commit types
        self.label_encoder.fit(['feat', 'fix', 'docs', 'style', 'refactor', 
                              'perf', 'test', 'build', 'ci', 'chore'])
        
        # Initialize model after vocabulary is built
        self.model = None
        self.optimizer = None
        self.epsilon = 0.1  # Exploration rate
        self.memory = []  # Store (state, action, reward) tuples
        self.max_memory = 1000

    def initialize_model(self):
        """Initialize the model after vocabulary is built"""
        self.model = CommitClassifierRL(
            vocab_size=len(self.vocab.word2idx),
            embedding_dim=100,
            hidden_dim=128,
            output_dim=len(self.label_encoder.classes_)
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def text_to_tensor(self, text, max_length=100):
        indices = self.vocab.text_to_indices(text, max_length)
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def predict(self, message):
        self.model.eval()
        with torch.no_grad():
            # Random exploration
            if np.random.random() < self.epsilon:
                prediction_idx = np.random.randint(len(self.label_encoder.classes_))
            else:
                text_tensor = self.text_to_tensor(message)
                outputs = self.model(text_tensor)
                prediction_idx = outputs.argmax(dim=1).item()
            
            return self.label_encoder.inverse_transform([prediction_idx])[0]
    
    def learn_from_feedback(self, message, action, reward):
        """Learn from user feedback"""
        self.memory.append((message, action, reward))
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
            
        if len(self.memory) >= 32:  # Batch size
            self.model.train()
            # Sample batch from memory
            batch = np.random.choice(len(self.memory), 32, replace=False)
            total_loss = 0
            
            for idx in batch:
                msg, act, rew = self.memory[idx]
                text_tensor = self.text_to_tensor(msg)
                
                # Convert action to tensor
                action_tensor = torch.tensor(
                    self.label_encoder.transform([act])[0],
                    dtype=torch.long
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(text_tensor)
                
                # Calculate loss (using cross-entropy with reward weighting)
                loss = -torch.log_softmax(outputs, dim=1)[0][action_tensor] * rew
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Decay epsilon (reduce exploration over time)
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            return total_loss / 32
        return None
    
    def save(self, path):
        """Save the model and its components"""
        model_dir = Path(path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state': self.model.state_dict(),
            'vocab': self.vocab,
            'label_encoder': self.label_encoder,
            'epsilon': self.epsilon,
            'memory': self.memory
        }
        
        torch.save(save_dict, path)
    
    def load(self, path):
        """Load the model and its components"""
        save_dict = torch.load(path, map_location=self.device)
        
        self.vocab = save_dict['vocab']
        self.label_encoder = save_dict['label_encoder']
        self.epsilon = save_dict['epsilon']
        self.memory = save_dict['memory']
        
        # Recreate model with correct dimensions
        self.model = CommitClassifierRL(
            vocab_size=len(self.vocab.word2idx),
            embedding_dim=100,
            hidden_dim=128,
            output_dim=len(self.label_encoder.classes_)
        ).to(self.device)
        
        self.model.load_state_dict(save_dict['model_state'])

def train_base_model():
    print("Loading dataset...")
    data = pd.read_csv('data/raw.csv')
    
    # Create the RL classifier
    classifier = RLCommitClassifier()
    
    # Build vocabulary from dataset
    print("Building vocabulary...")
    classifier.vocab.build(data['commit_message'])
    
    # Initialize model after vocabulary is built
    classifier.initialize_model()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data['commit_message'],
        data['type'],
        test_size=0.2,
        random_state=42
    )
    
    # Training parameters
    batch_size = 32
    epochs = 5
    
    # Convert to list for easier batch processing
    train_data = list(zip(X_train, y_train))
    test_data = list(zip(X_test, y_test))
    
    print("Training base model...")
    best_accuracy = 0
    
    for epoch in range(epochs):
        classifier.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Process in batches
        for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
            batch = train_data[i:i + batch_size]
            batch_messages, batch_types = zip(*batch)
            
            # Process each item in batch
            for message, true_type in zip(batch_messages, batch_types):
                try:
                    # Get model prediction
                    text_tensor = classifier.text_to_tensor(message)
                    outputs = classifier.model(text_tensor)
                    
                    # Convert true type to tensor
                    true_type_idx = classifier.label_encoder.transform([true_type])[0]
                    true_type_tensor = torch.tensor([true_type_idx], dtype=torch.long).to(classifier.device)
                    
                    # Calculate loss
                    loss = torch.nn.functional.cross_entropy(outputs, true_type_tensor)
                    
                    # Backward pass
                    classifier.optimizer.zero_grad()
                    loss.backward()
                    classifier.optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    pred_idx = outputs.argmax(dim=1).item()
                    correct += (pred_idx == true_type_idx)
                    total += 1
                except Exception as e:
                    print(f"Error processing message: {message}")
                    print(f"Error: {str(e)}")
                    continue
        
        # Evaluate on test set
        classifier.model.eval()
        test_correct = 0
        test_total = 0
        
        print("\nEvaluating...")
        with torch.no_grad():
            for message, true_type in tqdm(test_data, desc="Testing"):
                text_tensor = classifier.text_to_tensor(message)
                outputs = classifier.model(text_tensor)
                pred_idx = outputs.argmax(dim=1).item()
                true_idx = classifier.label_encoder.transform([true_type])[0]
                test_correct += (pred_idx == true_idx)
                test_total += 1
        
        # Calculate metrics
        train_accuracy = correct / total
        test_accuracy = test_correct / test_total
        avg_loss = total_loss / total
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            print(f"New best accuracy: {best_accuracy:.4f}! Saving model...")
            
            # Create model directory if it doesn't exist
            model_dir = Path('../src/git_cc/model')
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the model
            classifier.save(model_dir / 'rl.pt')
    
    print("\nTraining completed!")
    print(f"Best test accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    train_base_model()