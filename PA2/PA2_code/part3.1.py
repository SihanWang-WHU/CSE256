import os
import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformer_encoder_Alibi_Dropout import TransformerEncoder, FeedforwardClassifier, UnifiedClassifier
from utilities import Utilities
import torch.nn as nn
import torch.optim as optim

nltk.download('punkt')
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer_decoder import GPTLanguageModel
import numpy as np

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers

eval_interval = 10  # How often to evaluate train and test perplexity during training
max_iters = 500  # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 20  # Number of iterations to evaluate perplexity on the test set

## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15  # epochs for classifier training


def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data.
    """

    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts


def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])),
                                               "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels


def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def main():
    print("Loading data and creating tokenizer ...")
    texts = load_texts('../speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))  # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)
    vocab_size = tokenizer.vocab_size

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "../speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)
    val_CLS_dataset = SpeechesClassificationDataset(tokenizer, "../speechesdataset/test_CLS.tsv")
    val_CLS_loader = DataLoader(val_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

    ################################# Create the encoder and classifier models #################################
    # for the classification  task, you will train for a fixed number of epochs like this:
    print("Creating encoder and classifier models ...")
    encoder = TransformerEncoder(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
    classifier = FeedforwardClassifier(n_input, n_hidden, n_output).to(device)
    unified_classifier = UnifiedClassifier(encoder, classifier).to(device)
    print(sum(p.numel() for p in unified_classifier.parameters()), 'parameters')
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
    for epoch in range(epochs_CLS):
        total_loss = 0
        for xb, yb in train_CLS_loader:  # Assume train_CLS_loader is defined and contains input IDs and labels
            xb, yb = xb.to(device), yb.to(device)

            # Reset gradient
            optimizer.zero_grad()

            predictions = unified_classifier(xb)  # Pass the embeddings through the classifier

            # Compute loss
            loss = loss_function(predictions, yb)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        # Optionally print average loss per epoch
        print(f'Epoch {epoch + 1}, Average Loss: {total_loss / len(train_CLS_loader)}')

        # Optionally evaluate and print accuracy on a validation set every epoch
        if (epoch + 1) % 5 == 0:
            val_accuracy = compute_classifier_accuracy(unified_classifier,
                                                       val_CLS_loader)  # Assume val_CLS_loader is defined
            print(f'Validation Accuracy after Epoch {epoch + 1}: {val_accuracy}%')


if __name__ == "__main__":
    main()
