import os

import torch
from torch.utils.data import DataLoader

from dataset import LanguageModelingDataset
from tokenizer import SimpleTokenizer
from transformer_decoder import GPTLanguageModel as GPT

# hyperparameters
max_iters = 500
eval_interval = 10
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100

batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


# ------------
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


torch.manual_seed(42)

texts = load_texts('../speechesdataset')
tokenizer = SimpleTokenizer(' '.join(texts))  # create a tokenizer from the data
print("Vocabulary size is", tokenizer.vocab_size)
vocab_size = tokenizer.vocab_size
with open('../speechesdataset/train_LM.txt', 'r', encoding='utf-8') as f:
    text = f.read()

inputfile = "../speechesdataset/train_LM.txt"
with open(inputfile, 'r', encoding='utf-8') as f:
    lmtrainText = f.read()
input_val_file = "../speechesdataset/test_LM_obama.txt"
with open(input_val_file, 'r', encoding='utf-8') as f:
    lmvalText = f.read()

train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText, block_size)
train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
val_LM_dataset = LanguageModelingDataset(tokenizer, lmvalText, block_size)
val_LM_loader = DataLoader(val_LM_dataset, batch_size=batch_size, shuffle=True)


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        _, loss = decoderLMmodel(X, Y)  # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity


model = GPT(vocab_size, n_embd, n_head, n_layer, block_size)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_iter = iter(train_LM_loader)

for iterations in range(max_iters):
    # 每一次迭代，从加载器中获取一个批次的数据
    try:
        xb, yb = next(train_iter)
    except StopIteration:
        # 如果数据集结束了，重新开始迭代
        train_iter = iter(train_LM_loader)
        xb, yb = next(train_iter)

    # 将数据移到设备上（例如GPU）
    xb = xb.to(device)
    yb = yb.to(device)

    # 评估损失并更新模型参数
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # 每隔一段时间，评估训练集和验证集的损失
    if iterations % eval_interval == 0 or iterations == max_iters - 1:
        # 计算困惑度
        train_perplexity = compute_perplexity(model, train_LM_loader, eval_interval)
        val_perplexity = compute_perplexity(model, val_LM_loader, eval_interval)
        print(f'Epoch {iterations}, Train Perplexity: {train_perplexity:.4f}, Val Perplexity: {val_perplexity:.4f}')
