"""RNN по словам: по предыдущим словам предсказывает следующее.

У символьной модели алфавит маленький, поэтому вход — one-hot.
У слов словарь в сотни раз больше, поэтому слово сначала
переводится в индекс и проходит через Embedding. Дальше та же
рекуррентная ячейка: скрытое состояние переносится от слова к слову.
"""

import re

import torch
import torch.nn as nn


def load_words(path="saltan.txt"):
    with open(path, encoding="utf-8") as file:
        text = file.read().lower().replace("ё", "е")
    return re.findall(r"[а-я]+", text)


class WordRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=64, seq_len=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # вход ячейки: вектор слова + предыдущее скрытое состояние
        self.cell = nn.Linear(embed_size + hidden_size, hidden_size)
        self.activation = nn.Tanh()
        # логиты по словарю; softmax внутри CrossEntropyLoss
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens, hidden=None):
        # tokens: (batch, seq_len)
        batch_size, seq_len = tokens.shape
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=tokens.device)
        for t in range(seq_len):
            word = self.embedding(tokens[:, t])
            hidden = self.activation(self.cell(torch.cat((word, hidden), dim=1)))
        return self.output(hidden)

    def fit(self, word_ids, epochs=15, batch_size=64, lr=0.01):
        sequences = []
        targets = []
        for i in range(len(word_ids) - self.seq_len):
            sequences.append(word_ids[i:i + self.seq_len])
            targets.append(word_ids[i + self.seq_len])

        x = torch.tensor(sequences, dtype=torch.long)
        y = torch.tensor(targets, dtype=torch.long)

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        losses = []

        for epoch in range(epochs):
            order = torch.randperm(x.shape[0])
            epoch_loss = 0.0
            seen = 0
            for start in range(0, x.shape[0], batch_size):
                idx = order[start:start + batch_size]
                logits = self.forward(x[idx])
                loss = criterion(logits, y[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * idx.shape[0]
                seen += idx.shape[0]

            mean_loss = epoch_loss / seen
            losses.append(mean_loss)
            print(f"эпоха {epoch + 1:2d}/{epochs}  loss {mean_loss:.4f}")

        return losses

    def predict(self, word_ids):
        """Индекс следующего слова по последним seq_len индексам."""
        context = word_ids[-self.seq_len:]
        if len(context) < self.seq_len:
            raise ValueError(f"нужно хотя бы {self.seq_len} слов контекста")

        self.eval()
        tokens = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(tokens)
        return int(torch.argmax(logits, dim=1).item())

    def generate(self, word_ids, n=15):
        """Жадно дописывает n слов."""
        context = list(word_ids)
        for _ in range(n):
            context.append(self.predict(context))
        return context


def main():
    torch.manual_seed(0)

    words = load_words()
    vocab = sorted(set(words))
    stoi = {word: index for index, word in enumerate(vocab)}
    itos = vocab

    print(f"слов в тексте: {len(words)}")
    print(f"размер словаря: {len(vocab)}")

    word_ids = [stoi[word] for word in words]
    model = WordRNN(vocab_size=len(vocab))
    model.fit(word_ids)

    context_words = words[:model.seq_len]
    context_ids = [stoi[word] for word in context_words]
    next_id = model.predict(context_ids)
    generated = model.generate(context_ids, n=15)

    print("контекст:      ", " ".join(context_words))
    print("следующее слово:", itos[next_id])
    print("продолжение:   ", " ".join(itos[index] for index in generated))


if __name__ == "__main__":
    main()
