import torch
from torch import nn
import torch.optim as optim
from sklearn.utils import shuffle


def build_model(n_feats, layers_size):
    layers = [
        nn.Linear(n_feats, layers_size[0]),
        nn.Sigmoid()
    ]
    output_size = layers_size[0]

    if len(layers_size) == 2:
        layers += [
            nn.Linear(layers_size[0], layers_size[1]),
            nn.Sigmoid()
        ]
        output_size = layers_size[1]

    layers.append(nn.Linear(output_size, 1))
    model = nn.Sequential(*layers)
    return model


def train_model(X, y, model, epoches=10, batch_size=100):
    model.train()
    loss = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(params=model.parameters())

    for _ in range(epoches):
        X, y = shuffle(X, y)
        for j in range(batch_size, X.shape[0], batch_size):
            X_batch = X[j-batch_size:j, :]
            y_batch = y[j-batch_size:j].reshape(batch_size,1)
            X_batch = torch.tensor(X_batch, dtype=torch.float)
            y_batch = torch.tensor(y_batch, dtype=torch.float)
            y_pred = model(X_batch)

            output = loss(y_batch, y_pred)
            optimizer.zero_grad()
            output.backward()
            optimizer.step()


def predict(X, model):
    model.train(False)
    X_tensor = torch.tensor(X, dtype=torch.float)
    y_pred = model(X_tensor).data.cpu().numpy()
    return y_pred
