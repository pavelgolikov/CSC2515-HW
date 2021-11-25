import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as func
import q3_0
from q3_3 import summarize_and_save_model_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_data(filename, foldername, device):
    train_data, train_labels, test_data, test_labels = q3_0.data.load_all_data_from_zip(filename, foldername)
    train_data = torch.tensor(train_data).float().to(device)
    # One hot encoding the training labels to be able to compare with predictions
    train_labels = pd.get_dummies(train_labels)
    train_labels = train_labels.values.tolist()
    train_labels = torch.tensor(train_labels).type(torch.float).to(device)
    # convert test_data to tensor
    test_data = torch.tensor(test_data).float().to(device)
    # One hot encode and convert test_labels to tensor
    test_labels = pd.get_dummies(test_labels)
    test_labels = test_labels.values.tolist()
    test_labels = torch.tensor(test_labels).type(torch.float).to(device)

    return train_data, train_labels, test_data, test_labels


def setup_model():
    # model declaration
    model = nn.Sequential(
        nn.Linear(64, 100),
        nn.ReLU(),
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).to(device)
    # loss function declaration
    loss_fn = nn.MSELoss()
    # optimizer declaration
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    return model, loss_fn, optimizer


def train(train_data, train_labels, model, loss_fn, optim, epochs):
    model.train()
    for epoch in range(epochs):
        # zero out gradients on each training iteration
        optim.zero_grad()
        # forward pass through the network
        y_pred = model(train_data)
        # compute loss
        loss = loss_fn(y_pred, train_labels)
        # metric
        mean_avg_error = (torch.sum(torch.abs(y_pred - train_labels)))
        # compute gradients
        loss.backward()
        # print various stats
        print(f"Epoch {epoch}: traing loss: {loss.item()} \t Mean Average Error: {mean_avg_error}")
        # take a step
        optim.step()


def test(test_data, test_labels, model):
    model.eval()
    test_loss = 0
    correct_predictions = 0
    test_preds = []

    with torch.no_grad():
        # for each test example
        for i in range(test_data.size()[0]):
            # get prediction
            y_pred = model(test_data[i])
            # compute test loss on this prediction
            test_loss += func.mse_loss(y_pred, test_labels[i], reduction='mean').item()
            # compare the predicted value and test label
            if torch.argmax(y_pred.data).item() == torch.argmax(test_labels[i]).item():
                correct_predictions += 1
            # store prediction
            test_preds.append(torch.argmax(y_pred.data).item())

    #compute avg test loss and  
    test_loss = test_loss /  len(test_data)
    print('\ntest set loss (avg): {:.5f}, accuracy: {} / {} ({:.0f}%)\n'.format(
        test_loss, correct_predictions, len(test_data),
        100. * correct_predictions / len(test_data)))

    # save model summary for comparison
    test_labels = torch.argmax(test_labels, axis=-1).numpy().astype(np.float)
    summarize_and_save_model_report(np.array(test_preds, dtype=np.float), test_labels, "mlp")


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = preprocess_data('a3digits.zip', 'data', device)
    model, loss_fn, optim = setup_model()
    # I found that training for over 2000 iterations didn't produce higher accuracy, but gradients exploded.
    # I did train on GPU, so it will take longer on CPU
    train(train_data, train_labels, model, loss_fn, optim, 2000)
    test(test_data, test_labels, model)