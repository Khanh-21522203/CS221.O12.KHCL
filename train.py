import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import SGD
from torchtext.transforms import ToTensor
from src.dataset import AGNewDataSet
from src.model import VDCNN
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score


def get_args():
    parser = ArgumentParser(description="CNN training")
    parser.add_argument("--root", "-r", type=str, default="./data/AG_News", help="Root of the Dataset")
    parser.add_argument("--epochs", "-e", type=int, default=50)
    parser.add_argument("--batch_size", "-b", type=int, default=128)
    parser.add_argument("--max_length", "-m", type=int, default=1014)
    parser.add_argument("--logging", "-l", type=str, default="tensorboard")
    parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    args = parser.parse_args()
    return args

batch_size = 128
max_length = 1014
num_epochs = 50
root = "./data/AG_News"

if __name__ == "__main__":
    args = get_args()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    transform = ToTensor()
    train_set = AGNewDataSet(root=root, train=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_set = AGNewDataSet(root=root, train=False, transform=transform)
    test_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    model = VDCNN(n_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    num_iters = len(train_loader)

    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, colour='green')
        for iter, (texts, labels) in enumerate(progress_bar):
            texts = texts.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(texts)
            loss_value = criterion(outputs, labels)
            progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.5f}".format(epoch+1, num_epochs, iter+1, num_iters, loss_value))
            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (texts, labels) in enumerate(test_loader):
            all_labels.extend(labels)
            texts = texts.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                predictions = model(texts)
                indices = torch.argmax(predictions.cpu(), dim=1)
                all_predictions.extend(indices)
                loss_value = criterion(predictions, labels)
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch {}: Accuracy: {}".format(epoch+1, accuracy))
