import os
import csv
from torch.utils.data import Dataset

class AGNewDataSet(Dataset):
    def __init__(self, root="../data/AG_News", train=True, max_length=1014, transform=None):
        super(AGNewDataSet, self).__init__()
        self.vocab = list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        self.classes = ["World", "Sports", "Business", "ScienceAndTech"]

        texts, labels = [], []
        if train:
            self.file_path = os.path.join(root, "train.csv")
        else:
            self.file_path = os.path.join(root, "test.csv")
        with open(self.file_path, "r") as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            for line in reader:
                text = line[1] + line[2]
                texts.append(text)
                labels.append(int(line[0])-1)

        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        raw = self.texts[idx]
        encode = [self.vocab.index(i)+1 for i in raw if i in self.vocab]
        if len(encode) > self.max_length:
            encode = encode[:self.max_length]
        else:
            encode += [0]*(self.max_length - len(encode))
        if self.transform:
            encode = self.transform(encode)
        label = self.labels[idx]
        return encode, label
if __name__ == "__main__":
    data = AGNewDataSet()
    print(data.__len__())
    e, l = data.__getitem__(5100)
    print(e)
    print(l)
