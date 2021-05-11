import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split,DataLoader
from src.model import TeacherModel, StudentModel
import torch.optim as optimizers
from src.kd_loss.fitnet import HintLearningLoss
from src.kd_loss.st import SoftTargetLoss
from src.utils import EarlyStopping
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import random

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def main():
    seed_everything(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_dir = './data/cifar10'
    transform = transforms.Compose([transforms.ToTensor()
                                    ,transforms.Normalize(mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    cifar10_train = datasets.CIFAR10(root=data_dir,
                                     download=True,
                                     train=True,
                                     transform=transform)
    cifar10_test = datasets.CIFAR10(root=data_dir,
                                    download=True,
                                    train=False,
                                    transform=transform)
    
    n_samples = len(cifar10_train)
    n_train = int(n_samples * 0.8)
    n_val = n_samples - n_train
    cifar10_train, cifar10_val = random_split(cifar10_train, [n_train, n_val])
    
    train_dataloader = DataLoader(cifar10_train,
                                  batch_size=128,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=8)
    val_dataloader = DataLoader(cifar10_val,
                                batch_size=128,
                                shuffle=False)
    test_dataloader = DataLoader(cifar10_test,
                                 batch_size=128,
                                 shuffle=False)
    
    teacher = TeacherModel().to(device)
    teacher_optim = optimizers.Adam(teacher.parameters())
    loss_fn = nn.CrossEntropyLoss()
    epochs = 100
    teacher_hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    es = EarlyStopping(patience=10, verbose=1)
    # teacher最適化
    for epoch in range(epochs):
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        teacher.train()
        for (x,t) in tqdm(train_dataloader, leave=False):
            x, t= x.to(device),t.to(device)
            preds = teacher(x)
            loss = loss_fn(preds, t)
            teacher_optim.zero_grad()
            loss.backward()
            teacher_optim.step()
            train_loss += loss.item()
            train_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        # teacher validation
        teacher.eval()
        with torch.no_grad():
            for (x,t) in val_dataloader:
                x, t = x.to(device), t.to(device)
                preds = teacher(x)
                loss = loss_fn(preds, t)
                val_loss += loss.item()
                val_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)

        teacher_hist['loss'].append(train_loss)
        teacher_hist['accuracy'].append(train_acc)
        teacher_hist['val_loss'].append(val_loss)
        teacher_hist['val_accuracy'].append(val_acc)

        print(f'epoch: {epoch+1}, loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}')
        if es(val_loss):
            break

    # teacher test
    teacher.eval()
    test_loss = 0.
    test_acc = 0.
    with torch.no_grad():
        for (x,t) in test_dataloader:
            x, t = x.to(device),t.to(device)
            preds = teacher(x)
            loss = loss_fn(preds, t)
            test_loss += loss.item()
            test_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)
    print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')
    # test_loss: 0.662, test_accuracy: 0.819

    #todo: parameterの保存
    torch.save(teacher.state_dict(), './logs/train_teacher/teacher_param.pth')
    
if __name__ == '__main__':
    main()


