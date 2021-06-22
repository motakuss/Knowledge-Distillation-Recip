import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split,DataLoader
from src.model import TeacherAssistantModel, StudentModel
import torch.optim as optimizers
from src.kd_loss.fitnet import HintLearningLoss
from src.kd_loss.st import SoftTargetLoss
from src.utils import EarlyStopping
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import random
import pickle

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def main():
    for i in range(20):
        print(i)
        torch.manual_seed(i)
        np.random.seed(i)
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
        
        ta = TeacherAssistantModel().to(device)
        ta_optim = optimizers.Adam(ta.parameters())
        loss_fn = nn.CrossEntropyLoss()
        epochs = 100
        score = 0.
        ta_hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        #es = EarlyStopping(patience=10, verbose=1)
        # ta最適化
        for epoch in range(epochs):
            train_loss = 0.
            train_acc = 0.
            val_loss = 0.
            val_acc = 0.
            ta.train()
            for (x,t) in tqdm(train_dataloader, leave=False):
                x, t= x.to(device),t.to(device)
                preds = ta(x)
                loss = loss_fn(preds, t)
                ta_optim.zero_grad()
                loss.backward()
                ta_optim.step()
                train_loss += loss.item()
                train_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)

            # ta validation
            ta.eval()
            with torch.no_grad():
                for (x,t) in val_dataloader:
                    x, t = x.to(device), t.to(device)
                    preds = ta(x)
                    loss = loss_fn(preds, t)
                    val_loss += loss.item()
                    val_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
                val_loss /= len(val_dataloader)
                val_acc /= len(val_dataloader)
                
            if score <= val_acc:
                print('test')
                score = val_acc
                torch.save(ta.state_dict(), './logs/train_ta/ta_param' + str(i) + '.pth') 

            ta_hist['loss'].append(train_loss)
            ta_hist['accuracy'].append(train_acc)
            ta_hist['val_loss'].append(val_loss)
            ta_hist['val_accuracy'].append(val_acc)

            print(f'epoch: {epoch+1}, loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}')
            #if es(val_loss):
            #    break
        with open('./logs/train_ta/hist'+str(i)+'.pickle', mode='wb') as f:
            pickle.dump(ta_hist, f)

        # ta test
        ta.load_state_dict(torch.load('./logs/train_ta/ta_param' + str(i) + '.pth'))
        ta.eval()
        test_loss = 0.
        test_acc = 0.
        test = {'acc': [], 'loss': []}
        with torch.no_grad():
            for (x,t) in test_dataloader:
                x, t = x.to(device),t.to(device)
                preds = ta(x)
                loss = loss_fn(preds, t)
                test_loss += loss.item()
                test_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')
        # test_loss: 0.662, test_accuracy: 0.819
        test['acc'].append(test_acc)
        test['loss'].append(test_loss)
        with open('./logs/train_ta/test'+str(i)+'.pickle', mode='wb') as f:
            pickle.dump(test, f)
    
if __name__ == '__main__':
    main()