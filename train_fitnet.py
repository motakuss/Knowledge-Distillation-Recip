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
import pickle

'''
    FitNets: Hint for Thin Deep Nets
'''

def main():
    for i in range(20):
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
        
        teacher = TeacherModel().to(device)
        student = StudentModel().to(device)
        
        teacher.load_state_dict(torch.load('./logs/train_teacher/teacher_param' + str(i) + '.pth'))
        
        regressor = nn.Sequential(
                nn.Conv2d(32, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True) 
        ).to(device)
        #todo: Fitnets Training Stage1: Train guided layers with teach hints
        loss_hint = HintLearningLoss()
        hint_epochs = 100
        loss_score = float('inf')
        student_optim = optimizers.Adam(list(student.parameters()) + list(regressor.parameters()))
        student.train()
        regressor.train()
        for epoch in range(hint_epochs):
            hint_loss = 0.
            val_hint_loss = 0.
            for (x,t) in train_dataloader:
                x, t = x.to(device), t.to(device)
                #todo: ????????????????????????????????????(teacher hint layer outputs)?????????
                teacher.eval()
                with torch.no_grad():
                    teacher_features = teacher.extract_features(x)
                #todo: ??????????????????guide layer?????????????????????
                #todo: ????????????????????????????????????(student guide layer outptus)?????????
                student_features = student.extract_features(x)
                student_guide = regressor(student_features)
                #todo: loss?????????, ???????????????guide layer?????????
                loss = loss_hint(student_guide, teacher_features)
                student_optim.zero_grad()
                loss.backward()
                student_optim.step()
                hint_loss += loss.item()
            hint_loss /= len(train_dataloader)
                
            teacher.eval()
            regressor.eval()
            student.eval()
            with torch.no_grad():
                for (x,t) in val_dataloader:
                    x, t = x.to(device), t.to(device)
                    teacher_features = teacher.extract_features(x)
                    student_features = student.extract_features(x)
                    student_guide = regressor(student_features)
                    loss = loss_hint(student_guide, teacher_features)
                    val_hint_loss += loss.item()
            val_hint_loss /= len(val_dataloader)
            if loss_score >= val_hint_loss:
                print('test')
                loss_score = val_hint_loss
                torch.save(student.state_dict(), './logs/train_fitnet/student_middle_param' + str(i) + '.pth') 
            print(f'epoch: {epoch+1}, hint_loss: {hint_loss:.3f}, val_hint_loss: {val_hint_loss:.3f}')
            # epoch: 100, hint_loss: 0.051, val_hint_loss: 0.051
                
        #todo: second training(soft target)
        student_optim = optimizers.Adam(student.parameters())
        loss_fn = nn.CrossEntropyLoss()
        soft_loss = SoftTargetLoss()
        T = 10 #?????????????????????
        epochs = 100
        score = 0.
        student.load_state_dict(torch.load('./logs/train_fitnet/student_middle_param' + str(i) + '.pth'))
        #es = EarlyStopping(patience=10, verbose=1)
        student_hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        for epoch in range(epochs):
            train_loss = 0.
            train_acc = 0.
            val_loss = 0.
            val_acc = 0.
            student.train()
            teacher.eval()
            for (x, t) in tqdm(train_dataloader, leave=False):
                x, t = x.to(device), t.to(device)
                preds = student(x)
                with torch.no_grad():
                    targets = teacher(x)
                loss = loss_fn(preds, t) +  T * T * soft_loss(preds,targets)
                student_optim.zero_grad()
                loss.backward()
                student_optim.step()
                train_loss += loss.item()
                train_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)
            
            # student validation
            student.eval()
            with torch.no_grad():
                for (x,t) in val_dataloader:
                    x,t = x.to(device),t.to(device)
                    preds = student(x)
                    targets= teacher(x)
                    loss = loss_fn(preds, t) +  T * T * soft_loss(preds, targets)
                    val_loss += loss.item()
                    val_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
                val_loss /= len(val_dataloader)
                val_acc /= len(val_dataloader)

            student_hist['loss'].append(train_loss)
            student_hist['accuracy'].append(train_acc)
            student_hist['val_loss'].append(val_loss)
            student_hist['val_accuracy'].append(val_acc)

            print(f'epoch: {epoch+1}, loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}')
            #if es(val_loss):
            #    break
        
            if score <= val_acc:
                print('test')
                score = val_acc
                torch.save(student.state_dict(), './logs/train_fitnet/student_param' + str(i) + '.pth')
        
        with open('./logs/train_fitnet/hist'+str(i)+'.pickle', mode='wb') as f:
                pickle.dump(student_hist, f)
        
        # distillation student test
        student.load_state_dict(torch.load('./logs/train_fitnet/student_param' + str(i) + '.pth'))
        student.eval()
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for (x,t) in test_dataloader:
                x, t = x.to(device), t.to(device)
                preds = student(x)
                loss = loss_fn(preds, t)
                test_loss += loss.item()
                test_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')
        # test_loss: 0.711, test_accuracy: 0.834
        test = {'acc': [], 'loss': []}
        test['acc'].append(test_acc)
        test['loss'].append(test_loss)
        
        with open('./logs/train_fitnet/test' + str(i) + '.pickle', mode='wb') as f:
                pickle.dump(test, f)
    
if __name__ == '__main__':
    main()
