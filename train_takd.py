import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import random_split,DataLoader
from src.model import TeacherModel, StudentModel, TeacherAssistantModel
import torch.optim as optimizers
from src.kd_loss.st import SoftTargetLoss
from src.utils import EarlyStopping
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pickle

'''
    Improved Knowledge Distillation via Teacher Assistant
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
        ta = TeacherAssistantModel().to(device)
        student = StudentModel().to(device)
        
        teacher.load_state_dict(torch.load('./logs/train_teacher/teacher_param' + str(i) + '.pth'))
        loss_fn = nn.CrossEntropyLoss()
        # teacher test
        teacher.eval()
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for (x,t) in test_dataloader:
                x, t = x.to(device), t.to(device)
                preds = teacher(x)
                loss = loss_fn(preds, t)
                test_loss += loss.item()
                test_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')
        # test_loss: 0.662, test_accuracy: 0.819
        
        # teacher: teacher  student: teacher assistant train: distillation(softmax with temperature)
        T = 10
        epochs = 100
        score = 0.
        ta_optim = optimizers.Adam(ta.parameters())
        soft_loss = SoftTargetLoss()
        ta_hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        #ta_es = EarlyStopping(patience=10, verbose=1)
        for epoch in range(epochs):
            train_loss = 0.
            train_acc = 0.
            val_loss = 0.
            val_acc = 0.
            ta.train()
            teacher.eval()
            for (x,t) in tqdm(train_dataloader, leave=False):
                x, t = x.to(device), t.to(device)
                preds = ta(x)
                with torch.no_grad():
                    targets = teacher(x)
                loss = loss_fn(preds, t) + T * T * soft_loss(preds, targets)
                ta_optim.zero_grad()
                loss.backward()
                ta_optim.step()
                train_loss += loss.item()
                train_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)
            
            # ta validation
            ta.eval()
            teacher.eval()
            with torch.no_grad():
                for (x,t) in val_dataloader:
                    x,t = x.to(device), t.to(device)
                    preds = ta(x)
                    targets = teacher(x)
                    loss = loss_fn(preds, t) + T * T * soft_loss(preds, targets)
                    val_loss += loss.item()
                    val_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
                val_loss /= len(val_dataloader)
                val_acc /= len(val_dataloader)
                
            ta_hist['loss'].append(train_loss)
            ta_hist['accuracy'].append(train_acc)
            ta_hist['val_loss'].append(val_loss)
            ta_hist['val_accuracy'].append(val_acc)
            
            if score <= val_acc:
                print('test')
                score = val_acc
                torch.save(ta.state_dict(), './logs/train_takd/ta_param' + str(i) + '.pth') 

            
            print(f'epoch: {epoch+1}, loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}')
            #if ta_es(val_loss):
            #    break
            # epoch: 92, loss: 0.181, accuracy: 0.998, val_loss: 1.219, val_accuracy: 0.857                                                                                                          
            # early stopping
        
        # ta test
        ta.eval()
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for (x, t) in test_dataloader:
                x, t = x.to(device), t.to(device)
                preds = ta(x)
                loss = loss_fn(preds, t)
                test_loss += loss.item()
                test_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')
        # test_loss: 0.576, test_accuracy: 0.859
        
        # teacher: ta  student: student  train: distillation
        ta.load_state_dict(torch.load('./logs/train_takd/ta_param' + str(i) + '.pth'))
        student_optim = optimizers.Adam(student.parameters())
        student_hist = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        #student_es = EarlyStopping(patience=10, verbose=1)
        score = 0.
        for epoch in range(epochs):
            train_loss = 0.
            train_acc = 0.
            val_loss = 0.
            val_acc = 0.
            student.train()
            ta.eval()
            for (x,t) in train_dataloader:
                x, t = x.to(device), t.to(device)
                preds = student(x)
                with torch.no_grad():
                    targets = ta(x)
                loss = loss_fn(preds, t) + T * T * soft_loss(preds, targets)
                student_optim.zero_grad()
                loss.backward()
                student_optim.step()
                train_loss += loss.item()
                train_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)
            
            # student validation
            student.eval()
            ta.eval()
            with torch.no_grad():
                for (x, t) in val_dataloader:
                    x,t = x.to(device), t.to(device)
                    preds = student(x)
                    targets = ta(x)
                    loss = loss_fn(preds, t) + T * T * soft_loss(preds, targets)
                    val_loss += loss.item()
                    val_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
                val_loss /= len(val_dataloader)
                val_acc /= len(val_dataloader)
                
            student_hist['loss'].append(train_loss)
            student_hist['accuracy'].append(train_acc)
            student_hist['val_loss'].append(val_loss)
            student_hist['val_accuracy'].append(val_acc)
            
            if score <= val_acc:
                print('test')
                score = val_acc
                torch.save(student.state_dict(), './logs/train_takd/student_param' + str(i) + '.pth') 
            
            print(f'epoch: {epoch+1}, loss: {train_loss:.3f}, accuracy: {train_acc:.3f}, val_loss: {val_loss:.3f}, val_accuracy: {val_acc:.3f}')
            #if student_es(val_loss):
            #    break
            
        with open('./logs/train_takd/hist'+str(i)+'.pickle', mode='wb') as f:
            pickle.dump(student_hist, f)
            
        # student test
        student.load_state_dict(torch.load('./logs/train_takd/student_param' + str(i) + '.pth'))
        test = {'acc': [], 'loss': []}
        student.eval()
        test_loss = 0.
        test_acc = 0.
        with torch.no_grad():
            for (x, t) in test_dataloader:
                x, t = x.to(device), t.to(device)
                preds = student(x)
                loss = loss_fn(preds, t)
                test_loss += loss.item()
                test_acc += accuracy_score(t.tolist(), preds.argmax(dim=-1).tolist())
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f'test_loss: {test_loss:.3f}, test_accuracy: {test_acc:.3f}')
        # test_loss: 0.861, test_accuracy: 0.803
        test['acc'].append(test_acc)
        test['loss'].append(test_loss)
        with open('./logs/train_takd/test'+str(i)+'.pickle', mode='wb') as f:
            pickle.dump(test, f)
    
if __name__ == '__main__':
    main()
