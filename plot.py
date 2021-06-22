import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_hist(path):
    dic = {}
    acc = np.zeros(100,)
    loss = np.zeros(100,)
    val_acc = np.zeros(100,)
    val_loss = np.zeros(100,)
    # histの読み込み
    for i in range(20):
        with open(path+'/hist'+str(i)+'.pickle', mode='rb') as f:
            dic[i] = pickle.load(f)
            
    for i in range(20):
        acc += np.array(dic[i]['accuracy'])
        loss += np.array(dic[i]['loss'])
        val_acc += np.array(dic[i]['val_accuracy'])
        val_loss += np.array(dic[i]['val_loss'])
    
    acc = acc / 20
    loss = loss / 20
    val_acc = val_acc / 20
    val_loss = val_loss / 20
    
    return acc, loss, val_acc, val_loss


def load_test(path):
    dic = {}
    for i in range(20):
        with open(path+'/test'+str(i)+'.pickle', mode='rb') as f:
            dic[i] = pickle.load(f)
    
    acc = np.array(dic[0]['acc'])
    loss = np.array(dic[0]['loss'])
    
    for i in range(20):
        if i != 0:
            acc = np.append(acc, np.array(dic[i]['acc']))
            loss = np.append(loss, np.array(dic[i]['loss']))
    
    return acc, loss
               
    
teacher_path = '/home/morikawa/Documents/knowledge_distillation/pytorch/KD_recip/logs/train_teacher'
base_path = '/home/morikawa/Documents/knowledge_distillation/pytorch/KD_recip/logs/train_base'
fitnet_path = '/home/morikawa/Documents/knowledge_distillation/pytorch/KD_recip/logs/train_fitnet'
st_path = '/home/morikawa/Documents/knowledge_distillation/pytorch/KD_recip/logs/train_st'
takd_path = '/home/morikawa/Documents/knowledge_distillation/pytorch/KD_recip/logs/train_takd'
propose_path = '/home/morikawa/Documents/knowledge_distillation/pytorch/KD_recip/logs/train_propose_takd'
ta_path = '/home/morikawa/Documents/knowledge_distillation/pytorch/KD_recip/logs/train_ta'

teacher_acc, teacher_loss, teacher_valacc, teacher_valloss = load_hist(teacher_path)
ta_acc, ta_loss, ta_valacc, ta_valloss = load_hist(ta_path)
base_acc, base_loss, base_valacc, base_valloss = load_hist(base_path)
fitnet_acc, fitnet_loss, fitnet_valacc, fitnet_valloss = load_hist(fitnet_path)
st_acc, st_loss, st_valacc, st_valloss = load_hist(st_path)
takd_acc, takd_loss, takd_valacc, takd_valloss = load_hist(takd_path)
propose_acc, propsoe_loss, propose_valacc, propose_valloss = load_hist(propose_path)

x = np.arange(100)
plt.rcParams["font.size"] = 18

plt.figure(figsize=(10,8))
plt.plot(x, takd_valacc, ':', label='TAKD')
plt.plot(x, fitnet_valacc,  '--', label='Hint Learning and Distillaiton')
plt.plot(x, propose_valacc, '-', label='Propsoed TAKD')
plt.plot(x, st_valacc, '-.', label='Distillaiton')
plt.plot(x, base_valacc,'-',linewidth=4, label='Baseline BP')
plt.ylabel('validation accuracy')
plt.ylim(0.65, 0.90)
plt.xlabel('epoch')
plt.legend()
plt.savefig("acc.eps")

teacher_acc, teacher_loss = load_test(teacher_path)
base_acc, base_loss = load_test(base_path)
fitnet_acc, fitnet_loss = load_test(fitnet_path)
st_acc, st_loss = load_test(st_path)
takd_acc, takd_loss = load_test(takd_path)
propose_acc, propose_loss = load_test(propose_path)
ta_acc, ta_loss = load_test(ta_path)

from tabulate import tabulate

headers = ["teacher", "TA" ,"base", "KD", "HintKD", "TAKD", "Propose"]
table = [[np.mean(teacher_acc),np.mean(ta_acc) ,np.mean(base_acc), np.mean(st_acc), np.mean(fitnet_acc), np.mean(takd_acc), np.mean(propose_acc)],
         [np.max(teacher_acc),np.max(ta_acc) ,np.max(base_acc), np.max(st_acc), np.max(fitnet_acc), np.max(takd_acc), np.max(propose_acc)],
         [np.std(teacher_acc),np.std(ta_acc) ,np.std(base_acc), np.std(st_acc), np.std(fitnet_acc), np.std(takd_acc), np.std(propose_acc)]]
result = tabulate(table, headers, tablefmt="grid")
print(result)