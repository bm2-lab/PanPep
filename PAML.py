import joblib
import random
import argparse 
import torch, os
import numpy as np
import scipy.stats
import random, sys
import pandas as pd
import matplotlib.pyplot as plt
from Requirements.Memory_meta import Memory_Meta
from Requirements.Memory_meta import Memory_module
from sklearn.metrics import roc_auc_score

# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

# 模型参数输入
argparser = argparse.ArgumentParser()
argparser.add_argument('--learning_setting', type=str, help='choosing the learning setting: few-shot, zero-shot and majority',required=True)
argparser.add_argument('--input', type=str, help='the path to the input data file (*.csv)',required=True)
argparser.add_argument('--output', type=str, help='the path to the output data file (*.csv)', required=True)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)
argparser.add_argument('--C', type=int, help='Number of bases', default=3)
argparser.add_argument('--R', type=int, help='Peptide Index matrix vector length', default=3)
argparser.add_argument('--L', type=int, help='Peptide embedding length', default=75)
args = argparser.parse_args()

aa_dict = joblib.load("./Requirements/dic_Atchley_factors.pkl")
def aamapping(TCRSeq,encode_dim):
#the longest sting will probably be shorter than 80 nt
    TCRArray = []
    if len(TCRSeq)>encode_dim:
        print('Length: '+str(len(TCRSeq))+' over bound!')
        TCRSeq=TCRSeq[0:encode_dim]
    for aa_single in TCRSeq:
        try:
            TCRArray.append(aa_dict[aa_single])
        except KeyError:
            print('Not proper aaSeqs: '+TCRSeq)
            TCRArray.append(np.zeros(5,dtype='float64'))
    for i in range(0,encode_dim-len(TCRSeq)):
        TCRArray.append(np.zeros(5,dtype='float64'))
    return torch.FloatTensor(TCRArray) 

# 随机种子设置
torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
random.seed(222)
torch.cuda.manual_seed(222)

# position encoding
position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
position_encoding = torch.from_numpy(position_encoding)

def add_position_encoding(seq):
    padding_ids = torch.abs(seq).sum(dim=-1)==0
    seq[~padding_ids] += position_encoding[:seq[~padding_ids].size()[-2]]
    return seq

# 获取每一个task级别的embedding
def task_embedding(pep,tcr_data):
    # 获取support set的embedding
    spt_TCRs = tcr_data[0]
    ypt = tcr_data[1]
    support_x = torch.FloatTensor(1,len(spt_TCRs),25+15,5)
    support_y = np.zeros((1,len(ypt)), dtype=np.int)
    peptides = torch.FloatTensor(1,75)
    
    # 判断是否有query set
    if len(tcr_data) > 2:
        qry_TCRs = tcr_data[2]
    else:
        qry_TCRs = ['None']
    query_x = torch.FloatTensor(1,len(qry_TCRs),25+15,5)
    peptide_embedding = add_position_encoding(aamapping(pep,15))
    
    # 给support set赋值
    temp = torch.Tensor()
    for j in spt_TCRs:
        temp = torch.cat([temp,torch.cat([peptide_embedding,add_position_encoding(aamapping(j,25))]).unsqueeze(0)])
    #赋值
    support_x[0] = temp
    support_y[0] = np.array(ypt)
    peptides[0] = peptide_embedding.flatten()
    
    #如果有query set那么重复上述过程
    temp = torch.Tensor()
    if len(tcr_data) > 2:
        for j in qry_TCRs:
            temp = torch.cat([temp,torch.cat([peptide_embedding,add_position_encoding(aamapping(j,25))]).unsqueeze(0)])
        query_x[0] = temp
    else:
        query_x[0] = torch.FloatTensor(1,len(qry_TCRs),25+15,5)
    return peptides,support_x,torch.LongTensor(support_y),query_x


config = [
    ('self_attention',[[1,5,5],[1,5,5],[1,5,5]]),
    ('linear', [5, 5]),
    ('relu',[True]),
    ('conv2d', [16, 1, 2, 1, 1, 0]),
    ('relu', [True]),
    ('bn', [16]),
    ('max_pool2d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [2, 608])
]

device = torch.device('cuda')
model = Memory_Meta(args, config).to(device)

'''
tmp = filter(lambda x: x.requires_grad, model.parameters())
num = sum(map(lambda x: np.prod(x.shape), tmp))
print(model)
print('Total trainable tensors:', num)
'''

#载入模型
Path = f"./Requirements/model.pt"
device = torch.device('cuda')
model = Memory_Meta(args, config).to(device)
model.load_state_dict(torch.load(Path))
# 载入memory模块
model.Memory_module = Memory_module(args, model.meta_Parameter_nums).cuda()
content = joblib.load(f"./Requirements/Content_memory.pkl")
query = joblib.load(f"./Requirements/Query.pkl")
model.Memory_module.memory.content_memory = content
model.Memory_module.memory.Query.weight = query[0]
model.Memory_module.memory.Query.bias = query[1]

# few-shot
if args.learning_setting == 'few-shot':
    data = pd.read_csv(args.input)
    peptides = data['Peptide']
    TCRs = data['CDR3']
    labels = data['Label']
    F_data = {}
    for i,j in enumerate(peptides):
        if j not in F_data:
            F_data[j] = [[],[],[],[]]
        if labels[i] != 'Unknown':
            F_data[j][0].append(TCRs[i])
            F_data[j][1].append(labels[i])
        else:
            F_data[j][2].append(TCRs[i]) 

    ends = []
    for i in F_data:
        peptide_embedding, x_spt, y_spt, x_qry = task_embedding(i,F_data[i])
        end= model.finetunning(peptide_embedding[0].to(device), x_spt[0].to(device), y_spt[0].to(device), x_qry[0].to(device))
        ends += list(end[0])
    output_peps = []
    output_tcrs = []
    for i in F_data:
        output_peps += [i]*len(F_data[i][2])
        output_tcrs += F_data[i][2]
    output = pd.DataFrame({'Peptide':output_peps,'CDR3':output_tcrs,'Score':ends})
    output.to_csv(args.output,index=False)



# zero-shot
if args.learning_setting == 'zero-shot':
    # 获取每一个task级别的embedding
    def task_embedding(pep,tcr_data):
        # 获取support set的embedding
        spt_TCRs = tcr_data
        support_x = torch.FloatTensor(1,len(spt_TCRs),25+15,5)
        peptides = torch.FloatTensor(1,75)
        peptide_embedding = add_position_encoding(aamapping(pep,15))
        
        # 给support set赋值
        temp = torch.Tensor()
        for j in spt_TCRs:
            temp = torch.cat([temp,torch.cat([peptide_embedding,add_position_encoding(aamapping(j,25))]).unsqueeze(0)])
        #赋值
        support_x[0] = temp
        peptides[0] = peptide_embedding.flatten()
        
        return peptides,support_x

    data = pd.read_csv(args.input)
    peptides = data['Peptide']
    TCRs = data['CDR3']
    Z_data = {}
    for i,j in enumerate(peptides):
        if j not in Z_data:
            Z_data[j] = []
        Z_data[j].append(TCRs[i])

    starts = []
    for i in Z_data:
        peptide_embedding, x_spt = task_embedding(i,Z_data[i])
        start = model.meta_forward_score(peptide_embedding.to(device), x_spt.to(device))
        starts += list(torch.Tensor.cpu(start[0]).numpy())
    output = pd.DataFrame({'Peptide':peptides,'CDR3':TCRs,'Score':starts})
    output.to_csv(args.output,index=False)

# majority
if args.learning_setting == 'majority':
    data = pd.read_csv(args.input)
    peptides = data['Peptide']
    TCRs = data['CDR3']
    labels = data['Label']
    G_data = {}
    for i,j in enumerate(peptides):
        if j not in G_data:
            G_data[j] = [[],[],[],[]]
        if labels[i] != 'Unknown':
            G_data[j][0].append(TCRs[i])
            G_data[j][1].append(labels[i])
        else:
            G_data[j][2].append(TCRs[i])
    
    # model.update_step_test = 1000
    ends = []
    for i in G_data:
        peptide_embedding, x_spt, y_spt, x_qry = task_embedding(i,G_data[i])
        end= model.finetunning(peptide_embedding[0].to(device), x_spt[0].to(device), y_spt[0].to(device), x_qry[0].to(device))
        ends += list(end[0])
    output_peps = []
    output_tcrs = []
    for i in G_data:
        output_peps += [i]*len(G_data[i][2])
        output_tcrs += G_data[i][2]
    output = pd.DataFrame({'Peptide':output_peps,'CDR3':output_tcrs,'Score':ends})
    output.to_csv(args.output,index=False)