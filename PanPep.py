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

# The parameters of input
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

# Load the Atchley_factors for encoding the amino acid
aa_dict = joblib.load("./Requirements/dic_Atchley_factors.pkl")

def aamapping(TCRSeq, encode_dim):
    """
    this function is used for encoding the TCR sequence
    
    Parameters:
        param TCRSeq: the TCR original sequence
        param encode_dim: the first dimension of TCR sequence embedding matrix
        
    Returns:
        this function returns a TCR embedding matrix;
        e.g. the TCR sequence of ASSSAA
        return: (6 + encode_dim - 6) x 5 embedding matrix, in which (encode_dim - 6) x 5 will be zero matrix
        
    Raises:
        KeyError - using 0 vector for replacing the original amino acid encoding
    """
    
    TCRArray = []
    if len(TCRSeq)>encode_dim:
        print('Length: '+str(len(TCRSeq))+' over bound!')
        TCRSeq=TCRSeq[0:encode_dim]
    for aa_single in TCRSeq:
        try:
            TCRArray.append(aa_dict[aa_single])
        except KeyError:
            TCRArray.append(np.zeros(5,dtype='float64'))
    for i in range(0,encode_dim-len(TCRSeq)):
        TCRArray.append(np.zeros(5,dtype='float64'))
    return torch.FloatTensor(TCRArray) 

# Set the random seed
torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
random.seed(222)
torch.cuda.manual_seed(222)

# Sinusoidal position encoding
position_encoding = np.array([[pos / np.power(10000, 2.0 * (j // 2) / 5) for j in range(5)] for pos in range(40)])
position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
position_encoding = torch.from_numpy(position_encoding)

def add_position_encoding(seq):
    """
    this function is used to add position encoding for the TCR embedding
    
    Parameters:
        param seq: the TCR embedding matrix
        
    Returns:
        this function returns a TCR embedding matrix containing position encoding
    """
    
    padding_ids = torch.abs(seq).sum(dim=-1)==0
    seq[~padding_ids] += position_encoding[:seq[~padding_ids].size()[-2]]
    return seq

def task_embedding(pep,tcr_data):
    """
    this function is used to obtain the task-level embedding
    
    Parameters:
        param pep: peptide sequence
        param tcr_data: TCR and its label list in a pan-pep way;
        e.g. [[support TCRs],[support labels]] or [[support TCRs],[support labels],[query TCRs]]
        
    Returns:
        this function returns a peptide embedding, the embedding of support set, the labels of support set and the embedding of query set
    """
    
    # Obtain the TCRs of support set
    spt_TCRs = tcr_data[0]
    
    # Obtain the TCR labels of support set
    ypt = tcr_data[1]
    
    # Initialize the size of the Tensor for the support set and labels
    support_x = torch.FloatTensor(1,len(spt_TCRs),25+15,5)
    support_y = np.zeros((1,len(ypt)), dtype=np.int)
    peptides = torch.FloatTensor(1,75)
    
    # Determine whether there is a query set based on the length of input param2
    if len(tcr_data) > 2:
        qry_TCRs = tcr_data[2]
    else:
        qry_TCRs = ['None']
        
    # Initialize the size of the Tensor for the query set
    query_x = torch.FloatTensor(1,len(qry_TCRs),25+15,5)
    
    # Encoding for the peptide sequence
    peptide_embedding = add_position_encoding(aamapping(pep,15))
    
    # Put the embedding of support set, labels and peptide embedding into the initialized tensor
    temp = torch.Tensor()
    for j in spt_TCRs:
        temp = torch.cat([temp,torch.cat([peptide_embedding,add_position_encoding(aamapping(j,25))]).unsqueeze(0)])
    support_x[0] = temp
    support_y[0] = np.array(ypt)
    peptides[0] = peptide_embedding.flatten()
    
    # Put the embedding of query set into the initialized tensor
    temp = torch.Tensor()
    if len(tcr_data) > 2:
        for j in qry_TCRs:
            temp = torch.cat([temp,torch.cat([peptide_embedding,add_position_encoding(aamapping(j,25))]).unsqueeze(0)])
        query_x[0] = temp
    else:
        query_x[0] = torch.FloatTensor(1,len(qry_TCRs),25+15,5)
    
    return peptides,support_x,torch.LongTensor(support_y),query_x

# This is the model parameters
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

# The path of the trained model
Path = f"./Requirements/model.pt"

# Set the 'cuda' used for GPU testing
device = torch.device('cuda')

# Initialize a new model
model = Memory_Meta(args, config).to(device)

# Load the pretrained model
model.load_state_dict(torch.load(Path))

# Load the memory block
model.Memory_module = Memory_module(args, model.meta_Parameter_nums).cuda()
content = joblib.load(f"./Requirements/Content_memory.pkl")
query = joblib.load(f"./Requirements/Query.pkl")

# Load the content memory matrix and query matrix(read head)
model.Memory_module.memory.content_memory = content
model.Memory_module.memory.Query.weight = query[0]
model.Memory_module.memory.Query.bias = query[1]

# Few-shot setting; (default: fine-tuning 3 times)
if args.learning_setting == 'few-shot':
    
    # Read the data from the csv file
    data = pd.read_csv(args.input)
    peptides = data['Peptide']
    TCRs = data['CDR3']
    labels = data['Label']
    
    # Construct the episode, the input for the panpep in the few-shot setting
    F_data = {}
    for i,j in enumerate(peptides):
        if j not in F_data:
            F_data[j] = [[],[],[],[]]
        if labels[i] != 'Unknown':
            F_data[j][0].append(TCRs[i])
            F_data[j][1].append(labels[i])
        
        # If the label is unknown, we put the TCRs into the query set
        else:
            F_data[j][2].append(TCRs[i]) 

    # The variable "ends" is a list used for storing the predicted score for the 'Unknown' peptide-TCR pairs
    ends = []
    for i in F_data:
        
        # Convert the input into the embeddings
        peptide_embedding, x_spt, y_spt, x_qry = task_embedding(i,F_data[i])
        
        # Support set is used for fine-tune the model and the query set is used to test the performance
        end= model.finetunning(peptide_embedding[0].to(device), x_spt[0].to(device), y_spt[0].to(device), x_qry[0].to(device))
        ends += list(end[0])
    
    # Store the predicted result and output the result as .csv file
    output_peps = []
    output_tcrs = []
    for i in F_data:
        output_peps += [i]*len(F_data[i][2])
        output_tcrs += F_data[i][2]
    output = pd.DataFrame({'Peptide':output_peps,'CDR3':output_tcrs,'Score':ends})
    output.to_csv(args.output,index=False)



# Zero-shot setting; (no fine-tuning)
if args.learning_setting == 'zero-shot':
    
    def task_embedding(pep,tcr_data):
        """
        this function is used to obtain the task-level embedding for the zero-shot setting
        
        Parameters:
            param pep: peptide sequence
            param tcr_data: TCR list
            e.g. [query TCRs]
            
        Returns:
            this function returns a peptide embedding and the embedding of query TCRs
        """
        
        # Obtain the TCRs of support set
        spt_TCRs = tcr_data
        
        # Initialize the size of the Tensor for the query set and peptide encoding
        query_x = torch.FloatTensor(1,len(spt_TCRs),25+15,5)
        peptides = torch.FloatTensor(1,75)
        
        # Encoding for the peptide sequence
        peptide_embedding = add_position_encoding(aamapping(pep,15))
        
        # Put the embedding of query TCRs and peptide into the initialized tensor
        temp = torch.Tensor()
        for j in spt_TCRs:
            temp = torch.cat([temp,torch.cat([peptide_embedding,add_position_encoding(aamapping(j,25))]).unsqueeze(0)])
        query_x[0] = temp
        peptides[0] = peptide_embedding.flatten()
        
        return peptides, query_x

    # Read the data from the .csv file
    data = pd.read_csv(args.input)
    peptides = data['Peptide']
    TCRs = data['CDR3']
    Z_data = {}
    
    # Construct the episode, the input for the panpep in the zero-shot setting
    for i,j in enumerate(peptides):
        if j not in Z_data:
            Z_data[j] = []
        Z_data[j].append(TCRs[i])

    # The variable "starts" is a list used for storing the predicted score for the unseen peptide-TCR pairs
    starts = []
    for i in Z_data:
        
        # Convert the input into the embeddings
        peptide_embedding, x_spt = task_embedding(i,Z_data[i])
        
        # Memory block is used for predicting the binding scores of the unseen peptide-TCR pairs 
        start = model.meta_forward_score(peptide_embedding.to(device), x_spt.to(device))
        starts += list(torch.Tensor.cpu(start[0]).numpy())
    
    # Store the predicted result and output the result as .csv file
    output = pd.DataFrame({'Peptide':peptides,'CDR3':TCRs,'Score':starts})
    output.to_csv(args.output,index=False)

# Majority setting; (default: fine-tuning 1000 times)
if args.learning_setting == 'majority':
    
    # Read the data from the csv file
    data = pd.read_csv(args.input)
    peptides = data['Peptide']
    TCRs = data['CDR3']
    labels = data['Label']
    
    # Construct the episode, the input for the panpep in the majority setting
    G_data = {}
    for i,j in enumerate(peptides):
        if j not in G_data:
            G_data[j] = [[],[],[],[]]
        if labels[i] != 'Unknown':
            G_data[j][0].append(TCRs[i])
            G_data[j][1].append(labels[i])
            
        # If the label is unknown, we put the TCRs into the query set
        else:
            G_data[j][2].append(TCRs[i])
    
    # The variable "ends" is a list used for storing the predicted score for the 'Unknown' peptide-TCR pairs
    ends = []
    for i in G_data:
        
        # Convert the input into the embeddings
        peptide_embedding, x_spt, y_spt, x_qry = task_embedding(i,G_data[i])
        
        # Support set is used for fine-tune the model and the query set is used to test the performance
        end= model.finetunning(peptide_embedding[0].to(device), x_spt[0].to(device), y_spt[0].to(device), x_qry[0].to(device))
        ends += list(end[0])
    
    # Store the predicted result and output the result as .csv file
    output_peps = []
    output_tcrs = []
    for i in G_data:
        output_peps += [i]*len(G_data[i][2])
        output_tcrs += G_data[i][2]
    output = pd.DataFrame({'Peptide':output_peps,'CDR3':output_tcrs,'Score':ends})
    output.to_csv(args.output,index=False)
