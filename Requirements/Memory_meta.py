import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    Requirements.learner import Learner
from    copy import deepcopy
import  argparse
from tensorboardX import SummaryWriter
from datetime import datetime
from sklearn.metrics import roc_auc_score

# Read head
class ReadHead(nn.Module):
    def __init__(self,memory):
        super(ReadHead,self).__init__()
        self.memory = memory
        
    def forward(self,peptide):
        q = self.memory.Query(peptide)
        w = self.memory(q)
        # r = self.memory.readhead(w)
        return w

# Write head

class WriteHead(nn.Module):
    def __init__(self,C,L,R,V,memory):
        super(WriteHead,self).__init__()
        self.memory = memory
        self.R = R
        self.L = L
        self.C = C
        self.model_transform = nn.Linear(173,self.C)
        nn.init.xavier_uniform_(self.model_transform.weight, gain=1.4)
        nn.init.normal_(self.model_transform.bias, std=0.01)

    def forward(self, thetas):
        with torch.no_grad():
            models = thetas.T
        w = self.model_transform(models)
        self.memory.writehead(w)
    def size(self):
        return self.V


# Memory
class Memory(nn.Module):
    def __init__(self,L,C,R,V,num_task_batch=1):
        super(Memory,self).__init__()
        self.C = C
        self.R = R
        self.V = V
        self.num_task_batch = num_task_batch
        
        self.initial_state = torch.ones(C,V) * 1e-6
        self.register_buffer("content_memory",self.initial_state.data)
        
        self.diognal = torch.eye(C)
        self.register_buffer("peptide_index",self.diognal.data)
        # self.peptide_index = nn.Parameter(torch.FloatTensor(C,R))
        # nn.init.kaiming_normal_(self.peptide_index)
        self.Query = nn.Linear(L,R)
        nn.init.xavier_uniform_(self.Query.weight, gain=1.4)
        nn.init.normal_(self.Query.bias, std=0.01)
    def forward(self,query):
        query = query.view(self.num_task_batch,1,-1)
        w = F.softmax(F.cosine_similarity(self.peptide_index+1e-16,query+1e-16,dim=-1),dim=1)
        return w
    
    def reset(self):
        self.content_memory.data = self.initial_state.data.cuda()
    
    def size(self):
        return self.C, self.R, self.V
    
    def readhead(self, w):
        return torch.matmul(w.unsqueeze(1),self.content_memory).squeeze(1)
    
    def writehead(self,w):
        self.content_memory = w.T
        

# memory based net parameter reconstruction
def _split_parameters(x,memory_parameters):
    new_weights = []
    start_index = 0
    for i in range(len(memory_parameters)):
        end_index = np.prod(memory_parameters[i].shape)
        new_weights.append(x[:,start_index:start_index+end_index].reshape(memory_parameters[i].shape))
        start_index += end_index
    return new_weights


class Memory_module(nn.Module):
    def __init__(self,args,params_num):
        super(Memory_module,self).__init__()
        self.memory = Memory(args.L,args.C,args.R,params_num,num_task_batch=1)
        self.readhead = ReadHead(self.memory)
        self.writehead = WriteHead(args.C,args.L,args.R,params_num,self.memory)
        self.prev_loss = []
        self.prev_data = []
        self.models = torch.Tensor().cuda()
        self.optim = optim.Adam(self.parameters(), lr=5e-4)
    def forward(self,index):
        r = self.readhead(index)
        return r

    def reset(self):
        self.memory.reset()
        self.prev_data = []
        self.prev_loss = []
        self.models = torch.Tensor().cuda()
        
    def reinitialization(self):
        nn.init.xavier_uniform_(self.memory.Query.weight, gain=1.4)
        nn.init.normal_(self.memory.Query.bias, std=0.01)
        nn.init.xavier_uniform_(self.writehead.model_transform.weight, gain=1.4)
        nn.init.normal_(self.writehead.model_transform.bias, std=0.01)
        
class Memory_Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """
        :param args:
        """
        super(Memory_Meta, self).__init__()

        self.update_lr = args.update_lr
        self.update_step_test = args.update_step_test
        self.net = Learner(config)
        # Count the number of parameters
        tmp = filter(lambda x: x.requires_grad, self.net.parameters())
        self.meta_Parameter_nums = sum(map(lambda x: np.prod(x.shape), tmp))
        self.Memory_module = None
        # self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.prev_loss = []
        self.prev_data = []
        self.models = torch.Tensor().cuda()
        
    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter
    
    def reset(self):
        self.prev_data = []
        self.prev_loss = []
        self.models = torch.Tensor().cuda()


    def finetunning(self, peptide, x_spt, y_spt, x_qry):
                
        querysz = x_qry.size(0)

        start = []
        end = []
        
        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=False)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1)
            start.append(pred_q[:,1].cpu().numpy())
            # scalar
        
        # this is the loss and accuracy after the first update
        if self.update_step_test == 1:
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=False)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        else:
            with torch.no_grad():
                # [setsz, nway]
                logits_q = net(x_qry, fast_weights, bn_training=False)
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            for k in range(1, self.update_step_test):
                
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = net(x_spt, fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                
                logits_q = net(x_qry, fast_weights, bn_training=False)
                # loss_q will be overwritten and just keep the loss_q on last update step.

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1)
            end.append(pred_q[:,1].cpu().numpy())
            
        del net

        return end

    def meta_forward(self,peptide,x_spt,y_spt):  
        
        # auc version
        with torch.no_grad():
            scores_all = np.array([])
            labels  = np.array([])
            memory_parameters = deepcopy(self.net.parameters())
            for i in range(len(peptide)):
                r = self.Memory_module.readhead(peptide[i])[0]
                logits = []
                for m,n in enumerate(self.Memory_module.memory.content_memory):
                    weights_memory = _split_parameters(n.unsqueeze(0),memory_parameters)
                    logits.append(self.net(x_spt[i], weights_memory,bn_training=False))
                # pred = F.softmax(logits, dim=1).argmax(dim=1)
                pred = sum([r[k]*F.softmax(j) for k,j in enumerate(logits)])
                scores = pred[:,1].cpu().numpy()
                scores_all = np.r_[scores_all,scores]
                labels = np.r_[labels,y_spt[i].cpu().numpy()]
            return [scores_all,labels]
        
    def meta_forward_score(self,peptide,x_spt):
        with torch.no_grad():
            scores = []
            memory_parameters = deepcopy(self.net.parameters())
            for i in range(len(peptide)):
                r = self.Memory_module.readhead(peptide[i])[0]
                logits = []
                for m,n in enumerate(self.Memory_module.memory.content_memory):
                    weights_memory = _split_parameters(n.unsqueeze(0),memory_parameters)
                    logits.append(self.net(x_spt[i], weights_memory,bn_training=False))
                # pred = F.softmax(logits, dim=1).argmax(dim=1)
                pred = sum([r[k]*F.softmax(j) for k,j in enumerate(logits)])
                # print(sum([r[k]*F.softmax(j) for k,j in enumerate(logits)]))
                scores.append(pred[:,1])
            return scores    