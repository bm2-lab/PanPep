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


class ReadHead(nn.Module):
    """
    this is the readhead class of PanPep
    
    Parameters:
        param memory: a memory block used for retrieving the memory
    
    Returns:
        the similarity weights based on the memory basis, output by the forward function
    """
    
    def __init__(self,memory):
        super(ReadHead,self).__init__()
        self.memory = memory
        
    def forward(self,peptide):
        q = self.memory.Query(peptide)
        w = self.memory(q)
        return w

class WriteHead(nn.Module):
    """
    this is the writehead class of PanPep
    
    Parameters:
        param memory: a memory block used for retrieving the memory
        param C: the number of basis 
        
    Returns:
        the forward function of this class is used to write the model into the memory block
    """
    
    def __init__(self,C, memory):
        super(WriteHead,self).__init__()
        self.memory = memory
        self.C = C
        # linear layer for transforming the past models into the memory
        self.model_transform = nn.Linear(208,self.C)
        nn.init.xavier_uniform_(self.model_transform.weight, gain=1.4)
        nn.init.normal_(self.model_transform.bias, std=0.01)

    def forward(self, thetas):
        with torch.no_grad():
            models = thetas.T
        w = self.model_transform(models)
        self.memory.writehead(w)


# Memory
class Memory(nn.Module):
    """
    this is the writehead class of PanPep
    
    Parameters:
        param memory: a memory block used for retrieving the memory
        param R: the length of identity matrix
        param L: the length of peptide embedding
        param C: the number of basis
        param V: the length of model parameter vector
        param num_task_batch : the number of tasks in one batch
        
    Returns:
        the task-level similarity based on the basis matrix in the memory block, output by the forward function
    """
    
    def __init__(self,L,C,R,V,num_task_batch=1):
        super(Memory,self).__init__()
        self.C = C
        self.R = R
        self.V = V
        self.num_task_batch = num_task_batch
        
        # the content memory matrix
        self.initial_state = torch.ones(C,V) * 1e-6
        self.register_buffer("content_memory",self.initial_state.data)
        
        # the basis matrix
        self.diognal = torch.eye(C)
        self.register_buffer("peptide_index",self.diognal.data)
        
        # the query matrix
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
    """
    This function is used to rebuild the model parameter shape from the parameter vector
    
    Parameters:
        param x: parameter vector
        param memory_parameters: origin model parameter shape
        
    Returns:
        a new model parameter shape from the parameter vector
    """
    
    new_weights = []
    start_index = 0
    for i in range(len(memory_parameters)):
        end_index = np.prod(memory_parameters[i].shape)
        new_weights.append(x[:,start_index:start_index+end_index].reshape(memory_parameters[i].shape))
        start_index += end_index
    return new_weights


class Memory_module(nn.Module):
    """
    this is the Memory_module class of PanPep
    
    Parameters:
        param memory: the memory block object
        param readhead: the read head object
        param writehead: the write head object
        param prev_loss: store previous loss for disentanglement distillation
        param prev_data: store previous data for disentanglement distillation
        param models: store previous models for disentanglement distillation
        param optim: This is the optimizer for the disentanglement distillation
    """
    
    def __init__(self,args,params_num):
        super(Memory_module,self).__init__()
        self.memory = Memory(args.L,args.C,args.R,params_num,num_task_batch=1)
        self.readhead = ReadHead(self.memory)
        self.writehead = WriteHead(args.C, self.memory)
        self.prev_loss = []
        self.prev_data = []
        self.models = torch.Tensor().cuda()
        self.optim = optim.Adam(self.parameters(), lr=5e-4)
    def forward(self,index):
        r = self.readhead(index)
        return r

    def reset(self):
        # reset the stored elements in the memory
        self.memory.reset()
        self.prev_data = []
        self.prev_loss = []
        self.models = torch.Tensor().cuda()
        
    def reinitialization(self):
        # the memory module parameter reinitialization
        nn.init.xavier_uniform_(self.memory.Query.weight, gain=1.4)
        nn.init.normal_(self.memory.Query.bias, std=0.01)
        nn.init.xavier_uniform_(self.writehead.model_transform.weight, gain=1.4)
        nn.init.normal_(self.writehead.model_transform.bias, std=0.01)
        
class Memory_Meta(nn.Module):
    """
    Meta Learner
    
    Parameters:
        param update_lr: the update learning rate
        param update_step_test: update steps
        param net: the model from the config parameters
        param meta_Parameter_nums: the number of model parameter
        param Memory_module: the Memory_module block
        param prev_loss: store previous loss for disentanglement distillation
        param prev_data: store previous data for disentanglement distillation
        param models: store previous models for disentanglement distillation
    """
    
    def __init__(self, args, config):
        super(Memory_Meta, self).__init__()

        # Set the updating parameter
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
        this is the function for in-place gradient clipping.
        
        Parameters:
            param grad: list of gradients
            param max_norm: maximum norm allowable
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
        """
        this is the function used for fine-tuning on support set and test on the query set
         
        Parameters:
            param peptide: the embedding of peptide
            param x_spt: the embedding of support set
            param y_spt: the labels of support set
            param x_qry: the embedding of query set
        
        Return:
            the binding scores for the TCRs in the query set in the few-shot setting
        """
        
        querysz = x_qry.size(0)
        start = []
        end = []
        
        # in order not to ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # the loss and accuracy before first update
        with torch.no_grad():
            
            # predict logits
            logits_q = net(x_qry, net.parameters(), bn_training=False)
            
            # calculate the scores based on softmax
            pred_q = F.softmax(logits_q, dim=1)
            start.append(pred_q[:,1].cpu().numpy())
        
        # the loss and accuracy after the first update
        if self.update_step_test == 1:
            
            # predict logits
            logits_q = net(x_qry, fast_weights, bn_training=False)
            
            # calculate the scores based on softmax
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        else:
            with torch.no_grad():
                
                # predict logits
                logits_q = net(x_qry, fast_weights, bn_training=False)
                
                # calculate the scores based on softmax
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                
            for k in range(1, self.update_step_test):
                
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = net(x_spt, fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt)
                
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                
                # predict logits
                logits_q = net(x_qry, fast_weights, bn_training=False)

                with torch.no_grad():
                    # calculate the scores based on softmax
                    pred_q = F.softmax(logits_q, dim=1)
                
            end.append(pred_q[:,1].cpu().numpy())
            
        del net

        return end
        
    def meta_forward_score(self,peptide,x_spt):
        """
        This function is used to perform the zero-shot predition in the condition where you have peptide, TCRs
        
        Parameters:
            param peptide: the embedding of peptides
            param x_spt: the embedding of TCRs

        Returns:
            the predicted binding scores of the these TCRs
        """
        with torch.no_grad():
            scores = []
            
            # copy the origin model parameters for the baseline parameter shape
            memory_parameters = deepcopy(self.net.parameters())
            
            # predict the binding score based on the basis models in the memory block
            for i in range(len(peptide)):
                
                # retrieve the memory
                r = self.Memory_module.readhead(peptide[i])[0]
                logits = []
                for m,n in enumerate(self.Memory_module.memory.content_memory):
                    
                    # obtain the basis model
                    weights_memory = _split_parameters(n.unsqueeze(0),memory_parameters)
                    logits.append(self.net(x_spt[i], weights_memory,bn_training=False))
                
                # weighted the predicted result
                pred = sum([r[k]*F.softmax(j) for k,j in enumerate(logits)])
                scores.append(pred[:,1])
                
            return scores    