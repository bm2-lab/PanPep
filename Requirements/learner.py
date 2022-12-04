import  torch
import  math
from    torch import nn
from    torch.nn import functional as F
import  numpy as np



class Learner(nn.Module):
    """
    this is the class that we stored some usual model blocks which can be easily used to build the model
    """

    def __init__(self, config):
        """
        Parameters:
            param config: network config file, type:list of (string, list)
        """
        super(Learner, self).__init__()
        
        self.config = config
        
        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            
            # define a conv2d layer
            if name is 'conv2d':
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            # define a convt2d layer
            elif name is 'convt2d':
                w = nn.Parameter(torch.ones(*param[:4]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            # define a linear layer
            elif name is 'linear':
                w = nn.Parameter(torch.ones(*param))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            # define a batch normalization layer
            elif name is 'bn':
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            
            # define a attention layer
            elif name is 'attention':
                Q = nn.Parameter(torch.ones(param[0]))
                K = nn.Parameter(torch.ones(param[1]))
                V = nn.Parameter(torch.ones(param[2]))
                w = nn.Parameter(torch.ones(param[3]))
                torch.nn.init.kaiming_normal_(Q)
                torch.nn.init.kaiming_normal_(K)
                torch.nn.init.kaiming_normal_(V)
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(Q)
                self.vars.append(nn.Parameter(torch.zeros(param[0][:2])))
                self.vars.append(K)
                self.vars.append(nn.Parameter(torch.zeros(param[1][:2])))
                self.vars.append(V)
                self.vars.append(nn.Parameter(torch.zeros(param[2][:2])))
                self.vars.append(w)

            # define a self attention layer
            elif name is 'self_attention':
                Q = nn.Parameter(torch.ones(param[0]))
                K = nn.Parameter(torch.ones(param[1]))
                V = nn.Parameter(torch.ones(param[2]))
                torch.nn.init.kaiming_normal_(Q)
                torch.nn.init.kaiming_normal_(K)
                torch.nn.init.kaiming_normal_(V)
                self.vars.append(Q)
                self.vars.append(nn.Parameter(torch.zeros(param[0][:2])))
                self.vars.append(K)
                self.vars.append(nn.Parameter(torch.zeros(param[1][:2])))
                self.vars.append(V)
                self.vars.append(nn.Parameter(torch.zeros(param[2][:2])))
                
            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid','gelu']:
                continue
            else:
                raise NotImplementedError



    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'attention':
                tmp = 'attention:(Head:%d,Q_out:%d,Q_in:%d)'%(param[0][0],param[0][1],param[0][2])
                info += tmp + '\n'
                tmp = 'attention:(Head:%d,K_out:%d,K_in:%d)'%(param[1][0],param[0][1],param[0][2])
                info += tmp + '\n'
                tmp = 'attention:(Head:%d,V_out:%d,V_in:%d)'%(param[2][0],param[2][1],param[0][2])
                info += tmp + '\n'
                tmp = 'w:(Head:%d,V_out:%d,V_in:%d)'%(param[3][0],param[3][1],param[3][2])
                info += tmp + '\n'
            
            elif name is 'self_attention':
                tmp = 'self_attention:(Head:%d,Q_out:%d,Q_in:%d)'%(param[0][0],param[0][1],param[0][2])
                info += tmp + '\n'
                tmp = 'self_attention:(Head:%d,K_out:%d,K_in:%d)'%(param[1][0],param[0][1],param[0][2])
                info += tmp + '\n'
                tmp = 'self_attention:(Head:%d,V_out:%d,V_in:%d)'%(param[2][0],param[2][1],param[0][2])
                info += tmp + '\n'
            
            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn','gelu']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        
        Parameters:
            param x: [b, 1, 28, 28]
            param vars:
            param bn_training: set False to not update
            
        Returns:
            x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        
        # get mask index
        mask = torch.abs(x).sum(dim=-1)==0
        for name, param in self.config:
            
            # implementation of conv2d layer
            if name is 'conv2d':
                if len(x.size()) < 4:
                    x = x.unsqueeze(1)
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
                
            # implementation of convt2d layer
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            
            # implementation of linear layer
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            
            # implementation of batch normalization
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2
            
            # implementation of attention layer
            elif name is 'attention':
                Q, Qb = vars[idx],vars[idx+1]
                K, Kb = vars[idx+2], vars[idx+3]
                V, Vb = vars[idx+4], vars[idx+5]
                w = vars[idx+6]
                idx += 7
                q_value = torch.matmul(x.unsqueeze(1),Q.transpose(1,2))+Qb.unsqueeze(1)
                k_value = torch.matmul(x.unsqueeze(1),K.transpose(1,2))+Kb.unsqueeze(1)
                v_value = torch.matmul(x.unsqueeze(1),V.transpose(1,2))+Vb.unsqueeze(1)
                score = torch.matmul(q_value*k_value,w.transpose(-2,-1))
                score[mask.unsqueeze(1).repeat(1,1,score.size()[1]).view(score.size()[0],score.size()[1],score.size()[2])] = -1e9
                att = F.softmax(score,dim=-2)
                x = torch.matmul(att.transpose(-2,-1),v_value).squeeze(1)
                x = x.view(x.size()[0],-1)
            
            # implementation of self attention layer    
            elif name is 'self_attention':
                
                Q, Qb = vars[idx],vars[idx+1]
                K, Kb = vars[idx+2], vars[idx+3]
                V, Vb = vars[idx+4], vars[idx+5]
                idx += 6
                q_value = torch.matmul(x.unsqueeze(1),Q.transpose(1,2))+Qb.unsqueeze(1)
                k_value = torch.matmul(x.unsqueeze(1),K.transpose(1,2))+Kb.unsqueeze(1)
                v_value = torch.matmul(x.unsqueeze(1),V.transpose(1,2))+Vb.unsqueeze(1)
                score = torch.matmul(q_value,k_value.transpose(-2,-1))
                with torch.no_grad():
                    mask2 = mask.repeat(1,x.size()[1]).view(x.size()[0],x.size()[1],-1)
                    mask2 = mask2 + mask2.transpose(-2,-1)
                    score[mask2.unsqueeze(1).repeat(1,score.size()[1],1,1)] = -1e9        
                att = F.softmax(score,dim=-1)
                x = torch.matmul(att, v_value)
                x[mask.unsqueeze(1).repeat(1,score.size()[1],1)] = 0
                x = torch.mean(x,dim=1)
            
            # implementation of flatten layer   
            elif name is 'flatten':
                x = x.view(x.size(0), -1)
            
            # implementation of reshape
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            
            # implementation of relu
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            
            # implementation of gelu
            elif name is 'gelu':
                x = F.gelu(x)
            
            # implementation of leakyrelu
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            
            # implementation of tanh
            elif name is 'tanh':
                x = F.tanh(x)
            
            # implementation of sigmoid
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            
            # implementation of upsample
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            
            # implementation of max_pool2d
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            
            # implementation of avg_pool2d
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return x


    def zero_grad(self, vars=None):
        """
        Parameters:
            param vars:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
            override this function since initial parameters will return with a generator.
        """
        return self.vars