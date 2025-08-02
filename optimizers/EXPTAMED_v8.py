import torch
import torch.nn.parallel
import torch.optim
import numpy as np
import math
from torch.nn.parameter import Parameter





device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EXPTAMED_v8(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-1, eta=0, beta=1e14, r=3, eps=1e-8, weight_decay=0, sync_eps=False, averaging=False):
        defaults = dict(lr=lr, beta=beta, eta=eta, r=r, eps=eps, weight_decay=weight_decay, sync_eps=sync_eps, averaging=averaging)
        super(EXPTAMED_v8, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(EXPTAMED_v8, self).__setstate__(state)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        for group in self.param_groups:
            pnorm = 0
            eta = group['eta']
            r = group['r']
            if eta > 0:
                for p in group['params']:
                    pnorm += torch.sum(torch.pow(p.data, exponent=2))
                total_norm = torch.pow(pnorm, r)# ||theta||^(2r)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad # G
                state = self.state[p]

                eta, beta, lr, eps = group['eta'], group['beta'], group['lr'], group['eps']

                if len(state) == 0:
                    state['step'] = 0
                    state['eta'] = group['lr']
                    state['mu'] = 1
                    state['ax'] = torch.zeros_like(p, memory_format=torch.preserve_format)# is it the error?

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)
                if group['sync_eps']:
                    eps = lr

                abs_grad=torch.abs(grad)
                exp_abs_minus_grad_lr=torch.exp(-lr*abs_grad)
                scaled_grad=torch.mul( grad,( 1 + lr/ (eps+ abs_grad)))
                drift=scaled_grad
                
                if eta > 0:
                    drift=torch.add(scaled_grad,p.data,alpha=eta*total_norm)
                
                abs_drift=torch.abs(drift)
                exp_abs_minus_drift_lr=torch.exp(-lr*abs_drift)

                
               

                noise = math.sqrt(2 * lr / beta) * torch.randn(size=p.size(), device=device)*exp_abs_minus_grad_lr
                
                # sqrt(2 lambda/beta)*xi

                if beta == 1e0:
                  noise = 0

                
                numer = torch.sign(drift)  # boosting here, taken lambda or sqrt lambda,
                #numer=G*(1+sqrt(lambda)/(eps+|G|) )
                factor= (exp_abs_minus_drift_lr-1)  
                p.data.addcmul_(value=1, tensor1=numer, tensor2=factor).add_(noise)
                    

                #######

                
                    



                
            



                # averaging
                if group['averaging']:
                    state['step'] += 1

                    if state['mu'] != 1:
                        state['ax'].add_(p.sub(state['ax']).mul(state['mu']))
                    else:
                        state['ax'].copy_(p)
                    # update eta and mu
                    state['mu'] = 1 / state['step']
        return loss




