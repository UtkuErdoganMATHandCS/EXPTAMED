import math
import torch
from torch.optim.optimizer import Optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class THEOPOULA(Optimizer):

    def __init__(self, params, lr=1e-1, eta=0, beta=1e14, r=3, eps=1e-8, weight_decay=0, sync_eps=False, averaging=False):
        defaults = dict(lr=lr, beta=beta, eta=eta, r=r, eps=eps, weight_decay=weight_decay, sync_eps=sync_eps, averaging=averaging)
        super(THEOPOULA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(THEOPOULA, self).__setstate__(state)


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
                total_norm = torch.pow(pnorm, r)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                eta, beta, lr, eps = group['eta'], group['beta'], group['lr'], group['eps']

                if len(state) == 0:
                    state['step'] = 0
                    state['eta'] = group['lr']
                    state['mu'] = 1
                    state['ax'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                if eta > 0:
                    reg = eta  / (1 / total_norm + math.sqrt(lr))
                    grad.add_(reg, p.data)
                

                
                if group['sync_eps']:
                    eps = lr

                noise = math.sqrt(2 * lr / beta) * torch.randn(size=p.size(), device=device)
                if beta == 1e0:
                  noise = 0
                numer = grad * ( 1 + math.sqrt(lr)/ (eps+ torch.abs(grad)))
                denom = 1 + math.sqrt(lr) * torch.abs(grad)

                p.data.addcdiv_(value=-lr, tensor1=numer, tensor2=denom).add_(noise)

                

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




