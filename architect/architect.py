import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, rea_model, syn_model, opt):
        self.network_momentum = opt.momentum
        self.network_weight_decay = opt.weight_decay
        self.rea_model = rea_model
        self.syn_model = syn_model
        self.optimizer = torch.optim.Adam(self.rea_model.parameters(), lr=opt.start_lr, betas=opt.betas, eps=opt.eps)

    def _compute_unrolled_model(self, input_syn, target_syn, eta, network_optimizer):
        loss = self.syn_model.loss_syn(input_syn, target_syn)[1]
        theta = _concat(self.syn_model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.syn_model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.syn_model.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(moment + dtheta, alpha=eta))
        return unrolled_model

    def step(self, input_syn, target_syn, input_rea, r, eta, network_optimizer, unrolled):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = eta
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_syn, target_syn, input_rea, r, eta, network_optimizer)
        else:
            self._backward_step(input_rea, r)
        self.optimizer.step()

    def _backward_step(self, input_rea, r):
        loss = self.rea_model.loss_rea(input_rea, r)[1]
        loss.backward()

    def _backward_step_unrolled(self, input_syn, target_syn, input_rea, r, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_syn, target_syn, eta, network_optimizer)
        unrolled_loss = unrolled_model.loss_syn(input_syn, target_syn)[1]
        unrolled_loss.backward()

        rea_loss = self.rea_model.loss_rea(input_rea, r)[1]
        rea_loss.backward()

        dalpha = [v.grad for v in self.rea_model.parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]

        implicit_grads = self._hessian_vector_product(vector, input_syn, target_syn)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(ig.data, alpha=eta)

        for v, g in zip(self.rea_model.parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.syn_model.new()
        model_dict = self.syn_model.state_dict()

        params, offset = {}, 0
        for k, v in self.syn_model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input_syn, target_syn, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.syn_model.parameters(), vector):
            p.data.add_(v, alpha=R)
        loss = self.syn_model.loss_syn(input_syn, target_syn)[1]
        grads_p = torch.autograd.grad(loss, self.syn_model.parameters())

        for p, v in zip(self.syn_model.parameters(), vector):
            p.data.sub_(v, alpha=2 * R)
        loss = self.syn_model.loss_syn(input_syn, target_syn)[1]
        grads_n = torch.autograd.grad(loss, self.syn_model.parameters())

        for p, v in zip(self.syn_model.parameters(), vector):
            p.data.add_(v, alpha=R)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
