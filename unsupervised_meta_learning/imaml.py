# AUTOGENERATED! DO NOT EDIT! File to edit: 02b_iMAML.ipynb (unless otherwise specified).

__all__ = ['cg_solve', 'iMAML']

# Cell
#export
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import higher
import wandb
import numpy as np

import pytorch_lightning as pl
from itertools import repeat

from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from .pl_dataloaders import OmniglotDataModule
from .nn_utils import get_accuracy
from .maml import ConvolutionalNeuralNetwork

import unsupervised_meta_learning.hypergrad as hg
import gc

# Cell
def cg_solve(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10, x_init=None):
    """
    Goal: Solve Ax=b equivalent to minimizing f(x) = 1/2 x^T A x - x^T b
    Assumption: A is PSD, no damping term is used here (must be damped externally in f_Ax)
    Algorithm template from wikipedia
    Verbose mode works only with numpy
    """

    if type(b) == torch.Tensor:
        x = torch.zeros(b.shape[0]) if x_init is None else x_init
        x = x.to(b.device)
        if b.dtype == torch.float16:
            x = x.half()
        r = b - f_Ax(x)
        p = r.clone()
    elif type(b) == np.ndarray:
        x = np.zeros_like(b) if x_init is None else x_init
        r = b - f_Ax(x)
        p = r.copy()
    else:
        print("Type error in cg")

    fmtstr = "%10i %10.3g %10.3g %10.3g"
    titlestr = "%10s %10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
            norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
            print(fmtstr % (i, r.dot(r), norm_x, obj_fn))

        rdotr = r.dot(r)
        Ap = f_Ax(p)
        alpha = rdotr/(p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        beta = newrdotr/rdotr
        p = r + beta * p

        if newrdotr < residual_tol:
            # print("Early CG termination because the residual was small")
            break

    if callback is not None:
        callback(x)
    if verbose:
        obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
        norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
        print(fmtstr % (i, r.dot(r), norm_x, obj_fn))
    return x

# Cell
class iMAML(pl.LightningModule):
    def __init__(self, model, loss_function, inner_lr, outer_lr, lam_lr, inner_steps, cg_steps, cg_damping, lam=0., lam_min=0.):
        super().__init__()
        self.automatic_optimization = False
        self.accuracy = get_accuracy
        self.model = model
        self.loss_function = loss_function
        self.meta_lr = outer_lr
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.lam_lr = lam_lr
        self.inner_steps = inner_steps
        self.cg_steps = cg_steps
        self.n_params = len(list(model.parameters()))
        self.lam = lam
        self.lam_min = lam_min
        self.cg_damping = cg_damping

    def configure_optimizers(self):
        outer_opt = torch.optim.Adam(params=self.model.parameters(), lr=self.outer_lr)
        inner_opt = torch.optim.SGD(params=self.model.parameters(), lr=self.inner_lr)
        return outer_opt, inner_opt

    def regularization_loss(self, w_0, lam=0.):
        """
        Add a regularization loss onto the weights
        The proximal term regularizes around the point w_0
        Strength of regularization is lambda
        lambda can either be scalar (type float) or ndarray (numpy.ndarray)
        """
        regu_loss = 0.
        offset = 0

        regu_lam = lam if type(lam) == float or np.float64 else utils.to_tensor(lam)
        if w_0.dtype == torch.float16:
            try:
                regu_lam.half()
            except:
                regu_lam = np.float16(regu_lam)
        for param in self.model.parameters():
            delta = param.view(-1) - w_0[offset:offset + param.nelement()].view(-1)
            if type(regu_lam) == float or np.float64:
                regu_loss += 0.5 * regu_lam * torch.sum(delta ** 2)
            else:
                # import ipdb; ipdb.set_trace()
                param_lam = regu_lam[offset:offset + param.nelement()].view(-1)
                param_delta = delta * param_lam
                regu_loss += 0.5 * torch.sum(param_delta ** 2)
            offset += param.nelement()
        return regu_loss

    def get_loss(self, fmodel, x, y, return_np=False):
        y_hat = fmodel.forward(x)
        loss = self.loss_function(y_hat, y)

        if return_np:
            loss = loss.cpu().detach().numpy()
        return loss

    def get_params(self):
        return torch.cat([param.data.view(-1) for param in self.model.parameters()], 0).clone()


    def inner_loop(self, fmodel, diffopt, train_input, train_target, add_reg_loss=False, w_0=None, lam=0.):
        train_loss = []
        for i in range(self.inner_steps):
            train_logit = fmodel(train_input)
            tmpl = F.cross_entropy(train_logit, train_target)
            inner_loss = tmpl +   self.regularization_loss(w_0, lam) if add_reg_loss else tmpl
            diffopt.step(inner_loss)
            train_loss.append(inner_loss.detach())

        return train_loss

    def matrix_evaluator(self, task, lam, fmodel=None, regu_coef=1.0, lam_damping=10.0, x=None, y=None):
        """
        Constructor function that can be given to CG optimizer
        Works for both type(lam) == float and type(lam) == np.ndarray
        """
        if type(lam) == np.ndarray:
            lam = torch.from_numpy(lam).float().to(self.device)
        def evaluator(v):
            hvp = self.hessian_vector_product(fmodel, task, v, x=x, y=y)
            Av = (1.0 + regu_coef) * v + hvp / (lam + lam_damping)
            return Av
        return evaluator

    def hessian_vector_product(self, fmodel, task, vector, params=None, x=None, y=None):
        """
        Performs hessian vector product on the train set in task with the provided vector
        """
        if x is not None and y is not None:
            xt, yt = x, y
        else:
            xt, yt = task['train']
        if params is not None:
            self.set_params(params)
        tloss = self.get_loss(fmodel, xt, yt)
        grad_ft = torch.autograd.grad(tloss, fmodel.parameters(), create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
        vec = vector.to(self.device)
        h = torch.sum(flat_grad * vec)
        hvp = torch.autograd.grad(h, fmodel.parameters())
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat

    def outer_step_with_grad(self, grad, meta_opt, flat_grad=False):
        """
        Given the gradient, step with the outer optimizer using the gradient.
        Assumed that the gradient is a tuple/list of size compatible with model.parameters()
        If flat_grad, then the gradient is a flattened vector
        """
        check = 0
        for p in self.model.parameters():
            check = check + 1 if type(p.grad) == type(None) else check
        if check > 0:
            # init grad fields as needed
            dumdum_loss = self.regularization_loss(self.get_params())
            dumdum_loss.backward()
        if flat_grad:
            offset = 0
            grad = grad.to(self.device)
            for p in self.model.parameters():
                this_grad = grad[offset:offset + p.nelement()].view(p.size())
                p.grad.copy_(this_grad)
                offset += p.nelement()
        else:
            for i, p in enumerate(self.model.parameters()):
                p.grad = grad[i]
        meta_opt.step()

#     @torch.enable_grad()
    def meta_learn(self, batch, batch_idx):
        meta_optimizer, inner_optimizer = self.optimizers(use_pl_optimizer=False)
        tr_xs, tr_ys = batch["train"][0].to(self.device), batch["train"][1].to(self.device)
        tst_xs, tst_ys = batch["test"][0].to(self.device), batch["test"][1].to(self.device)

        lam_grad = torch.tensor(0., device=self.device)
        batch_size = tr_xs.shape[0]
        outer_loss, acc = torch.tensor(0., device=self.device), torch.tensor(0., device=self.device)

        meta_grad = 0.
        inner_opt_kwargs = {'step_size': self.inner_lr}

        torch.cuda.memory_summary(0)

        meta_optimizer.zero_grad()
        for t_idx, (tr_x, tr_y, tst_x, tst_y) in enumerate(zip(tr_xs, tr_ys, tst_xs, tst_ys)):
            with higher.innerloop_ctx(self.model, inner_optimizer, copy_initial_weights=True) as (fmodel, diffopt):
                train_losses = self.inner_loop(fmodel, diffopt, tr_x, tr_y)

            regu_loss = self.regularization_loss(self.get_params(), self.lam)
            diffopt.step(regu_loss)

            tst_loss = self.get_loss(fmodel, tst_x, tst_y)
            outer_loss += tst_loss

            with torch.no_grad():
                test_logit = fmodel(tst_x)
                preds = test_logit.softmax(dim=-1)
                acc += self.accuracy(test_logit, tst_y)

            tst_grad = torch.autograd.grad(tst_loss, fmodel.parameters())

            flat_grad = torch.cat([g.contiguous().view(-1) for g in tst_grad])

            if self.cg_steps <= 1:
                outer_grad = flat_grad
            else:
                task_matrix_eval = self.matrix_evaluator(self.lam, self.cg_damping, fmodel=fmodel, x=tr_x, y=tr_y)
                outer_grad = cg_solve(task_matrix_eval, flat_grad, self.cg_steps, x_init=None)
            # grad collection based on the CG solver, instead of having a normal outer grad it has to be calculated from the cg solver
            # see MAML for what is supposedly normal
            meta_grad += outer_grad

            if self.lam_lr <= 0.:
                task_lam_grad = torch.tensor(0., device=self.device)
            else:
                # TODO: lambda learning
                train_loss = self.get_loss(fmodel, tr_x, tr_y)
                train_grad = torch.autograd.grad(train_loss, fmodel.parameters())
                train_grad = torch.cat([g.contiguous().view(-1) for g in train_grad])
                inner_prod = train_grad.dot(outer_grad)
                task_lam_grad = inner_prod / (self.lam**2 + 0.1)

            lam_grad += (task_lam_grad / batch_size)
        meta_grad.div_(batch_size)
        self.outer_step_with_grad(meta_grad, meta_optimizer, flat_grad=True)
        lam_delta = - self.lam_lr * lam_grad
        self.lam = torch.clamp(self.lam + lam_delta, self.lam_min, 5000.)
        outer_loss.div_(batch_size).detach_()
        acc.div_(batch_size).detach_()
        return outer_loss, acc

    def training_step(self, batch, batch_idx, optimizer_idx):
        train_loss, train_acc = self.meta_learn(batch, batch_idx)

        self.log_dict({
            'tr_accuracy': train_acc,
            'tr_loss': train_loss
        }, prog_bar=True, logger=True)
        return {'tr_loss': train_loss, 'tr_acc': train_acc}


    def validation_step(self, batch, batch_idx):
        val_loss, val_acc = self.meta_learn(batch, batch_idx)

        self.log_dict({
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })

        return val_loss

    def test_step(self, batch, batch_idx):
        test_loss, test_acc = self.meta_learn(batch, batch_idx)
        self.log_dict({
            'test_loss': test_loss,
            'test_accuracy': test_acc
        })
        return test_loss