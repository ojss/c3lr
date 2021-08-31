# AUTOGENERATED! DO NOT EDIT! File to edit: 03b_ProtoCLR.ipynb (unless otherwise specified).

__all__ = ['Classifier', 'ProtoCLR']

# Cell
#export
import copy
import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from tqdm.auto import tqdm
from torch.autograd import Variable
from pytorch_lightning.loggers import WandbLogger
from .proto_utils import prototypical_loss, get_prototypes, CAE
from .pl_dataloaders import UnlabelledDataModule, UnlabelledDataset, get_episode_loader

# Cell
class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

    def _set_params(self, weight, bias):
        state_dict = dict(weight=weight, bias=bias)
        self.fc.load_state_dict(state_dict)

    def init_params_from_prototypes(self, z_support, n_way, n_support):
        z_support   = z_support.contiguous()
        z_proto     = z_support.view(n_way, n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        # Interpretation of ProtoNet as linear layer (see Snell et al. (2017))
        self._set_params(weight=2*z_proto, bias=-torch.norm(z_proto, dim=-1)**2)

# Cell
class ProtoCLR(pl.LightningModule):
    def __init__(self, model, n_support, n_query, batch_size,
    lr_decay_step, lr_decay_rate, classifier=None, lr=1e-3, inner_lr=1e-3,
    ae=False, distance='euclidean', mode='trainval', eval_ways=5,
    sup_finetune=True, sup_finetune_lr=1e-3, sup_finetune_epochs=15,
    ft_freeze_backbone=True, finetune_batch_norm=False):
        super().__init__()
        self.model = model
        self.ae = ae
        self.batch_size = batch_size
        self.n_support = n_support
        self.n_query = n_query

        self.distance = distance

        self.lr = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step
        self.inner_lr = inner_lr

        self.mode = mode
        self.eval_ways = eval_ways
        self.sup_finetune = sup_finetune
        self.sup_finetune_lr = sup_finetune_lr
        self.sup_finetune_epochs = sup_finetune_epochs
        self.ft_freeze_backbone = ft_freeze_backbone
        self.finetune_batch_norm = finetune_batch_norm

        self.automatic_optimization = False

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=self.lr_decay_step, gamma=self.lr_decay_rate)
        return {'optimizer': opt, 'lr_scheduler': sch}

    def _get_pixelwise_reconstruction_loss(self, supp, query):
        return F.mse_loss(supp.view(1, supp.shape[1], self.n_support, 1, 28, 28),
                          query.view(1, supp.shape[1], self.n_query, 1, 28, 28),
                          reduction='none').sum(dim=[1, 2, 3, 4, 5]).mean(dim=[0])

    def calculate_protoclr_loss(self, batch, ae=True):
        r_supp, r_query = None, None

        # Treat the first dim as way, the second as shots
        # and use the first shot as support (like 1-shot setting)
        # the remaining shots as query
        data = batch['data'].to(self.device) # [batch_size x ways x shots x image_dim]
        data = data.unsqueeze(0)
        # e.g. 50 images, 2 support, 2 query, miniImageNet: torch.Size([1, 50, 4, 3, 84, 84])
        batch_size = data.size(0)
        ways = data.size(1)

        # Divide into support and query shots
        x_support = data[:,:,:self.n_support]
        x_support = x_support.reshape((batch_size, ways * self.n_support, *x_support.shape[-3:])) # e.g. [1,50*n_support,*(3,84,84)]
        x_query = data[:,:,self.n_support:]
        x_query = x_query.reshape((batch_size, ways * self.n_query, *x_query.shape[-3:])) # e.g. [1,50*n_query,*(3,84,84)]

        # Create dummy query labels
        y_query = torch.arange(ways).unsqueeze(0).unsqueeze(2) # batch and shot dim
        y_query = y_query.repeat(batch_size, 1, self.n_query)
        y_query = y_query.view(batch_size, -1).to(self.device)

        y_support = torch.arange(ways).unsqueeze(0).unsqueeze(2) # batch and shot dim
        y_support = y_support.repeat(batch_size, 1, self.n_support)
        y_support = y_support.view(batch_size, -1).to(self.device)

        # Extract features (first dim is batch dim)
        x = torch.cat([x_support, x_query], 1) # e.g. [1,50*(n_support+n_query),*(3,84,84)]
        if ae:
            z, r = self.model.forward(x)
            r_supp = r[:,:ways * self.n_support]
            r_query = r[:,ways * self.n_support:]
        else:
            z = self.model.forward(x)
        z_support = z[:,:ways * self.n_support] # e.g. [1,50*n_support,*(3,84,84)]
        z_query = z[:,ways * self.n_support:] # e.g. [1,50*n_query,*(3,84,84)]
        # Get prototypes
        if self.n_support == 1:
            z_proto = z_support # in 1-shot the prototypes are the support samples
        else:
            z_proto = get_prototypes(z_support, y_support, ways)

        loss, accuracy = prototypical_loss(z_proto, z_query, y_query,
                                           distance=self.distance)
        return loss, accuracy, r_supp, r_query

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        opt.zero_grad()

        loss, accuracy, r_supp, r_query = self.calculate_protoclr_loss(batch, ae=self.ae)
        # adding the pixelwise reconstruction loss at the end
        # it has been broadcasted such that each support source image is broadcasted thrice over the three
        # query set images - which are the augmentations of the support image
        if self.ae:
            loss += self._get_pixelwise_reconstruction_loss(r_supp, r_query)

        self.manual_backward(loss)
        opt.step()
        sch.step()

        self.log_dict({
            'loss': loss.item(),
            'train_accuracy': accuracy
        }, prog_bar=True)

        return loss, accuracy

    @torch.enable_grad()
    def supervised_finetuning(self, encoder, episode, device='cpu', proto_init=True,
                        freeze_backbone=False, finetune_batch_norm=False,
                        inner_lr = 0.001, total_epoch=15, n_way=5):
        x_support = episode['train'][0][0] # only take data & only first batch
        x_support = x_support.to(device)
        x_support_var = Variable(x_support)
        x_query = episode['test'][0][0] # only take data & only first batch
        x_query = x_query.to(device)
        x_query_var = Variable(x_query)
        n_support = x_support.shape[0] // n_way
        n_query = x_query.shape[0] // n_way

        batch_size = n_way
        support_size = n_way * n_support

        y_a_i = Variable(torch.from_numpy(np.repeat(range( n_way ), n_support ) )).to(self.device) # (25,)

        x_b_i = x_query_var
        x_a_i = x_support_var
        encoder.eval()
        z_a_i = encoder(x_a_i.to(device))
        encoder.train()

        # Define linear classifier
        input_dim=z_a_i.shape[1]
        classifier = Classifier(input_dim, n_way=n_way)
        classifier.to(device)
        classifier.train()
        ###############################################################################################
        loss_fn = nn.CrossEntropyLoss().to(device)
        if proto_init: # Initialise as distance classifer (distance to prototypes)
            classifier.init_params_from_prototypes(z_a_i, n_way, n_support)
        classifier_opt = torch.optim.Adam(classifier.parameters(), lr = inner_lr)
        if freeze_backbone is False:
            delta_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=inner_lr)
        # Finetuning
        if freeze_backbone is False:
            encoder.train()
        else:
            encoder.eval()
        classifier.train()
        if not finetune_batch_norm:
            for module in encoder.modules():
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()

        for epoch in tqdm(range(total_epoch), total=total_epoch, leave=False):
            rand_id = np.random.permutation(support_size)

            for j in range(0, support_size, batch_size):
                classifier_opt.zero_grad()
                if freeze_backbone is False:
                    delta_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).to(device)

                z_batch = x_a_i[selected_id]
                y_batch = y_a_i[selected_id]
                #####################################

                output = encoder(z_batch)
                output = classifier(output)
                loss = loss_fn(output, y_batch)

                #####################################
                loss.backward()

                classifier_opt.step()

                if freeze_backbone is False:
                    delta_opt.step()
        classifier.eval()
        encoder.eval()

        output = encoder(x_b_i.to(device))
        scores = classifier(output)

        y_query = torch.tensor(np.repeat(range( n_way ), n_query)).to(device)
        loss = F.cross_entropy(scores, y_query, reduction='mean')
        _, predictions = torch.max(scores, dim=1)
        accuracy = torch.mean(predictions.eq(y_query).float())
        return loss, accuracy.item()
    def validation_step(self, batch, batch_idx):
        original_encoder_state = copy.deepcopy(self.model.encoder.state_dict())
        if not self.mode == 'trainval':
            original_encoder_state = copy.deepcopy(self.model.encoder.state_dict())

        if self.sup_finetune:
            loss, accuracy = self.supervised_finetuning(self.model.encoder,
                                                            episode=batch,
                                                            inner_lr=self.sup_finetune_lr,
                                                            total_epoch=self.sup_finetune_epochs,
                                                            freeze_backbone=self.ft_freeze_backbone,
                                                            finetune_batch_norm=self.finetune_batch_norm,
                                                            device=self.device,
                                                            n_way=self.eval_ways)
            self.model.encoder.load_state_dict(original_encoder_state)
        elif self.mode == 'trainval':
            with torch.no_grad():
                loss, accuracy, _, _ = self.calculate_protoclr_loss(batch, ae=False)
        else:
            with torch.no_grad():
                loss, accuracy, _, _ = self.calculate_protoclr_loss(batch, ae=False)
        self.log_dict({
            'val_loss': loss.detach(),
            'val_accuracy': accuracy
        }, prog_bar=True)

        return loss.item(), accuracy

    def test_step(self, batch, batch_idx):
        original_encoder_state = copy.deepcopy(self.model.encoder.state_dict())
        # if self.sup_finetune:
        loss, accuracy = self.supervised_finetuning(self.model.encoder,
                                                        episode=batch,
                                                        inner_lr=self.sup_finetune_lr,
                                                        total_epoch=self.sup_finetune_epochs,
                                                        freeze_backbone=self.ft_freeze_backbone,
                                                        finetune_batch_norm=self.finetune_batch_norm,
                                                        device=self.device,
                                                        n_way=self.eval_ways)
        self.model.encoder.load_state_dict(original_encoder_state)
        self.log("test_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss.item(), accuracy
