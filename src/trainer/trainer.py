import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.metric import psnr
from torchvision.utils import save_image
import torch.nn.functional as F
from torch import autograd
import os
import torch.nn as nn
from model.measure import compute_measure

patch_size = 512

class Gradient_Net(nn.Module):
  def __init__(self):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

    self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

  def forward(self, x):
    grad_x = F.conv2d(x, self.weight_x)
    grad_y = F.conv2d(x, self.weight_y)
    gradient = torch.abs(grad_x)*torch.exp_(-torch.abs(grad_x)) + torch.abs(grad_y)*torch.exp_(-torch.abs(grad_y))
    return gradient

def gradient(x):
    gradient_model = Gradient_Net().cuda()
    g = torch.mean(gradient_model(x))
    return g
def sgd(weight: torch.Tensor, grad: torch.Tensor, meta_lr) -> torch.Tensor:
    weight = weight - meta_lr * grad
    return weight

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, test_data_loader,
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer=optimizer, config=config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.test_data_loader = test_data_loader
        self.do_test = self.test_data_loader is not None
        self.do_test = True
        self.gamma = 1.00
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('Total_loss', writer=self.writer)
        self.test_metrics = MetricTracker('psnr', 'ssim', writer=self.writer)
        if os.path.isdir('../output') == False:
            os.makedirs('../output/')
        if os.path.isdir('../output/C') == False:
            os.makedirs('../output/C/')
        if os.path.isdir('../output/GT') == False:
            os.makedirs('../output/GT/')
        if os.path.isdir('../output/I') == False:
            os.makedirs('../output/I/')

    def _train_epoch(self, epoch):

        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (target, input_noisy, input_GT, input_normal) in enumerate(self.data_loader):
            input_noisy = input_noisy.to(self.device)

            self.optimizer.zero_grad()

            '''待发布'''

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} TotalLoss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss_total.item()
                ))

            if batch_idx == self.len_epoch:
                break

            del target
            del loss_total

        log = self.train_metrics.result()

        if self.do_test:
            if epoch > 100 or epoch % 1 == 0:
                test_log = self._test_epoch(epoch, save=False)
                log.update(**{'test_' + k: v for k, v in test_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.writer.close()

        return log

    def _test_epoch(self, epoch, save=False):

        self.test_metrics.reset()
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        # with torch.no_grad():
        if save == True:
            os.makedirs('../output/C/' + str(epoch))
        for batch_idx, (target, input_noisy, input_GT, input_normal) in enumerate(self.test_data_loader):
            input_noisy = input_noisy.to(self.device)
            input_GT = input_GT.to(self.device)


            _, _, clean = self.model(input_noisy)

            clean = self.trunc(self.denormalize_(clean.cpu().detach()))
            input_GT = self.trunc(self.denormalize_(input_GT.cpu().detach()))
            input_noisy = self.trunc(self.denormalize_(input_noisy.cpu().detach()))

            # red-cnn指标计算
            original_result, pred_result = compute_measure(input_noisy, input_GT, clean, 400)
            ori_psnr_avg += original_result[0]
            ori_ssim_avg += original_result[1]
            ori_rmse_avg += original_result[2]
            pred_psnr_avg += pred_result[0]
            pred_ssim_avg += pred_result[1]
            pred_rmse_avg += pred_result[2]

            size = [clean.shape[0], clean.shape[1], clean.shape[2] * clean.shape[3]]
            clean = (clean - torch.min(clean.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1)) / (
                    torch.max(clean.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1) -
                    torch.min(clean.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1))
            input_GT = (input_GT - torch.min(input_GT.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1)) / (
                    torch.max(input_GT.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1) -
                    torch.min(input_GT.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1))
            input_noisy = (input_noisy - torch.min(input_noisy.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1)) / (
                    torch.max(input_noisy.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1) -
                    torch.min(input_noisy.view(size), -1)[0].unsqueeze(-1).unsqueeze(-1))

            if save == True:
                for i in range(input_noisy.shape[0]):
                    save_image(torch.clamp(clean[i, ::, :], min=0, max=1).detach().cpu(),
                               '../output/C/' + str(epoch) + '/' + target['dir_idx'][i] + '.PNG')
                    save_image(torch.clamp(input_GT[i, :, :, :], min=0, max=1).detach().cpu(),
                               '../output/GT/' + target['dir_idx'][i] + '.PNG')
                    save_image(torch.clamp(input_noisy[i, :, :, :], min=0, max=1).detach().cpu(),
                               '../output/I/' + target['dir_idx'][i] + '.PNG')

            self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'test')
            for met in self.metric_ftns:
                if met.__name__ == "psnr":
                    self.test_metrics.update('psnr', pred_result[0])
                elif met.__name__ == "ssim":
                    self.test_metrics.update('ssim', pred_result[1])
            self.writer.close()

            del target

        self.writer.close()
        return self.test_metrics.result()

    def denormalize_(self, image):
        image = image * (3072 - (-1024)) + (-1024)
        return image

    def trunc(self, mat):
        mat[mat <= -160] = -160
        mat[mat >= 240] = 240
        return mat

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

