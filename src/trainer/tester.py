import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from model.metric import psnr
from torchvision.utils import save_image
import torch.nn.functional as F
from torch import autograd
import os
from model.measure import compute_measure
import matplotlib.pyplot as plt
import time
from thop import profile
from thop import clever_format
patch_size = 512


def sgd(weight: torch.Tensor, grad: torch.Tensor, meta_lr) -> torch.Tensor:
    weight = weight - meta_lr * grad
    return weight

class Test(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, test_data_loader,
                  lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer=optimizer, config=config, is_test=True)
        self.config = config
        self.len_epoch = len_epoch
        self.test_data_loader = test_data_loader
        self.do_test = self.test_data_loader is not None
        self.gamma = 1.0
        self.lr_scheduler = lr_scheduler

        self.train_metrics = MetricTracker('Total_loss', writer=self.writer)
        self.test_metrics = MetricTracker('psnr', 'ssim', writer=self.writer)
        if os.path.isdir('../output')==False:
           os.makedirs('../output/')
        if os.path.isdir('../output/C')==False:
           os.makedirs('../output/C/')
        if os.path.isdir('../output/GT')==False:
           os.makedirs('../output/GT/')
        if os.path.isdir('../output/I')==False:
           os.makedirs('../output/I/')

    def _train_epoch(self, epoch):

        self.model.train()
        self.train_metrics.reset()

        log = self.train_metrics.result()
        self.writer.set_step(epoch)
        test_log = self._test_epoch(epoch,save=True)
        log.update(**{'test_' + k: v for k, v in test_log.items()})


        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.writer.close()

        return log

    def _test_epoch(self, epoch,save=False):


        self.test_metrics.reset()

        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0
        #with torch.no_grad():

        if save == True:
            if not os.path.exists('../output/C/' + str(epoch)):
                os.makedirs('../output/C/' + str(epoch))
            else:
                print('../output/C/' + str(epoch) + 'exists!!!')
            if not os.path.exists('../output/index_img/' + str(epoch)):
                os.makedirs('../output/index_img/' + str(epoch))
            else:
                print('../output/index_img/' + str(epoch) + 'exists!!!')
        time_record = []
        for batch_idx, (target, input_noisy, input_GT) in enumerate(self.test_data_loader):
                input_noisy = input_noisy.to(self.device)
                input_GT = input_GT.to(self.device)
                startTime = time.time()

                clean = self.model(input_noisy)
                endTime = time.time()
                if batch_idx != 0:
                    time_record.append(endTime - startTime)
                clean = torch.max(clean, torch.tensor([0.]).cuda())

                clean = self.trunc(self.denormalize_(clean.cpu().detach()))
                input_GT = self.trunc(self.denormalize_(input_GT.cpu().detach()))
                input_noisy = self.trunc(self.denormalize_(input_noisy.cpu().detach()))


                original_result, pred_result = compute_measure(input_noisy[:,:, : , : ], input_GT[:,:, : , : ], clean[:,:, : , : ], 400)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]
                if save == True:
                    for i in range(input_noisy.shape[0]):
                        self.save_fig(input_noisy[i, 0, :, :], input_GT[i, 0, :, :], clean[i, 0, :, :],
                                      target['dir_idx'][i], original_result, pred_result,
                                      '../output/index_img/' + str(epoch))
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

                if save==True:
                    for i in range(input_noisy.shape[0]):
                        save_image(torch.clamp(clean[i,:, : , : ],min=0,max=1).detach().cpu(), '../output/C/'+str(epoch)+'/'+target['dir_idx'][i]+'.PNG')
                        save_image(torch.clamp(input_GT[i,:, : , : ],min=0,max=1).detach().cpu(), '../output/GT/' +target['dir_idx'][i]+'.PNG')
                        save_image(torch.clamp(input_noisy[i,:, : , : ],min=0,max=1).detach().cpu(), '../output/I/' +target['dir_idx'][i]+'.PNG')

                self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'test')
                for met in self.metric_ftns:
                    if met.__name__ == "psnr":
                        self.test_metrics.update('psnr', pred_result[0])
                    elif met.__name__ == "ssim":
                        self.test_metrics.update('ssim', pred_result[1])
                self.writer.close()

                del target
        self.writer.close()
        print('\n')
        print(f"time per image(s)：{sum(time_record)/len(time_record)}")
        print('Original\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
            ori_psnr_avg / len(self.test_data_loader),
            ori_ssim_avg / len(self.test_data_loader),
            ori_rmse_avg / len(self.test_data_loader)))
        print('After learning\nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
            pred_psnr_avg / len(self.test_data_loader),
            pred_ssim_avg / len(self.test_data_loader),
            pred_rmse_avg / len(self.test_data_loader)))
        print('\n')
        total_params = sum(p.numel() for p in self.model.parameters())
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        trainable_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Total parameters in the Genclean: {total_params}")
        print(f"Trainable parameters in the Genclean: {trainable_params}")
        print('\n')
        input = torch.randn(1, 1, 512, 512).cuda()
        macs, _ = profile(self.model.cuda(), (input,))
        macs, _ = clever_format([macs, _], "%.3f")
        print('MACS:', macs)
        return self.test_metrics.result()

    def denormalize_(self, image):
        image = image * (3072 - (-1024)) + (-1024)
        return image

    def trunc(self, mat):
        mat[mat <= -160] = -160
        mat[mat >= 240] = 240
        return mat
    def save_fig(self, x, y, pred, fig_name, original_result, pred_result, save_path):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))

        ax[0].imshow(x, cmap=plt.cm.gray, vmin=-160, vmax=240)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=-160, vmax=240)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=-160, vmax=240)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(save_path, 'result_{}.png'.format(fig_name)))
        plt.close()
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
