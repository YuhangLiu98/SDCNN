import torch.nn.functional as F
import torch.nn as nn
import torch




mse = nn.MSELoss(reduction='mean') 


def loss_dis(output1, output2, margin=1.0):
    euclidean_distance = F.pairwise_distance(output1, output2)
    loss_contrastive = torch.mean(torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive
    
def loss_aug(clean, clean1, noise_w, noise_w1, noise_b, noise_b1):
    loss1 = mse(clean1,clean)
    loss2 = mse(noise_w1,noise_w)
    loss3 = mse(noise_b1,noise_b)
    loss = loss1 + loss2 + loss3
    return loss



    
def loss_main(input_noisy, input_noisy_pred, clean, clean1, clean2, clean3, noise_b, noise_b1, noise_b2, noise_b3, noise_w, noise_w1, noise_w2):
    loss1 = mse(input_noisy_pred, input_noisy)
      
    loss2 = mse(clean1,clean)
    loss3 = mse(noise_b3, noise_b)
    loss4 = mse(noise_w2, noise_w)
    loss5 = mse(clean2, clean)
    

    loss6 = mse(clean3, torch.zeros_like(clean3))
    loss7 = mse(noise_w1, torch.zeros_like(noise_w1))
    loss8 = mse(noise_b1, torch.zeros_like(noise_b1))
    loss9 = mse(noise_b2, torch.zeros_like(noise_b2))

    loss = loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9
    return loss

def loss_pre(input_clean, input_clean_pred, input_noisy, input_noisy_pred, input_noise_pred_clean):
    loss1 = mse(input_clean,input_clean_pred)
    # loss2 = mse(input_noisy,input_noisy_pred)
    # loss3 = 1 - mse(input_noisy, input_noise_pred_clean)
    loss = loss1
    return loss

if __name__ == '__main__':
    print('loss')
