import torch
import time
import os
from torch import nn
from sklearn.metrics import mean_squared_error
from utils import AverageMeter, calculate_accuracy, calculate_accuracy_binary_multilabel


def train_epoch(epoch, data_loader, model_G,model_D, criterion_G,criterion_D, optimizer_G,optimizer_D, scheduler, opt,
                epoch_logger, batch_logger, writer):
    print('Train at epoch {}'.format(epoch))

    model_G.train()
    model_D.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    g_losses = AverageMeter()
    l1 = AverageMeter()
    ssim = AverageMeter()
    ssim_piqa = AverageMeter()
    mse = AverageMeter()
    kld = AverageMeter()
    adv = AverageMeter()
    dice = AverageMeter()
    d_losses = AverageMeter()

    end_time = time.time()
    num_batches = len(data_loader)

    for i, inputs in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        # pdb.set_trace()

        if not opt.no_cuda:
            inputs = inputs.cuda(non_blocking=True)
        
        # NEW
        if (opt.model == 'resnet') or (opt.model == 'resnet_fm') or (opt.model == 'resnet_fm_pyramid_sum_Jiapan') or (opt.model == 'resnet_fm_pyramid_sum_CIT') or (opt.model == 'resnet_fm_pyramid_concat_Jiapan') or (opt.model == 'resnet_fm_pyramid_concat_CIT'):
           

            label_1 = torch.ones([inputs.size(0),1], dtype=torch.float32).cuda(non_blocking=True) 
            label_0 = torch.zeros([inputs.size(0),1], dtype=torch.float32).cuda(non_blocking=True) 
            
        
            # Determine MSE
            # outputs_sig_ = torch.flatten(outputs).cpu().detach()
            # inputs_ = torch.flatten(inputs).cpu().detach()
            # mse_error = mean_squared_error(outputs_sig_, inputs_)
            # here maybe the tumor region can be enhanced
        # GAN loss
        optimizer_G.zero_grad()
        recons, outputs_latent = model_G(inputs)
        logits_fake = model_D(recons)
        g_loss_dict = criterion_G(inputs=inputs, recons=recons, logits = logits_fake, labels = label_1)
        g_loss = g_loss_dict['loss']
        g_loss.backward()
        optimizer_G.step()            
            
        
        #model_D.zero_grad()
        optimizer_D.zero_grad()
        logits_real = model_D(inputs)        
        d_loss_real = criterion_D(logits_real,label_1 )      

        logits_fake = model_D(recons.detach())        
        d_loss_fake = criterion_D(logits_fake,label_0 )      

        d_loss  = 0.5*(d_loss_real+d_loss_fake)
        d_loss.backward()
        optimizer_D.step()
        
        
        # NEW
        
        g_losses.update(g_loss.item(), inputs.size(0))
        l1.update(g_loss_dict['L1'].item(), inputs.size(0))
        ssim.update(g_loss_dict['SSIM'].item(), inputs.size(0))
        ssim_piqa.update(g_loss_dict['SSIM_piqa'].item(), inputs.size(0))
        mse.update(g_loss_dict['MSE'].item(), inputs.size(0))
        kld.update(g_loss_dict['KLD'].item(), inputs.size(0))
        adv.update(g_loss_dict['ADV'].item(), inputs.size(0))
        dice.update(g_loss_dict['DICE'].item(), inputs.size(0))
        d_losses.update(d_loss, inputs.size(0))
        
        if opt.lr_scheduler == 'cycliclr':
            scheduler.step()  # For CyclicLR

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        # NEW
        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'g_loss': g_losses.val,
            'l1': l1.val,
            'ssim': ssim.val,
            'ssim_piqa': ssim_piqa.val,
            'mse': mse.val,
            'kld': kld.val,
            'adv': adv.val,
            'dice': dice.val,
            'd_loss': d_losses.val,
            'lr': optimizer_G.param_groups[0]['lr']
        })

        # NEW
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'G_Loss {g_loss.val:.4f} ({g_loss.avg:.4f})\t'
              'L1 {l1.val:.4f} ({l1.avg:.4f})\t'
              'SSIM {ssim.val:.4f} ({ssim.avg:.4f})\t'
              'SSIM_piqa {ssim_piqa.val:.4f} ({ssim_piqa.avg:.4f})\t'
              'MSE {mse.val:.4f} ({mse.avg:.4f})\t'
              'KLD {kld.val:.4f} ({kld.avg:.4f})\t'
              'ADV {adv.val:.4f} ({adv.avg:.4f})\t'
              'DICE {dice.val:.4f} ({dice.avg:.4f})\t'
              'D_Loss {d_loss.val:.4f} ({d_loss.avg:.4f})'.format(
            epoch,
            i + 1,
            len(data_loader),
            batch_time=batch_time,
            data_time=data_time,
            g_loss=g_losses,
            l1=l1,
            ssim=ssim,
            ssim_piqa=ssim_piqa,
            mse=mse,
            kld=kld,
            adv=adv,
            dice=dice,
            d_loss=d_losses))

        # NEW
        n_iter = (epoch - 1) * len(data_loader) + (i + 1)
        writer.add_scalar('G_Loss/train', g_loss.item(), n_iter)
        writer.add_scalar('L1/train', g_loss_dict['L1'].item(), n_iter)
        writer.add_scalar('SSIM/train', g_loss_dict['SSIM'].item(), n_iter)
        writer.add_scalar('SSIM_piqa/train', g_loss_dict['SSIM_piqa'].item(), n_iter)
        writer.add_scalar('MSE/train', g_loss_dict['MSE'].item(), n_iter)
        writer.add_scalar('KLD/train', g_loss_dict['KLD'].item(), n_iter)
        writer.add_scalar('ADV/train', g_loss_dict['ADV'].item(), n_iter)
        writer.add_scalar('DICE/train', g_loss_dict['DICE'].item(), n_iter)
        writer.add_scalar('D_Loss/train', d_loss, n_iter)

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict_G': model_G.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'state_dict_D': model_D.state_dict(),
            'optimizer_D': optimizer_D.state_dict()
        }
        torch.save(states, save_file_path)

    # NEW
    epoch_logger.log({
        'epoch': epoch,
        'g_loss': g_losses.avg,
        'l1': l1.avg,
        'ssim': ssim.avg,
        'ssim_piqa': ssim_piqa.avg,
        'mse': mse.avg,
        'kld': kld.avg,
        'adv': adv.avg,
        'dice': dice.avg,
        'd_loss': d_losses.avg,
        'lr': optimizer_G.param_groups[0]['lr'],
    })
    
    # NEW
    batch_time.reset()
    data_time.reset()
    g_losses.reset()
    l1.reset()
    ssim.reset()
    ssim_piqa.reset()
    mse.reset()
    kld.reset()
    adv.reset()
    dice.reset()
    d_losses.reset()
    
    # NEW
    # return losses.avg, l1.avg, ssim.avg, ssim_piqa.avg, mse.avg, kld.avg
