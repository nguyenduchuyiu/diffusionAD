"""Train segmentation model only with frozen diffusion model"""
import torch
import os
import json
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from models import UNetModel, SegmentationSubNetwork, GaussianDiffusionModel, get_beta_schedule
from utils import RealIADTrainDataset

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=4):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        return torch.mean(self.alpha * (1-pt)**self.gamma * bce)


def train_seg(args, sub_class, device, ckpt_path):
    # Load frozen diffusion model
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    unet = UNetModel(
        args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'],
        dropout=args["dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
        in_channels=args["channels"]
    ).to(device)
    unet.load_state_dict(checkpoint['unet_model_state_dict'])
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    # Use DataParallel for unet if multiple GPUs are available (mostly defensive, since it's frozen)
    if torch.cuda.device_count() > 1:
        unet = torch.nn.DataParallel(unet)

    # Segmentation model (trainable)
    seg_model = SegmentationSubNetwork(in_channels=6, out_channels=1).to(device)
    seg_model.load_state_dict(checkpoint['seg_model_state_dict'])
    # Use DataParallel for segmentation model if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        seg_model = torch.nn.DataParallel(seg_model)
    # Diffusion sampler
    betas = get_beta_schedule(args['T'], args['beta_schedule'])
    ddpm = GaussianDiffusionModel(
        args['img_size'], betas, loss_weight=args['loss_weight'],
        loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=args["channels"]
    )
    
    # Dataset
    subclass_path = os.path.join(args["data_root_path"], args['data_name'], sub_class)
    dataset = RealIADTrainDataset(subclass_path, sub_class, img_size=args["img_size"], args=args)
    loader = DataLoader(dataset, batch_size=args['Batch_Size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    
    # Optimizer & Loss
    # Only the parameters of the underlying model are passed to the optimizer
    # (DataParallel exposes a .module attribute)
    optimizer = optim.Adam(
        seg_model.module.parameters() if isinstance(seg_model, torch.nn.DataParallel) else seg_model.parameters(),
        lr=args.get('seg_lr', 1e-4)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    focal_loss = BinaryFocalLoss().to(device)
    smL1_loss = nn.SmoothL1Loss().to(device)
    
    epochs = args['EPOCHS']
    # lấy trong checkpoint
    losses = {
        'total': checkpoint.get('train_loss_list', []),
        'focal': checkpoint.get('train_focal_loss_list', []),
        'smL1': checkpoint.get('train_smL1_loss_list', [])
    }
    best_loss = float('inf')
    
    for epoch in range(epochs):
        seg_model.train()
        epoch_loss, epoch_focal, epoch_smL1 = 0, 0, 0
        
        tbar = tqdm(loader, desc=f'Epoch {epoch}')
        for sample in tbar:
            aug_image = sample['augmented_image'].to(device)
            anomaly_mask = sample['anomaly_mask'].to(device)
            anomaly_label = sample['has_anomaly'].to(device).squeeze()
            
            with torch.no_grad():
                # If unet is DataParallel, call .module for compatibility with state_dict (already handled above)
                unet_forward = unet
                # Don't need .module here because DataParallel respects __call__
                _, pred_x0, *_ = ddpm.norm_guided_one_step_denoising(unet_forward, aug_image, anomaly_label, args)
            
            pred_mask = seg_model(torch.cat((aug_image, pred_x0), dim=1))
            fl = focal_loss(pred_mask, anomaly_mask)
            sl = smL1_loss(pred_mask, anomaly_mask)
            loss = 5*fl + sl
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_focal += fl.item()
            epoch_smL1 += sl.item()
            tbar.set_postfix(loss=f'{epoch_loss/(tbar.n+1):.4f}')
        
        scheduler.step()
        n_batches = len(loader)
        epoch_loss /= n_batches
        epoch_focal /= n_batches
        epoch_smL1 /= n_batches
        losses['total'].append(epoch_loss)
        losses['focal'].append(epoch_focal)
        losses['smL1'].append(epoch_smL1)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}'
            os.makedirs(save_path, exist_ok=True)
            # Save underlying model's state_dict if using DataParallel
            torch.save({
                'seg_model_state_dict': seg_model.module.state_dict() if isinstance(seg_model, torch.nn.DataParallel) else seg_model.state_dict(),
                'epoch': epoch, 'loss': best_loss
            }, f'{save_path}/seg-best.pt')
            print(f'Best model saved at epoch {epoch}, loss={best_loss:.4f}')
        
        if epoch % 100 == 0:
            save_path = f'{args["output_path"]}/model/diff-params-ARGS={args["arg_num"]}/{sub_class}'
            os.makedirs(save_path, exist_ok=True)
            torch.save({
                'seg_model_state_dict': seg_model.module.state_dict() if isinstance(seg_model, torch.nn.DataParallel) else seg_model.state_dict(),
                'epoch': epoch, 'loss': epoch_loss
            }, f'{save_path}/seg-last.pt')
            print(f'Checkpoint saved at epoch {epoch}, loss={epoch_loss:.4f}')
        
        if epoch % 10 == 0:
            plot_losses(losses, sub_class, args)
            
        
    
    plot_losses(losses, sub_class, args, final=True)
    return losses


def plot_losses(losses, sub_class, args, final=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(len(losses['total']))
    ax.plot(epochs, losses['total'], 'b-', label='Total', linewidth=2)
    ax.plot(epochs, losses['focal'], 'r-', label='Focal')
    ax.plot(epochs, losses['smL1'], 'g-', label='SmoothL1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{sub_class} - Segmentation Training')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
    plt.tight_layout()
    
    save_dir = f'{args["output_path"]}/learning_curves/ARGS={args["arg_num"]}'
    os.makedirs(save_dir, exist_ok=True)
    suffix = '_final' if final else ''
    plt.savefig(f'{save_dir}/{sub_class}_seg{suffix}.png', dpi=150)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open('./args/args1.json', 'r') as f:
        args = json.load(f)
    ckpt_path = "outputs/model/diff-params-ARGS=1/metal_nut/params-last.pt"
    args['arg_num'] = '1'
    args = defaultdict(str, args)
    args['EPOCHS'] = 1800 # can override
    classes = os.listdir(os.path.join(args["data_root_path"], args['data_name']))
    for sub_class in classes:
        print(f"\n=== Training segmentation for {sub_class} ===")
        train_seg(args, sub_class, device, ckpt_path)


if __name__ == '__main__':
    main()

