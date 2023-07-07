from config import CFG
from torch import nn, optim
from torch.utils.data import DataLoader
from model import USPPPMModel
from dataset import USPPPMDataset
from utils import AverageMeter
import os, argparse
import torch, wandb
from tqdm import tqdm
from addict import Dict
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy as sp

def make_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--group', type=str, default=CFG.group, help='Group of the experiment; (default: %(default)s)')
    arg('--name', type=str, default=CFG.name, help='Name of the experiment; (default: %(default)s)')
    # hyper parameters
    arg('--n_epochs', type=int, default=CFG.n_epochs, help='Number of Epoch; (default: %(default)s)')
    arg('--batch_size', type=int, default=CFG.batch_size, help='Batch Size; (default: %(default)s)')
    arg('--learning_rate', type=float, default=CFG.learning_rate, help='Learning Rate; (default: %(default)s)')
    arg('--weight_decay', type=float, default=CFG.weight_decay, help='Weight Decay; (default: %(default)s)')
    # others
    arg('--num_workers', type=int, default=CFG.num_workers, help='Number of dataloader workers; (default: %(default)s)')
    arg("--debug", action="store_true", help="debug")
    arg("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Which device to train on; (default: %(default)s)")
    return parser


def get_score(y_true, y_pred):
    score = sp.stats.pearsonr(y_true, y_pred)[0]
    return score


def train(data, model, optimizer, loss_fn, scaler):
    inputs, targets = data
    model.train()
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=CFG.amp):
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    #preds = outputs.argmax(-1)
    score = get_score(targets.detach().cpu().numpy(), outputs.detach().cpu().numpy())
    return loss, score


@torch.no_grad()
def validate(data, model, loss_fn):
    model.eval()
    inputs, targets = data
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    
    #preds = outputs.argmax(-1)
    score = get_score(targets.detach().cpu().numpy(), outputs.detach().cpu().numpy())
    return loss, score, outputs.to('cpu').numpy(), targets.to('cpu').numpy()


if __name__ == '__main__':
    args = make_parser().parse_args()
    device = args.device
    #torch.multiprocessing.set_start_method('spawn')
    print('num_workers', args.num_workers)
    
    if args.debug:
        print('DEBUG')
        limit = 2
    else:
        limit = None
        
    wandb.login()
    run = wandb.init(
        project='WriteAssist',
        name=args.name,
        config = {
            'backbone': CFG.backbone,
            'n_epochs': args.n_epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay,
        },
        group=args.group,
        anonymous=None
    )
    
    model = USPPPMModel(CFG.backbone).to(device)
    
    train_dataset = USPPPMDataset('./Data/USPPPM 4Fold/fold-0-train.csv', model.tokenizer, 133, limit)
    valid_dataset = USPPPMDataset('./Data/USPPPM 4Fold/fold-0-test.csv', model.tokenizer, 133, limit)
    
    train_dl = DataLoader(
        train_dataset, batch_size = args.batch_size,
        shuffle = True, num_workers = args.num_workers, 
        pin_memory = True, drop_last = True,
    )
    
    val_dl = DataLoader(
        valid_dataset, batch_size = args.batch_size,
        shuffle = False, num_workers = args.num_workers, 
        pin_memory = True, drop_last = False,
    )
    
    
    loss_fn = nn.SmoothL1Loss(reduction='mean')
    #loss_fn = nn.CrossEntropyLoss()
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
          'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
          'lr': args.learning_rate, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
          'lr': args.learning_rate, 'weight_decay': 0.0}  # Decoder lr
    ]
    optimizer = optim.Adam(optimizer_parameters, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.amp)
    
    for epoch in tqdm(range(args.n_epochs), desc='EPOCH'):
        
        # Train Model
        train_losses, train_scores = AverageMeter(), AverageMeter()
        n_batch = len(train_dl)
        for i, (inputs, targets) in enumerate(tqdm(train_dl, desc="TRAIN")):
            inputs = inputs.to(device)
            targets = targets.to(device)
            train_loss, train_score = train((inputs, targets), model, optimizer, loss_fn, scaler)
            train_losses.update(train_loss.item(), n_batch)
            train_scores.update(train_score, n_batch)
            wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_score": train_score, 'train_step':epoch*len(train_dl)+i})
        
        # Validate Model
        val_losses, val_scores = AverageMeter(), AverageMeter()
        n_batch = len(val_dl)
        for i, (inputs, targets) in enumerate(tqdm(val_dl, desc="VALID")):
            inputs = inputs.to(device)
            targets = targets.to(device)
            val_loss, val_score, pred, target = validate((inputs, targets), model, loss_fn)
            val_losses.update(val_loss.item(), n_batch)
            val_scores.update(val_score, n_batch)
            wandb.log({"epoch": epoch+1, "val_loss": val_loss, "val_score": val_score, 'val_step':epoch*len(val_dl)+i})
        
        # Log Results
        print(f'\nEPOCH: {epoch+1: >2}  train_loss:{train_losses.avg: .3f}  valid_loss:{val_losses.avg: .3f}  train_scores:{train_scores.avg: .4f}  val_scores:{val_scores.avg: .4f}\n')
        wandb.log({
            "epoch": epoch+1,
            "train_loss(avg)": train_losses.avg,
            "valid_loss(avg)": val_losses.avg,
            "train_scores(avg)": train_scores.avg,
            "valid_scores(avg)": val_scores.avg,
        })
        
        scheduler.step()
    
    
    if not os.path.exists('./Output'): os.mkdir('./Output')
    torch.save(model.state_dict(), './Output/model_weights.pth')
    wandb.finish()

