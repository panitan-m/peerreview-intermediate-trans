import os
import time
import logging
import argparse

from peerkit.evaluator import AverageMeter, AverageMeterHMS, ProgressMeter
from peerkit.evaluator.metrics import f1, acc, f1_macro
from peerkit import utils as U

from intermediate_dataset import get_train_val

from models.bert import BERT

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)

def parser_args():
    parser = argparse.ArgumentParser()
    #Dataset configs
    parser.add_argument('--dataset', default='iclr', type=str)
    parser.add_argument('--aspect', default='clarity', #['clarity', 'meaningful_comparison', 'motivation', 'originality', 'replicability', 'soundness', 'substance']
                        type=str, help='review aspect')
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--check_score', default=False, action='store_true')

    #Model configs
    parser.add_argument('--pretrained', default='allenai/scibert_scivocab_uncased', type=str)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--max_len', default=512, type=int)

    # Training configs
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument("--anneal_strategy", default="linear", type=str)
    parser.add_argument('--eval_metrics', default=['acc', 'f1'], type=str, nargs="+")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--print-freq', default=1, type=int)
    parser.add_argument('--val_interval', default=1, type=int)

    parser.add_argument('--out_dir', default='results', type=str)
    # parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--seed', default=1234, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parser_args()
    U.setup(args)
    
    summary_writer = SummaryWriter(log_dir=args.out_dir)

    # train_dataset, val_dataset, test_dataset = get_dataset(
    #     args.dataset,
    #     args.pretrained, max_length=args.max_len, 
    #     aspect=args.aspect, num_classes=args.num_classes,
    #     seed=args.seed)
    
    train_dataset, val_dataset = get_train_val(
        args.dataset,
        args.pretrained, max_length=args.max_len, 
        aspect=args.aspect, num_classes=args.num_classes,
        check_score=args.check_score,
        seed=args.seed)

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.batch_size)
    # test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size)
        
    model = BERT(args)
    model.cuda()
    
    if args.num_classes == 1:
        criterion = torch.nn.MSELoss()
    else:    
        criterion = torch.nn.CrossEntropyLoss()
        
    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses],
        prefix='=> Val Epoch: ')
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        anneal_strategy=args.anneal_strategy,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2
    )
        
    end = time.time()
    best_epoch = -1
    best_loss = float('inf')
    best_test_loss = float('inf')
    best_f1 = 0
    best_acc = 0
    best_test_f1 = 0
    best_test_acc = 0
    torch.cuda.empty_cache()
    for epoch in range(args.epochs):
        loss = train(train_loader, model, criterion, optimizer, scheduler, epoch, args, logger)

        if summary_writer:
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
        if epoch % args.val_interval == 0:
            val_loss, val_preds, val_targets = validate(val_loader, model, criterion, args)
            # test_loss, test_preds, test_targets = validate(test_loader, model, criterion, args)
            losses.update(val_loss)
            epoch_time.update(time.time() - end)
            end = time.time()
            eta.update(epoch_time.avg * (args.epochs - epoch - 1))
            
            progress.display(epoch, logger)
            
            if summary_writer:
                summary_writer.add_scalar('val_loss', val_loss, epoch)
                # summary_writer.add_scalar('test_loss', test_loss, epoch)
                
            if args.num_classes == 2: f1_fn = f1
            else: f1_fn = f1_macro
            
            if args.num_classes == 1:
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                    best_epoch = epoch
                logger.info("[Dev]  Loss: {:.3f}".format(val_loss))
                logger.info("Best@{}   Loss: {:.3f}".format(best_epoch, best_test_loss))         
            else:    
                val_f1 = f1_fn(torch.tensor(val_preds), torch.tensor(val_targets))
                val_acc = acc(torch.tensor(val_preds), torch.tensor(val_targets))
                
                # test_f1 = f1_fn(torch.tensor(test_preds), torch.tensor(test_targets))
                # test_acc = acc(torch.tensor(test_preds), torch.tensor(test_targets))
                
                is_best = val_acc > best_acc
                if is_best:
                    best_f1 = val_f1
                    best_epoch = epoch
                    # best_test_f1 = test_f1
                    # best_test_acc = test_acc
                    best_acc = val_acc
                    
                logger.info("[Dev]  Loss: {:.3f}   F1: {:.3f}   ACC: {:.3f}".format(val_loss, val_f1, val_acc))
                # logger.info("[Test] Loss: {:.3f}   F1: {:.3f}   ACC: {:.3f}".format(test_loss, test_f1, test_acc))
                # logger.info("Best@{}   F1: {:.3f}   ACC: {:.3f}".format(best_epoch, best_test_f1, best_test_acc))
                logger.info("Best@{}   F1: {:.3f}   ACC: {:.3f}".format(best_epoch, best_f1, best_acc))
                
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best, filename=os.path.join(args.out_dir, 'checkpoint.e{}.pth.tar'.format(epoch+1)))
    return 0

def batch_content(_batch):
    batch = {}
    batch['labels'] = _batch['labels']
    batch['input_ids'] = _batch['paper_input_ids']
    batch['mask'] = _batch['paper_attention_mask']
    
    for key in batch:
        if key == 'labels': batch[key] = batch[key].flatten()
        batch[key] = batch[key].cuda()
    return batch

def train(train_loader, model, criterion, optimizer, scheduler, epoch, args, logger):
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.2e', val_only=True)
    progress = ProgressMeter(
        len(train_loader),
        [lr, losses],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))
    
    model.train()
    
    for i, batch in enumerate(train_loader):
        inputs = batch_content(batch)
        target = inputs.pop('labels')
        if args.num_classes != 1: target = target.long()
            
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(**inputs)
            loss = criterion(outputs['logits'], target)
            
        losses.update(loss.item(), target.size(0))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        lr.update(optimizer.param_groups[0]["lr"])
        
        if i % args.print_freq == 0:
            progress.display(i, logger)

    return losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':5.3f')
    preds, targets = [], []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch_content(batch)
            target = inputs.pop('labels')
            if args.num_classes != 1: target = target.long()
            
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(**inputs)
                loss = criterion(outputs['logits'], target)
                if args.num_classes == 1:
                    pred = outputs['logits']
                else:
                    pred = torch.argmax(outputs['logits'], dim=1)
            
            losses.update(loss.item(), target.size(0))
            
            preds.extend(pred.tolist())
            targets.extend(target.tolist())
            
    return losses.avg, preds, targets


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')
    
    
if __name__ == '__main__':
    main()