import os
import time
import logging
import argparse
import json
from sklearn.model_selection import KFold

from peerkit.datasets import peerread, evaluate_major
from peerkit.evaluator import Evaluator, AverageMeter, AverageMeterHMS, ProgressMeter, create_kfold_results
from peerkit.evaluator.metrics import f1, acc
from peerkit import utils as U

from transformers import AutoTokenizer
from models.bert import BERT

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)

def parser_args():
    parser = argparse.ArgumentParser()
    #Dataset configs
    parser.add_argument('--data_dir', default='PeerRead/data')
    parser.add_argument('--dataset', default='acl_2017', type=str)
    parser.add_argument('--aspects', default=['recommendation'], #['recommendation', 'substance', 'meaningful_comparison', 'soundness_correctness', 'originality', 'clarity', 'impact']
                        type=str, nargs="+", help='review aspects')
    parser.add_argument('--task', default='cls', type=str)

    #Model configs
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--pretrained', default='allenai/scibert_scivocab_uncased', type=str)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--max_len', default=512, type=int)

    # Training configs
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument("--anneal_strategy", default="linear", type=str)
    parser.add_argument('--eval_metrics', default=['acc', 'f1_macro'], type=str, nargs="+")
    parser.add_argument('--epochs', type=int, default=30)
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

    if args.task == 'bcls': args.num_classes = 2
    elif args.task == 'cls': args.num_classes = 5
    else: raise ValueError
    
    dataset = peerread.all_sets(args.data_dir, args.dataset, args.aspects, task=args.task)
    logging.info('Total labeled: {}'.format(len(dataset)))

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)
    
    all_pred = []
    all_target = []

    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    kfold_results = create_kfold_results(args.eval_metrics)
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        logging.info('Fold {}'.format(fold_i))
        (train_dataset, test_dataset), vocab = peerread.kfold(dataset, train_idx, test_idx, tokenizer,
                                                            paper_length=args.max_len)
        evaluate_major(train_dataset, test_dataset, args.aspects)
        # # print(train_dataset.paper_ids)
        # # print(test_dataset.paper_ids)
        # print(test_dataset.y.avg.squeeze(1))
        # continue
        
        train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size)
        
        model = BERT(args)
        if args.checkpoint is not None:
            logging.info('Loading checkpoint from {}'.format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            state_dict = checkpoint['state_dict']
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            model.load_state_dict(state_dict, strict=False)
        model.cuda()
        
        criterion = torch.nn.CrossEntropyLoss()
        
        epoch_time = AverageMeterHMS('TT')
        eta = AverageMeterHMS('ETA', val_only=True)
        losses = AverageMeter('Loss', ':5.3f', val_only=True)
        progress = ProgressMeter(
            args.epochs,
            [eta, epoch_time, losses],
            prefix='=> Test Epoch: ')
        
        evaluator = Evaluator(args.out_dir, args.aspects, args.eval_metrics)
        
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
        best_f1 = 0
        best_acc = 0
        torch.cuda.empty_cache()
        for epoch in range(args.epochs):
            loss = train(train_loader, model, criterion, optimizer, scheduler, epoch, args, logger)

            if summary_writer:
                summary_writer.add_scalar('train_loss', loss, epoch)
                summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                
            if epoch % args.val_interval == 0:
                loss, preds, targets = validate(test_loader, model, criterion, args)
                losses.update(loss)
                epoch_time.update(time.time() - end)
                end = time.time()
                eta.update(epoch_time.avg * (args.epochs - epoch - 1))
                
                progress.display(epoch, logger)
                
                if summary_writer:
                    summary_writer.add_scalar('val_loss', loss, epoch)
                    
                # f1_score = f1(torch.tensor(preds), torch.tensor(targets))
                test_acc = acc(torch.tensor(preds), torch.tensor(targets))
                
                # is_best = f1_score > best_f1
                is_best = test_acc > best_acc
                if is_best:
                    # best_f1 = f1_score
                    best_acc = test_acc
                    best_epoch = epoch
                    best_preds = preds
                    
                # logger.info("Loss: {:.3f}   F1: {:.3f}".format(loss, f1_score))
                # logger.info("Best F1: {:.3f} in ep {}".format(best_f1, best_epoch))
                logger.info("Loss: {:.3f}   ACC: {:.3f}".format(loss, test_acc))
                logger.info("Best ACC: {:.3f} in ep {}".format(best_acc, best_epoch))
                
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(args.out_dir, 'checkpoint.pth.tar'))
               
                
        eval_out = evaluator.evaluate(torch.tensor(best_preds), torch.tensor(targets))
        for metric in args.eval_metrics:
            kfold_results[metric].update(eval_out[metric])
            
        all_pred.extend(best_preds)
        all_target.extend(targets)

    logger.info('Cross-validation is complete')
    description = '\t%20s'%('ASPECTS')
    for metric in args.eval_metrics:
        description += '\t%8s'%(metric.upper())
    logger.info(description)
    description = '\t%20s'%(args.aspects[0].upper())
    for metric in args.eval_metrics:
        description += '\t%8.4f'%(kfold_results[metric].avg)
    logger.info(description)
    
    with open(os.path.join(args.out_dir, 'preds.txt'), 'w') as f:
        f.write(json.dumps(all_pred))
    with open(os.path.join(args.out_dir, 'targets.txt'), 'w') as f:
        f.write(json.dumps(all_target))
    
    return 0

def batch_content(_batch):
    batch = {}
    batch['labels'] = _batch['labels']
    batch['input_ids'] = _batch['paper_input_ids']
    batch['mask'] = _batch['paper_attention_mask']
    
    for key in batch:
        if key == 'labels': batch[key] = batch[key].flatten().long()
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
            
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(**inputs)
                loss = criterion(outputs['logits'], target)
                pred = torch.argmax(outputs['logits'], dim=1)
            
            losses.update(loss.item(), target.size(0))
            
            preds.extend(pred.tolist())
            targets.extend(target.tolist())
            
    return losses.avg, preds, targets


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')
    
    
if __name__ == '__main__':
    main()