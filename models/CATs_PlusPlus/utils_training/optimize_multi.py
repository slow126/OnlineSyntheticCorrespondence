import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from models.CATs_PlusPlus.utils_training.utils import flow2kps
from models.CATs_PlusPlus.utils_training.eval_instance import MultiBenchmarkEvaluator

r'''
    Multi-benchmark validation functions for training with multiple evaluation sets
'''

def EPE(input_flow, target_flow, sparse=True, mean=True, sum=False):
    """End-Point Error loss function"""
    EPE_map = torch.norm(target_flow-input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/torch.sum(~mask)


def validate_epoch_multi_benchmark(net,
                                  val_loaders,
                                  device,
                                  epoch,
                                  multi_evaluator,
                                  primary_benchmark=None):
    """
    Validate on multiple benchmarks during training
    
    Args:
        net: The model to evaluate
        val_loaders: Dict of {benchmark: dataloader} for different benchmarks
        device: Device to run on
        epoch: Current epoch number
        multi_evaluator: MultiBenchmarkEvaluator instance
        primary_benchmark: Which benchmark to use for the main loss (if None, uses first benchmark)
    
    Returns:
        dict: Results for each benchmark with 'loss' and 'pck' keys
    """
    net.eval()
    
    if primary_benchmark is None:
        primary_benchmark = list(val_loaders.keys())[0]
    
    results = {}
    
    with torch.no_grad():
        for benchmark, val_loader in val_loaders.items():
            print(f"Validating on {benchmark}...")
            
            running_total_loss = 0
            pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Val {benchmark}")
            pck_array = []
            
            for i, mini_batch in pbar:
                flow_gt = mini_batch['flow'].to(device)
                pred_flow = net(mini_batch['trg_img'].to(device),
                               mini_batch['src_img'].to(device))

                # Convert flow to keypoints for evaluation
                estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))

                # Evaluate using the specific benchmark evaluator
                eval_result = multi_evaluator.evaluate(benchmark, estimated_kps.cpu(), mini_batch)
                
                # Compute loss
                Loss = EPE(pred_flow, flow_gt) 

                pck_array += eval_result['pck']
                running_total_loss += Loss.item()
                
                pbar.set_description(
                    f'Val {benchmark} R_total_loss: {running_total_loss / (i + 1):.3f}/{Loss.item():.3f}')
            
            mean_pck = sum(pck_array) / len(pck_array) if pck_array else 0.0
            avg_loss = running_total_loss / len(val_loader)
            
            results[benchmark] = {
                'loss': avg_loss,
                'pck': mean_pck
            }
            
            print(f"{benchmark} - Loss: {avg_loss:.4f}, PCK: {mean_pck:.2f}%")

    return results


def validate_epoch_single_benchmark(net,
                                   val_loader,
                                   device,
                                   epoch,
                                   evaluator):
    """
    Validate on a single benchmark (backward compatibility)
    
    Args:
        net: The model to evaluate
        val_loader: DataLoader for validation
        device: Device to run on
        epoch: Current epoch number
        evaluator: EvaluatorInstance for the benchmark
    
    Returns:
        tuple: (average_loss, mean_pck)
    """
    net.eval()
    running_total_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        pck_array = []
        for i, mini_batch in pbar:
            flow_gt = mini_batch['flow'].to(device)
            pred_flow = net(mini_batch['trg_img'].to(device),
                           mini_batch['src_img'].to(device))

            estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))

            eval_result = evaluator.evaluate(estimated_kps.cpu(), mini_batch)
            
            Loss = EPE(pred_flow, flow_gt) 

            pck_array += eval_result['pck']

            running_total_loss += Loss.item()
            pbar.set_description(
                ' validation R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), Loss.item()))
        mean_pck = sum(pck_array) / len(pck_array)

    return running_total_loss / len(val_loader), mean_pck

# Duplicated from optimize.py. Probably don't need and can delete.
# TODO: Delete this function.
# def train_epoch(net,
#                 optimizer,
#                 train_loader,
#                 device,
#                 epoch,
#                 train_writer):
#     """Training epoch function (unchanged from original)"""
#     n_iter = epoch*len(train_loader)
    
#     net.train()
#     running_total_loss = 0

#     pbar = tqdm(enumerate(train_loader), total=len(train_loader))
#     for i, mini_batch in pbar:
#         optimizer.zero_grad()
#         flow_gt = mini_batch['flow'].to(device)

#         pred_flow = net(mini_batch['trg_img'].to(device),
#                          mini_batch['src_img'].to(device))
        
#         Loss = EPE(pred_flow, flow_gt) 
#         Loss.backward()
#         optimizer.step()

#         running_total_loss += Loss.item()
#         train_writer.add_scalar('train_loss_per_iter', Loss.item(), n_iter)
#         n_iter += 1
#         pbar.set_description(
#                 'training: R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), Loss.item()))
#     running_total_loss /= len(train_loader)
#     return running_total_loss
