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

############# Motion Aware Section ########
def compute_zero_flow_accuracy(pred_flow, gt_flow, pred_kps, gt_kps, trg_kps, n_pts, 
                               zero_threshold=0.5):
    """
    Compute zero-flow prediction accuracy metrics:
    
    - Zero-flow precision: When zero is predicted, how often is it correct?
      TP_zero / (TP_zero + FP_zero)
      where TP_zero = pred=zero AND gt=zero, FP_zero = pred=zero AND gt≠zero
    
    - Zero-flow recall: When GT is zero, how often is zero predicted?
      TP_zero / (TP_zero + FN_zero)
      where FN_zero = pred≠zero AND gt=zero
    
    - Zero-flow F1: Harmonic mean of precision and recall
    
    - Static bias: Ratio of zero predictions to zero GT
      (pred=zero) / (gt=zero)
    """
    batch_size = pred_flow.shape[0]
    
    metrics = {
        'zero_tp': 0,  # True positive: pred=zero, gt=zero
        'zero_fp': 0,  # False positive: pred=zero, gt≠zero
        'zero_fn': 0,  # False negative: pred≠zero, gt=zero
        'zero_tn': 0,  # True negative: pred≠zero, gt≠zero
        'total_pixels': 0,
        'zero_pred_count': 0,
        'zero_gt_count': 0
    }
    
    # Flow-based metrics (dense)
    for b in range(batch_size):
        gt_flow_mag = torch.norm(gt_flow[b], dim=0)  # (H, W)
        pred_flow_mag = torch.norm(pred_flow[b], dim=0)  # (H, W)
        
        # Valid flow mask (not inf)
        valid_mask = torch.isfinite(gt_flow_mag) & torch.isfinite(pred_flow_mag)
        
        if valid_mask.sum() > 0:
            gt_mag_valid = gt_flow_mag[valid_mask]
            pred_mag_valid = pred_flow_mag[valid_mask]
            
            # Classify as zero or non-zero
            pred_zero = pred_mag_valid < zero_threshold
            gt_zero = gt_mag_valid < zero_threshold
            
            # Confusion matrix
            metrics['zero_tp'] += (pred_zero & gt_zero).sum().item()
            metrics['zero_fp'] += (pred_zero & ~gt_zero).sum().item()
            metrics['zero_fn'] += (~pred_zero & gt_zero).sum().item()
            metrics['zero_tn'] += (~pred_zero & ~gt_zero).sum().item()
            
            metrics['total_pixels'] += valid_mask.sum().item()
            metrics['zero_pred_count'] += pred_zero.sum().item()
            metrics['zero_gt_count'] += gt_zero.sum().item()
    
    # Keypoint-based metrics (sparse)
    kp_metrics = {
        'zero_tp': 0,
        'zero_fp': 0,
        'zero_fn': 0,
        'zero_tn': 0,
        'total_kps': 0
    }
    
    for b in range(batch_size):
        npt = n_pts[b].item()
        if npt > 0:
            # Compute motion magnitude from keypoints
            gt_kp_motion = torch.norm(gt_kps[b][:, :npt] - trg_kps[b][:, :npt], dim=0)
            pred_kp_motion = torch.norm(pred_kps[b][:, :npt] - trg_kps[b][:, :npt], dim=0)
            
            # Classify as zero or non-zero
            pred_kp_zero = pred_kp_motion < zero_threshold
            gt_kp_zero = gt_kp_motion < zero_threshold
            
            # Confusion matrix
            kp_metrics['zero_tp'] += (pred_kp_zero & gt_kp_zero).sum().item()
            kp_metrics['zero_fp'] += (pred_kp_zero & ~gt_kp_zero).sum().item()
            kp_metrics['zero_fn'] += (~pred_kp_zero & gt_kp_zero).sum().item()
            kp_metrics['zero_tn'] += (~pred_kp_zero & ~gt_kp_zero).sum().item()
            
            kp_metrics['total_kps'] += npt
    
    # Compute precision, recall, F1 for flow (dense)
    tp = metrics['zero_tp']
    fp = metrics['zero_fp']
    fn = metrics['zero_fn']
    
    zero_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    zero_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    zero_f1 = 2 * (zero_precision * zero_recall) / (zero_precision + zero_recall) if (zero_precision + zero_recall) > 0 else 0.0
    
    # Static bias: ratio of zero predictions to zero GT
    static_bias_ratio = (metrics['zero_pred_count'] / metrics['total_pixels']) / \
                       (metrics['zero_gt_count'] / metrics['total_pixels']) \
                       if metrics['zero_gt_count'] > 0 else float('inf')
    
    # Compute precision, recall, F1 for keypoints (sparse)
    kp_tp = kp_metrics['zero_tp']
    kp_fp = kp_metrics['zero_fp']
    kp_fn = kp_metrics['zero_fn']
    
    kp_zero_precision = kp_tp / (kp_tp + kp_fp) if (kp_tp + kp_fp) > 0 else 0.0
    kp_zero_recall = kp_tp / (kp_tp + kp_fn) if (kp_tp + kp_fn) > 0 else 0.0
    kp_zero_f1 = 2 * (kp_zero_precision * kp_zero_recall) / (kp_zero_precision + kp_zero_recall) \
                 if (kp_zero_precision + kp_zero_recall) > 0 else 0.0
    
    return {
        # Flow-based (dense) metrics
        'zero_precision': zero_precision,  # When zero is predicted, how often is it correct?
        'zero_recall': zero_recall,       # When GT is zero, how often is zero predicted?
        'zero_f1': zero_f1,
        'static_bias_ratio': static_bias_ratio,  # >1 means over-predicting zero, <1 means under-predicting
        'zero_pred_rate': metrics['zero_pred_count'] / metrics['total_pixels'] if metrics['total_pixels'] > 0 else 0.0,
        'zero_gt_rate': metrics['zero_gt_count'] / metrics['total_pixels'] if metrics['total_pixels'] > 0 else 0.0,
        
        # Keypoint-based (sparse) metrics
        'kp_zero_precision': kp_zero_precision,
        'kp_zero_recall': kp_zero_recall,
        'kp_zero_f1': kp_zero_f1,
        
        # Confusion matrix counts (for debugging)
        'confusion_matrix': {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': metrics['zero_tn']
        },
        'kp_confusion_matrix': {
            'tp': kp_tp, 'fp': kp_fp, 'fn': kp_fn, 'tn': kp_metrics['zero_tn']
        }
    }
############# End Motion Aware Section ########

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
                                  primary_benchmark=None,
                                  use_motion_aware=True,
                                  min_motion_pixels=5.0,
                                  zero_threshold=0.5):
    """
    Validate on multiple benchmarks during training
    
    Args:
        net: The model to evaluate
        val_loaders: Dict of {benchmark: dataloader} for different benchmarks
        device: Device to run on
        epoch: Current epoch number
        multi_evaluator: MultiBenchmarkEvaluator instance
        primary_benchmark: Which benchmark to use for the main loss (if None, uses first benchmark)
        use_motion_aware: If True, compute motion-aware metrics
        min_motion_pixels: Minimum motion magnitude for motion-aware PCK
        zero_threshold: Flow magnitude threshold below which flow is considered "zero"
    
    Returns:
        dict: Results for each benchmark with 'loss', 'pck', and motion-aware metrics
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
            pck_by_category = {}  # Track per-category PCK for TSS
            
            ############# Motion Aware Section ########
            # Accumulate metrics for zero-flow analysis and motion-aware evaluation
            all_pred_flows = []
            all_gt_flows = []
            all_pred_kps = []
            all_gt_kps = []
            all_trg_kps = []
            all_n_pts = []
            motion_pck_array = []
            motion_binned_pck = {'small': [], 'medium': [], 'large': []}
            motion_binned_counts = {'small': 0, 'medium': 0, 'large': 0}
            ############# End Motion Aware Section ########
            
            for i, mini_batch in pbar:
                flow_gt = mini_batch['flow'].to(device)
                pred_flow = net(mini_batch['trg_img'].to(device),
                               mini_batch['src_img'].to(device))

                
                
                # Convert flow to keypoints for evaluation
                estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))

                ############# Motion Aware Section ########
                # Store for zero-flow analysis
                all_pred_flows.append(pred_flow.cpu())
                all_gt_flows.append(flow_gt.cpu())
                all_pred_kps.append(estimated_kps.cpu())
                all_gt_kps.append(mini_batch['src_kps'])
                all_trg_kps.append(mini_batch['trg_kps'])
                all_n_pts.append(mini_batch['n_pts'])
                
                # Motion-aware evaluation (if enabled) - aggregate during loop
                if use_motion_aware:
                    motion_eval = multi_evaluator.evaluators[benchmark].eval_kps_transfer_with_motion_prior(
                        estimated_kps.cpu(), mini_batch, min_motion_pixels=min_motion_pixels
                    )
                    motion_pck_array += motion_eval['pck']
                    
                    # Motion-binned evaluation - aggregate properly
                    motion_binned = multi_evaluator.evaluators[benchmark].eval_kps_transfer_motion_binned(
                        estimated_kps.cpu(), mini_batch
                    )
                    # For each sample, get PCK per bin
                    for idx, (pk, tk, trk) in enumerate(zip(estimated_kps.cpu(), mini_batch['src_kps'], mini_batch['trg_kps'])):
                        thres = mini_batch['pckthres'][idx]
                        npt = mini_batch['n_pts'][idx]
                        motion = trk[:, :npt] - tk[:, :npt]
                        motion_magnitude = torch.norm(motion, dim=0)
                        
                        # Classify into bins and compute PCK per bin
                        for bin_name, (min_motion, max_motion) in [('small', (0, 5)), ('medium', (5, 20)), ('large', (20, float('inf')))]:
                            bin_mask = (motion_magnitude >= min_motion) & (motion_magnitude < max_motion)
                            if bin_mask.sum() > 0:
                                pk_bin = pk[:, :npt][:, bin_mask]
                                tk_bin = tk[:, :npt][:, bin_mask]
                                _, correct_ids, _ = multi_evaluator.evaluators[benchmark].classify_prd(pk_bin, tk_bin, thres)
                                bin_pck = (len(correct_ids) / bin_mask.sum().item()) * 100
                                motion_binned_pck[bin_name].append(bin_pck)
                                motion_binned_counts[bin_name] += bin_mask.sum().item()
                ############# End Motion Aware Section ########

                # Evaluate using the specific benchmark evaluator
                eval_result = multi_evaluator.evaluate(benchmark, estimated_kps.cpu(), mini_batch)
                
                # Track per-category results for TSS
                if benchmark == 'tss' and 'category' in mini_batch:
                    categories = mini_batch['category']
                    pck_values = eval_result['pck']
                    
                    # Handle both batched (list) and single category values
                    if isinstance(categories, (list, tuple)):
                        # DataLoader batches strings into lists
                        category_list = categories
                    else:
                        # Single value (shouldn't happen with DataLoader, but handle it)
                        category_list = [categories]
                    
                    # Aggregate PCK per category
                    for cat, pck in zip(category_list, pck_values):
                        # Convert category to string (handle both tensor and string types)
                        if isinstance(cat, str):
                            cat_name = cat
                        elif hasattr(cat, 'item'):
                            cat_name = cat.item()
                        else:
                            cat_name = str(cat)
                        
                        if cat_name not in pck_by_category:
                            pck_by_category[cat_name] = []
                        pck_by_category[cat_name].append(pck)
                
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
            
            ############# Motion Aware Section ########
            # Compute zero-flow accuracy across entire validation set
            if all_pred_flows:
                pred_flow_all = torch.cat(all_pred_flows, dim=0)
                gt_flow_all = torch.cat(all_gt_flows, dim=0)
                pred_kps_all = torch.cat(all_pred_kps, dim=0)
                gt_kps_all = torch.cat(all_gt_kps, dim=0)
                trg_kps_all = torch.cat(all_trg_kps, dim=0)
                n_pts_all = torch.cat(all_n_pts, dim=0)
                
                zero_flow_metrics = compute_zero_flow_accuracy(
                    pred_flow_all, gt_flow_all, pred_kps_all, gt_kps_all, 
                    trg_kps_all, n_pts_all, zero_threshold=zero_threshold
                )
                results[benchmark]['zero_flow_metrics'] = zero_flow_metrics
            
            # Motion-aware results (already computed during loop)
            if use_motion_aware:
                mean_motion_pck = sum(motion_pck_array) / len(motion_pck_array) if motion_pck_array else 0.0
                results[benchmark]['pck_motion_aware'] = mean_motion_pck
                
                # Motion-binned results
                motion_binned_final = {}
                for bin_name in ['small', 'medium', 'large']:
                    mean_pck = sum(motion_binned_pck[bin_name]) / len(motion_binned_pck[bin_name]) if motion_binned_pck[bin_name] else 0.0
                    motion_binned_final[bin_name] = {
                        'mean_pck': mean_pck,
                        'count': motion_binned_counts[bin_name]
                    }
                results[benchmark]['motion_binned'] = motion_binned_final
            ############# End Motion Aware Section ########
            
            # Add per-category results for TSS
            if benchmark == 'tss' and pck_by_category:
                results[benchmark]['pck_by_category'] = {
                    cat: sum(pcks) / len(pcks) if pcks else 0.0 
                    for cat, pcks in pck_by_category.items()
                }
            
            print(f"{benchmark} - Loss: {avg_loss:.4f}, PCK: {mean_pck:.2f}%")
            
            ############# Motion Aware Section ########
            # Print motion-aware results
            if use_motion_aware and 'pck_motion_aware' in results[benchmark]:
                print(f"{benchmark} - PCK (motion-aware, >{min_motion_pixels}px): {results[benchmark]['pck_motion_aware']:.2f}%")
            
            if use_motion_aware and 'motion_binned' in results[benchmark]:
                print(f"{benchmark} - PCK by motion:")
                for bin_name, bin_data in results[benchmark]['motion_binned'].items():
                    if bin_data['count'] > 0:
                        print(f"  {bin_name}: {bin_data['mean_pck']:.2f}% (n={bin_data['count']})")
            
            # Print static bias metrics
            if 'zero_flow_metrics' in results[benchmark]:
                zfm = results[benchmark]['zero_flow_metrics']
                print(f"{benchmark} - Zero-flow Precision: {zfm['zero_precision']:.2%}")
                print(f"  (When zero is predicted, {zfm['zero_precision']:.2%} of the time it's correct)")
                print(f"{benchmark} - Zero-flow Recall: {zfm['zero_recall']:.2%}")
                print(f"  (When GT is zero, {zfm['zero_recall']:.2%} of the time zero is predicted)")
                print(f"{benchmark} - Zero-flow F1: {zfm['zero_f1']:.2%}")
                print(f"{benchmark} - Static Bias Ratio: {zfm['static_bias_ratio']:.2f}")
                print(f"  (Ratio of zero predictions to zero GT: >1 = over-predicting zero, <1 = under-predicting)")
            ############# End Motion Aware Section ########
            
            if benchmark == 'tss' and 'pck_by_category' in results[benchmark]:
                print(f"  TSS Subcategories:")
                for cat, pck in results[benchmark]['pck_by_category'].items():
                    print(f"    {cat}: {pck:.2f}%")

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

