import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from models.CATs_PlusPlus.utils_training.utils import flow2kps
from models.CATs_PlusPlus.utils_training.eval_instance import MultiBenchmarkEvaluator
from src.data.synth.datasets.flow_utils import flow_from_kps

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
    # Ensure all tensors are on the same device (CPU for memory efficiency)
    device = pred_flow.device
    pred_flow = pred_flow.to(device, non_blocking=True)
    gt_flow = gt_flow.to(device, non_blocking=True)
    pred_kps = pred_kps.to(device, non_blocking=True)
    gt_kps = gt_kps.to(device, non_blocking=True)
    trg_kps = trg_kps.to(device, non_blocking=True)
    n_pts = n_pts.to(device, non_blocking=True) if isinstance(n_pts, torch.Tensor) else n_pts
    
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
        npt = n_pts[b].item() if isinstance(n_pts[b], torch.Tensor) else int(n_pts[b])
        if npt > 0:
            # Ensure all keypoint tensors are on the same device for this batch
            gt_kps_b = gt_kps[b].to(device, non_blocking=True)
            trg_kps_b = trg_kps[b].to(device, non_blocking=True)
            pred_kps_b = pred_kps[b].to(device, non_blocking=True)
            
            # Compute motion magnitude from keypoints
            gt_kp_motion = torch.norm(gt_kps_b[:, :npt] - trg_kps_b[:, :npt], dim=0)
            pred_kp_motion = torch.norm(pred_kps_b[:, :npt] - trg_kps_b[:, :npt], dim=0)
            
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
                                  zero_threshold=0.5,
                                  mmd_every_n_epochs=0):
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
        mmd_every_n_epochs: Compute MMD every N epochs (0 = disabled, 1 = every epoch, 2 = every other, etc.)
    
    Returns:
        dict: Results for each benchmark with 'loss', 'pck', motion-aware metrics, and MMD metrics
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
            # Only initialize if motion-aware evaluation is enabled
            if use_motion_aware:
                all_pred_flows = []
                all_gt_flows = []
                all_pred_kps = []
                all_gt_kps = []
                all_trg_kps = []
                all_n_pts = []
                motion_pck_array = []
                motion_binned_pck = {'small': [], 'medium': [], 'large': []}
                motion_binned_counts = {'small': 0, 'medium': 0, 'large': 0}
            else:
                # Initialize empty lists to avoid NameError, but they won't be used
                all_pred_flows = None
                all_gt_flows = None
                all_pred_kps = None
                all_gt_kps = None
                all_trg_kps = None
                all_n_pts = None
            ############# End Motion Aware Section ########
            
            ############# MMD Section ########
            # Initialize streaming MMD if enabled for this epoch
            streaming_mmd = None
            if mmd_every_n_epochs > 0 and epoch % mmd_every_n_epochs == 0:
                try:
                    from models.CATs_PlusPlus.utils_training.mmd_validation import create_mmd_streaming
                    streaming_mmd = create_mmd_streaming(device=device)
                    print(f"MMD enabled for {benchmark} (epoch {epoch}, every {mmd_every_n_epochs} epochs)")
                except Exception as e:
                    print(f"ERROR: Failed to initialize MMD streaming for {benchmark}: {e}")
                    import traceback
                    traceback.print_exc()
                    streaming_mmd = None
            ############# End MMD Section ########
            
            for i, mini_batch in pbar:
                # Transfer entire batch to GPU once at the start (efficient - single operation)
                # This ensures all tensors are on the same device throughout the loop
                # Use explicit device comparison to handle "cuda" vs "cuda:0" properly
                gpu_batch = {}
                for key, value in mini_batch.items():
                    if isinstance(value, torch.Tensor):
                        # Explicit device comparison: check type and index separately
                        # This handles cases where device is "cuda" vs "cuda:0"
                        value_device = value.device
                        needs_transfer = (
                            value_device.type != device.type or
                            (value_device.index if value_device.index is not None else 0) != 
                            (device.index if device.index is not None else 0)
                        )
                        if needs_transfer:
                            gpu_batch[key] = value.to(device, non_blocking=True)
                        else:
                            gpu_batch[key] = value  # Already on correct device
                    else:
                        gpu_batch[key] = value  # Keep non-tensors as-is (e.g., 'category' strings)
                
                # Ensure all async transfers complete before using tensors
                # This is critical when mixing datasets (synthetic on GPU, others on CPU)
                if device.type == 'cuda' and any(isinstance(v, torch.Tensor) and v.device.type == 'cuda' for v in gpu_batch.values()):
                    torch.cuda.synchronize(device)
                
                # Use flow_downsampled if available (CorrespondenceDataset), otherwise use flow (old datasets)
                if 'flow_downsampled' in gpu_batch:
                    flow_gt = gpu_batch['flow_downsampled']
                else:
                    flow_gt = gpu_batch['flow']


                pred_flow = net(gpu_batch['trg_img'], gpu_batch['src_img'])
                
                # Convert flow to keypoints for evaluation
                estimated_kps = flow2kps(gpu_batch['trg_kps'], pred_flow, gpu_batch['n_pts'])

                ############# Motion Aware Section ########
                # Store for zero-flow analysis (move to CPU for memory efficiency)
                # Only collect data if motion-aware evaluation is enabled
                if use_motion_aware:
                    all_pred_flows.append(pred_flow.cpu())
                    all_gt_flows.append(flow_gt.cpu())
                    all_pred_kps.append(estimated_kps.cpu())
                    # Ensure all keypoints are on CPU for consistency
                    all_gt_kps.append(gpu_batch['src_kps'].cpu())
                    all_trg_kps.append(gpu_batch['trg_kps'].cpu())
                    all_n_pts.append(gpu_batch['n_pts'].cpu())
                    
                    # Motion-aware evaluation - aggregate during loop
                    # All tensors already on GPU in gpu_batch
                    motion_eval = multi_evaluator.evaluators[benchmark].eval_kps_transfer_with_motion_prior(
                        estimated_kps, gpu_batch, min_motion_pixels=min_motion_pixels
                    )
                    motion_pck_array += motion_eval['pck']
                    
                    # Motion-binned evaluation - aggregate properly
                    motion_binned = multi_evaluator.evaluators[benchmark].eval_kps_transfer_motion_binned(
                        estimated_kps, gpu_batch
                    )
                    # For each sample, get PCK per bin
                    # All tensors are already on GPU from gpu_batch
                    src_kps = gpu_batch['src_kps']
                    trg_kps = gpu_batch['trg_kps']
                    
                    for idx, (pk, tk, trk) in enumerate(zip(estimated_kps, src_kps, trg_kps)):
                        thres = gpu_batch['pckthres'][idx]
                        npt = gpu_batch['n_pts'][idx]
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
                # All tensors already on GPU in gpu_batch
                eval_result = multi_evaluator.evaluate(benchmark, estimated_kps, gpu_batch)
                
                ############# MMD Section ########
                # Update streaming MMD if enabled for this epoch
                if streaming_mmd is not None:
                    try:
                        # Get flow_full for MMD calculation (must be in pixel space at full resolution)
                        if 'flow_full' in gpu_batch:
                            flow_full = gpu_batch['flow_full']
                        else:
                            # Some datasets provide feature-grid flow in 'flow'; rebuild pixel flow if needed.
                            if 'trg_img' in gpu_batch:
                                _, _, img_h, img_w = gpu_batch['trg_img'].shape
                            elif 'src_img' in gpu_batch:
                                _, _, img_h, img_w = gpu_batch['src_img'].shape
                            else:
                                raise ValueError("Cannot determine image size to rebuild flow for MMD.")

                            def rebuild_flow_full():
                                flows = []
                                for b in range(gpu_batch['src_kps'].shape[0]):
                                    flows.append(flow_from_kps(gpu_batch['src_kps'][b], gpu_batch['trg_kps'][b], (img_h, img_w)))
                                return torch.stack(flows, dim=0)

                            if 'flow' in gpu_batch:
                                flow_candidate = gpu_batch['flow']
                                # Use directly only if already pixel-aligned
                                if flow_candidate.shape[2:] == (img_h, img_w):
                                    flow_full = flow_candidate
                                else:
                                    flow_full = rebuild_flow_full()
                            else:
                                flow_full = rebuild_flow_full()
                        
                        # Get correct/incorrect keypoint IDs for MMD calculation
                        eval_result_with_correct, correct_id_list = multi_evaluator.evaluators[benchmark].eval_kps_transfer_with_correct(
                            estimated_kps, gpu_batch
                        )
                        from models.CATs_PlusPlus.utils_training.mmd_validation import update_mmd_streaming
                        update_mmd_streaming(streaming_mmd, pred_flow, flow_full, gpu_batch['trg_kps'], 
                                            correct_id_list, gpu_batch['n_pts'], device)
                    except Exception as e:
                        # Don't break validation if MMD fails, but print error for debugging
                        print(f"ERROR: Failed to update MMD streaming for {benchmark} batch {i}: {e}")
                        import traceback
                        traceback.print_exc()
                ############# End MMD Section ########
                
                # Track per-category results for TSS
                if benchmark == 'tss' and 'category' in gpu_batch:
                    categories = gpu_batch['category']
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
            
            ############# MMD Section ########
            # Compute MMD if streaming was enabled for this epoch
            if streaming_mmd is not None:
                try:
                    from models.CATs_PlusPlus.utils_training.mmd_validation import compute_mmd_from_streaming
                    mmd_results = compute_mmd_from_streaming(streaming_mmd)
                    results[benchmark].update(mmd_results)
                    
                    # Print MMD results with clear formatting
                    print(f"\n{benchmark} - MMD^2 Results:")
                    mmd_val = mmd_results['mmd2_pred_corr_vs_pred_miss']
                    if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:  # Check for NaN (NaN != NaN)
                        print(f"  pred_corr vs pred_miss: {mmd_val:.6f}")
                    else:
                        print(f"  pred_corr vs pred_miss: NaN (insufficient samples)")
                    
                    mmd_val = mmd_results['mmd2_pred_corr_vs_gt']
                    if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:  # Check for NaN
                        print(f"  pred_corr vs gt: {mmd_val:.6f}")
                    else:
                        print(f"  pred_corr vs gt: NaN (insufficient samples)")
                    
                    mmd_val = mmd_results['mmd2_pred_miss_vs_gt']
                    if isinstance(mmd_val, (int, float)) and mmd_val == mmd_val:  # Check for NaN
                        print(f"  pred_miss vs gt: {mmd_val:.6f}")
                    else:
                        print(f"  pred_miss vs gt: NaN (insufficient samples)")
                    print()
                except Exception as e:
                    print(f"ERROR: Failed to compute MMD for {benchmark}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Still add NaN values so CSV logging doesn't break
                    results[benchmark].update({
                        'mmd2_pred_corr_vs_pred_miss': float('nan'),
                        'mmd2_pred_corr_vs_gt': float('nan'),
                        'mmd2_pred_miss_vs_gt': float('nan')
                    })
            else:
                # MMD not enabled for this epoch - add NaN values for consistency
                if mmd_every_n_epochs > 0:
                    results[benchmark].update({
                        'mmd2_pred_corr_vs_pred_miss': float('nan'),
                        'mmd2_pred_corr_vs_gt': float('nan'),
                        'mmd2_pred_miss_vs_gt': float('nan')
                    })
            ############# End MMD Section ########
            
            ############# Motion Aware Section ########
            # Compute zero-flow accuracy across entire validation set
            # Only compute if motion-aware evaluation is enabled
            if use_motion_aware and all_pred_flows is not None and len(all_pred_flows) > 0:
                # All tensors should be on CPU at this point (moved during append)
                pred_flow_all = torch.cat(all_pred_flows, dim=0)
                gt_flow_all = torch.cat(all_gt_flows, dim=0)
                
                # Handle variable-sized keypoints (when using dense keypoints, batches may have different sizes)
                # Find max keypoint size across all batches
                # Keypoint tensors have shape [B, 2, N] where N is the number of keypoints
                # Note: All tensors are already on CPU (moved during loop for memory efficiency)
                max_kps = 0
                kps_device = None
                for kps_tensor in all_pred_kps + all_gt_kps + all_trg_kps:
                    if isinstance(kps_tensor, torch.Tensor):
                        if kps_device is None:
                            kps_device = kps_tensor.device
                        if kps_tensor.shape[2] > max_kps:
                            max_kps = kps_tensor.shape[2]
                
                # Pad all keypoint tensors to max_kps (ensure all on same device)
                if max_kps > 0 and kps_device is not None:
                    padded_pred_kps = []
                    padded_gt_kps = []
                    padded_trg_kps = []
                    
                    for kps_tensor in all_pred_kps:
                        kps_tensor = kps_tensor.to(kps_device, non_blocking=True)  # Ensure same device
                        if kps_tensor.shape[2] < max_kps:
                            pad_size = max_kps - kps_tensor.shape[2]
                            padding = torch.ones(kps_tensor.shape[0], 2, pad_size, dtype=kps_tensor.dtype, device=kps_device) * -1
                            kps_tensor = torch.cat([kps_tensor, padding], dim=2)
                        padded_pred_kps.append(kps_tensor)
                    
                    for kps_tensor in all_gt_kps:
                        if isinstance(kps_tensor, torch.Tensor):
                            kps_tensor = kps_tensor.to(kps_device, non_blocking=True)  # Ensure same device
                            if kps_tensor.shape[2] < max_kps:
                                pad_size = max_kps - kps_tensor.shape[2]
                                padding = torch.ones(kps_tensor.shape[0], 2, pad_size, dtype=kps_tensor.dtype, device=kps_device) * -1
                                kps_tensor = torch.cat([kps_tensor, padding], dim=2)
                        padded_gt_kps.append(kps_tensor)
                    
                    for kps_tensor in all_trg_kps:
                        if isinstance(kps_tensor, torch.Tensor):
                            kps_tensor = kps_tensor.to(kps_device, non_blocking=True)  # Ensure same device
                            if kps_tensor.shape[2] < max_kps:
                                pad_size = max_kps - kps_tensor.shape[2]
                                padding = torch.ones(kps_tensor.shape[0], 2, pad_size, dtype=kps_tensor.dtype, device=kps_device) * -1
                                kps_tensor = torch.cat([kps_tensor, padding], dim=2)
                        padded_trg_kps.append(kps_tensor)
                    
                    pred_kps_all = torch.cat(padded_pred_kps, dim=0)
                    gt_kps_all = torch.cat(padded_gt_kps, dim=0)
                    trg_kps_all = torch.cat(padded_trg_kps, dim=0)
                else:
                    # Ensure all tensors are on the same device before concatenating
                    if all_pred_kps:
                        kps_device = all_pred_kps[0].device
                        pred_kps_all = torch.cat([kps.to(kps_device, non_blocking=True) for kps in all_pred_kps], dim=0)
                        gt_kps_all = torch.cat([kps.to(kps_device, non_blocking=True) if isinstance(kps, torch.Tensor) else kps for kps in all_gt_kps], dim=0)
                        trg_kps_all = torch.cat([kps.to(kps_device, non_blocking=True) if isinstance(kps, torch.Tensor) else kps for kps in all_trg_kps], dim=0)
                    else:
                        pred_kps_all = torch.cat(all_pred_kps, dim=0)
                        gt_kps_all = torch.cat(all_gt_kps, dim=0)
                        trg_kps_all = torch.cat(all_trg_kps, dim=0)
                
                # Ensure n_pts is on the same device
                if all_n_pts:
                    n_pts_device = all_n_pts[0].device if isinstance(all_n_pts[0], torch.Tensor) else torch.device('cpu')
                    n_pts_all = torch.cat([n_pts.to(n_pts_device, non_blocking=True) if isinstance(n_pts, torch.Tensor) else n_pts for n_pts in all_n_pts], dim=0)
                else:
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
            # Transfer entire batch to GPU once at the start (efficient - single operation)
            # Use explicit device comparison to handle "cuda" vs "cuda:0" properly
            gpu_batch = {}
            for key, value in mini_batch.items():
                if isinstance(value, torch.Tensor):
                    # Explicit device comparison: check type and index separately
                    value_device = value.device
                    needs_transfer = (
                        value_device.type != device.type or
                        (value_device.index if value_device.index is not None else 0) != 
                        (device.index if device.index is not None else 0)
                    )
                    if needs_transfer:
                        gpu_batch[key] = value.to(device, non_blocking=True)
                    else:
                        gpu_batch[key] = value  # Already on correct device
                else:
                    gpu_batch[key] = value  # Keep non-tensors as-is
            
            # Ensure all async transfers complete before using tensors
            if device.type == 'cuda' and any(isinstance(v, torch.Tensor) and v.device.type == 'cuda' for v in gpu_batch.values()):
                torch.cuda.synchronize(device)
            
            # Use flow_downsampled if available (CorrespondenceDataset), otherwise use flow (old datasets)
            if 'flow_downsampled' in gpu_batch:
                flow_gt = gpu_batch['flow_downsampled']
            else:
                flow_gt = gpu_batch['flow']
            pred_flow = net(gpu_batch['trg_img'], gpu_batch['src_img'])

            estimated_kps = flow2kps(gpu_batch['trg_kps'], pred_flow, gpu_batch['n_pts'])

            # All tensors already on GPU in gpu_batch
            eval_result = evaluator.evaluate(estimated_kps, gpu_batch)
            
            Loss = EPE(pred_flow, flow_gt) 

            pck_array += eval_result['pck']

            running_total_loss += Loss.item()
            pbar.set_description(
                ' validation R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), Loss.item()))
        mean_pck = sum(pck_array) / len(pck_array)

    return running_total_loss / len(val_loader), mean_pck
