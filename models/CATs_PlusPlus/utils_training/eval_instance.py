"""Instance-based evaluator for multiple benchmarks during training"""
from skimage import draw
import numpy as np
import torch

from . import utils


class EvaluatorInstance:
    r"""Instance-based evaluator that supports multiple benchmarks simultaneously"""
    
    def __init__(self, benchmark, alpha=0.1):
        self.benchmark = benchmark
        self.alpha = alpha
        if benchmark == 'caltech':
            self.eval_func = self.eval_mask_transfer
        else:
            self.eval_func = self.eval_kps_transfer

    def evaluate(self, prd_kps, batch):
        r"""Compute evaluation metric"""
        return self.eval_func(prd_kps, batch)

    def eval_kps_transfer(self, prd_kps, batch):
        r"""Compute percentage of correct key-points (PCK) based on prediction"""

        easy_match = {'src': [], 'trg': [], 'dist': []}
        hard_match = {'src': [], 'trg': []}

        pck = []
        for idx, (pk, tk) in enumerate(zip(prd_kps, batch['src_kps'])):
            thres = batch['pckthres'][idx]
            npt = batch['n_pts'][idx]
            npt_val = npt.item()
            
            if npt_val > 0:
                # Filter out invalid keypoints (marked with -1)
                pk_slice = pk[:, :npt]
                tk_slice = tk[:, :npt]
                valid_mask = (pk_slice[0, :] >= 0) & (pk_slice[1, :] >= 0) & (tk_slice[0, :] >= 0) & (tk_slice[1, :] >= 0)
                
                if valid_mask.sum() > 0:
                    # Only evaluate valid keypoints
                    pk_valid = pk_slice[:, valid_mask]
                    tk_valid = tk_slice[:, valid_mask]
                    _, correct_ids, _ = self.classify_prd(pk_valid, tk_valid, thres)
                    pck.append((len(correct_ids) / valid_mask.sum().item()) * 100)
                else:
                    # All keypoints are invalid
                    pck.append(0.0)
            else:
                # No valid points - PCK is undefined, set to 0
                pck.append(0.0)

        eval_result = {'pck': pck}

        return eval_result

    def eval_kps_transfer_with_correct(self, prd_kps, batch):
        r"""Compute percentage of correct key-points (PCK) based on prediction"""

        easy_match = {'src': [], 'trg': [], 'dist': []}
        hard_match = {'src': [], 'trg': []}

        pck = []
        correct_id_list = []
        for idx, (pk, tk) in enumerate(zip(prd_kps, batch['src_kps'])):
            thres = batch['pckthres'][idx]
            npt = batch['n_pts'][idx]
            npt_val = npt.item()
            
            if npt_val > 0:
                # Filter out invalid keypoints (marked with -1)
                pk_slice = pk[:, :npt]
                tk_slice = tk[:, :npt]
                valid_mask = (pk_slice[0, :] >= 0) & (pk_slice[1, :] >= 0) & (tk_slice[0, :] >= 0) & (tk_slice[1, :] >= 0)
                
                if valid_mask.sum() > 0:
                    # Only evaluate valid keypoints
                    pk_valid = pk_slice[:, valid_mask]
                    tk_valid = tk_slice[:, valid_mask]
                    _, correct_ids, _ = self.classify_prd(pk_valid, tk_valid, thres)
                    correct_id_list.append(correct_ids)
                    pck.append((len(correct_ids) / valid_mask.sum().item()) * 100)
                else:
                    # All keypoints are invalid
                    correct_id_list.append([])
                    pck.append(0.0)
            else:
                # No valid points - PCK is undefined, set to 0
                correct_id_list.append([])
                pck.append(0.0)

        eval_result = {'pck': pck}

        return eval_result, correct_id_list

    def eval_mask_transfer(self, prd_kps, batch):
        r"""Compute LT-ACC and IoU based on transferred points"""

        ltacc = []
        iou = []

        for idx, prd in enumerate(prd_kps):
            trg_n_pts = (batch['trg_kps'][idx] > 0)[0].sum()
            prd_kp = prd[:, :batch['n_pts'][idx]]
            trg_kp = batch['trg_kps'][idx][:, :trg_n_pts]

            imsize = list(batch['trg_img'].size())[2:]
            trg_xstr, trg_ystr = self.pts2ptstr(trg_kp)
            trg_mask = self.ptstr2mask(trg_xstr, trg_ystr, imsize[0], imsize[1])
            prd_xstr, pred_ystr = self.pts2ptstr(prd_kp)
            prd_mask = self.ptstr2mask(prd_xstr, pred_ystr, imsize[0], imsize[1])

            ltacc.append(self.label_transfer_accuracy(prd_mask, trg_mask))
            iou.append(self.intersection_over_union(prd_mask, trg_mask))

        eval_result = {'ltacc': ltacc,
                       'iou': iou}

        return eval_result

    def classify_prd(self, prd_kps, trg_kps, pckthres):
        r"""Compute the number of correctly transferred key-points"""
        l2dist = (prd_kps - trg_kps).pow(2).sum(dim=0).pow(0.5)
        thres = pckthres.expand_as(l2dist).float() * self.alpha
        correct_pts = torch.le(l2dist, thres)

        correct_ids = utils.where(correct_pts == 1)
        incorrect_ids = utils.where(correct_pts == 0)
        correct_dist = l2dist[correct_pts]

        return correct_dist, correct_ids, incorrect_ids

    def intersection_over_union(self, mask1, mask2):
        r"""Computes IoU between two masks"""
        rel_part_weight = torch.sum(torch.sum(mask2.gt(0.5).float(), 2, True), 3, True) / \
                          torch.sum(mask2.gt(0.5).float())
        part_iou = torch.sum(torch.sum((mask1.gt(0.5) & mask2.gt(0.5)).float(), 2, True), 3, True) / \
                   torch.sum(torch.sum((mask1.gt(0.5) | mask2.gt(0.5)).float(), 2, True), 3, True)
        weighted_iou = torch.sum(torch.mul(rel_part_weight, part_iou)).item()

        return weighted_iou

    def label_transfer_accuracy(self, mask1, mask2):
        r"""LT-ACC measures the overlap with emphasis on the background class"""
        return torch.mean((mask1.gt(0.5) == mask2.gt(0.5)).double()).item()

    def pts2ptstr(self, pts):
        r"""Convert tensor of points to string"""
        x_str = str(list(pts[0].cpu().numpy()))
        x_str = x_str[1:len(x_str)-1]
        y_str = str(list(pts[1].cpu().numpy()))
        y_str = y_str[1:len(y_str)-1]

        return x_str, y_str

    def pts2mask(self, x_pts, y_pts, shape):
        r"""Build a binary mask tensor base on given xy-points"""
        x_idx, y_idx = draw.polygon(x_pts, y_pts, shape)
        mask = np.zeros(shape, dtype=np.bool)
        mask[x_idx, y_idx] = True

        return mask

    def ptstr2mask(self, x_str, y_str, out_h, out_w):
        r"""Convert xy-point mask (string) to tensor mask"""
        x_pts = np.fromstring(x_str, sep=',')
        y_pts = np.fromstring(y_str, sep=',')
        mask_np = self.pts2mask(y_pts, x_pts, [out_h, out_w])
        mask = torch.tensor(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).float()

        return mask

    ############# Motion Aware Section ########
    def eval_kps_transfer_with_motion_prior(self, prd_kps, batch, min_motion_pixels=5.0):
        r"""Compute PCK but only on keypoints with motion >= min_motion_pixels"""
        pck = []
        motion_stats = {'total_moving': 0, 'total_static': 0}
        
        for idx, (pk, tk, trk) in enumerate(zip(prd_kps, batch['src_kps'], batch['trg_kps'])):
            thres = batch['pckthres'][idx]
            npt = batch['n_pts'][idx]
            
            # Filter out invalid keypoints (marked with -1) first
            pk_slice = pk[:, :npt]
            tk_slice = tk[:, :npt]
            trk_slice = trk[:, :npt]
            valid_mask = (pk_slice[0, :] >= 0) & (pk_slice[1, :] >= 0) & (tk_slice[0, :] >= 0) & (tk_slice[1, :] >= 0) & (trk_slice[0, :] >= 0) & (trk_slice[1, :] >= 0)
            
            if valid_mask.sum() == 0:
                # All keypoints are invalid
                pck.append(0.0)
                continue
            
            # Compute motion magnitude for each valid keypoint
            motion = trk_slice[:, valid_mask] - tk_slice[:, valid_mask]
            motion_magnitude = torch.norm(motion, dim=0)
            
            # Only evaluate keypoints with significant motion
            motion_mask = motion_magnitude >= min_motion_pixels
            
            motion_stats['total_moving'] += motion_mask.sum().item()
            motion_stats['total_static'] += (~motion_mask).sum().item()
            
            if motion_mask.sum() > 0:
                # Only evaluate on moving keypoints
                pk_moving = pk_slice[:, valid_mask][:, motion_mask]
                tk_moving = tk_slice[:, valid_mask][:, motion_mask]
                _, correct_ids, _ = self.classify_prd(pk_moving, tk_moving, thres)
                
                pck.append((len(correct_ids) / motion_mask.sum().item()) * 100)
            else:
                # No motion in this sample
                pck.append(0.0)
        
        eval_result = {'pck': pck, 'motion_stats': motion_stats}
        return eval_result
    
    def eval_kps_transfer_motion_binned(self, prd_kps, batch):
        r"""Compute PCK for different motion ranges (standard in optical flow evaluation)"""
        # Motion bins: small (< 5px), medium (5-20px), large (> 20px)
        motion_bins = {
            'small': (0, 5),
            'medium': (5, 20),
            'large': (20, float('inf'))
        }
        
        results = {bin_name: {'pck': [], 'count': 0} for bin_name in motion_bins.keys()}
        
        for idx, (pk, tk, trk) in enumerate(zip(prd_kps, batch['src_kps'], batch['trg_kps'])):
            thres = batch['pckthres'][idx]
            npt = batch['n_pts'][idx]
            
            # Filter out invalid keypoints (marked with -1) first
            pk_slice = pk[:, :npt]
            tk_slice = tk[:, :npt]
            trk_slice = trk[:, :npt]
            valid_mask = (pk_slice[0, :] >= 0) & (pk_slice[1, :] >= 0) & (tk_slice[0, :] >= 0) & (tk_slice[1, :] >= 0) & (trk_slice[0, :] >= 0) & (trk_slice[1, :] >= 0)
            
            if valid_mask.sum() == 0:
                # All keypoints are invalid, skip this sample
                continue
            
            # Compute motion magnitude for each valid keypoint
            motion = trk_slice[:, valid_mask] - tk_slice[:, valid_mask]
            motion_magnitude = torch.norm(motion, dim=0)
            
            # Classify each valid keypoint into motion bins
            for bin_name, (min_motion, max_motion) in motion_bins.items():
                bin_mask = (motion_magnitude >= min_motion) & (motion_magnitude < max_motion)
                
                if bin_mask.sum() > 0:
                    pk_bin = pk_slice[:, valid_mask][:, bin_mask]
                    tk_bin = tk_slice[:, valid_mask][:, bin_mask]
                    _, correct_ids, _ = self.classify_prd(pk_bin, tk_bin, thres)
                    
                    results[bin_name]['pck'].append((len(correct_ids) / bin_mask.sum().item()) * 100)
                    results[bin_name]['count'] += bin_mask.sum().item()
        
        # Compute average PCK for each bin
        eval_result = {
            'pck_by_motion': {
                bin_name: {
                    'mean_pck': sum(pcks) / len(pcks) if pcks else 0.0,
                    'count': results[bin_name]['count']
                }
                for bin_name, pcks in [(k, v['pck']) for k, v in results.items()]
            }
        }
        
        return eval_result
    ############# End Motion Aware Section ########


class MultiBenchmarkEvaluator:
    r"""Manages multiple evaluator instances for different benchmarks"""
    
    def __init__(self, benchmarks_and_alphas):
        """
        Initialize multiple evaluators
        
        Args:
            benchmarks_and_alphas: List of tuples (benchmark, alpha) or dict {benchmark: alpha}
        """
        self.evaluators = {}
        
        if isinstance(benchmarks_and_alphas, dict):
            for benchmark, alpha in benchmarks_and_alphas.items():
                self.evaluators[benchmark] = EvaluatorInstance(benchmark, alpha)
        else:
            for benchmark, alpha in benchmarks_and_alphas:
                self.evaluators[benchmark] = EvaluatorInstance(benchmark, alpha)
    
    def evaluate(self, benchmark, prd_kps, batch):
        """Evaluate on a specific benchmark"""
        if benchmark not in self.evaluators:
            raise ValueError(f"Benchmark '{benchmark}' not initialized. Available: {list(self.evaluators.keys())}")
        
        return self.evaluators[benchmark].evaluate(prd_kps, batch)
    
    def evaluate_all(self, prd_kps, batch):
        """Evaluate on all benchmarks"""
        results = {}
        for benchmark, evaluator in self.evaluators.items():
            results[benchmark] = evaluator.evaluate(prd_kps, batch)
        return results
    
    def get_available_benchmarks(self):
        """Get list of available benchmarks"""
        return list(self.evaluators.keys())
