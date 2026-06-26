"""Generate GLU-Net + FlowFormer configs for ALL tss_* kubric motion variants.

CATs (a ceiling-y semantic matcher) learned from every kubric variant, so it couldn't
discriminate between motion modes. GLU-Net/FlowFormer are the discriminating flow models,
and they were only ever run on ns4 (+ glunet on some zoom). This fills the gap: train both
flow models on every controlled kubric motion variant and eval on TSS + TAP-Vid + kitti, so
we can finally see whether high-motion kubric trains the flow models better than near-static
(the motion-coverage prediction) or whether they all sit near the floor.

Templates: src/configs/lightning/ns4_{glunet,flowformer}_tt_tssgrid.yaml (kubric_intervention).
Output: tap_vid_probe/configs/kubvar/kv_<variant>_<model>_<regime>.yaml   (32 configs)
Run: python tap_vid_probe/gen_kubric_variant_grid.py
"""
import copy, os, yaml

REPO = "/home/spencer/Projects/OnlineSyntheticCorrespondence"
OUT = os.path.join(REPO, "tap_vid_probe/configs/kubvar")
DS = "/mnt/nvme_1tb_a/kubric_interventions/datasets/tss_{}_1000"
# the 8 controlled TSS motion variants (low->high motion)
VARIANTS = ["ns4", "so1", "camonly", "zoom1obj", "zoom1obj_focal", "zoom2obj", "zoom1obj_big", "mm4"]
TEMPLATE = {"glunet": "ns4_glunet_tt_tssgrid", "flowformer": "ns4_flowformer_tt_tssgrid"}
REGIME = {  # model -> regime -> model-block flag overrides
    "glunet":     {"tt": {"pretrained_backbone": True, "freeze": True},
                   "tf": {"pretrained_backbone": True, "freeze": False}},
    "flowformer": {"tt": {"pretrain": True, "freeze": True},
                   "tf": {"pretrain": True, "freeze": False}},
}


def eval_block(base_eval):
    e = dict(base_eval)
    e["eval_benchmarks"] = ["tss", "tapvid_davis", "kitti2015", "kitti2012"]
    e["eval_alphas"] = [0.05, 0.03, 0.05, 0.05]
    e["tss_root"] = "/home/spencer/Data/correspondence/TSS_CVPR2016"
    e["kitti_root"] = "/home/spencer/Data/correspondence/kitti"
    e["kitti_val_use_full_training"] = True   # match the working ft_* configs
    e["tapvid_davis_root"] = "/mnt/nvme_1tb_a/tapvid/probe_cache"
    e["val_datasets"] = {
        "tapvid_davis": {"tapvid_stride": 6, "tapvid_frame_step": 5, "tapvid_min_pts": 1,
                         "reverse_flow": True, "normalize_images": True},
        "kitti2015": {"split": "val", "normalize_images": True},
        "kitti2012": {"split": "val", "normalize_images": True},
    }
    e["val_batch_size"] = 2          # low-RAM (avoid the host-OOM when concurrent)
    e["val_num_workers"] = 2
    e["prefetch_factor"] = 2
    return e


def main():
    os.makedirs(OUT, exist_ok=True)
    names = []
    for model, tmpl in TEMPLATE.items():
        base = yaml.safe_load(open(os.path.join(REPO, f"src/configs/lightning/{tmpl}.yaml")))
        for variant in VARIANTS:
            for regime, flags in REGIME[model].items():
                c = copy.deepcopy(base)
                c["dataset"]["datapath"] = DS.format(variant)
                c["model"].update(flags)
                c["evaluation"] = eval_block(base["evaluation"])
                c["training"]["n_threads"] = 2          # low-RAM
                c["training"]["eval_initial"] = True
                c["training"]["check_val_every_n_epoch"] = 1   # eval every epoch (catch early/sharp peaks)
                c["paths"]["snapshots"] = f"/mnt/nvme_1tb_a/snapshots/kubvar/{variant}_{model}_{regime}"
                c["paths"]["save_epoch_checkpoints"] = False
                name = f"kv_{variant}_{model}_{regime}"
                yaml.dump(c, open(os.path.join(OUT, name + ".yaml"), "w"),
                          default_flow_style=False, sort_keys=False)
                names.append(name)
    print(f"wrote {len(names)} configs to {OUT}")
    for n in names:
        print(" ", n)


if __name__ == "__main__":
    main()
