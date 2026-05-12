# Supplementary: Synthetic Pipelines, Variants, and Mixed Training Configurations

This note extracts the exact dataset-generation logic used in code for the two synthetic pipelines discussed in the paper:

- `synthetic`: the base procedural 3D pipeline (`SDF-Fractal3D`)
- `synthetic_2d_warp`: the 2D-warp ablation built from the same base pipeline

For code-generated feature-space, transfer-estimator, and training-recipe tables, see `docs/supplementary_feature_and_training_details.md`.

The implementation lives primarily in:

- `src/data/synth/datasets/OnlineGeometryDataset.py`
- `src/data/synth/datasets/processors/SyntheticCorrespondenceProcessor.py`
- `src/data/synth/datasets/CorrespondenceDataset.py`
- `src/data/synth/datasets/sampler.py`
- `src/configs/online_synth_configs/OnlineGeometryConfig.yaml`

## 1. Base 3D pipeline: `synthetic` / `SDF-Fractal3D`

Each training pair is generated on the fly in three stages:

1. sample scene geometry and camera/view parameters
2. render two views of the same underlying 3D scene
3. recover dense correspondence from matched 3D surface coordinates

### 1.1 Scene parameter sampling

For each object, the renderer samples a latent fractal parameter
\[
\mathbf{c} \in \mathbb{R}^4
\]
from a curvature-weighted sampler (`CurvatureMapSampler`). In the default configuration, the probability of selecting a cell in the precomputed parameter grid is uniform over curvature scores in a selected interval:
\[
p(\mathbf{c}) \propto \mathbf{1}\!\left[\kappa(\mathbf{c}) \in [\ell,\ell+s]\right],
\]
with `distribution: uniform`, `loc = 3`, and `scale = 9` in `OnlineGeometryConfig.yaml`.

The curvature score used by the sampler is
\[
\kappa = 2\frac{\mu \sigma}{\mu+\sigma}(1+r),
\]
where \(\mu\) and \(\sigma\) are stored curvature statistics and \(r\) is an edge/internal-surface ratio term (`CurvatureMapSampler.convert_curvature`).

The two camera angles are sampled as a bounded pair. Let
\[
v_x \sim p_x,\qquad v_y \sim p_y,
\]
then the two views are
\[
(\theta^{(1)}_x,\theta^{(1)}_y) = (c_x-v_x/2,\; c_y-v_y/2),
\]
\[
(\theta^{(2)}_x,\theta^{(2)}_y) = (c_x+v_x/2,\; c_y+v_y/2),
\]
where the shared pair center \((c_x,c_y)\) is sampled so both views remain inside fixed bounds (`AngleSampler.sample`).

With the default config:

- \(v_x \sim \mathcal{U}(-0.16667,\;0.16666)\)
- \(v_y \sim \mathcal{U}(-0.16667,\;0.16666)\)
- bounds are \((0.5,\;0.25)\) in units of \(\pi\)

The code converts these normalized angles to a 3D camera position by
\[
\phi_x = \pi \theta_x,\qquad \phi_y = -\pi \theta_y,
\]
\[
\mathbf{cam}(\theta_x,\theta_y; d)
=
d\begin{bmatrix}
\sin(\phi_x)\cos(\phi_y) \\
-\sin(\phi_y) \\
\cos(\phi_x)\cos(\phi_y)
\end{bmatrix},
\]
with \(d = 5\) (`convert_location`, `iViewDistance`).

Scale is sampled as a dependent pair:
\[
a \sim p_{\text{abs}},\qquad r \sim p_{\text{rel}},
\]
\[
s^{(1)} = a,\qquad s^{(2)} = a+r
\]
(`ScaleSampler.sample`). In the default config:

- \(a \sim \mathcal{U}(0.5,\;0.9)\)
- \(r \sim \mathcal{U}(-0.25,\;0)\)

The shader zoom parameter is then
\[
z^{(i)} = \frac{s^{(i)}}{0.36}.
\]

Lighting/material parameters are sampled independently per view:

- ambient in \([0.1, 0.4]\)
- diffuse in \([0.2, 0.4]\)
- specular in \([0.7, 0.99]\)
- specular exponent in \(\{32,\dots,74\}\)

The light azimuth is sampled from a Gaussian centered near the current view angle (`sample_light_and_material`).

### 1.2 Rendering

The renderer instantiates several procedural SDF objects (Julia-set, Mandelbulb, sphere, etc.), each with its own sampled latent and spatial offset, then renders two views of the same scene using GLSL shaders (`OnlineGeometryDataset.__getitem__`).

For each view, the renderer outputs at least:

- `geometry`: a dense 3D coordinate map \(\mathbf{X}^{(i)} \in \mathbb{R}^{H\times W\times 3}\)
- `normals`: a dense normal map \(\mathbf{N}^{(i)} \in \mathbb{R}^{H\times W\times 3}\)
- `object_id`: an integer object label map

Texturing and shading are then applied in `SyntheticCorrespondenceProcessor.process_scene`, producing RGB images `src_img` and `trg_img`.

### 1.3 Dense correspondence from rendered 3D geometry

Ground-truth flow is not taken from a projected camera transform. It is recovered by matching rendered 3D surface coordinates across views (`flow_by_coordinate_matching`).

Let
\[
\mathbf{X}^{(1)}(u_1),\quad \mathbf{X}^{(2)}(u_2) \in \mathbb{R}^3
\]
denote the 3D coordinates stored at source pixel \(u_1\) and target pixel \(u_2\). For each valid target pixel \(u_2\), the code finds the nearest source pixel
\[
u_1^\star = \arg\min_{u_1} \left\|\mathbf{X}^{(1)}(u_1) - \mathbf{X}^{(2)}(u_2)\right\|_2^2
\]
using FAISS, then accepts the match only if
\[
\left\|\mathbf{X}^{(1)}(u_1^\star) - \mathbf{X}^{(2)}(u_2)\right\|_2^2 \le 5\times 10^{-5}.
\]

The stored flow follows the convention "target to source":
\[
\mathbf{f}(u_2) = u_1^\star - u_2.
\]

Pixels with no valid 3D match are marked invalid by setting the flow to `inf`.

This means the base synthetic dataset contains:

- true multi-view 3D motion
- appearance changes from re-rendering under a new view/light configuration
- occlusions/disocclusions induced by view change

## 2. 2D-warp pipeline: `synthetic_2d_warp`

The `synthetic_2d_warp` dataset does **not** create a new scene generator. It first runs the full 3D pipeline above, computes the same dense flow field, and then replaces the target image with a pure image-space warp of the source (`CorrespondenceDataset._process_synthetic_batch`).

Let the base source image be \(I_s\) and the target-to-source flow be \(\mathbf{f}(u)\). The new target image is constructed by inverse warping:
\[
I_t^{2D}(u) = I_s\!\left(u + \mathbf{f}(u)\right),
\]
implemented with bilinear `grid_sample` in `_warp_src_with_flow`.

Equivalently, if \(\mathbf{m}(u)=u+\mathbf{f}(u)\) is the source lookup location for target pixel \(u\), then
\[
I_t^{2D}(u) = \text{bilinear-sample}(I_s,\mathbf{m}(u)).
\]

Invalid flow locations are mapped outside the image and therefore become zero-padded.

An optional swap augmentation (`synthetic_flow_warp_swap`) randomly exchanges source and target and inverts the flow, but for clean visualization it is best disabled.

### 2.1 What changes relative to the base 3D pipeline?

The flow field is the same dense field recovered from 3D geometry. The difference is only in image formation:

- `synthetic` uses two independently rendered views: \(I_s, I_t^{3D}\)
- `synthetic_2d_warp` uses \(I_s\) and a warped copy \(I_t^{2D}\)

So `synthetic_2d_warp` removes view-dependent effects such as:

- newly visible surfaces
- view-conditioned shading changes
- independent re-rendered texture/layout changes caused by camera change

while preserving the same displacement field used to define correspondence.

## 3. Concise paper-ready distinction

If you want a compact supplementary description, the implementation supports the following wording:

> **SDF-Fractal3D.** We render procedural signed-distance scenes composed of fractal primitives under two sampled camera views. Dense ground-truth flow is recovered by nearest-neighbor matching of rendered 3D surface coordinates across views, yielding target-to-source displacement at each valid target pixel.

> **SDF-Fractal3D (2D warp).** Starting from the same SDF-Fractal3D sample and the same dense target-to-source flow field, we replace the independently rendered target view with a bilinear inverse warp of the source image. This keeps the displacement field fixed while removing true multi-view appearance changes and occlusion effects.

## 4. Synthetic fractal variants

The paper uses a small family of closely related synthetic generators built from the same `synthetic` base config. Table 1 summarizes the exact dataset-level differences.

| Variant | Dataset config | Exact change relative to base `synthetic` | Practical effect |
| --- | --- | --- | --- |
| `SDF-Fractal3D` | `src/configs/lightning/datasets/synthetic.yaml` | Base config: `random_flip = 0.0`; `angle_sampler.x_components = {distribution: uniform, loc: -0.16667, scale: 0.33333}`; `angle_sampler.y_components = {distribution: uniform, loc: -0.16667, scale: 0.33333}`; `scale_sampler.abs_components = {distribution: uniform, loc: 0.5, scale: 0.4}`; `scale_sampler.rel_components = {distribution: uniform, loc: -0.25, scale: 0.25}` | Full procedural 3D re-rendering with moderate view change and zoom change. |
| `SDF-Fractal2D` | `src/configs/lightning/datasets/synthetic_2d_warp_rc.yaml` | Keeps the same geometry config, but sets `synthetic_flow_warp: true` and `synthetic_flow_warp_swap: true` | Replaces the independently rendered target with a bilinear warp of the source using the same target-to-source flow. |
| `SDF-Fractal3D (small zoom)` | `src/configs/lightning/datasets/synthetic_rc_small_zoom.yaml` | Overrides `angle_sampler.{x,y}_components` to `{loc: 0, scale: 0.01}`; overrides `scale_sampler.abs_components.scale` to `0.2`; overrides `scale_sampler.rel_components` to `{loc: -0.1, scale: 0.2}` | Near-zero camera motion with a narrower zoom range. |
| `SDF-Fractal3D (large zoom)` | `src/configs/lightning/datasets/synthetic_rc_large_zoom.yaml` | Overrides `angle_sampler.{x,y}_components` to `{loc: 0, scale: 0.01}`; overrides `scale_sampler.abs_components.scale` to `0.5`; overrides `scale_sampler.rel_components` to `{loc: -0.35, scale: 0.5}` | Near-zero camera motion with substantially larger zoom variation. |
| `SDF-Fractal3D (random flipping)` | `src/configs/lightning/datasets/synthetic_rc_random_flipping.yaml` | Sets `geometry_config_overrides.random_flip: 0.5` | Adds left-right flipping to otherwise standard synthetic sampling. |

For the supplement, a concise description that matches the configs is:

> **Synthetic variant family.** All synthetic variants share the same procedural SDF-fractal renderer and dense 3D correspondence extraction pipeline. The ablations differ only through dataset-level overrides: either replacing the target image with a 2D inverse warp (`synthetic_2d_warp`), constraining the angle sampler and zoom sampler (`small_zoom`, `large_zoom`), or enabling random image flips (`random_flipping`).

## 5. Mixed dataset training configurations

All RC mixed-dataset configs share the same high-level training-data loader settings:

| Setting | Value |
| --- | --- |
| `split` | `train` |
| `size` | `[512, 512]` |
| `downsample_flow` | `32` |
| `max_kps` | `null` |
| `dense_kps_use_all` | `true` |
| `epoch_size` | `null` |
| `verbose`, `debug` | `false`, `false` |
| mixing seed | `42` for all `*_rc.yaml` configs |

### 5.1 Real + synthetic RC mixtures

| Config | Mixture ratio | Dataset-specific overrides that differ from the common settings |
| --- | --- | --- |
| `spair_synthetic_30_70_rc.yaml` | `30%` SPair, `70%` synthetic | SPair: `datapath=./models/Datasets_CATs`, `normalize_images=false`. Synthetic: `geometry_config_path=OnlineGeometryConfig.yaml`, `processor_config_path=OnlineProcessorConfig.yaml`, `normalize_images=false`, `geometry_config_overrides=null`. |
| `spair_synthetic_rc.yaml` | `50%` SPair, `50%` synthetic | Same overrides as above. |
| `spair_synthetic_70_30_rc.yaml` | `70%` SPair, `30%` synthetic | Same overrides as above. |
| `sintel_synthetic_30_70_rc.yaml` | `30%` Sintel, `70%` synthetic | Sintel: `sintel_root=/home/slow1/Data`, `pass_name=clean`, `reverse_flow=true`, `normalize_images=true`. Synthetic override block matches the SPair+synthetic configs. |
| `sintel_synthetic_70_30_rc.yaml` | `70%` Sintel, `30%` synthetic | Same overrides as above. |
| `flyingthings_synthetic_30_70_rc.yaml` | `30%` FlyingThings, `70%` synthetic | FlyingThings: `datapath=/home/slow1/Data/FlyingThings3D_Pytorch`, `reverse_flow=true`, `normalize_images=true`. Synthetic override block matches the SPair+synthetic configs. |
| `flyingthings_synthetic_70_30_rc.yaml` | `70%` FlyingThings, `30%` synthetic | Same overrides as above. |
| `pointodyssey_synthetic_30_70_rc.yaml` | `30%` PointOdyssey, `70%` synthetic | PointOdyssey: `dataset_location=/home/slow1/Data/PointOdyssey`, `pointodyssey_sequence_length=2`, `pointodyssey_num_pts_to_track=32`, `pointodyssey_strides=[1,2,4]`, `reverse_flow=true`, `use_all_valid=true`, `pointodyssey_disable_motion_filter=true`, `normalize_images=true`. Synthetic override block matches the SPair+synthetic configs. |
| `pointodyssey_synthetic_50_50_rc.yaml` | `50%` PointOdyssey, `50%` synthetic | Same overrides as above. |
| `pointodyssey_synthetic_70_30_rc.yaml` | `70%` PointOdyssey, `30%` synthetic | Same overrides as above. |
| `mixed_5datasets_20pct_rc.yaml` | `20%` each of Sintel, synthetic, SPair, FlyingThings, PointOdyssey | Combines the same per-dataset override blocks listed above into a uniform five-way mixture. |

### 5.2 SPair + ImageNet2DWarp mixtures

These configs are separate from the SDF-fractal generator, but they are useful to document alongside the synthetic mixtures because they are the main 2D-only mixed-data controls.

| Config | Mixture ratio | Dataset-specific overrides |
| --- | --- | --- |
| `spair_2d_warp_30_70_rc.yaml` | `30%` SPair, `70%` ImageNet2DWarp | ImageNet2DWarp uses `datapath=hf://ILSVRC/imagenet-1k`, `hf_streaming=false`, `hf_cache_dir=/home/slow1/Data/hf_cache/datasets`, `normalize_images=true`, `rotation_range=[-30,30]`, `scale_range=[0.5,2.5]`, `translation_range=[-0.1,0.1]`, `shear_range=[-0.2,0.2]`, `cache_warp_params=true`, `cache_dir=./cache/imagenet2dwarp`, `seed=42`. SPair uses the same override block as above. |
| `spair_2d_warp_50_50_rc.yaml` | `50%` SPair, `50%` ImageNet2DWarp | Same overrides as above. |
| `spair_2d_warp_70_30_rc.yaml` | `70%` SPair, `30%` ImageNet2DWarp | Same overrides as above. |

For paper text, a concise summary is:

> **Mixed-data training.** All mixed RC training datasets use `512\times512` crops, dense correspondence supervision with `downsample_flow=32`, and deterministic mixture sampling with seed 42. We instantiate pairwise mixtures of real and synthetic data by changing only the dataset weights, while keeping each constituent dataset's native normalization and loader-specific options fixed.

## 6. Code references

- 3D scene sampling and rendering:
  - `src/data/synth/datasets/OnlineGeometryDataset.py`
- Camera and scale pair samplers:
  - `src/data/synth/datasets/sampler.py`
- Default parameter values:
  - `src/configs/online_synth_configs/OnlineGeometryConfig.yaml`
- Dense 3D correspondence extraction:
  - `src/flow.py`
- 2D warp ablation logic:
  - `src/data/synth/datasets/CorrespondenceDataset.py`
- Synthetic dataset configs:
  - `src/configs/lightning/datasets/synthetic.yaml`
  - `src/configs/lightning/datasets/synthetic_2d_warp_rc.yaml`
  - `src/configs/lightning/datasets/synthetic_rc_small_zoom.yaml`
  - `src/configs/lightning/datasets/synthetic_rc_large_zoom.yaml`
  - `src/configs/lightning/datasets/synthetic_rc_random_flipping.yaml`
- Mixed dataset configs:
  - `src/configs/lightning/datasets/spair_synthetic_rc.yaml`
  - `src/configs/lightning/datasets/spair_synthetic_30_70_rc.yaml`
  - `src/configs/lightning/datasets/spair_synthetic_70_30_rc.yaml`
  - `src/configs/lightning/datasets/sintel_synthetic_30_70_rc.yaml`
  - `src/configs/lightning/datasets/sintel_synthetic_70_30_rc.yaml`
  - `src/configs/lightning/datasets/flyingthings_synthetic_30_70_rc.yaml`
  - `src/configs/lightning/datasets/flyingthings_synthetic_70_30_rc.yaml`
  - `src/configs/lightning/datasets/pointodyssey_synthetic_30_70_rc.yaml`
  - `src/configs/lightning/datasets/pointodyssey_synthetic_50_50_rc.yaml`
  - `src/configs/lightning/datasets/pointodyssey_synthetic_70_30_rc.yaml`
  - `src/configs/lightning/datasets/mixed_5datasets_20pct_rc.yaml`
  - `src/configs/lightning/datasets/spair_2d_warp_30_70_rc.yaml`
  - `src/configs/lightning/datasets/spair_2d_warp_50_50_rc.yaml`
  - `src/configs/lightning/datasets/spair_2d_warp_70_30_rc.yaml`
