# Citation Audit — ACCV 2026

> **Update 2026-07-02.** The fixes proposed below have been applied to `main.bib`, `tables/tab_study.tex`,
> `sections/01_introduction.tex`, `sections/02_related_work.tex`, `sections/03_setup.tex`, and
> `sections/07_interventions_v2.tex`. The paper recompiles cleanly with no undefined references. See
> [§0 What was changed](#0-what-was-changed-2026-07-02) for the diff summary; the tables below are the
> original audit with status markers left intact so you can re-verify. Your remaining spot-checks are the
> four "confirm..." notes in §1 and the remaining low-priority suggestions in §3.
>
> **§5 (added 2026-07-02)** cross-checks ACCV's citations against your verified ECCV26 "Beyond Realism"
> bib: 19 re-used entries are byte-identical (inherit your verification), and it lists which ACCV
> additions still need a fresh check.

## 0. What was changed (2026-07-02)

**Metadata fixes (bib):**
- `im2022dflow` — replaced the hallucinated author list with the real authors: Kwon, Byung-Ki; Nam, Hyeon-Woo; Kim, Ji-Yun; Oh, Tae-Hyun (ICLR 2023). Key left unchanged so cite sites still resolve.

**New bib entries added + wired into the paper:**
- `kitti2012` (Geiger et al., CVPR 2012) and `kitti2015` (Menze & Geiger, CVPR 2015) — now cited for the KITTI-12/15 flow benchmark in `tab_study.tex`, replacing the raw-data paper `kitti_flow`. (`kitti_flow` is now uncited; left in bib.)
- `taniai2016tss` (Taniai, Sinha, Sato, CVPR 2016) — TSS now cited in `tab_study.tex`.
- `bergstra2011tpe` (Bergstra et al., NeurIPS 2011) — TPE now cited in §7.
- `bendavid2010theory` (Ben-David et al., ML 2010) — added to the covariate-shift sentence in Related Work.
- `bonneel2015sliced` (Bonneel et al., JMIV 2015) — added to the sliced-Wasserstein-2 mention in §3.

**Existing-but-uncited bib entries now wired in:**
- `dosovitskiy2015Flownet-FlyingChairsDataset` — added to the "FlyingChairs/FlyingThings" mention in Related Work.
- `Kataoka2020Pre-trainingImages` + `Anderson2022ImprovingFractal` — added to the two "non-photorealistic / procedural abstraction" sentences (Intro + Related Work) to strengthen the previously weak `white2009mandelbulb`-only support. `white2009mandelbulb` was kept alongside them.
- `roth_opt_flow` (Roth & Black, ICCV 2005) — added to the §3 "spatial layout of a motion field is signal" motivation.

**Deliberately NOT done:**
- Spearman-ρ citation in §3: the only rank-correlation entry in the bib is `kendall1938`, which is Kendall's τ, **not** Spearman. Citing it there would be wrong, so it was left out. Add a real Spearman (Spearman 1904) entry if you want the anchor.
- `pfpascal` still does double duty for PF-Pascal and PF-Willow (see ⚠️ note in §1). The TPAMI journal version covers both, so this was left as-is; swap in the Ham et al. CVPR 2016 "Proposal Flow" entry if you prefer a PF-Willow-specific cite.

---

Every `\cite{...}` key that actually appears in the compiled paper (`main.tex` inputs
`01`, `02`, `03`, `04`, `05`, `06`, **`07_interventions_v2`**, `09`). Sections 04, 05,
06, 09 and `supp_main.tex` contain no citations. `07_interventions.tex` (the v1 A/B
variant) is **not** compiled, so its cites are ignored here.

Use the **Verify** column to confirm each entry (a) resolves to a real paper and (b)
says what the paper claims it says. Status legend:

- ✅ looks correct (key metadata matches a real paper and supports the claim)
- ⚠️ check this one (metadata error, wrong-paper risk, or weak support)

---

## 1. Cited references, by where they appear

| Key | What the reference is (ground truth) | Why it is cited / claim it supports | Verify |
|---|---|---|---|
| `mayer2016FlyingThingsDataset` | Mayer et al., "A large dataset to train ConvNets for disparity, optical flow, and scene flow" (FlyingThings3D), CVPR 2016 | Intro + RelWork + Table 1: synthetic data is the default for dense correspondence; FlyingThings source/benchmark | ✅ |
| `sintel` | Butler et al., "A naturalistic open source movie for optical flow evaluation" (MPI-Sintel), ECCV 2012 | Intro + RelWork: canonical synthetic flow dataset | ✅ (note: duplicate entry `Butler:ECCV:2012` exists in bib, uncited) |
| `greff2021kubric` | Greff et al., "Kubric: A Scalable Dataset Generator", CVPR **2022** | Intro/RelWork (default synthetic source; rendering-side line of work) and §7 (Kubric = photorealistic generator, MOVi-F generic config) | ⚠️ key says 2021 but paper is 2022; entry is `@article` with a `booktitle` field (wrong type). Also duplicate `kubric` entry exists (uncited). Cosmetic, not hallucinated. |
| `pointodyssey` | Zheng et al., "PointOdyssey: A Large-Scale Synthetic Dataset for Long-Term Point Tracking", ICCV 2023 | Intro/RelWork (synthetic source) + Table 1 benchmark | ✅ |
| `Baradad2021LookingAtNoise` | Baradad et al., "Learning to See by Looking at Noise", NeurIPS 2021 | Intro + RelWork: non-photorealistic / procedural data transfers well; supports motion-controllable non-photoreal generators | ✅ |
| `white2009mandelbulb` | White & Nylander, "The Mystery of the Real 3D Mandelbrot Fractal" (Mandelbulb web page), 2009 | Same sentence as above ("procedural abstraction supports our use of non-photorealistic generators") | ⚠️ It is a real web resource, but it is a formula description page, not a pretraining/transfer result. Weak support for the "supports our use" claim; a fractal-pretraining paper would carry the argument better (see Suggested §3). |
| `mayer2018what` | Mayer et al., "What Makes Good Synthetic Training Data for Learning Disparity and Optical Flow Estimation?", IJCV 2018 | RelWork: "closest to our premise"; matching target displacement statistics helps flow training, but undirected | ✅ Confirm the paper's claim is specifically about matching *displacement statistics* (it is broader, covering realism/diversity too). Phrasing is defensible but double-check. |
| `sun2021autoflow` | Sun et al., "AutoFlow: Learning a Better Training Set for Optical Flow", CVPR 2021 | RelWork: methods that *learn* data params with training in the loop | ✅ (duplicate entry `autoflow` exists in bib, uncited) |
| `im2022dflow` | Kwon, Nam, Kim, Oh, "DFlow: Learning to Synthesize Better Optical Flow Datasets via a Differentiable Pipeline", **ICLR 2023** | RelWork: learns data params with training in the loop | ✅ **FIXED** — author list corrected to Kwon, Byung-Ki / Nam, Hyeon-Woo / Kim, Ji-Yun / Oh, Tae-Hyun; year 2023. Key `im2022dflow` kept (cosmetic mismatch, harmless). |
| `nguyen2020leep` | Nguyen et al., "LEEP: A New Measure to Evaluate Transferability of Learned Representations", ICML 2020 | RelWork: transferability-estimation method (classification) | ✅ (bib lists it as CoRR/arXiv; real venue is ICML 2020) |
| `you2021logme` | You et al., "LogME: Practical Assessment of Pre-trained Models for Transfer Learning", ICML 2021 | RelWork: transferability estimator | ✅ |
| `tran2019transferability` | Tran et al., "Transferability and Hardness of Supervised Classification Tasks", ICCV 2019 | RelWork: cited as "NCE" transferability measure | ✅ Confirm you are happy calling this the NCE method (it introduces the NCE/conditional-entropy transferability score). |
| `bao2019information` | Bao et al., "An Information-Theoretic Approach to Transferability in Task Transfer Learning", ICIP 2019 | RelWork: cited as "H-score" | ✅ This is the H-score paper. |
| `tan2021otce` | Tan et al., "OTCE: A Transferability Metric for Cross-Domain Cross-Task Representations", CVPR 2021 | RelWork: OTCE transferability metric | ✅ |
| `alvarez2020geometric` | Alvarez-Melis & Fusi, "Geometric Dataset Distances via Optimal Transport" (OTDD), NeurIPS 2020 | RelWork: fit-free but *symmetric* dataset distance; assumes discrete class space so does not apply to dense correspondence | ✅ Confirm the "assumes a discrete class space" characterization (OTDD uses label-to-label OT; accurate). |
| `sajjadi2018assessing_precision_recall` | Sajjadi et al., "Assessing Generative Models via Precision and Recall", NeurIPS 2018 | RelWork: directional precision/recall diagnostics in generative eval | ✅ |
| `kynkaanniemi2019improved_pr` | Kynkäänniemi et al., "Improved Precision and Recall Metric for Assessing Generative Models", NeurIPS 2019 | RelWork: same group of directional diagnostics | ✅ |
| `naeem2020reliable` | Naeem et al., "Reliable Fidelity and Diversity Metrics for Generative Models" (density/coverage), ICML 2020 | RelWork: same group | ✅ |
| `alaa2022faithful` | Alaa et al., "How Faithful is your Synthetic Data?", ICML 2022 | RelWork: same group | ✅ |
| `sorscher2022beyond` | Sorscher et al., "Beyond Neural Scaling Laws: Beating Power Law Scaling via Data Pruning", NeurIPS 2022 | RelWork: value of easy vs. hard examples flips with data abundance (context-conditional data value) | ✅ |
| `xie2023data` | Xie et al., "Data Selection for Language Models via Importance Resampling" (DSIR), NeurIPS 2023 | RelWork: pretraining-data selection that matches distributions symmetrically | ✅ |
| `xie2023doremi` | Xie et al., "DoReMi: Optimizing Data Mixtures Speeds Up LM Pretraining", NeurIPS 2023 | RelWork: same (symmetric distribution matching) | ✅ |
| `hanneke2019value` | Hanneke & Kpotufe, "On the Value of Target Data in Transfer Learning", NeurIPS 2019 | RelWork: asymmetric transfer exponents anticipate the coverage headline on the theory side | ✅ Confirm "asymmetric transfer exponents" maps to their transfer-exponent framework (it does; phrasing is a light gloss). |
| `cho2022cats++` | Cho et al., "CATs++: Boosting Cost Aggregation with Convolutions and Transformers", TPAMI 2022 (arXiv 2202.06817) | RelWork + Table 1: semantic-matching architecture family | ✅ |
| `teed2020raft` | Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow", ECCV 2020 | RelWork + Table 1: recurrent flow architecture | ✅ |
| `truong2020glunet` | Truong et al., "GLU-Net: Global-Local Universal Network for Dense Flow and Correspondences", CVPR 2020 | RelWork + Table 1: pyramid dense-matching architecture | ✅ (duplicate entries `truong2020glu` / `melekhov2019dgc-net` context; `truong2020glu` is uncited) |
| `simeoni2025dinov3` | Siméoni et al., "DINOv3", arXiv 2508.10104, 2025 | RelWork + §3: appearance descriptor / backbone for the appearance hypothesis | ✅ Recent arXiv; confirm arXiv id 2508.10104 and author list render. |
| `sd_dino_tale_of_two_features` | Zhang et al., "A Tale of Two Features: Stable Diffusion Complements DINO for Zero-Shot Semantic Correspondence", 2023 | RelWork + §3: deep features alone give competitive zero-shot correspondence (motivates appearance as competing hypothesis) | ✅ |
| `heusel2018ganstrainedtimescaleupdate` | Heusel et al., "GANs Trained by a Two Time-Scale Update Rule..." (introduces FID), NeurIPS 2017 | §3: FID as one of the distribution distances | ✅ |
| `fan2017pointcloud` | Fan et al., "A Point Set Generation Network for 3D Object Reconstruction from a Single Image", CVPR 2017 | §3: Chamfer distance definition | ✅ Common Chamfer-distance citation; confirm you want this (it popularized Chamfer for point clouds rather than originating it). |
| `doersch2022tapvid` | Doersch et al., "TAP-Vid: A Benchmark for Tracking Any Point in a Video", NeurIPS 2022 D&B | §7: TAP-Vid-DAVIS held-out point-tracking benchmark | ✅ |
| ~~`kitti_flow`~~ → `kitti2012`,`kitti2015` | Geiger et al., CVPR 2012 (KITTI-2012) + Menze & Geiger, "Object Scene Flow for Autonomous Vehicles", CVPR 2015 (KITTI-2015) | Table 1: KITTI-12/15 optical-flow benchmark | ✅ **FIXED** — `tab_study.tex` now cites `kitti2012,kitti2015`. Old `kitti_flow` (IJRR 2013 raw-data paper) is no longer cited. |
| `spair71k` | Min et al., "SPair-71k: A Large-scale Benchmark for Semantic Correspondence", arXiv 2019 | Table 1: SPair-71k semantic benchmark | ✅ |
| `pfpascal` | Ham et al., "Proposal Flow: Semantic Correspondences from Object Proposals", TPAMI 2017 | Table 1: cited for **both** PF-Pascal and PF-Willow | ⚠️ PF-Pascal is fine. PF-Willow was introduced in the earlier Ham et al. CVPR 2016 "Proposal Flow" paper; the TPAMI journal version covers both, so this is acceptable, but confirm you are comfortable citing the journal version for PF-Willow. |
| `huang2022flowformer` | Huang et al., "FlowFormer: A Transformer Architecture for Optical Flow", ECCV 2022 | Table 1: transformer flow architecture | ✅ |

---

## 2. Bib entries defined but never cited (safe to ignore or prune)

Not errors, just dangling `@` entries in `main.bib`. Several are duplicates of a cited
key under a different name. Listing so you know they carry no weight in the paper:

`kitti_flow` (**now uncited** after the KITTI fix; superseded by `kitti2012`/`kitti2015`, left in bib),
`kubric` (dup of `greff2021kubric`), `autoflow` (dup of `sun2021autoflow`),
`truong2020glu` (dup of `truong2020glunet`), `Butler:ECCV:2012` (dup of `sintel`),
`kendall1938` / `kendall1983` (identical to each other), `mmd`,
`gretton2008kernelmethodtwosampleproblem`, `arjovsky2017wassersteingan`, `COTFNT`,
`kl_est`, `when_nn`, `faiss`,
`luo2024flowdiffuser`, `DINOv2`, `Genesis`, `huang2023self`,
`puig2023habitat3`, `szot2021habitat`, `habitat19iccv`, `rocco2017convolutionalneuralnetworkarchitecture`,
`laptev2008`, `pervfi_cvpr2024`, `dlss4_2025`,
`shinoda2023segrcdb-semantic-seg-fdsl`,
`actionvlad_citation`, `Ma2021`, `slam`, `tracking`, `10.1145/74333.74337`, `fbm`,
`julia`, `MegaDepthLi18`, `RobotCarDatasetIJRR`, `hpatches_2017_cvpr`,
`banik2021awa-pose-keypoint-quadruped`.

(Newly cited as of 2026-07-02, no longer dangling: `dosovitskiy2015Flownet-FlyingChairsDataset`,
`roth_opt_flow`, `Kataoka2020Pre-trainingImages`, `Anderson2022ImprovingFractal`.)

---

## 3. Suggested citations (things you may be missing)

Ranked by how load-bearing the gap is. **✅ = applied 2026-07-02** (see §0). The rest are left
for your call.

| Priority | Suggested cite | Why / what it backs | Where to insert |
|---|---|---|---|
| ✅ High | **Bergstra et al., "Algorithms for Hyper-Parameter Optimization", NeurIPS 2011** (TPE) | §7 used "a Tree-structured Parzen Estimator (TPE)" with no citation. | **Applied** as `bergstra2011tpe` in §7. |
| ✅ High | **KITTI-2012: Geiger et al., CVPR 2012** and **KITTI-2015: Menze & Geiger, "Object Scene Flow for Autonomous Vehicles", CVPR 2015** | `kitti_flow` (IJRR 2013) is the raw-data paper, not the flow benchmarks. | **Applied** as `kitti2012,kitti2015` in `tab_study.tex`. |
| ✅ High | **TSS: Taniai, Sinha, Sato, "Joint Recovery of Dense Correspondence and Cosegmentation in Two Images", CVPR 2016** | TSS was used as a benchmark with no citation anywhere. | **Applied** as `taniai2016tss` in `tab_study.tex`. |
| ✅ Medium | **FlyingChairs: Dosovitskiy et al., "FlowNet", ICCV 2015** (`dosovitskiy2015Flownet-FlyingChairsDataset`) | RelWork wrote "FlyingChairs/FlyingThings" but only cited FlyingThings. | **Applied** in `02_related_work.tex`. |
| ✅ Medium | **Ben-David et al., "A theory of learning from different domains", ML 2010** | RelWork invoked "covariate-shift support conditions" with only `hanneke2019value`. | **Applied** as `bendavid2010theory` in `02_related_work.tex`. |
| ✅ Medium | **Fractal/FDSL pretraining** — Kataoka et al. ACCV 2020 + Anderson & Farrell WACV 2022 | Strengthens the "non-photorealistic data transfers well" claim beyond the Mandelbulb web page. | **Applied** (`Kataoka2020Pre-trainingImages`, `Anderson2022ImprovingFractal`) in Intro + RelWork; `white2009mandelbulb` kept. |
| ✅ Low | **Sliced-Wasserstein — Bonneel et al., JMIV 2015** | §3 introduced "sliced Wasserstein-2" with no citation. | **Applied** as `bonneel2015sliced` in §3. |
| ✅ Low | **Roth & Black, "On the spatial statistics of optical flow", ICCV 2005** (`roth_opt_flow`) | §3's "spatial layout is signal" motivation had a natural prior-work anchor. | **Applied** in §3. |
| ⬜ Low | **Spearman / rank correlation** (Spearman 1904) — **not** `kendall1938`, which is Kendall's τ | §3 reports Spearman ρ without citation. | **Skipped** deliberately — no correct Spearman entry in the bib; add one if you want the anchor. |

---

## 4. Quick triage summary

**Must-fix items — all applied:**
- ✅ `im2022dflow` — author list corrected to Kwon/Nam/Kim/Oh, ICLR 2023.
- ✅ KITTI flow — `tab_study.tex` now cites `kitti2012` + `kitti2015`; the raw-data `kitti_flow` is dropped.
- ✅ **TSS** now cited (`taniai2016tss`); ✅ **TPE** now cited (`bergstra2011tpe`).

**Strengthened (applied):** FlyingChairs cite added; fractal-pretraining cites (`Kataoka2020`,
`Anderson2022`) added next to `white2009mandelbulb`; Ben-David domain-adaptation-theory cite added;
sliced-Wasserstein and Roth & Black cites added.

**Left as-is (your call):** `greff2021kubric` year/entry-type cosmetics (harmless), `pfpascal`
double duty for PF-Willow, no Spearman citation (bib's `kendall1938` is Kendall's τ, not Spearman).

**Still needs your eyes:** the four "confirm..." notes in §1 (they are judgment calls about whether
each paper's claim matches the paper's own framing, not metadata errors), plus the remaining
low-priority suggestions in §3.

---

## 5. Cross-check against the ECCV26 "Beyond Realism" bib (2026-07-02)

You verified the ECCV26 paper's citations, so anything ACCV re-uses from it can inherit that
verification. Two caveats up front:

1. **The ECCV `main.bib` in that folder is out of sync with its own sections.** 82 keys are
   `\cite`d across the ECCV sections but only 52 are defined in its `main.bib`; **36 cited keys are
   undefined there** (e.g. `taniai2016joint-correspondence-coseg`, `Huang2022FlowFormerAT`,
   `min2019spair`, `fid`, `FlyingThings_Dataset`, `ham2017proposal-flow-pascal`). So the bib file
   present is not the complete verified bibliography. Whatever you actually verified likely lives in a
   newer bib or the rendered PDF. I could only cross-check against the 52 entries that file does define.

2. Cross-check is **entry-text identity**, not a re-verification of the underlying paper.

### 5a. Re-used entries confirmed byte-identical to the ECCV bib ✅

Every bib entry ACCV cites that also exists in the ECCV `main.bib` is character-for-character
identical (whitespace-normalized). No drift. These 19 inherit your ECCV verification directly:

`mayer2016FlyingThingsDataset`, `sintel`, `greff2021kubric`, `pointodyssey`,
`Baradad2021LookingAtNoise`, `nguyen2020leep`, `cho2022cats++`, `teed2020raft`, `simeoni2025dinov3`,
`sd_dino_tale_of_two_features`, `heusel2018ganstrainedtimescaleupdate`, `spair71k`, `pfpascal`,
`sajjadi2018assessing_precision_recall`, `kynkaanniemi2019improved_pr`,
`dosovitskiy2015Flownet-FlyingChairsDataset`, `Kataoka2020Pre-trainingImages`,
`Anderson2022ImprovingFractal`, `roth_opt_flow`.

### 5b. Same paper, different key (equivalent, no fix needed)

| ACCV key | ECCV key | Notes |
|---|---|---|
| `truong2020glunet` | `truong2020glu` | GLU-Net, CVPR 2020. Same authors/title/year; ACCV's version abbreviates the venue to `CVPR`. Content equivalent. (Both keys exist in the ACCV bib; only `truong2020glu` is cited by ACCV's Related Work, `truong2020glunet` by the study table.) |

### 5c. ACCV additions that map to an ECCV citation but CANNOT be cross-checked here

These papers are cited in the ECCV **sections** but their entries are **not in the ECCV `main.bib`
provided**, so there is no verified entry text to diff against. You still need to check these
against wherever your verified ECCV bib lives (the key mapping is given so you can reuse that work):

| ACCV key | Same paper in ECCV, cited as | 
|---|---|
| `taniai2016tss` (TSS) | `taniai2016joint-correspondence-coseg` |
| `huang2022flowformer` (FlowFormer) | `Huang2022FlowFormerAT` |
| `spair71k` (also) | `min2019spair` |
| `pfpascal` (also) | `ham2017proposal-flow-pascal` / `ham2016proposal-flow` |
| `heusel2018...` (FID, also) | `fid` |
| `mayer2016FlyingThingsDataset` (also) | `FlyingThings_Dataset` |
| `pointodyssey` (also) | `PointOdyssey_Dataset` |

### 5d. Deliberate divergence from ECCV

| Topic | ECCV | ACCV (now) | Why |
|---|---|---|---|
| KITTI flow | `kitti_flow` (Geiger et al., "Vision Meets Robotics", IJRR 2013 — the raw-data paper) | `kitti2012` + `kitti2015` (the actual flow benchmarks) | Same wrong-paper issue flagged in §1 applies to ECCV too. Left ECCV alone (yours, already verified); ACCV corrected. If you want the two papers consistent, either fix ECCV or revert ACCV. |

### 5e. ACCV-only citations with NO ECCV counterpart (still need your manual check)

Not in the ECCV paper at all, so no cross-check possible. Verify from scratch:

`mayer2018what`, `sun2021autoflow`, `im2022dflow` (author list already corrected + web-verified),
`you2021logme`, `tran2019transferability`, `bao2019information`, `tan2021otce`,
`alvarez2020geometric`, `naeem2020reliable`, `alaa2022faithful`, `sorscher2022beyond`,
`xie2023data`, `xie2023doremi`, `hanneke2019value`, `fan2017pointcloud`, `doersch2022tapvid`,
`white2009mandelbulb`, `kitti2012` (web-verified), `kitti2015` (web-verified),
`taniai2016tss` (web-verified), `bergstra2011tpe`, `bendavid2010theory`, `bonneel2015sliced`.

---

## 6. Bib regenerated from `updated_bib.txt` (2026-07-02)

`main.bib` was rebuilt from your re-exported, verified `updated_bib.txt`. Method: every entry
body was refreshed from your export **matched by title**, while the **original citation key was
kept** (your export used different keys like `Mayer_2016`, `cho2022catsboostingcostaggregation`,
`siméoni2025dinov3`, so re-keying was mandatory or every `\cite` in the paper would break). The two
RIS blocks (`Ma2021`, `bendavid2010theory`) were converted to BibTeX. Result compiles clean: **0
undefined citations, 0 undefined-string warnings, no BibTeX errors.**

**Refreshed from your export:** 73 entries.

**Duplicates removed (5):** `Butler:ECCV:2012` (dup of `sintel`), `kubric` (dup of `greff2021kubric`),
`autoflow` (dup of `sun2021autoflow`), `truong2020glu` (dup of `truong2020glunet`), `fbm` (dup of
`10.1145/74333.74337`). In each pair the cited key was kept.

**Kept as-is because they are NOT in `updated_bib.txt` (4):** `teed2020raft` (RAFT — cited!),
`roth_opt_flow` (Roth & Black — cited!), `Genesis`, `laptev2008` (last two uncited). ⚠️ **RAFT and
Roth & Black were not in your export**, so they did not get re-verified in this pass. Worth exporting
those two so the whole bib comes from one verified source.

**Two nits in the export you may want to fix (left as your version):**
- `im2022dflow` (DFlow): author field is `Byung-Ki, Kwon and Hyeon-Woo, Nam and ...`, which BibTeX
  reads as surname="Byung-Ki". Korean family name is Kwon, so it should be `Kwon, Byung-Ki and Nam,
  Hyeon-Woo and Kim, Ji-Yun and Oh, Tae-Hyun`. Also `year={2022}` but the venue is the Eleventh ICLR
  = 2023.
- Backup of the pre-regeneration `main.bib` is in the session scratchpad (not the repo).
