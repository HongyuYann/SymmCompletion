### <div align="center">SymmCompletion: High-Fidelity and High-Consistency Point Cloud Completion with Symmetry Guidance<div> 
#####  <p align="center"> [Hongyu Yan<sup>*1</sup>](https://scholar.google.com/citations?user=TeKnXhkAAAAJ&hl=zh-CN), Zijun Li<sup>*2</sup>, [Kunming Luo<sup>1</sup>](https://coolbeam.github.io/index.html), Li Lu<sup>2</sup>, [Ping Tan<sup>1</sup>](https://ece.hkust.edu.hk/pingtan)
#####  <p align="center"> <sup>1</sup>Hong Kong University of Science and Technology, <sup>2</sup>Sichuan University</p>
<div align="center">
  <a href="https://drive.google.com/drive/folders/1JRdZvdEuPDzXbiLTTvt3pYjGC3Yj3z6p?usp=drive_link">Pretrained Models</a> &ensp;
  <a href="https://arxiv.org/abs/2503.18007">Paper</a> &ensp;
</div>

# ✨ News
- We open-source the 3D native generation model [CraftsMan3D](https://github.com/wyysf-98/CraftsMan3D/tree/main). Welcome to discuss the next-generation method for point cloud completion.

# Introduction
This repository is the code for SymmCompletion: High-Fidelity and High-Consistency Point Cloud Completion with Symmetry Guidance (AAAI 2025 Oral Presentation)

SymmCompletion is a symmetry-based method for point cloud completion. It efficiently estimates point-wise local symmetry transformation to generate geometry-align partial-missing pairs and initial point clouds. Then, it leverages the geometric features of partial-missing pairs as the explicit symmetric guidance to refine initial point clouds. Qualitative and quantitative evaluations on several benchmark datasets demonstrate that SymmCompletion outperforms state-of-the-art completion networks. 

<p align="center">
    <img src="assets/teaser.png"/>
</p>

# Installation
```
git clone https://github.com/HongyuYann/SymmCompletion.git
cd SymmCompletion
conda create --name SymmCompletion python=3.11.0
conda activate SymmCompletion
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
sh extensions/install.sh
```

# Training and Testing
1. Download datasets

    [PCN dataset](https://gateway.infinitescript.com/s/ShapeNetCompletion)

    [MVP dataset](https://drive.google.com/drive/folders/1ylC-dYFM45KW4K9tPyljBSVyetazCEeH?usp=sharing)

    [ShapeNet55/34](https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view)

    [KITTI](https://drive.google.com/drive/folders/1JRdZvdEuPDzXbiLTTvt3pYjGC3Yj3z6p?usp=drive_link)
    ❗Note: The official link for the KITTI dataset generated by [PCN](https://github.com/wentaoyuan/pcn) is currently returning a 404 error, we provide a replacement link to download this dataset. This replacement link will be removed once the official link is functional again.


2. Replace the original data path in dataset_configs to your path
    ```
    PCN dataset:
    PARTIAL_POINTS_PATH: your path/%s/partial/%s/%s/%02d.pcd
    COMPLETE_POINTS_PATH: your path/%s/complete/%s/%s.pcd

    MVP dataset (16K):
    N_POINTS: 16384
    PARTIAL_POINTS_PATH: your path/mvp_%s_input.h5
    COMPLETE_POINTS_PATH: your path/mvp_%s_gt_%dpts.h5

    ShapeNet55/34 dataset:
    PC_PATH: your path/shapenet_pc

    KITTI dataset:
    CLOUD_PATH: your path/KITTI/cars/%s.pcd
    BBOX_PATH: your path/KITTI/bboxes/%s.txt
    ```
3. Training
    ```
    python main.py --config cfgs/PCN_models/SymmCompletion.yaml --val_freq 10 --val_interval 50 --exp_name train_pcn
    ```
4. Testing
    ```
    python main.py --config cfgs/PCN_models/SymmCompletion.yaml --test --test_interval 50 --ckpt ./ckpts/PCN/ckpt-best.pth --exp_name test_pcn
    ```
# Pretrained Models
We provide all pretrained models in [Google Drive](https://drive.google.com/drive/folders/1JRdZvdEuPDzXbiLTTvt3pYjGC3Yj3z6p?usp=drive_link)

# Visualized Results
<p align="center">
    <img src="assets/PCN-vis.png"/>
</p>

# 🤗 Acknowledgements
Our code is built on [AnchorFormer](https://github.com/chenzhik/AnchorFormer) codebase. Our work is also inspired by these projects:
- [FBNet](https://github.com/hikvision-research/3DVision/)
- [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch)
- [SnowflakeNet](https://github.com/AllenXiangX/SnowflakeNet)
- [PCN](https://github.com/wentaoyuan/pcn)
- [VRCNet](https://github.com/paul007pl/VRCNet/tree/main)
- [PoinTr](https://github.com/yuxumin/PoinTr)
- [GRNet](https://github.com/hzxie/GRNet)

# 📖 BibTeX

    @misc{SymmCompletion,
    title         = {SymmCompletion: High-Fidelity and High-Consistency Point Cloud Completion with Symmetry Guidance}, 
    author        = {Hongyu Yan, Zijun Li, Kunming Luo, Li Lu, Ping Tan},
    year          = {2025},
    eprint        = {2503.18007},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV}
    }
