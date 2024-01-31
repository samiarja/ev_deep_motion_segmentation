# Motion Segmentation for Neuromorphic Airborne Surveillance
Pytorch implementation

**Authors**: *[Sami Arja](https://samiarja.com/)*

# Installation
```
conda create --name ev_motion_segmentation python=3.9
conda activate ev_motion_segmentation
python3 -m pip install -e .
pip install torch
pip install tqdm
pip install plotly
pip install scikit-image
pip install loris
pip install PyYAML
pip install opencv-python
conda install -c conda-forge pydensecrf
sudo apt install -y build-essentials
```


## Acknowledgement
This code is built on top of [TokenCut](https://github.com/YangtaoWANG95/TokenCut_video), [DINO](https://github.com/facebookresearch/dino), [RAFT](https://github.com/princeton-vl/RAFT), and [event_warping](https://github.com/neuromorphicsystems/event_warping). We would like to sincerely thanks those authors for their great works. 
