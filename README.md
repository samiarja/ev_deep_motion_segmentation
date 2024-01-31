# Motion Segmentation for Neuromorphic Airborne Surveillance
Pytorch implementation

**Authors**: *[Sami Arja](https://samiarja.com/)*

# EV-Airborne Dataset Results

|        |        |        |     |         |
|--------|--------|--------|-----|---------|
| <img src="./output/EV-Airborne_recording_2023-04-26_15-30-21_cut3/input_frames/EV-Airborne_recording_2023-04-26_15-30-21_cut3_input_vid.gif" loading="lazy" width="300px"> | <img src="./output/EV-Airborne_recording_2023-04-26_15-30-21_cut/input_frames/EV-Airborne_recording_2023-04-26_15-30-21_cut_input_vid.gif" loading="lazy" width="300px"> | <img src="./output/EV-Airborne_recording_2023-04-26_14-53-22_cut/input_frames/EV-Airborne_recording_2023-04-26_14-53-22_cut_input_vid.gif" loading="lazy" width="300px"> | <img src="./output/EV-Airborne_recording_2023-04-26_14-53-22_cut2/input_frames/EV-Airborne_recording_2023-04-26_14-53-22_cut2_input_vid.gif" width="300px"> | <img src="./output/EV-Airborne_recording_2023-04-26_15-30-21_cut2/input_frames/EV-Airborne_recording_2023-04-26_15-30-21_cut2_input_vid.gif" width="300px"> |
| <img src="./output/EV-Airborne_recording_2023-04-26_15-30-21_cut3/motion_comp/my_animation.gif" loading="lazy" width="300px"> | <img src="./output/EV-Airborne_recording_2023-04-26_15-30-21_cut/motion_comp/EV-Airborne_recording_2023-04-26_15-30-21_cut_motion_segmentation.gif" loading="lazy" width="300px"> | <img src="./output/EV-Airborne_recording_2023-04-26_14-53-22_cut/motion_comp/EV-Airborne_recording_2023-04-26_14-53-22_cut_motion_segmentation.gif" loading="lazy" width="300px"> | <img src="./output/EV-Airborne_recording_2023-04-26_14-53-22_cut2/motion_comp/EV-Airborne_recording_2023-04-26_14-53-22_cut2_motion_segmentation.gif" width="300px"> | <img src="./output/EV-Airborne_recording_2023-04-26_15-30-21_cut2/motion_comp/EV-Airborne_recording_2023-04-26_15-30-21_cut2_motion_segmentation.gif" width="300px"> |


|        |        |        |     |         |
|--------|--------|--------|-----|---------|
| <img src="./output/EV-Airborne_congenial-turkey-b1-54-e2-00_cars/input_frames/EV-Airborne_congenial-turkey-b1-54-e2-00_cars_input_vid.gif" loading="lazy" width="300px"> | <img src="./output/EV-Airborne_recording_2023-04-26_15-30-21_cut4/input_frames/EV-Airborne_recording_2023-04-26_15-30-21_cut4_input_vid.gif" loading="lazy" width="300px"> | <img src="./output/EV-Airborne_recording_2023-04-26_15-30-21_cut5/input_frames/EV-Airborne_recording_2023-04-26_15-30-21_cut5_input_vid.gif" loading="lazy" width="300px"> | <img src="/home/samiarja/Desktop/PhD/Code/ev_deep_motion_segmentation/output/EV-Airborne_recording_2023-04-26_14-54-52_cut4/input_frames/EV-Airborne_recording_2023-04-26_14-54-52_cut4_input_vid.gif" width="300px"> | <img src="/home/samiarja/Desktop/PhD/Code/ev_deep_motion_segmentation/output/EV-Airborne_recording_2023-04-26_14-54-52_cut3/input_frames/EV-Airborne_recording_2023-04-26_14-54-52_cut3_input_vid.gif" width="300px"> |
| <img src="./output/EV-Airborne_congenial-turkey-b1-54-e2-00_cars/motion_comp/EV-Airborne_congenial-turkey-b1-54-e2-00_cars_motion_segmentation.gif" loading="lazy" width="300px"> | <img src="./output/EV-Airborne_recording_2023-04-26_15-30-21_cut4/motion_comp/EV-Airborne_recording_2023-04-26_15-30-21_cut4_motion_segmentation.gif" loading="lazy" width="300px"> | <img src="./output/EV-Airborne_recording_2023-04-26_15-30-21_cut5/motion_comp/EV-Airborne_recording_2023-04-26_15-30-21_cut5_motion_segmentation.gif" loading="lazy" width="300px"> | <img src="./output/EV-Airborne_recording_2023-04-26_14-53-22_cut2/motion_comp/EV-Airborne_recording_2023-04-26_14-53-22_cut2_motion_segmentation.gif" width="300px"> | <img src="./output/EV-Airborne_recording_2023-04-26_15-30-21_cut2/motion_comp/EV-Airborne_recording_2023-04-26_15-30-21_cut2_motion_segmentation.gif" width="300px"> |



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
MSNAS code is built on top of [TokenCut](https://github.com/YangtaoWANG95/TokenCut_video), [DINO](https://github.com/facebookresearch/dino), [RAFT](https://github.com/princeton-vl/RAFT), and [event_warping](https://github.com/neuromorphicsystems/event_warping). We would like to sincerely thanks those authors for their great works. 
