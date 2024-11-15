import os
import shutil
import yaml
import time
import torch
import h5py
import warnings
import glob as gb
import numpy as np
import graph
import tools
import networks
import datetime
from torchvision import transforms 
from datasets import extract_feat_info
from PIL import Image, ImageDraw
from tqdm import tqdm
import event_warping
from DataLoader import DataLoader
import numpy.lib.recfunctions as rfn

'''
Motion Segmentation for Neuromorphic Aerial Surveillance
'''

start_time = time.time()

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

config_path = './config/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

arch                    = config['arch']
data                    = config['data']
seq                     = config['seq']
patch_size              = config['patch_size']
min_size                = config['min_size']
time_surface_decay      = config['time_surface_decay']
tau                     = config['tau']
gap                     = config['gap']
fusion_mode             = config['fusion_mode']
flow_model              = config['flow_model']
alpha                   = config['alpha']
max_frame               = config['max_frame']
bs                      = config['bilateral_solver']['use']
sigma_spatial           = config['bilateral_solver']['sigma_spatial']
sigma_luma              = config['bilateral_solver']['sigma_luma']
sigma_chroma            = config['bilateral_solver']['sigma_chroma']
crf                     = config['crf']
single_frame            = config['single_frame']
out_dir                 = config['out_dir']
n_last_frames           = config['n_last_frames']
size_mask_neighborhood  = config['size_mask_neighborhood']
batch_size              = config['batch_size']
topk                    = config['topk']
radius                  = config['radius']
chunk_size              = config['chunk_size']
timediff                = float(config['timediff'])
start_time              = float(config['start_time'])
finish_time             = float(config['finish_time'])
timediff_timecode       = event_warping.seconds_to_timecode(timediff)
start_time_timecode     = event_warping.seconds_to_timecode(start_time)
finish_time_timecode    = event_warping.seconds_to_timecode(finish_time)

event_name             = f"{out_dir}/{data}_{seq}"
flow_model             = "./raft-sintel.pth"
data_path              = os.path.join(event_name, "input_frames")
out_vis                = os.path.join(event_name, 'coarse/')
out_vis_rgb            = os.path.join(event_name, 'rgb/')
if bs:
    seg_path           = os.path.join(event_name, 'bs/')
else:
    seg_path            = os.path.join(event_name, 'crf/')
tt_output_path          = os.path.join(event_name, 'tt_adapt/')
motioncomp_output_path  = os.path.join(event_name, 'motion_comp/')
motioncompL_output_path = os.path.join(event_name, 'motion_comp_large_delta/')
flow_img_dir            = os.path.join(event_name, f'RAFT_FlowImages_gap{gap}')
flow_dir                = os.path.join(event_name, f'RAFT_Flows_gap{gap}')
model                   = networks.get_model(arch, patch_size, device)
transform               = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),])


event_warping.print_message(f"events loaded, {data} from {seq} dataset",color='green', style='bold')
width, height, events = event_warping.read_es_file(f"./Dataset/{data}/{data}_{seq}_events.es")
sensor_size           = (width, height)

if not os.path.exists(event_name):
        os.makedirs(event_name, exist_ok=True)
if not os.path.exists(event_name+"/input_frames"):
        os.makedirs(event_name+"/input_frames", exist_ok=True)

event_warping.print_message(f"Convert events to .ppm frames", color='red', style='bold')
os.chdir(os.path.expanduser("./command_line_tools/build/release"))

image_directory = f"{event_name}/input_frames"
if os.path.exists(image_directory):
    for file_name in os.listdir(image_directory):
        if file_name.endswith('.png') or file_name.endswith('.ppm'):
            file_path = os.path.join(image_directory, file_name)
            os.unlink(file_path)


start_time = time.time()
cmd1 = (
    f"./es_to_frames"
    f" -i ../../../Dataset/{data}/{data}_{seq}_events.es"
    f" -o ../../../{image_directory}"
    f" -s linear"
    f" -t {time_surface_decay}"
    f" -f {timediff_timecode}"
    f" -b {start_time_timecode}"
    f" -e {finish_time_timecode}"
)
os.system(cmd1)

end_time = time.time()
execution_time = (end_time - start_time) * 1000
print(f"Execution time for time surface: {execution_time} ms")


event_warping.print_message(f"Convert .ppm to .png with vertical flip", color='red', style='bold')
cmd2 = (
    f"cd ../../../{image_directory} &&"
    f" for file in *.ppm; do"
    f"   ffmpeg -i \"$file\" -vf 'format=rgb24,vflip' \"${{file%.ppm}}.png\" >/dev/null 2>&1;"
    f" done &&"
    f" rm *.ppm &&"
    f" ls -1 | grep '\.png$' | sort | head -n 1 | xargs -d '\\n' rm --"
)
os.system(cmd2)
os.chdir(os.path.expanduser("../../../"))


event_warping.print_message(f'Calculate optical flow with RAFT', color='blue', style='bold')
# Copy input images to the ./raft directory
shutil.copytree(data_path, './raft/input', dirs_exist_ok=True) # type: ignore 

# Change directory to ./raft
start_time = time.time()
os.chdir(os.path.abspath("./raft"))
for g in [gap]:  # assuming gap is defined    
    cmd = (f"python predict.py"
           f" --gap {g}"
           f" --model {flow_model}"
           f" --path ./input"
           f" --outroot ./RAFT_FlowImages_gap{g}"
           f" --reverse 0"
           f" --raw_outroot ./RAFT_Flows_gap{g}"
           f" --resize {min_size}")
    os.system(cmd)

end_time = time.time()
execution_time = (end_time - start_time) * 1000
print(f"Execution time for RAFT: {execution_time} ms")

# Change back to the main directory
os.chdir(os.path.abspath(".."))

# Loop through the output folders
for output_dir in ['RAFT_Flows_gap', 'RAFT_FlowImages_gap']:
    src_path = os.path.join('./raft', f'{output_dir}{gap}')
    dest_path = os.path.join(event_name, f'{output_dir}{gap}')
    
    # Check if "input" folder exists
    input_dir = os.path.join(src_path, 'input')
    if os.path.exists(input_dir):
        # Move the contents of "input" folder up one level
        for filename in os.listdir(input_dir):
            shutil.move(os.path.join(input_dir, filename), src_path)
        # Remove the "input" folder
        shutil.rmtree(input_dir)
    
    # Move the cleaned output directories to the event_name directory
    if os.path.exists(src_path):
        shutil.rmtree(dest_path, ignore_errors=True) # Ensure the destination is clear
        shutil.move(src_path, dest_path)

# Remove the "input" folder inside ./raft directory
raft_input_dir = './raft/input'
if os.path.exists(raft_input_dir):
    shutil.rmtree(raft_input_dir)

start_time = time.time()
event_warping.print_message("Extract features from frame", color='cyan', style='bold')
img_names, nb_node, nb_img, feat_h, feat_w, feats, arr_h, arr_w, frame_id, pil, _ = extract_feat_info('',  
                                                                                        data_path,
                                                                                        patch_size,
                                                                                        min_size,
                                                                                        arch,
                                                                                        model,
                                                                                        transform,
                                                                                        )
end_time = time.time()
execution_time = (end_time - start_time) * 1000
print(f"Execution time for frame features: {execution_time} ms")

start_time = time.time()
event_warping.print_message("Extract features from flow", color='magenta', style='bold')
img_names, _, nb_flow, feat_h_flow, feat_w_flow, feats_flow, arr_h_flow, arr_w_flow, frame_id, _, flow = extract_feat_info('',
                                                                                        data_path,
                                                                                        patch_size,
                                                                                        min_size,
                                                                                        arch,
                                                                                        model,
                                                                                        transform,
                                                                                        flow_img_dir,
                                                                                        flow_dir,
                                                                                        )
end_time = time.time()
execution_time = (end_time - start_time) * 1000
print(f"Execution time for flow features: {execution_time} ms")

assert nb_flow == nb_img 
assert feat_h == feat_h_flow
assert feat_w == feat_w_flow

event_warping.print_message(f"Building the graph, {nb_node} nodes", color='red', style='bold')
if not single_frame:
    start_time = time.time()
    foreground = graph.build_graph(nb_img, nb_node, feats, feats_flow, frame_id, arr_w, arr_h, tau, alpha, fusion_mode = fusion_mode, max_frame=max_frame)
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000
    print(f"Execution time for graph: {execution_time} ms")
else:
    foreground = graph.build_graph_single_frame(nb_img, feats, feats_flow, frame_id, arr_w, arr_h, tau, alpha, fusion_mode = fusion_mode)
foreground = foreground.reshape(nb_img, feat_h, feat_w)

event_warping.print_message(f"Generating masks for input video", color='green', style='bold')
if bs:
    crf = False
    out_vis_refine = os.path.join(event_name, 'bs/')
elif crf:
    out_vis_refine = os.path.join(event_name, 'crf/')
if not os.path.exists(out_vis):
    os.makedirs(out_vis, exist_ok=True)
    os.makedirs(out_vis_rgb, exist_ok=True)
    os.makedirs(out_vis_refine, exist_ok=True) # type: ignore

for img_id in tqdm(range(nb_img)):
    rgb, mask_coarse, mask_refine = tools.vis_mask_pil(pil[img_id], foreground[img_id], crf, bs, sigma_spatial, sigma_luma, sigma_chroma)
    base_filename_coarse = os.path.splitext(img_names[img_id])[0]
    base_filename_refine = os.path.splitext(img_names[img_id])[0]
    filename_coarse = os.path.join(out_vis, base_filename_coarse + '.png')
    filename_refine = os.path.join(out_vis_refine, base_filename_refine + '.png') # type: ignore
    Image.fromarray(mask_coarse.astype(np.uint8) * 255).save(filename_coarse)
    Image.fromarray(mask_refine.astype(np.uint8) * 255).save(filename_refine)


event_warping.print_message(f"Running test-time adaptation to enhance flow-predicted masks", color='cyan', style='bold')
if not os.path.exists(tt_output_path):
    os.makedirs(tt_output_path)

start_time = time.time()
cmd = (
    f"python dino/eval_adaptation.py"
    f" --arch vit_small"
    f" --patch_size {patch_size}"
    f" --n_last_frames {n_last_frames}"
    f" --size_mask_neighborhood {size_mask_neighborhood}"
    f" --topk {topk}"
    f" --bs {batch_size}"
    f" --data_path {data_path}"
    f" --seg_path {seg_path}"
    f" --output_path {tt_output_path}"
)
os.system(cmd)

end_time = time.time()
execution_time = (end_time - start_time) * 1000
print(f"Execution time for DMR: {execution_time} ms")

event_warping.print_message(f"Overlay test-time adaptation masks on original images", color='red', style='bold')
num_files = len([f for f in os.listdir(tt_output_path) if os.path.isfile(os.path.join(tt_output_path, f))])
for img_id in tqdm(range(num_files)):
    filename = sorted(os.listdir(tt_output_path))[img_id]
    input_image_path = os.path.join(data_path, filename)
    input_image = Image.open(input_image_path)
    filepath = os.path.join(tt_output_path, filename)
    refined_test_time_mask_image = Image.open(filepath)
    refined_test_time_mask_np = np.array(refined_test_time_mask_image)
    refined_test_time_mask_binary = refined_test_time_mask_np // 255
    refined_test_time_mask_boolean = refined_test_time_mask_np.astype(bool)
    rgb, mask_coarse, mask_refine = tools.vis_mask_pil(input_image, refined_test_time_mask_boolean, crf, bs, sigma_spatial, sigma_luma, sigma_chroma)
    rgb.save(os.path.join(out_vis_rgb, filename))


event_warping.print_message(f"Salient objects frames to events interpolation", color='magenta', style='bold')
image_paths   = gb.glob(f"{data_path}/*.png") 
mask_path     = gb.glob(f"{tt_output_path}/*.png")
image_paths.sort()
mask_path.sort()
events = rfn.append_fields(events, ['l','cl', 'vx', 'vy'], [np.zeros(len(events["x"]), dtype=np.float64)] * 4, usemask=False) # type: ignore


for i in tqdm(range(0, len(image_paths)-1)):
    previous_frame_path = image_paths[i]
    prev_mask = np.array(Image.open(mask_path[i]), dtype=bool)
    next_frame_path = image_paths[i+1]
    next_mask = np.array(Image.open(mask_path[i+1]), dtype=bool)
    
    previous_frame = os.path.basename(previous_frame_path)
    next_frame = os.path.basename(next_frame_path)
    previous_timestamp = int(previous_frame.split('_')[1].split('.')[0])
    next_timestamp = int(next_frame.split('_')[1].split('.')[0])
    
    ii          = np.where(np.logical_and(events["t"] >= previous_timestamp, events["t"] <= next_timestamp))
    sub_events  = events[ii]
    
    # Initialize an empty array to store the labels
    labels = np.zeros(len(sub_events), dtype=int)

    # prev_mask = np.flipud(prev_mask)
    # next_mask = np.flipud(next_mask)
    
    # Loop through the sub_events
    for j, ev in enumerate(sub_events):
        x, y = int(ev[1]), int(ev[2]) 
        
        label_assigned  = False  # Flag to check if a label has been assigned
        
        # Check the pixels within the defined radius
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                new_x, new_y = x+dx, y+dy
                
                # Ensure new coordinates are within the image boundaries
                if 0 <= new_x < prev_mask.shape[1] and 0 <= new_y < prev_mask.shape[0]:
                    if prev_mask[new_y, new_x] and next_mask[new_y, new_x]:
                        labels[j] = 1
                        label_assigned = True
                        break # Exit the loop once a label is assigned
                        
            if label_assigned: # Exit the outer loop if a label is assigned
                break

    events['l'][ii] = labels


event_warping.print_message(f"Saving events with motion segmentation labels", color='red', style='bold')
with h5py.File(f'{event_name}/{data}_{seq}_events_with_motion_inter.h5', 'w') as hf:
    hf.create_dataset("events", data=np.asarray(events))


event_warping.print_message(f"Motion compensation by contrast maximisation", color='green', style='bold')
with h5py.File(f'{event_name}/{data}_{seq}_events_with_motion_inter.h5', 'r') as hf:
    events = hf['events'][:]

BBOX           = False
image_paths    = gb.glob(f"{data_path}/*.png")
bbox_path      = f'../../Dataset/{data}/{seq}/boundingbox.txt'
bbox_exists = False
if os.path.exists(bbox_path):
    bounding_boxes = event_warping.read_bbox_file(bbox_path)
    bbox_exists = True
if not os.path.exists(motioncompL_output_path):
        os.makedirs(motioncompL_output_path, exist_ok=True)
image_paths.sort()
first_timestamp = int(os.path.basename(image_paths[0]).split('_')[1].split('.')[0])
last_timestamp  = int(os.path.basename(image_paths[-1]).split('_')[1].split('.')[0])
sequence_number = 1
unique_labels   = np.unique(events["l"])

for chunk_start in np.arange(first_timestamp, last_timestamp, chunk_size):
    chunk_end       = chunk_start + chunk_size
    ii              = np.where(np.logical_and(events["t"] >= chunk_start, events["t"] < chunk_end))
    selected_events = events[ii]
    event_warping.print_message(f"Processing chunk: {int(chunk_start)}", color='yellow', style='bold')

    if np.any(selected_events["l"] > 0):
        unique_labels = np.unique(selected_events["l"])
        for current_label in unique_labels:
            event_warping.print_message(f"Processing events where label = {current_label}", color='blue', style='bold')
            selected_indices                = np.where(selected_events["l"] == current_label)
            best_velocity, highest_variance = event_warping.find_best_velocity_with_iteratively(sensor_size, selected_events[selected_indices], increment=100)
            warped_image_before             = event_warping.accumulate_pixel_map(sensor_size, selected_events[selected_indices], best_velocity) # type: ignore
            cumulative_map                  = warped_image_before['cumulative_map']
            event_indices                   = warped_image_before['event_indices']
            warped_image                    = event_warping.render(cumulative_map, colormap_name="magma", gamma=lambda image: image ** (1 / 3))
            events["vx"][ii[0][selected_indices]] = best_velocity[0] # type: ignore
            events["vy"][ii[0][selected_indices]] = best_velocity[1] # type: ignore

        cumulative_map_object, seg_label = event_warping.accumulate_cnt_rgb((width, height),
                                                                            events[ii],
                                                                            events["l"][ii].astype(np.int32),
                                                                            (events["vx"][ii],events["vy"][ii]))
        warped_image_segmentation_rgb = event_warping.rgb_render(cumulative_map_object, seg_label)
        if BBOX and bbox_exists:
            image_height = warped_image_segmentation_rgb.height
            draw = ImageDraw.Draw(warped_image_segmentation_rgb)
            for bbox in [b for b in bounding_boxes if b["timestamp"] >= chunk_start and b["timestamp"] < chunk_end]:
                # Adjust y-coordinates for vertical flip
                flipped_y_top = image_height - 1 - bbox["y"] - bbox["h"]
                flipped_y_bottom = image_height - 1 - bbox["y"]
                draw.rectangle([bbox["x"], flipped_y_top, bbox["x"] + bbox["w"], flipped_y_bottom], outline="red")
        output_filename = f"{sequence_number:06d}_{chunk_start}.png"
        flipped_warped_image_segmentation_rgb = warped_image_segmentation_rgb.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_warped_image_segmentation_rgb.save(os.path.join(motioncompL_output_path, output_filename))
        sequence_number += 1
    else:
        event_warping.print_message("No labels > 0, processing all events.", color='magenta', style='bold')
        selected_indices = np.where(selected_events)
        best_velocity, highest_variance = event_warping.find_best_velocity_with_iteratively(sensor_size, selected_events[selected_indices], increment=100)
        warped_image_before             = event_warping.accumulate_pixel_map(sensor_size, selected_events[selected_indices], best_velocity) # type: ignore 
        cumulative_map                  = warped_image_before['cumulative_map']
        event_indices                   = warped_image_before['event_indices']
        warped_image                    = event_warping.render(cumulative_map, colormap_name="magma",gamma=lambda image: image ** (1 / 3))
        events["vx"][ii[0][selected_indices]] = best_velocity[0] # type: ignore
        events["vy"][ii[0][selected_indices]] = best_velocity[1] # type: ignore
        
        cumulative_map_object, seg_label = event_warping.accumulate_cnt_rgb((width, height),
                                                                            events[ii],
                                                                            events["l"][ii].astype(np.int32),
                                                                            (events["vx"][ii],events["vy"][ii]))
        warped_image_segmentation_rgb = event_warping.rgb_render(cumulative_map_object, seg_label)
        if BBOX and bbox_exists:
            image_height = warped_image_segmentation_rgb.height
            draw = ImageDraw.Draw(warped_image_segmentation_rgb)
            for bbox in [b for b in bounding_boxes if b["timestamp"] >= chunk_start and b["timestamp"] < chunk_end]:
                # Adjust y-coordinates for vertical flip
                flipped_y_top = image_height - 1 - bbox["y"] - bbox["h"]
                flipped_y_bottom = image_height - 1 - bbox["y"]
                draw.rectangle([bbox["x"], flipped_y_top, bbox["x"] + bbox["w"], flipped_y_bottom], outline="red")
        output_filename = f"{sequence_number:06d}_{chunk_start}.png"
        flipped_warped_image_segmentation_rgb = warped_image_segmentation_rgb.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_warped_image_segmentation_rgb.save(os.path.join(motioncompL_output_path, output_filename))
        sequence_number += 1

    event_warping.print_message(f"Overwriting the events with motion segmentation labels", color='red', style='bold')
    with h5py.File(f'{event_name}/{data}_{seq}_events_with_motion_inter.h5', 'w') as hf:
        hf.create_dataset("events", data=np.asarray(events))

event_warping.print_message(f"Produce motion segmentation frames", color='blue', style='bold')
if not os.path.exists(motioncomp_output_path):
    os.makedirs(motioncomp_output_path, exist_ok=True)
image_paths = gb.glob(f"{data_path}/*.png")
image_paths.sort()
previous_timestamp = int(os.path.basename(image_paths[0]).split('_')[1].split('.')[0])
next_timestamp = int(os.path.basename(image_paths[1]).split('_')[1].split('.')[0])
last_timestamp = int(os.path.basename(image_paths[-1]).split('_')[1].split('.')[0])
chunk_timediff = next_timestamp - previous_timestamp
sequence_number = 1
for chunk_start in np.arange(previous_timestamp, last_timestamp + chunk_timediff, chunk_timediff):
    chunk_end = chunk_start + chunk_timediff
    ii = np.where(np.logical_and(events["t"] >= chunk_start, events["t"] < chunk_end))
    cumulative_map_object, seg_label = event_warping.accumulate_cnt_rgb(
        (width, height),
        events[ii],
        events[ii]["l"].astype(np.int32),
        (events["vx"][ii], events["vy"][ii])
    )
    warped_image_segmentation_rgb = event_warping.rgb_render(cumulative_map_object, seg_label)
    output_filename = f"{sequence_number:06d}_{chunk_start}.png"
    flipped_warped_image_segmentation_rgb = warped_image_segmentation_rgb.transpose(Image.FLIP_TOP_BOTTOM)
    flipped_warped_image_segmentation_rgb.save(os.path.join(motioncomp_output_path, output_filename))
    sequence_number += 1

event_warping.print_message(f"Convert the frames to video for illustration", color='cyan', style='bold')
folders = {
    f"{data}_{seq}_input_vid": data_path,
    f"{data}_{seq}_raft_optical_flow": flow_img_dir,
    f"{data}_{seq}_bilateral_solver": seg_path,
    f"{data}_{seq}_test_time_adapt": tt_output_path,
    f"{data}_{seq}_final_overlay": out_vis_rgb,
    f"{data}_{seq}_motion_segmentation": motioncomp_output_path,
    f"{data}_{seq}_motion_segmentation_L": motioncompL_output_path
}

lower_frame_rate = 10 # for motion_segmentation_L

for folder_name, folder_path in folders.items():
    frame_rate = lower_frame_rate if folder_name.endswith("_L") else 30
    if any(file.endswith('.jpg') for file in os.listdir(folder_path)):
        cmd_jpg = (
            f"ffmpeg -y -framerate {frame_rate} -pattern_type glob -i '{folder_path}/*.jpg' "
            f"{folder_path}/{folder_name}.gif"
        )
        os.system(f"{cmd_jpg} >/dev/null 2>&1")
    elif any(file.endswith('.png') for file in os.listdir(folder_path)):
        cmd_png = (
            f"ffmpeg -y -framerate {frame_rate} -pattern_type glob -i '{folder_path}/*.png' "
            f"{folder_path}/{folder_name}.gif"
        )
        os.system(f"{cmd_png} >/dev/null 2>&1")

cmd = (
    f"ffmpeg -y"
    f" -i {folders[f'{data}_{seq}_input_vid']}/{data}_{seq}_input_vid.gif"
    f" -i {folders[f'{data}_{seq}_raft_optical_flow']}/{data}_{seq}_raft_optical_flow.gif"
    f" -i {folders[f'{data}_{seq}_bilateral_solver']}/{data}_{seq}_bilateral_solver.gif"
    f" -i {folders[f'{data}_{seq}_test_time_adapt']}/{data}_{seq}_test_time_adapt.gif"
    # f" -i {folders[f'{data}_{seq}_final_overlay']}/{data}_{seq}_final_overlay.gif"
    f" -i {folders[f'{data}_{seq}_motion_segmentation']}/{data}_{seq}_motion_segmentation.gif"
    f" -filter_complex \""
    f" [0:v]scale=-1:480:flags=lanczos[0s];"
    f" [1:v]scale=-1:480:flags=lanczos[1s];"
    f" [2:v]scale=-1:480:flags=lanczos[2s];"
    f" [3:v]scale=-1:480:flags=lanczos[3s];"
    # f" [4:v]scale=-1:480:flags=lanczos[4s];"
    f" [4:v]scale=-1:480:flags=lanczos[4s];"
    f" [0s][1s][2s][3s][4s]hstack=inputs=5[v]\""
    f" -map \"[v]\""
    f" {event_name}/motion_segmentation_network_{data}_{seq}.gif"
)
os.system(f"{cmd} >/dev/null 2>&1")


event_warping.print_message(f"Save configuration", color='green', style='bold')
shutil.copy(config_path, f'{event_name}/config_{data}_{seq}.yaml')

end_time = time.time()
event_warping.print_message(f'Time cost: {str(datetime.timedelta(milliseconds=int((end_time - start_time)*1000)))}', color='magenta', style='bold')
