from __future__ import annotations
import cmaes
import os
import copy
import dataclasses
import event_stream
import event_warping_extension
import h5py
import matplotlib
import matplotlib.colors
import matplotlib.pyplot
import matplotlib.pyplot as plt
import pathlib
import numpy
import PIL.Image
import PIL.ImageChops
import PIL.ImageDraw
import PIL.ImageFont
import scipy.optimize
import typing
import torch
import random
from torch import optim
from math import inf, nan
from scipy.linalg import expm
from typing import Tuple, List
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm
import plotly.graph_objects as go
import scipy.io as sio
from PIL import Image, ImageDraw, ImageFont, ImageOps
from skimage.metrics import structural_similarity as ssim
import colorsys
import scipy.io
from skimage import filters
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage import morphology

def print_message(message, color='default', style='normal'):
    styles = {
        'default': '\033[0m',  # Reset to default
        'bold': '\033[1m',
        'underline': '\033[4m'
    }
    
    colors = {
        'default': '',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m'
    }
    
    print(f"{styles[style]}{colors[color]}{message}{styles['default']}")

def seconds_to_timecode(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"

def read_es_file(
    path: typing.Union[pathlib.Path, str]
) -> tuple[int, int, numpy.ndarray]:
    with event_stream.Decoder(path) as decoder:
        return (
            decoder.width,
            decoder.height,
            numpy.concatenate([packet for packet in decoder]),
        )


def read_h5_file(
    path: typing.Union[pathlib.Path, str]
) -> tuple[int, int, numpy.ndarray]:
    data = numpy.asarray(h5py.File(path, "r")["/FalconNeuro"], dtype=numpy.uint32)
    events = numpy.zeros(data.shape[1], dtype=event_stream.dvs_dtype)
    events["t"] = data[3]
    events["x"] = data[0]
    events["y"] = data[1]
    events["on"] = data[2] == 1
    return numpy.max(events["x"].max()) + 1, numpy.max(events["y"]) + 1, events  # type: ignore


def read_es_or_h5_file(
    path: typing.Union[pathlib.Path, str]
) -> tuple[int, int, numpy.ndarray]:
    if pathlib.Path(path).with_suffix(".es").is_file():
        return read_es_file(path=pathlib.Path(path).with_suffix(".es"))
    elif pathlib.Path(path).with_suffix(".h5").is_file():
        return read_h5_file(path=pathlib.Path(path).with_suffix(".h5"))
    raise Exception(
        f"neither \"{pathlib.Path(path).with_suffix('.es')}\" nor \"{pathlib.Path(path).with_suffix('.h5')}\" exist"
    )

def read_bbox_file(file_path):
    bounding_boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            timestamp = float(parts[0])
            x = int(parts[1].split('=')[1].strip())
            y = int(parts[2].split('=')[1].strip())
            w = int(parts[3].split('=')[1].strip())
            h = int(parts[4].split('=')[1].strip())
            bounding_boxes.append({"timestamp": timestamp, "x": x, "y": y, "w": w, "h": h})
    return bounding_boxes

def convert_mat_to_h5(mat_file, h5_file):
    # Load the .mat file
    mat_data = scipy.io.loadmat(mat_file)

    # Access the 'events' data
    events = mat_data['events']

    # Create a new .h5 file
    with h5py.File(h5_file, 'w') as h5f:
        for field in events.dtype.names:
            # Extract the field data from the record array
            field_data = numpy.concatenate(events[field].ravel())

            # Save each field as a separate dataset in the .h5 file
            h5f.create_dataset(field, data=field_data)

@dataclasses.dataclass
class RotatedEvents:
    eventsrot: numpy.ndarray

@dataclasses.dataclass
class CumulativeMap:
    pixels: numpy.ndarray


def without_most_active_pixels(events: numpy.ndarray, ratio: float):
    assert ratio >= 0.0 and ratio <= 1.0
    count = numpy.zeros((events["x"].max() + 1, events["y"].max() + 1), dtype="<u8")
    numpy.add.at(count, (events["x"], events["y"]), 1)  # type: ignore
    return events[count[events["x"], events["y"]]<= numpy.percentile(count, 100.0 * (1.0 - ratio))]

def with_most_active_pixels(events: numpy.ndarray):
    return events[events["x"], events["y"]]

def warp_seg(events: numpy.ndarray, velocity: tuple[numpy.ndarray, numpy.ndarray]):
    warped_events = events.copy()
    velocity_x = numpy.asarray(velocity[0])
    velocity_y = numpy.asarray(velocity[1])
    warped_x = warped_events['x'].astype(numpy.float64) - (velocity_x * warped_events['t'])
    warped_y = warped_events['y'].astype(numpy.float64) - (velocity_y * warped_events['t'])
    warped_x = numpy.clip(warped_x, 0, numpy.iinfo(numpy.uint16).max)
    warped_y = numpy.clip(warped_y, 0, numpy.iinfo(numpy.uint16).max)
    warped_events['x'] = numpy.round(warped_x).astype(numpy.uint16)
    warped_events['y'] = numpy.round(warped_y).astype(numpy.uint16)
    return warped_events

# velocity in px/us
def warp(events: numpy.ndarray, velocity: tuple[float, float]):
    warped_events = numpy.array(
        events, dtype=[("t", "<u8"), ("x", "<f8"), ("y", "<f8"), ("on", "?")]
    )
    warped_events["x"] -= velocity[0] * warped_events["t"]
    warped_events["y"] -= velocity[1] * warped_events["t"]
    warped_events["x"] = numpy.round(warped_events["x"])
    warped_events["y"] = numpy.round(warped_events["y"])
    return warped_events

def unwarp(warped_events: numpy.ndarray, velocity: tuple[float, float]):
    events = numpy.zeros(
        len(warped_events),
        dtype=[("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")],
    )
    events["t"] = warped_events["t"]
    events["x"] = numpy.round(
        warped_events["x"] + velocity[0] * warped_events["t"]
    ).astype("<u2")
    events["y"] = numpy.round(
        warped_events["y"] + velocity[1] * warped_events["t"]
    ).astype("<u2")
    events["on"] = warped_events["on"]
    return events



def smooth_histogram(warped_events: numpy.ndarray):
    return event_warping_extension.smooth_histogram(warped_events)

def accumulate(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    return CumulativeMap(
        pixels=event_warping_extension.accumulate(  # type: ignore
            sensor_size[0],
            sensor_size[1],
            events["t"].astype("<f8"),
            events["x"].astype("<f8"),
            events["y"].astype("<f8"),
            velocity[0],
            velocity[1],
        ),
        offset=0
    )

def accumulate_timesurface(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
    tau: int,
):
    return CumulativeMap(
        pixels=event_warping_extension.accumulate_timesurface(  # type: ignore
            sensor_size[0],
            sensor_size[1],
            events["t"].astype("<f8"),
            events["x"].astype("<f8"),
            events["y"].astype("<f8"),
            velocity[0],
            velocity[1],
            tau,
        ),
        offset=0
    )

def accumulate_pixel_map(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    accumulated_pixels, event_indices_list = event_warping_extension.accumulate_pixel_map(  # type: ignore
        sensor_size[0],
        sensor_size[1],
        events["t"].astype("<f8"),
        events["x"].astype("<f8"),
        events["y"].astype("<f8"),
        velocity[0],
        velocity[1],
    )
    
    # Convert event_indices_list to a numpy array if needed
    event_indices_np = numpy.array(event_indices_list, dtype=object)

    return {
        'cumulative_map': CumulativeMap(
            pixels=accumulated_pixels,
            offset=0
        ),
        'event_indices': event_indices_np
    }


def accumulate_cnt(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    return CumulativeMap(
        pixels=event_warping_extension.accumulate_cnt(  # type: ignore
            sensor_size[0],
            sensor_size[1],
            events["t"].astype("<f8"),
            events["x"].astype("<f8"),
            events["y"].astype("<f8"),
            velocity[0],
            velocity[1],
        ),
        offset=0
    )

def accumulate_cnt_rgb(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    label: numpy.ndarray,
    velocity: tuple[float, float],
):
    accumulated_pixels, label_image = event_warping_extension.accumulate_cnt_rgb(  # type: ignore
        sensor_size[0],
        sensor_size[1],
        events["t"].astype("<f8"),
        events["x"].astype("<f8"),
        events["y"].astype("<f8"),
        label,  # Assuming 'l' labels are 32-bit integers
        velocity[0],
        velocity[1],
    )
    return CumulativeMap(
        pixels=accumulated_pixels,
        offset=0
    ), label_image

class CumulativeMap:
    def __init__(self, pixels, offset=0):
        self.pixels = pixels
        self.offset = offset

def accumulate4D_placeholder(sensor_size, events, linear_vel, angular_vel, zoom):
    # Placeholder function to simulate accumulate4D.
    # You'll need to replace this with the actual PyTorch-compatible implementation.
    return torch.randn(sensor_size[0], sensor_size[1])

def accumulate4D_torch(sensor_size, events, linear_vel, angular_vel, zoom):
    # Convert tensors back to numpy arrays for the C++ function
    t_np = events["t"].cpu().numpy()
    x_np = events["x"].cpu().numpy()
    y_np = events["y"].cpu().numpy()

    # Get the 2D image using the C++ function
    image_np = event_warping_extension.accumulate4D(
        sensor_size[0],
        sensor_size[1],
        t_np.astype("<f8"),
        x_np.astype("<f8"),
        y_np.astype("<f8"),
        linear_vel[0],
        linear_vel[1],
        angular_vel[0],
        angular_vel[1],
        angular_vel[2],
        zoom,
    )

    # Convert numpy array to PyTorch tensor
    image_tensor = torch.tensor(image_np).float().to(linear_vel.device)
    return image_tensor

def accumulate4D(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    linear_vel: tuple[float, float],
    angular_vel: tuple[float, float, float],
    zoom: float,
):
    return CumulativeMap(
        pixels=event_warping_extension.accumulate4D(  # type: ignore
            sensor_size[0],
            sensor_size[1],
            events["t"].astype("<f8"),
            events["x"].astype("<f8"),
            events["y"].astype("<f8"),
            linear_vel[0],
            linear_vel[1],
            angular_vel[0],
            angular_vel[1],
            angular_vel[2],
            zoom,
        ),
        offset=0
    )

def accumulate4D_cnt(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    linear_vel: numpy.ndarray,
    angular_vel: numpy.ndarray,
    zoom: numpy.ndarray,
):
    return CumulativeMap(
        pixels=event_warping_extension.accumulate4D_cnt(  # type: ignore
            sensor_size[0],
            sensor_size[1],
            events["t"].astype("<f8"),
            events["x"].astype("<f8"),
            events["y"].astype("<f8"),
            linear_vel[0],
            linear_vel[1],
            angular_vel[0],
            angular_vel[1],
            angular_vel[2],
            zoom,
        ),
        offset=0
    )


def geometric_transformation(
        resolution: float, 
        rotation_angle: float
):
    rotated_particles = event_warping_extension.geometricTransformation(
        resolution, 
        rotation_angle)
    return rotated_particles


def render(
    cumulative_map: CumulativeMap,
    colormap_name: str,
    gamma: typing.Callable[[numpy.ndarray], numpy.ndarray],
    bounds: typing.Optional[tuple[float, float]] = None,
):
    colormap = matplotlib.pyplot.get_cmap(colormap_name) # type: ignore
    if bounds is None:
        bounds = (cumulative_map.pixels.min(), cumulative_map.pixels.max())
    scaled_pixels = gamma(
        numpy.clip(
            ((cumulative_map.pixels - bounds[0]) / (bounds[1] - bounds[0])),
            0.0,
            1.0,
        )
    )
    image = PIL.Image.fromarray(
        (colormap(scaled_pixels)[:, :, :3] * 255).astype(numpy.uint8)
    )
    return image.transpose(PIL.Image.FLIP_TOP_BOTTOM)

def generate_palette(cluster_count):
    """Generates a color palette for a given number of clusters."""
    palette = []
    for i in range(cluster_count):
        hue = i / cluster_count
        lightness = 0.5  # Middle value ensures neither too dark nor too light
        saturation = 0.9  # High saturation for vibrant colors
        rgb = tuple(int(c * 255) for c in colorsys.hls_to_rgb(hue, lightness, saturation))
        palette.append(rgb)
    return palette

def see_cluster_color(events, cluster):
    """Processes events and generates an image."""
    # Generate color palette
    palette = generate_palette(max(cluster))
    
    # Extract dimensions and event count
    xs, ys = int(events["x"].max()) + 1, int(events["y"].max()) + 1
    event_count = events.shape[0]
    
    # Initialize arrays
    wn = numpy.full((xs, ys), -numpy.inf)
    img = numpy.full((xs, ys, 3), 255, dtype=numpy.uint8)
    
    # Process each event
    for idx in tqdm(range(event_count)):
        x = events["x"][idx]
        y = events["y"][idx]
        label = cluster[idx]
        if label < 0:
            label = 1
        
        wn[x, y] = label + 1
        img[wn == 0] = [0, 0, 0]
        img[wn == label + 1] = palette[label - 1]
    return numpy.rot90(img, -1)

def get_high_intensity_bbox(image):
    """Return the bounding box of the region with the highest intensity in the image."""
    # Convert the image to grayscale
    gray = image.convert("L")
    arr = numpy.array(gray)
    threshold_value = arr.mean() + arr.std()
    high_intensity = (arr > threshold_value).astype(numpy.uint8)
    labeled, num_features = scipy.ndimage.label(high_intensity)
    slice_x, slice_y = [], []

    for i in range(num_features):
        slice_xi, slice_yi = scipy.ndimage.find_objects(labeled == i + 1)[0]
        slice_x.append(slice_xi)
        slice_y.append(slice_yi)

    if not slice_x:
        return None

    max_intensity = -numpy.inf
    max_intensity_index = -1
    for i, (slice_xi, slice_yi) in enumerate(zip(slice_x, slice_y)):
        if arr[slice_xi, slice_yi].mean() > max_intensity:
            max_intensity = arr[slice_xi, slice_yi].mean()
            max_intensity_index = i

    return (slice_y[max_intensity_index].start, slice_x[max_intensity_index].start, 
            slice_y[max_intensity_index].stop, slice_x[max_intensity_index].stop)



def process_blurry_image(input_image, dilation_size=7, size_threshold=50):
    """
    Process a blurry image to select pixels with the highest intensity, apply dilation,
    and remove outlier pixels.
    
    Parameters:
    - input_image: PIL Image object of the blurry image.
    - dilation_size: Size of the structuring element for dilation.
    - size_threshold: Minimum size of objects to keep (removes smaller outliers).

    Returns:
    - A NumPy array of the processed binary image.
    """
    # Convert the image to grayscale
    gray_image = input_image.convert('L')
    gray_array = numpy.array(gray_image)

    # Apply a threshold at the average of the maximum pixel intensity
    threshold_value = gray_array.max() / 2
    thresholded_image = gray_array >= threshold_value

    # Dilate the thresholded image to thicken the high-intensity regions
    dilation_structure = morphology.square(dilation_size)
    dilated_image = morphology.dilation(thresholded_image, footprint=dilation_structure)

    # Perform an opening operation to remove small outliers
    opened_image = morphology.opening(dilated_image, footprint=dilation_structure)

    # Remove small objects to clean up isolated outlier pixels
    labeled_image = label(opened_image)
    cleaned_image = remove_small_objects(labeled_image, min_size=size_threshold)

    # Convert the labeled image back to a binary image
    cleaned_binary_image = cleaned_image > 0

    return cleaned_binary_image

def generate_combined_image(sensor_size, events, labels, vx, vy):
    unique_labels = numpy.unique(labels)
    total_events = len(events)

    # Generate the first warped image to determine its width and height
    first_warped_image = accumulate_cnt(sensor_size, 
                                        events=events[labels == unique_labels[0]], 
                                        velocity=(vx[labels == unique_labels[0]], vy[labels == unique_labels[0]]))
    warped_image_rendered = render(first_warped_image, colormap_name="magma", gamma=lambda image: image ** (1 / 3))
    
    num_cols = min(4, len(unique_labels))
    num_rows = int(numpy.ceil(len(unique_labels) / 4))
    combined_final_segmentation = Image.new('RGB', (num_cols * warped_image_rendered.width, num_rows * warped_image_rendered.height))
    
    try:
        font = ImageFont.truetype("./src/Roboto-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    variances = []
    max_variance_index = -1
    previous_coordinates = (0, 0)
    
    # Initialize variables to store the required additional information
    max_variance_events = None
    max_variance_velocity = None
    max_intensity_pixel_center = None
    
    for i, label in enumerate(unique_labels):
        sub_events = events[labels == label]
        sub_vx = vx[labels == label]
        sub_vy = vy[labels == label]
        
        warped_image = accumulate_pixel_map(sensor_size, events=sub_events, velocity=(sub_vx[0], sub_vy[0]))
        cumulative_map = warped_image['cumulative_map']
        event_indices = warped_image['event_indices']
        flipped_event_indices = event_indices[::-1]
        warped_image_rendered = render(cumulative_map, colormap_name="magma", gamma=lambda image: image ** (1 / 3))
        variance = variance_loss_calculator(cumulative_map.pixels)

        x_coordinate = (i % 4) * warped_image_rendered.width
        y_coordinate = (i // 4) * warped_image_rendered.height
        
        combined_final_segmentation.paste(warped_image_rendered, (x_coordinate, y_coordinate))
        variances.append(variance)
        max_var = numpy.argmax(variances)
        
        if i == max_var:
            if max_variance_index != -1:
                combined_final_segmentation.paste(warped_image_rendered, previous_coordinates)
            
            draw = ImageDraw.Draw(combined_final_segmentation)
            
            # Update the additional information variables
            max_variance_events = sub_events
            max_variance_velocity = (sub_vx[0], sub_vy[0])
            
            # Draw bounding box on the new image with max variance
            draw.rectangle([(x_coordinate, y_coordinate), 
                            (x_coordinate + warped_image_rendered.width, y_coordinate + warped_image_rendered.height)],
                           outline=(255, 0, 0), width=5)
            
            # Draw circle for the pixel with maximum intensity
            max_intensity_pixel = numpy.unravel_index(numpy.argmax(cumulative_map.pixels), cumulative_map.pixels.shape)
            
            # Flip the y-coordinate vertically
            flipped_y = cumulative_map.pixels.shape[0] - max_intensity_pixel[0]
            
            # Update the center of the pixel with the highest intensity
            max_intensity_pixel_center = (x_coordinate + max_intensity_pixel[1], y_coordinate + flipped_y)
            
            circle_radius = 50  # radius of the circle
            draw.ellipse([(max_intensity_pixel_center[0] - circle_radius, max_intensity_pixel_center[1] - circle_radius),
                          (max_intensity_pixel_center[0] + circle_radius, max_intensity_pixel_center[1] + circle_radius)],
                         outline=(0, 255, 0), width=2)  # green color
            
            max_variance_index = i
            previous_coordinates = (x_coordinate, y_coordinate)

    return combined_final_segmentation, max_variance_events, max_variance_velocity, max_intensity_pixel_center


def motion_selection(sensor_size, events, labels, vx, vy):
    '''
    Apply the same speed on all the cluster, pick the cluster that has the maximum contrast
    '''
    variances       = []
    unique_labels   = numpy.unique(labels)
    for i, label in enumerate(unique_labels):
        sub_events      = events[labels == label]
        warped_image    = accumulate(sensor_size, events=sub_events, velocity=(vx[0], vy[0]))
        variance        = variance_loss_calculator(warped_image.pixels)
        variances.append(variance)
    return unique_labels[numpy.argmax(variances)]


def events_trimming(sensor_size, events, labels, vx, vy, winner, circle_radius, nearby_radius):
    """
    Trims events based on intensity and a specified circle radius and nearby pixels.

    Parameters:
    - sensor_size: Tuple indicating the dimensions of the sensor.
    - events: Array containing event data.
    - labels: Array of labels corresponding to events.
    - vx, vy: Arrays of x and y velocities for each event.
    - winner: The winning label for which events are to be trimmed.
    - circle_radius: Radius for trimming around high intensity pixel.
    - nearby_radius: Radius to select nearby pixels around each pixel.

    Returns:
    - List of selected event indices after trimming.
    - Centroid of the selected events (x, y).
    """
    # Filter events, velocities by winner label
    sub_events, sub_vx, sub_vy = events[labels == winner], vx[labels == winner], vy[labels == winner]
    # Compute warped image and retrieve cumulative map and event indices
    warped_image = accumulate_pixel_map(sensor_size, events=sub_events, velocity=(sub_vx[0], sub_vy[0]))
    cumulative_map, event_indices = warped_image['cumulative_map'], warped_image['event_indices']
    # Determine max intensity pixel and its flipped y-coordinate
    max_y, max_x = numpy.unravel_index(numpy.argmax(cumulative_map.pixels), cumulative_map.pixels.shape)
    flipped_y = cumulative_map.pixels.shape[0] - max_y
    # Create a mask centered on the max intensity pixel
    y, x = numpy.ogrid[:cumulative_map.pixels.shape[0], :cumulative_map.pixels.shape[1]]
    main_mask = (x - max_x)**2 + (y - flipped_y)**2 <= circle_radius**2
    
    # Mask for nearby pixels
    nearby_mask = numpy.zeros_like(main_mask)
    for i in range(cumulative_map.pixels.shape[0]):
        for j in range(cumulative_map.pixels.shape[1]):
            if main_mask[i, j]:
                y_nearby, x_nearby = numpy.ogrid[max(0, i-nearby_radius):min(cumulative_map.pixels.shape[0], i+nearby_radius+1), 
                                                 max(0, j-nearby_radius):min(cumulative_map.pixels.shape[1], j+nearby_radius+1)]
                mask = (x_nearby - j)**2 + (y_nearby - i)**2 <= nearby_radius**2
                nearby_mask[y_nearby, x_nearby] = mask
    
    combined_mask = main_mask | nearby_mask

    # Mask the flipped pixels to identify high intensity regions
    marked_image_np = numpy.flipud(cumulative_map.pixels) * combined_mask
    # Extract event indices for non-zero pixels
    selected_events_indices = numpy.concatenate(event_indices[::-1][numpy.where(marked_image_np != 0)]).tolist()
    return selected_events_indices


def compute_centroid(selected_events, label, winner_class, events_filter_raw, current_indices, 
                     label_after_segmentation, vx_after_segmentation, vy_after_segmentation, sub_vx, sub_vy):
    """
    Compute the centroid for selected events based on the winner class.

    Parameters:
    - selected_events: Array of selected events.
    - label: Array of labels corresponding to the events.
    - winner_class: The winning class for which centroid is computed.
    - events_filter_raw: Raw event data.
    - current_indices: Current indices of the events.
    - label_after_segmentation: Global array or passed array for labels after segmentation.
    - vx_after_segmentation: Global array or passed array for vx after segmentation.
    - vy_after_segmentation: Global array or passed array for vy after segmentation.
    - sub_vx: Array of x velocities for each event.
    - sub_vy: Array of y velocities for each event.

    Returns:
    - centroid_x: x-coordinate of the centroid.
    - centroid_y: y-coordinate of the centroid.
    """
    selected_indices = numpy.where(label == winner_class)[0][selected_events]
    label_after_segmentation[current_indices[selected_indices]] = winner_class
    vx_after_segmentation[current_indices[selected_indices]] = sub_vx[0]
    vy_after_segmentation[current_indices[selected_indices]] = sub_vy[0]

    selected_events_for_centroid = events_filter_raw[label_after_segmentation == winner_class]
    centroid_x = numpy.mean(selected_events_for_centroid['x'])
    centroid_y = numpy.mean(selected_events_for_centroid['y'])

    return centroid_x, centroid_y

def generate_combined_image_no_label(sensor_size, events, labels, vx, vy):
    """
    Generate a combined image for given events, labels, and velocities.

    Parameters:
    - events: Array of event data.
    - labels: Array of labels for each event.
    - vx: Array of x velocities.
    - vy: Array of y velocities.

    Returns:
    - A PIL Image combining the warped images for each label.
    """
    
    unique_labels = numpy.unique(labels)
    sub_vx = vx[labels == unique_labels[0]]
    sub_vy = vy[labels == unique_labels[0]]

    # Generate the first warped image to determine its width and height
    first_warped_image = accumulate(sensor_size, 
                                        events=events, 
                                        velocity=(sub_vx[0],sub_vy[0]))
    warped_image_rendered = render(first_warped_image, colormap_name="magma", gamma=lambda image: image ** (1 / 3))
    
    # Determine the number of rows and columns for the final image
    num_cols = min(4, len(unique_labels))
    num_rows = int(numpy.ceil(len(unique_labels) / 4))
    
    # Initialize the final combined image
    combined_final_segmentation = Image.new('RGB', (num_cols * warped_image_rendered.width, num_rows * warped_image_rendered.height))
    
    # Load a font for the text
    try:
        font = ImageFont.truetype("./src/Roboto-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for i, label in enumerate(unique_labels):
        # Filter events, vx, and vy based on the label value
        # sub_events = events[labels == label]
        sub_vx = vx[labels == label]
        sub_vy = vy[labels == label]
        
        # Generate the warped image for this subset of events
        warped_image = accumulate(sensor_size, events=events, velocity=(sub_vx[0], sub_vy[0]))
        warped_image_rendered = render(warped_image, colormap_name="magma", gamma=lambda image: image ** (1 / 3))

        # Compute the x and y coordinates for pasting based on the index
        x_coordinate = (i % 4) * warped_image_rendered.width
        y_coordinate = (i // 4) * warped_image_rendered.height
        
        # Paste this warped image into the final combined image
        combined_final_segmentation.paste(warped_image_rendered, (x_coordinate, y_coordinate))
    return combined_final_segmentation


def generate_overlay_and_indices(blur_map, warped_image, removeFactor=1, flipped_event_indices=None):
    """
    Generate an overlay image and unique indices to remove based on the provided blur_map and warped_image.

    Parameters:
    - blur_map: Numpy array representing the blur map.
    - warped_image: PIL Image object.
    - removeFactor: Factor to determine the intensity of the overlay.
    - flipped_event_indices: Numpy array representing the indices of flipped events.

    Returns:
    - overlay_image: PIL Image object.
    - unique_indices_to_remove: Numpy array of indices.
    """
    
    # Convert blur_map to PIL Image and adjust range
    blur_map_image = Image.fromarray(numpy.uint8(blur_map * 255), 'L')
    blur_map_image = ImageOps.flip(blur_map_image)
    
    blur_image_np = (blur_map - blur_map.min()) / (blur_map.max() - blur_map.min())
    blur_image_np = numpy.flipud(blur_image_np)
    
    image_np = numpy.array(warped_image.convert('RGBA'))
    marked_image_np = image_np.copy()
    marked_image_np[..., :3] = [0, 0, 255]
    marked_image_np[..., 3] = (blur_image_np * removeFactor * 255).astype(numpy.uint8)
    marked_image_np = numpy.where(marked_image_np > numpy.mean(marked_image_np.flatten()), marked_image_np, 0).astype(numpy.uint8)
    
    overlay_image = Image.alpha_composite(Image.fromarray(image_np), Image.fromarray(marked_image_np))
    
    if flipped_event_indices is not None:
        sharp_pixel_coords = numpy.where(marked_image_np[..., 3] != 0)
        all_indices_to_remove = numpy.concatenate(flipped_event_indices[sharp_pixel_coords]).astype(int)
        unique_indices_to_remove = numpy.unique(all_indices_to_remove)
        unique_indices_to_remove = numpy.sort(unique_indices_to_remove)
    else:
        unique_indices_to_remove = None

    return blur_map_image, overlay_image, unique_indices_to_remove


def rgb_render_white(cumulative_map_object, l_values):
    """Render the cumulative map using RGB values based on the frequency of each class, with a white background and save as PNG."""
    
    def generate_intense_palette(n_colors):
        """Generate an array of intense and bright RGB colors, avoiding blue."""
        base_palette = numpy.array([
            [255, 255, 255],  # White (will invert to black)
            [0, 255, 255],    # Cyan (will invert to red)
            [255, 0, 255],    # Magenta (will invert to green)
            [255, 255, 0],    # Yellow (will invert to blue)
            [0, 255, 0],      # Green (will invert to magenta)
            [255, 0, 0],      # Red (will invert to cyan)
            [0, 128, 128],    # Teal (will invert to a light orange)
            [128, 0, 128],    # Purple (will invert to a light green)
            [255, 165, 0],    # Orange (will invert to a blue-green)
            [128, 128, 0],    # Olive (will invert to a light blue)
            [255, 192, 203],  # Pink (will invert to a light green-blue)
            [165, 42, 42],    # Brown (will invert to a light blue-green)
            [0, 100, 0],      # Dark Green (will invert to a light magenta)
            [173, 216, 230],  # Light Blue (will invert to a darker yellow)
            [245, 222, 179],  # Wheat (will invert to a darker blue)
            [255, 20, 147],   # Deep Pink (will invert to a light cyan-green)
            [75, 0, 130],     # Indigo (will invert to a lighter yellow)
            [240, 230, 140],  # Khaki (will invert to a light blue)
            [0, 0, 128],      # Navy (will invert to a light yellow)
        ], dtype=numpy.uint8)
        palette = numpy.tile(base_palette, (int(numpy.ceil(n_colors / base_palette.shape[0])), 1))
        return palette[:n_colors]


    cumulative_map = cumulative_map_object.pixels
    height, width = cumulative_map.shape
    rgb_image = numpy.ones((height, width, 3), dtype=numpy.uint8) * 255  # Start with a white background

    unique, counts = numpy.unique(l_values[l_values != 0], return_counts=True)
    sorted_indices = numpy.argsort(counts)[::-1]
    sorted_unique = unique[sorted_indices]
    sorted_unique = numpy.concatenate(([0], sorted_unique))

    palette = generate_intense_palette(len(sorted_unique))
    color_map = dict(zip(sorted_unique, palette))
    
    for label, color in color_map.items():
        mask = l_values == label
        if numpy.any(mask):
            norm_intensity = cumulative_map[mask] / (cumulative_map[mask].max() + 1e-9)

            norm_intensity = numpy.power(norm_intensity, 0.01)
            blended_color = color * norm_intensity[:, numpy.newaxis]
            rgb_image[mask] = numpy.clip(blended_color, 0, 255)
    
    image = Image.fromarray(rgb_image, 'RGB')
    inverted_image = ImageOps.invert(image)
    inverted_image = inverted_image.transpose(Image.FLIP_TOP_BOTTOM)
    return inverted_image


def rgb_render(cumulative_map_object, l_values):
    """Render the cumulative map using RGB values based on the frequency of each class."""
    
    def generate_intense_palette(n_colors):
        """Generate an array of intense and bright RGB colors, avoiding blue."""
        # Define a base palette with intense and bright colors
        base_palette = numpy.array([
            [255, 255, 255],  # Bright white
            [255, 150, 150],  # Intense red
            [255, 255, 150],  # Intense yellow
            [21,  185, 200],  # blue
            [255, 175, 150],  # Coral
            [255, 150, 200],  # Magenta
            [150, 255, 150],  # Intense green
            [150, 255, 200],  # Aqua
            [255, 200, 150],  # Orange
            [200, 255, 150],  # Light green with more intensity
            [255, 225, 150],  # Gold
            [255, 150, 175],  # Raspberry
            [175, 255, 150],  # Lime
            [255, 150, 255],  # Strong pink
            # Add more colors if needed
        ], dtype=numpy.uint8)
        # Repeat the base palette to accommodate the number of labels
        palette = numpy.tile(base_palette, (int(numpy.ceil(n_colors / base_palette.shape[0])), 1))
        return palette[:n_colors]  # Select only as many colors as needed

    cumulative_map = cumulative_map_object.pixels
    height, width = cumulative_map.shape
    rgb_image = numpy.zeros((height, width, 3), dtype=numpy.uint8)  # Start with a black image

    unique, counts = numpy.unique(l_values[l_values != 0], return_counts=True)
    # Sort the indices of the unique array based on the counts in descending order
    sorted_indices = numpy.argsort(counts)[::-1]
    # Retrieve the sorted labels, excluding label 0
    sorted_unique = unique[sorted_indices]

    # Now we explicitly add the label 0 at the beginning of the sorted_unique array
    sorted_unique = numpy.concatenate(([0], sorted_unique))

    palette = generate_intense_palette(len(sorted_unique))
    color_map = dict(zip(sorted_unique, palette))
    
    for label, color in color_map.items():
        mask = l_values == label
        norm_intensity = cumulative_map[mask] / (cumulative_map[mask].max() + 1e-9)
        norm_intensity = numpy.power(norm_intensity, 0.3)  # Increase the color intensity
        blended_color = color * norm_intensity[:, numpy.newaxis]
        rgb_image[mask] = numpy.clip(blended_color, 0, 255)
    
    image = Image.fromarray(rgb_image)
    return image.transpose(Image.FLIP_TOP_BOTTOM)



def render_3d(variance_loss_3d: numpy.ndarray):
    x, y, z = numpy.indices(variance_loss_3d.shape)
    values = variance_loss_3d.flatten()
    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=values,
        isomin=0.2,
        isomax=numpy.max(variance_loss_3d),
        opacity=0.6,
        surface_count=1,
    ))
    fig.show()


def render_histogram(cumulative_map: CumulativeMap, path: pathlib.Path, title: str):
    matplotlib.pyplot.figure(figsize=(16, 9))
    matplotlib.pyplot.hist(cumulative_map.pixels.flat, bins=200, log=True)
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.xlabel("Event count")
    matplotlib.pyplot.ylabel("Pixel count")
    matplotlib.pyplot.savefig(path)
    matplotlib.pyplot.close()


def intensity_variance(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    return event_warping_extension.intensity_variance(  # type: ignore
        sensor_size[0],
        sensor_size[1],
        events["t"].astype("<f8"),
        events["x"].astype("<f8"),
        events["y"].astype("<f8"),
        velocity[0],
        velocity[1],
    )


def intensity_variance_ts(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
    tau: int,
):
    return event_warping_extension.intensity_variance_ts(  # type: ignore
        sensor_size[0],
        sensor_size[1],
        events["t"].astype("<f8"),
        events["x"].astype("<f8"),
        events["y"].astype("<f8"),
        velocity[0],
        velocity[1],
        tau,
    )

@dataclasses.dataclass
class CumulativeMap:
    pixels: numpy.ndarray
    offset: Tuple[float, float]

def accumulate_warped_events_square(warped_x: torch.Tensor, warped_y: torch.Tensor):
    x_minimum = float(warped_x.min())
    y_minimum = float(warped_y.min())
    xs = warped_x - x_minimum + 1.0
    ys = warped_y - y_minimum + 1.0
    pixels = torch.zeros((int(torch.ceil(ys.max())) + 2, int(torch.ceil(xs.max())) + 2))
    xis = torch.floor(xs).long()
    yis = torch.floor(ys).long()
    xfs = xs - xis.float()
    yfs = ys - yis.float()
    for xi, yi, xf, yf in zip(xis, yis, xfs, yfs):
        pixels[yi, xi] += (1.0 - xf) * (1.0 - yf)
        pixels[yi, xi + 1] += xf * (1.0 - yf)
        pixels[yi + 1, xi] += (1.0 - xf) * yf
        pixels[yi + 1, xi + 1] += xf * yf
    return CumulativeMap(
        pixels=pixels,
        offset=(-x_minimum + 1.0, -y_minimum + 1.0),
    )

def center_events(eventx, eventy):
    center_x = eventx.max() / 2
    center_y = eventy.max() / 2
    eventsx_centered = eventx - center_x
    eventsy_centered = eventy - center_y
    return eventsx_centered, eventsy_centered


def warp_4D(events, linear_vel, angular_vel, zoom, deltat):
    wx, wy, wz = angular_vel
    vx, vy = linear_vel
    eventsx, eventsy = center_events(events[0,:], events[1,:])
    rot_mat = torch.tensor([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=torch.float32)
    event_stack = torch.stack((eventsx, eventsy, torch.ones(len(events[0,:])))).t().float()
    deltat = torch.from_numpy(deltat).float()
    rot_exp = (rot_mat * deltat[:, None, None]).float()
    rot_exp = torch.matrix_exp(rot_exp)
    rot_events = torch.einsum("ijk,ik->ij", rot_exp, event_stack)
    warpedx_scale = (1 - deltat * zoom) * rot_events[:, 0]
    warpedy_scale = (1 - deltat * zoom) * rot_events[:, 1]
    warpedx_trans = warpedx_scale - deltat * vx
    warpedy_trans = warpedy_scale - deltat * vy
    return warpedx_trans, warpedy_trans


def opt_loss_py(events, linear_vel, angular_vel, zoom, deltat):
    warpedx, warpedy    = warp_4D(events, linear_vel, angular_vel, zoom, deltat)
    warped_image        = accumulate_warped_events_square(warpedx, warpedy)
    objective_func      = variance_loss_calculator(warped_image)
    save_img(warped_image, "./")
    return objective_func

def opt_loss_cpp(events, sensor_size, linear_vel, angular_vel, zoom):
    # Convert events numpy array to a PyTorch tensor
    events_tensor = {}
    for key in events.dtype.names:
        if events[key].dtype == numpy.uint64:
            events_tensor[key] = torch.tensor(numpy.copy(events[key]).astype(numpy.int64)).to(linear_vel.device)
        elif events[key].dtype == numpy.uint16:
            events_tensor[key] = torch.tensor(numpy.copy(events[key]).astype(numpy.int32)).to(linear_vel.device)
        elif events[key].dtype == numpy.bool_:
            events_tensor[key] = torch.tensor(numpy.copy(events[key]).astype(numpy.int8)).to(linear_vel.device)

    warped_image = accumulate4D_torch(sensor_size=sensor_size,
                                events=events_tensor,
                                linear_vel=linear_vel,
                                angular_vel=angular_vel,
                                zoom=zoom)

    # Convert warped_image to a PyTorch tensor if it's not already one
    if not isinstance(warped_image, torch.Tensor):
        warped_image_tensor = torch.tensor(warped_image.pixels).float()
    else:
        warped_image_tensor = warped_image
        
    objective_func = variance_loss_calculator_torch(warped_image_tensor)
    objective_func = objective_func.float()
    return objective_func


def variance_loss_calculator_torch(evmap):
    flattening = evmap.view(-1)  # Flatten the tensor
    res = flattening[flattening != 0]
    return -torch.var(res)

def variance_loss_calculator(evmap):
    pixels = evmap
    flattening = pixels.flatten()
    res = flattening[flattening != 0]
    return torch.var(torch.from_numpy(res))


def save_img(warped_image, savefileto):
    image = render(
    warped_image,
    colormap_name="magma",
    gamma=lambda image: image ** (1 / 3))
    filename = "eventmap_wz.jpg"
    filepath = os.path.join(savefileto, filename)
    image.save(filepath)

def rad2degree(val):
    return val/numpy.pi*180.

def degree2rad(val):
    return val/180*numpy.pi

def generate_warped_images(events: numpy.ndarray,
                           sensor_size: Tuple[int, int],
                           linear_velocity: numpy.ndarray, 
                           angular_velocity: numpy.ndarray, 
                           scale: numpy.ndarray, 
                           tmax: float,
                           savefileto: str) -> None:

    for iVel in tqdm(range(len(linear_velocity))):
        linear          = linear_velocity[iVel]
        angular         = angular_velocity[iVel]
        zoom            = scale[iVel]
        vx              = -linear / 1e6
        vy              = -43 / 1e6
        wx              = 0.0 / 1e6
        wy              = 0.0 / 1e6
        wz              = (0.0 / tmax) / 1e6
        zooms           = (0.0 / tmax) / 1e6

        warped_image = accumulate4D(sensor_size=sensor_size,
                                    events=events,
                                    linear_vel=(vx,vy),
                                    angular_vel=(wx,wy,wz),
                                    zoom=zooms)

        image = render(warped_image,
                       colormap_name="magma",
                       gamma=lambda image: image ** (1 / 3))
        new_image = image.resize((500, 500))
        filename = f"eventmap_wz_{wz*1e6:.2f}_z_{zooms*1e6:.2f}_vx_{vx*1e6:.4f}_vy_{vy*1e6:.4f}_wx_{wx*1e6:.2f}_wy_{wy*1e6:.2f}.jpg"
        filepath = os.path.join(savefileto, filename)
        new_image.save(filepath)
    return None


def generate_3Dlandscape(events: numpy.ndarray,
                       sensor_size: Tuple[int, int],
                       linear_velocity: numpy.ndarray, 
                       angular_velocity: numpy.ndarray, 
                       scale: numpy.ndarray, 
                       tmax: float,
                       savefileto: str) -> None:
    nvel = len(angular_velocity)
    trans=0
    rot=0
    variance_loss = numpy.zeros((nvel*nvel,nvel))
    for iVelz in tqdm(range(nvel)):
        wx              = 0.0 / 1e6
        wy              = 0.0 / 1e6
        wz              = (angular_velocity[iVelz] / tmax) / 1e6
        for iVelx in range(nvel):
            vx          = linear_velocity[iVelx] / 1e6
            for iVely in range(nvel):
                vy          = linear_velocity[iVely] / 1e6
                warped_image = accumulate4D(sensor_size=sensor_size,
                                            events=events,
                                            linear_vel=(vx,vy),
                                            angular_vel=(wx,wy,wz),
                                            zoom=0)
                var = variance_loss_calculator(warped_image.pixels)
                variance_loss[trans,rot] = var
                trans+=1
        rot+=1
        trans=0
    
    reshaped_variance_loss = variance_loss.reshape(nvel, nvel, nvel)
    sio.savemat(savefileto+"reshaped_variance_loss.mat",{'reshaped_variance_loss':numpy.asarray(reshaped_variance_loss)})
    render_3d(reshaped_variance_loss)
    return None


def find_best_velocity_with_iteratively(sensor_size: Tuple[int, int], events: numpy.ndarray, increment=100):
    """
    Finds the best optimized velocity over a number of iterations.
    """
    best_velocity = None
    highest_variance = float('-inf')
    
    variances = []  # Storing variances for each combination of velocities
    
    for vy in tqdm(range(-1000, 1001, increment)):
        for vx in range(-1000, 1001, increment):
            current_velocity = (vx / 1e6, vy / 1e6)
            
            optimized_velocity = optimize_local(sensor_size=sensor_size,
                                                events=events,
                                                initial_velocity=current_velocity,
                                                tau=1000,
                                                heuristic_name="variance",
                                                method="Nelder-Mead",
                                                callback=None)
            
            objective_loss = intensity_variance(sensor_size, events, optimized_velocity)
            
            variances.append((optimized_velocity, objective_loss))
            
            if objective_loss > highest_variance:
                highest_variance = objective_loss
                best_velocity = optimized_velocity
    
    # Converting variances to a numpy array for easier handling
    variances = numpy.array(variances, dtype=[('velocity', float, 2), ('variance', float)])
    print(f"vx: {best_velocity[0] * 1e6} vy: {best_velocity[1] * 1e6} contrast: {highest_variance}")
    return best_velocity, highest_variance


def random_velocity(opt_range):
    return (random.uniform(-opt_range / 1e6, opt_range / 1e6), 
            random.uniform(-opt_range / 1e6, opt_range / 1e6))

def find_best_velocity_with_initialisation(sensor_size: Tuple[int, int], events: numpy.ndarray, initial_velocity:int, iterations: int):
    """
    Finds the best optimized velocity over a number of iterations.
    """
    best_velocity = None
    highest_variance = float('-inf')
    for _ in range(iterations):
        optimized_velocity = optimize_local(sensor_size=sensor_size,
                                                          events=events,
                                                          initial_velocity=initial_velocity,
                                                          tau=1000,
                                                          heuristic_name="variance",
                                                          method="Nelder-Mead",
                                                          callback=None)
        objective_loss = intensity_variance(sensor_size, events, optimized_velocity)
        print("iter. vx: {}    iter. vy: {}    contrast: {}".format(optimized_velocity[0] * 1e6, optimized_velocity[1] * 1e6, objective_loss))
        if objective_loss > highest_variance:
            highest_variance = objective_loss
            best_velocity = optimized_velocity
        initial_velocity = optimized_velocity
    return best_velocity, highest_variance

def find_best_velocity(sensor_size: Tuple[int, int], events: numpy.ndarray, opt_range:int, iterations: int):
    """
    Finds the best optimized velocity over a number of iterations.
    """
    best_velocity    = None
    highest_variance = float('-inf')
    for _ in range(iterations):
        initial_velocity   = random_velocity(opt_range)
        optimized_velocity = optimize_local(sensor_size=sensor_size,
                                                          events=events,
                                                          initial_velocity=initial_velocity,
                                                          tau=1000,
                                                          heuristic_name="variance",
                                                          method="Nelder-Mead",
                                                          callback=None)
        objective_loss = intensity_variance(sensor_size, events, optimized_velocity)
        print("iter. vx: {}    iter. vy: {}    contrast: {}".format(optimized_velocity[0] * 1e6, optimized_velocity[1] * 1e6, objective_loss))
        if objective_loss > highest_variance:
            highest_variance = objective_loss
            best_velocity = optimized_velocity
        initial_velocity = optimized_velocity
    return best_velocity, highest_variance

def find_best_velocity_advanced(sensor_size: Tuple[int, int], events: numpy.ndarray, opt_range:int, iterations: int, previous_velocities: typing.List[tuple[float, float]]):
    """
    Finds the best optimized velocity over a number of iterations.
    """
    best_velocity = None
    highest_variance = float('-inf')
    DISTANCE_THRESHOLD = 0 #0.01  # Adjust as needed
    PENALTY = 0 #0.5  # Adjust as needed

    for _ in range(iterations):
        initial_velocity   = random_velocity(opt_range)
        optimized_velocity = optimize_local(sensor_size=sensor_size,
                                            events=events,
                                            initial_velocity=initial_velocity,
                                            heuristic_name="variance",
                                            tau=10000,
                                            method="Nelder-Mead",
                                            callback=None)
        objective_loss = intensity_variance(sensor_size, events, optimized_velocity)
        
        # Penalty for being close to previous velocities
        for prev_velocity in previous_velocities:
            dist = numpy.linalg.norm(numpy.array(optimized_velocity) - numpy.array(prev_velocity))
            if dist < DISTANCE_THRESHOLD:
                objective_loss -= PENALTY
        
        print("iter. vx: {}    iter. vy: {}    contrast: {}".format(optimized_velocity[0] * 1e6, optimized_velocity[1] * 1e6, objective_loss))
        
        if objective_loss > highest_variance:
            highest_variance = objective_loss
            best_velocity = optimized_velocity
            previous_velocities.append(optimized_velocity)  # Update previous velocities list
        initial_velocity = optimized_velocity
    return best_velocity, highest_variance


def calculate_patch_variance(sensor_size, events, x_start, y_start, window_size, optimized_velocity):
    """
    Calculate the variance for a specific patch of events using a given velocity.
    
    Parameters:
    - events: The events data.
    - x_start: The starting x-coordinate of the patch.
    - y_start: The starting y-coordinate of the patch.
    - window_size: The size of the patch.
    - optimized_velocity: The velocity value to use.
    
    Returns:
    - The variance of the warped patch.
    """
    mask = (
        (events["x"] >= x_start) & (events["x"] < x_start + window_size) &
        (events["y"] >= y_start) & (events["y"] < y_start + window_size)
    )

    # Extract the patch of events
    patch_events = {
        "x": events["x"][mask],
        "y": events["y"][mask],
        "p": events["on"][mask],
        "t": events["t"][mask]
    }

    # Warp the patch using the optimized_velocity
    # (Assuming you have a warp function. Modify as needed.)
    warped_patch = accumulate(sensor_size, patch_events, optimized_velocity)
    
    # Calculate the variance of the warped patch
    variance = numpy.var(warped_patch.pixels)
    return variance


def optimization(events, sensor_size, initial_linear_vel, initial_angular_vel, initial_zoom, max_iters, lr, lr_step, lr_decay):
    optimizer_name = 'Adam'
    optim_kwargs = dict()  # Initialize as empty dict by default

    # lr = 0.005
    # iters = 100
    lr_step = max(1, lr_step)  # Ensure lr_step is at least 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # linear_vel = torch.tensor(initial_linear_vel).float().to(device)
    # linear_vel.requires_grad = True
    linear_vel = torch.tensor(initial_linear_vel, requires_grad=True)
    print(linear_vel.grad)

    optimizer = optim.__dict__[optimizer_name]([linear_vel], lr=lr, **optim_kwargs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, lr_step, lr_decay)

    print_interval = 1
    min_loss = float('inf')  # Use Python's float infinity
    best_poses = linear_vel.clone()  # Clone to ensure we don't modify the original tensor
    best_it = 0

    if optimizer_name == 'Adam':
        for it in range(max_iters):
            optimizer.zero_grad()
            poses_val = linear_vel.cpu().detach().numpy()
            
            if numpy.isnan(poses_val).any():  # Proper way to check for NaN in numpy
                print("nan in the estimated values, something wrong takes place, please check!")
                exit()

            # Use linear_vel directly in the loss computation
            loss = opt_loss_cpp(events, sensor_size, linear_vel, initial_angular_vel, initial_zoom)

            if it == 0:
                print('[Initial]\tloss: {:.12f}\tposes: {}'.format(loss.item(), poses_val))
            elif (it + 1) % print_interval == 0:
                print('[Iter #{}/{}]\tloss: {:.12f}\tposes: {}'.format(it + 1, max_iters, loss.item(), poses_val))
            
            # Store a copy of the best linear_vel tensor
            if loss < min_loss:
                best_poses = linear_vel.clone()
                min_loss = loss.item()
                best_it = it
            try:
                loss.requires_grad = True
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                
            except Exception as e:
                print(e)
                return poses_val, loss.item()
            
            print("Loss before step:", loss.item())
            optimizer.step()
            print("Loss after step:", loss.item())
            scheduler.step()
    else:
        print("The optimizer is not supported.")

    best_poses = best_poses.cpu().detach().numpy()
    print('[Final Result]\tloss: {:.12f}\tposes: {} @ {}'.format(min_loss, best_poses, best_it))
    if device == torch.device('cuda:0'):
        torch.cuda.empty_cache()
    
    return best_poses, min_loss


def correction(i: numpy.ndarray, j: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray, vx: int, vy: int, width: int, height: int):
    return {
        '1': (1, vx / width, vy / height),
        '2': vx / x[i, j],
        '3': vy / y[i, j],
        '4': vx / (-x[i, j] + width + vx),
        '5': vy / (-y[i, j] + height + vy),
        '6': (vx*vy) / (vx*y[i, j] + vy*width - vy*x[i, j]),
        '7': (vx*vy) / (vx*height - vx*y[i, j] + vy*x[i, j]),
    }


def alpha_1(warped_image: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray, vx: int, vy: int, width: int, height: int, edgepx: int):
    """
    Input:
    warped_image: A 2D numpy array representing the warped image, where pixel values represent the event count.

    This function apply a correction on the warped image based on a set of conditions where vx < w and vy < h. The conditions are designed based on the pixel's 
    x and y positions and additional parameters (vx, vy, width, height).

    Output:
    A 2D numpy array (image) that has been corrected based on a set of conditions. This image is then fed to the Contrast Maximization algorithm
    to estimate the camera.
    """
    conditions = [
        (x > vx) & (x < width) & (y >= vy) & (y <= height),
        (x > 0) & (x < vx) & (y <= height) & (y >= ((vy*x) / vx)),
        (x >= 0) & (x <= width) & (y > 0) & (y < vy) & (y < ((vy*x) / vx)),
        (x >= width) & (x <= width+vx) & (y >= vy) & (y <= (((vy*(x-width)) / vx) + height)),
        (x > vx) & (x < width+vx) & (y > height) & (y > (((vy*(x-width)) / vx) + height)) & (y < height+vy),
        (x > width) & (x < width+vx) & (y >= ((vy*(x-width)) / vx)) & (y < vy),
        (x > 0) & (x < vx) & (y > height) & (y <= (((vy*x) / vx) + height))
    ]

    for idx, condition in enumerate(conditions, start=1):
        i, j = numpy.where(condition)            
        correction_func = correction(i, j, x, y, vx, vy, width, height)
        if idx == 1:
            warped_image[i+1, j+1] *= correction_func[str(idx)][0]
        else:    
            warped_image[i+1, j+1] *= correction_func[str(idx)]

    warped_image[x > width+vx-edgepx] = 0
    warped_image[x < edgepx] = 0
    warped_image[y > height+vy-edgepx] = 0
    warped_image[y < edgepx] = 0
    warped_image[y < ((vy*(x-width)) / vx) + edgepx] = 0
    warped_image[y > (((vy*x) / vx) + height) - edgepx] = 0
    warped_image[numpy.isnan(warped_image)] = 0
    return warped_image


def alpha_2(warped_image: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray, vx: int, vy: int, width: int, height: int, edgepx: int, section: int):
    """
    Input:
    warped_image: A 2D numpy array representing the warped image, where pixel values represent the event count.

    This function apply a correction on the warped image based on a set of conditions where vx > w and vy > h. The conditions are designed based on the pixel's 
    x and y positions and additional parameters (vx, vy, width, height).

    Output:
    A 2D numpy array (image) that has been corrected based on a set of conditions. This image is then fed to the Contrast Maximization algorithm
    to estimate the camera.
    """
    conditions_1 = [
        (x >= width) & (x <= vx) & (y >= (vy*x)/vx) & (y <= (vy/vx)*(x-width-vx)+vy+height), 
        (x > 0) & (x < width) & (y >= (vy*x)/vx) & (y < height), 
        (x > 0) & (x <= width) & (y > 0) & (y < (vy*x)/vx), 
        (x > vx) & (x < vx+width) & (y > vy) & (y <= (vy/vx)*(x-width-vx)+vy+height), 
        (x > vx) & (x < vx+width) & (y > (vy/vx)*(x-width-vx)+vy+height) & (y < height+vy), 
        (x > width) & (x <= vx+width) & (y >= (vy*(x-width))/vx) & (y < vy) & (y < (vy*x)/vx) & (y > 0), 
        (x > 0) & (x <= vx) & (y < (vy/vx)*x+height) & (y >= height) & (y > (vy/vx)*(x-width-vx)+vy+height) 
    ]

    conditions_2 = [
        (x >= 0) & (x < vx+width) & (y > ((vy*(x-width))/vx)+height) & (y < vy) & (y > height) & (y < (vy*x)/vx), 
        (x >= 0) & (x <= vx) & (y > (vy*x)/vx) & (y < height),
        (x >= 0) & (x < width) & (y >= 0) & (y < (vy*x)/vx) & (y < height), 
        (x > width) & (x < vx+width) & (y <= ((vy*(x-width))/vx)+height) & (y > vy), 
        (x >= vx) & (x < vx+width) & (y > ((vy*(x-width))/vx)+height) & (y < vy+height) & (y > vy), 
        (x >= width) & (x <= vx+width) & (y > (vy/vx)*(x-width)) & (y < ((vy*(x-width))/vx)+height) & (y > 0) & (y <vy), 
        (x >= 0) & (x <= vx) & (y <= (vy/vx)*x+height) & (y > (vy/vx)*x) & (y > height) & (y <= height+vy) 
    ]

    conditions = [conditions_1, conditions_2]
    for idx, condition in enumerate(conditions[section-1], start=1):
        i, j = numpy.where(condition)
        correction_func = correction(i, j, x, y, vx, vy, width, height)
        if idx == 1:
            warped_image[i+1, j+1] *= correction_func[str(idx)][section]
        else:    
            warped_image[i+1, j+1] *= correction_func[str(idx)]

    warped_image[x > width+vx-edgepx] = 0
    warped_image[x < edgepx] = 0
    warped_image[y > height+vy-edgepx] = 0
    warped_image[y < edgepx] = 0
    warped_image[y < ((vy*(x-width)) / vx) + edgepx] = 0
    warped_image[y > (((vy*x) / vx) + height) - edgepx] = 0
    warped_image[numpy.isnan(warped_image)] = 0
    return warped_image

def mirror(warped_image: numpy.ndarray):
    mirrored_image = []
    height, width = len(warped_image), len(warped_image[0])
    for i in range(height):
        mirrored_row = []
        for j in range(width - 1, -1, -1):
            mirrored_row.append(warped_image[i][j])
        mirrored_image.append(mirrored_row)
    return numpy.array(mirrored_image)

def intensity_weighted_variance(sensor_size: tuple[int, int],events: numpy.ndarray,velocity: tuple[float, float]):
    numpy.seterr(divide='ignore', invalid='ignore')
    t               = (events["t"][-1]-events["t"][0])/1e6
    edgepx          = t
    width           = sensor_size[0]
    height          = sensor_size[1]
    fieldx          = velocity[0] / 1e-6
    fieldy          = velocity[1] / 1e-6
    velocity        = (fieldx * 1e-6, fieldy * 1e-6)
    warped_image    = accumulate(sensor_size, events, velocity)
    vx              = numpy.abs(fieldx*t)
    vy              = numpy.abs(fieldy*t)
    x               = numpy.tile(numpy.arange(1, warped_image.pixels.shape[1]+1), (warped_image.pixels.shape[0], 1))
    y               = numpy.tile(numpy.arange(1, warped_image.pixels.shape[0]+1), (warped_image.pixels.shape[1], 1)).T
    corrected_iwe   = None
    var             = 0.0
    
    if (fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)<=height) or (fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)<=height):
        corrected_iwe            = alpha_1(warped_image.pixels, x, y, vx, vy, width, height, edgepx)
        
    if (fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)<=height) or (fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)<=height):
        warped_image.pixels     = mirror(warped_image.pixels)
        corrected_iwe           = alpha_1(warped_image.pixels, x, y, vx, vy, width, height, edgepx)
        
    if (fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)<=height) or (fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)<=height) or ((((vy/vx)*width)-height)/(numpy.sqrt(1+(vy/vx)**2)) <= 0 and fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height) or ((((vy/vx)*width)-height)/(numpy.sqrt(1+(vy/vx)**2)) <= 0 and fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height):
        corrected_iwe            = alpha_2(warped_image.pixels, x, y, vx, vy, width, height, edgepx, 1)

    if (fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)>=height) or (fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)>=height) or ((((vy/vx)*width)-height)/(numpy.sqrt(1+(vy/vx)**2)) > 0 and fieldx*t >= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height) or ((((vy/vx)*width)-height)/(numpy.sqrt(1+(vy/vx)**2)) > 0 and fieldx*t <= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height):
        corrected_iwe            = alpha_2(warped_image.pixels, x, y, vx, vy, width, height, edgepx, 2)

    if (fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)<=height) or (fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)<=height) or ((height+vy-(vy/vx)*(width+vx))/(numpy.sqrt(1+(-vy/vx)**2)) >= 0 and fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height) or ((height+vy-(vy/vx)*(width+vx))/(numpy.sqrt(1+(-vy/vx)**2)) >= 0 and fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height):
        warped_image.pixels     = mirror(warped_image.pixels)
        corrected_iwe           = alpha_2(warped_image.pixels, x, y, vx, vy, width, height, edgepx, 1)

    if (fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)>=height) or (fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)<=width and numpy.abs(fieldy*t)>=height) or ((height+vy-(vy/vx)*(width+vx))/(numpy.sqrt(1+(-vy/vx)**2)) < 0 and fieldx*t >= 0.0 and fieldy*t <= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height) or ((height+vy-(vy/vx)*(width+vx))/(numpy.sqrt(1+(-vy/vx)**2)) < 0 and fieldx*t <= 0.0 and fieldy*t >= 0.0 and numpy.abs(fieldx*t)>=width and numpy.abs(fieldy*t)>=height):
        warped_image.pixels     = mirror(warped_image.pixels)
        corrected_iwe           = alpha_2(warped_image.pixels, x, y, vx, vy, width, height, edgepx, 2)
    
    if corrected_iwe is not None:
        var = variance_loss_calculator(corrected_iwe)
    return var

def intensity_maximum(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    velocity: tuple[float, float],
):
    return event_warping_extension.intensity_maximum(  # type: ignore
        sensor_size[0],
        sensor_size[1],
        events["t"].astype("<f8"),
        events["x"].astype("<f8"),
        events["y"].astype("<f8"),
        velocity[0],
        velocity[1],
    )

def calculate_heuristic(self, velocity: Tuple[float, float]):
        if self.heuristic == "variance":
            return intensity_variance(
                (self.width, self.height), self.events, velocity
            )
        if self.heuristic == "variance_ts":
            return intensity_variance_ts(
                (self.width, self.height), self.events, velocity, self.tau
            )
        if self.heuristic == "weighted_variance":
            return intensity_weighted_variance(
                (self.width, self.height), self.events, velocity
            )
        if self.heuristic == "max":
            return intensity_maximum(
                (self.width, self.height), self.events, velocity
            )
        raise Exception("unknown heuristic")

def optimize_local(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    initial_velocity: tuple[float, float],  # px/µs
    tau: int,
    heuristic_name: str,  # max or variance
    method: str,  # Nelder-Mead, Powell, L-BFGS-B, TNC, SLSQP
    # see Constrained Minimization in https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    callback: typing.Callable[[numpy.ndarray], None],
):
    def heuristic(velocity):
        if heuristic_name == "max":
            return -intensity_maximum(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3),
            )
        elif heuristic_name == "variance":
            return -intensity_variance(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3),
            )
        elif heuristic_name == "variance_ts":
            return -intensity_variance_ts(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3), 
                tau=tau)
        elif heuristic_name == "weighted_variance":
            return -intensity_weighted_variance(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3))
        else:
            raise Exception(f'unknown heuristic name "{heuristic_name}"')

    if method == "Nelder-Mead":
        result = scipy.optimize.minimize(
            fun=heuristic,
            x0=[initial_velocity[0] * 1e3, initial_velocity[1] * 1e3],
            method=method,
            bounds=scipy.optimize.Bounds([-1.0, -1.0], [1.0, 1.0]),
            options={'maxiter': 100},
            callback=callback
        ).x
        return (float(result[0]) / 1e3, float(result[1]) / 1e3)
    elif method == "BFGS":
        result = scipy.optimize.minimize(
            fun=heuristic,
            x0=[initial_velocity[0] / 1e2, initial_velocity[1] / 1e2],
            method=method,
            options={'ftol': 1e-9,'maxiter': 50},
            callback=callback
        ).x
        return (float(result[0]) / 1e3, float(result[1]) / 1e3)
    elif method == "Newton-CG":
        result = scipy.optimize.minimize(
            fun=heuristic,
            x0=[initial_velocity[0] / 1e2, initial_velocity[1] / 1e2],
            method=method,
            jac=True,
            options={'ftol': 1e-9,'maxiter': 50},
            callback=callback
        ).x
        return (float(result[0]) / 1e3, float(result[1]) / 1e3)
    else:
        raise Exception(f'unknown optimisation method: "{method}"')

def optimize_cma(
    sensor_size: tuple[int, int],
    events: numpy.ndarray,
    initial_velocity: tuple[float, float],
    initial_sigma: float,
    heuristic_name: str,
    iterations: int,
):
    def heuristic(velocity):
        if heuristic_name == "max":
            return -intensity_maximum(
                sensor_size,
                events,
                velocity=velocity,
            )
        elif heuristic_name == "variance":
            return -intensity_variance(
                sensor_size,
                events,
                velocity=velocity,
            )
        elif heuristic_name == "weighted_variance":
            return -intensity_weighted_variance(
                sensor_size,
                events,
                velocity=(velocity[0] / 1e3, velocity[1] / 1e3))
        else:
            raise Exception(f'unknown heuristic name "{heuristic_name}"')

    optimizer = cmaes.CMA(
        mean=numpy.array(initial_velocity) * 1e3,
        sigma=initial_sigma * 1e3,
        bounds=numpy.array([[-1.0, 1.0], [-1.0, 1.0]]),
    )
    best_velocity: tuple[float, float] = copy.copy(initial_velocity)
    best_heuristic = numpy.Infinity
    for _ in range(0, iterations):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = heuristic((x[0] / 1e3, x[1] / 1e3))
            solutions.append((x, value))
        optimizer.tell(solutions)
        velocity_array, heuristic_value = sorted(
            solutions, key=lambda solution: solution[1]
        )[0]
        velocity = (velocity_array[0] / 1e3, velocity_array[1] / 1e3)
        if heuristic_value < best_heuristic:
            best_velocity = velocity
            best_heuristic = heuristic_value
    return (float(best_velocity[0]), float(best_velocity[1]))
