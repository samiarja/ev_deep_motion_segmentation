import typing
import pathlib
import numpy as np
import event_stream
from typing import Tuple
import event_warping

class DataLoader:
    DATASET_PATHS = {
        "EV-Airborne": "/home/samiarja/Desktop/PhD/Dataset/EV-Airborne/sequences/",
        "DistSurf": "/home/samiarja/Desktop/PhD/Dataset/DistSurf/",
        "EV-IMO": "/home/samiarja/Desktop/PhD/Dataset/EV-IMO/",
        "EV-IMO2": "/home/samiarja/Desktop/PhD/Dataset/EV-IMO2/",
        "EED": "/home/samiarja/Desktop/PhD/Dataset/EED/",
        "HKUST-EMS": "/home/samiarja/Desktop/PhD/Dataset/HKUST-EMS/",
        "stars": "/home/samiarja/Desktop/PhD/Dataset/NORALPH_ICNS_EB_Space_Imaging_Speed_Dataset/2022-03-31T11-51-20Z_speed_survey-WINDY/files/"
    }
    
    def __init__(self, dataset: str, seq: str, tstart: float = 0.0, windowsize: float = 0.3e6):
        self.dataset = dataset
        
        if self.dataset == "EV-IMO":
            # Split the seq argument for the EV-IMO dataset
            seq_parts = seq.split("_", 1)
            self.main_sequence = seq_parts[0]
            self.sub_sequence = seq_parts[1] if len(seq_parts) > 1 else None
        else:
            # For other datasets, use the full seq as the main sequence
            self.main_sequence = seq
        
        self.tstart = tstart
        self.windowsize = windowsize
        self.tfinish = self.tstart + self.windowsize

        # Determine data path based on dataset, main sequence category, and sub-sequence identifier
        if self.dataset == "EV-Airborne":
            self.data_path = pathlib.Path(self.DATASET_PATHS[self.dataset])
        elif self.dataset == "EV-IMO" and self.main_sequence in ["box", "table", "fast", "floor", "tabletop", "wall"]:
            # Construct the correct path based on the main sequence category and sub-sequence identifier
            if self.sub_sequence:
                self.data_path = pathlib.Path(self.DATASET_PATHS[self.dataset]) / self.main_sequence / self.sub_sequence
            else:
                self.data_path = pathlib.Path(self.DATASET_PATHS[self.dataset]) / self.main_sequence
        elif self.dataset == "EV-IMO2":
            # The path structure for EV-IMO2 is more complex, so we search through possible subdirectories
            subdirs = ["samsung_mono_low_light/imo_ll/eval", 
                       "samsung_mono_low_light/imo_ll/train", 
                       "samsung_mono_obj_det/imo/eval", 
                       "samsung_mono_obj_det/imo/train"]
            for subdir in subdirs:
                potential_path = pathlib.Path(self.DATASET_PATHS[self.dataset]) / subdir / self.main_sequence
                if potential_path.exists():
                    self.data_path = potential_path
                    break
        else:
            # Use the provided main sequence category as is
            self.data_path = pathlib.Path(self.DATASET_PATHS[self.dataset]) / self.main_sequence


    def load_data(self):
        if self.dataset == "stars":
            data_name = "psee400.es"
        elif self.dataset == "EV-Airborne":
            data_name = f"{self.main_sequence}.es"
        else:
            data_name = "events.txt"

        file_extension = pathlib.Path(data_name).suffix

        if file_extension == ".es":
            return self._process_data(self._load_es(data_name))
        elif file_extension == ".txt":
            return self._process_data(self._load_txt(data_name))
        else:
            raise ValueError(f"Unsupported data type: {file_extension}")

    def _process_data(self, data: Tuple[int, int, np.ndarray]) -> Tuple[int, int, np.ndarray]:
        width, height, events = data
        
        # Normalize timestamps
        if events["t"][0] != 0:
            events["t"] = events["t"] - events["t"][0]

        # Filter events based on tstart and tfinish
        ii = np.where(np.logical_and(events["t"] > self.tstart, events["t"] < self.tfinish))
        events = events[ii]
        events = event_warping.without_most_active_pixels(events, ratio=0.0)
        return width, height, events

    def _load_es(self, data_name: str) -> Tuple[int, int, np.ndarray]:
        with event_stream.Decoder(self.data_path / data_name) as decoder:
            return (
                decoder.width,
                decoder.height,
                np.concatenate([packet for packet in decoder]),
            )

    def _load_txt(self, data_name: str) -> Tuple[int, int, np.ndarray]:
        """
        Load data from a .txt file and return it in the same format as .es data.
        """
        # Define the dtype for the structured array
        dtype = [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('on', '?')]
        
        # List to store the events
        events_list = []
        
        with open(self.data_path / data_name, 'r') as file:
            for line in file:
                # Convert the first value to microseconds (assuming it's in seconds)
                t = int(float(line.split()[0]) * 1e6)
                # t = int(float(line.split()[0]))
                x = int(line.split()[1])
                y = int(line.split()[2])
                on = bool(int(line.split()[3]))
                
                events_list.append((t, x, y, on))
        
        # Convert the list to a structured np array
        events_array = np.array(events_list, dtype=dtype)
        
        # Making an assumption about the width and height based on the max x and y values.
        # Adjust if you have a different way to determine width and height for .txt files.
        width  = events_array['x'].max() + 1
        height = events_array['y'].max() + 1
        return width, height, events_array
    