import numpy as np
from skimage.transform import resize
from multiprocessing import Pool
import h5py
import csv
import os
import random
import argparse

def process_sevir(data_collection_path, data_stats_path, path_to_dataset, path_to_catalog, target_size, time_frames):
    img_types = ['vis', 'ir069', 'ir107', 'vil', 'lght']
    num_workers_for_pool = 100

    os.mkdir(data_collection_path)
    os.mkdir(os.path.join(data_collection_path, "train"))
    os.mkdir(os.path.join(data_collection_path, "valid"))
    os.mkdir(os.path.join(data_collection_path, "test"))

    # ---------------------------------------------------------------- read catalog csv file
    # events = dict mapping {event_id} -> {dict}
    #          where keys = dict from {modality} to {filename + fileindex}
    # e.g. events[event_id][modality] = (filename, fileindex)
    events = {}
    with open(path_to_catalog) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in list(csv_reader)[1:]:
            pc_missing = row[-1]
            if float(pc_missing) > 0.0:
                continue
            event_id, file_name, file_index, img_type, time_utc = row[:5]
            if event_id not in events.keys():
                events[event_id] = {}
            events[event_id][img_type] = (file_name, int(file_index))

    # ----------------------------------------------------------------train, valid, test splits
    all_event_ids = list(events.keys())
    random.shuffle(all_event_ids)
    total_len = len(all_event_ids)
    train_split, valid_split = int(total_len * 0.8), int(total_len * 0.9)

    train_event_ids = all_event_ids[:train_split]
    valid_event_ids = all_event_ids[train_split:valid_split]
    test_event_ids  = all_event_ids[valid_split:]

    train_events = {k:events[k] for k in train_event_ids}
    valid_events = {k:events[k] for k in valid_event_ids}
    test_events = {k:events[k] for k in test_event_ids}

    print ("len(train_events) = {}".format(len(train_events)))
    print ("len(valid_events) = {}".format(len(valid_events)))
    print ("len(test_events) = {}".format(len(test_events)))

    constants = (img_types, target_size, time_frames, num_workers_for_pool)
    os_paths = (path_to_dataset, data_stats_path, data_collection_path)

    write_datasplit_to_disk(train_events, "train", constants, os_paths)
    write_datasplit_to_disk(valid_events, "valid", constants, os_paths)
    write_datasplit_to_disk(test_events, "test",  constants, os_paths)


def write_datasplit_to_disk(events,
                            split,
                            constants,
                            os_paths):

    img_types, target_size, time_frames, num_workers_for_pool = constants
    path_to_dataset, data_stats_path, data_collection_path = os_paths
    print("Writing split {} to disk".format(split))

    file_min = open(os.path.join(data_stats_path, 'min_{}.txt'.format(split)), 'w')
    file_max = open(os.path.join(data_stats_path, 'max_{}.txt'.format(split)), 'w')

    data_index = 0
    for j, e_id in enumerate(events):
        if len(events[e_id].keys()) == 5:  # skip all events without all modalities
            modality_data = {}
            for img_type in img_types:
                file_name = events[e_id][img_type][0]
                with h5py.File(os.path.join(path_to_dataset, file_name), 'r') as hf:
                    # get image
                    if img_type != 'lght':
                        file_index = events[e_id][img_type][1]
                        x = hf[img_type][file_index]
                    else:
                        x = hf[e_id][:]
                        x = _lght_to_grid(x, 48)
                    # resize
                    x = list(np.transpose(x, (2, 0, 1)))
                    arguments = [(xi, target_size) for xi in x]
                    with Pool(num_workers_for_pool) as p:
                        x = p.starmap(wrapper, arguments)  # resize
                    modality_data[img_type] = x
                    localmin = np.inf
                    localmax = -np.inf
                    for xi in x:
                        localmin = min(localmin, np.min(xi))
                        localmax = max(localmax, np.max(xi))
                    file_min.write(f"{localmin}\n")
                    file_min.flush()
                    file_max.write(f"{localmax}\n")
                    file_max.flush()

            event_data = [np.array([modality_data[img_type][i] for img_type in img_types]) for i in
                          range(time_frames)]
            print(f"Saving event #{data_index}")
            np.save(os.path.join(data_collection_path, split, str(j)), np.array(event_data))
            data_index += 1

def wrapper(example, target_size):
    return resize(example, (target_size, target_size), preserve_range=True)

def _lght_to_grid(lght_data, img_size):
    """
    Credit: the SEVIR notebook.

    Converts SEVIR lightning lght_data stored in Nx5 matrix to an LxLx49 tensor representing
    flash counts per pixel per frame

    Parameters
    ----------
    lght_data  np.array
       SEVIR lightning event (Nx5 matrix)

    Returns
    -------
    np.array
       LxLx49 tensor containing pixel counts
    """
    FRAME_TIMES = np.arange(-120.0, 125.0, 5) * 60  # in seconds
    out_size = (img_size, img_size, len(FRAME_TIMES))
    if lght_data.shape[0] == 0:
        return np.zeros(out_size, dtype=np.float32)

    # filter out points outside the grid
    x, y = lght_data[:, 3], lght_data[:, 4]
    m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
    lght_data = lght_data[m, :]
    if lght_data.shape[0] == 0:
        return np.zeros(out_size, dtype=np.float32)

    # Filter/separate times
    # compute z coodinate based on bin locaiton times
    t = lght_data[:, 0]
    z = np.digitize(t, FRAME_TIMES) - 1
    z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1

    x = lght_data[:, 3].astype(np.int64)
    y = lght_data[:, 4].astype(np.int64)

    k = np.ravel_multi_index(np.array([y, x, z]), out_size)
    n = np.bincount(k, minlength=np.prod(out_size))
    return np.reshape(n, out_size).astype(np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset', type=str,
                        default='/home/gridsan/groups/EarthIntelligence/datasets/SEVIR/data')
    parser.add_argument('--path_to_catalog', type=str,
                        default='/home/gridsan/groups/EarthIntelligence/datasets/SEVIR/CATALOG.csv')

    parser.add_argument('--data_collection_path', type=str, default='./sevir/')
    parser.add_argument('--target_size', type=int, default=384)
    parser.add_argument('--time_frames', type=int, default=49)
    args = parser.parse_args()

    dataset = process_sevir(args.data_collection_path, args.data_collection_path, args.path_to_dataset, args.path_to_catalog, args.target_size, args.time_frames)

