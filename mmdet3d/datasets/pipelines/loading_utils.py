import os
import random
import numpy as np
import torch

__all__ = ["load_augmented_point_cloud", "reduce_LiDAR_beams"]


def load_augmented_point_cloud(path, virtual=False, reduce_beams=32):
    # NOTE: following Tianwei's implementation, it is hard coded for nuScenes
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    # NOTE: path definition different from Tianwei's implementation.
    tokens = path.split("/")
    vp_dir = "_VIRTUAL" if reduce_beams == 32 else f"_VIRTUAL_{reduce_beams}BEAMS"
    seg_path = os.path.join(
        *tokens[:-3],
        "virtual_points",
        tokens[-3],
        tokens[-2] + vp_dir,
        tokens[-1] + ".pkl.npy",
    )
    assert os.path.exists(seg_path)
    data_dict = np.load(seg_path, allow_pickle=True).item()

    virtual_points1 = data_dict["real_points"]
    # NOTE: add zero reflectance to virtual points instead of removing them from real points
    virtual_points2 = np.concatenate(
        [
            data_dict["virtual_points"][:, :3],
            np.zeros([data_dict["virtual_points"].shape[0], 1]),
            data_dict["virtual_points"][:, 3:],
        ],
        axis=-1,
    )

    points = np.concatenate(
        [
            points,
            np.ones([points.shape[0], virtual_points1.shape[1] - points.shape[1] + 1]),
        ],
        axis=1,
    )
    virtual_points1 = np.concatenate(
        [virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1
    )
    # note: this part is different from Tianwei's implementation, we don't have duplicate foreground real points.
    if len(data_dict["real_points_indice"]) > 0:
        points[data_dict["real_points_indice"]] = virtual_points1
    if virtual:
        virtual_points2 = np.concatenate(
            [virtual_points2, -1 * np.ones([virtual_points2.shape[0], 1])], axis=1
        )
        points = np.concatenate([points, virtual_points2], axis=0).astype(np.float32)
    return points



def reduce_LiDAR_beams(pts, reduce_beams_to=32):
    if isinstance(pts, np.ndarray):
        is_numpy = True
        pts = torch.from_numpy(pts)
    else:
        is_numpy = False

    radius = torch.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)
    sine_theta = pts[:, 2] / radius
    theta = torch.asin(sine_theta)

    top_ang = 0.1862
    down_ang = -0.5353
    beam_range = torch.linspace(top_ang, down_ang, steps=32)

    if reduce_beams_to == 0:
        # 1. Random Point Drop: Every point has an equal chance to be dropped
        drop_probability = 0.1  # Example drop probability
        drop_mask = torch.rand(len(pts)) > drop_probability
        pts = pts[drop_mask]

        # 2. Drop Points within Certain Random View Field Angles
        start_angle, end_angle = sorted(random.sample(list(beam_range.tolist()), 2))
        view_field_mask = (theta >= start_angle) & (theta <= end_angle)
        pts = pts[~view_field_mask]

        # 3. Beam Drop: Randomly select the number of beams to drop and which beams they are
        num_beams_to_drop = random.randint(1, 5)  # Randomly choose how many beams to drop
        beams_to_drop = random.sample(range(32), num_beams_to_drop)  # Which beams to drop
        beam_drop_mask = torch.ones(len(pts), dtype=torch.bool)
        for beam_id in beams_to_drop:
            beam_angle = beam_range[beam_id]
            next_beam_angle = beam_range[beam_id + 1] if beam_id + 1 < len(beam_range) else beam_angle - 0.023275
            beam_drop_mask &= ~((theta >= beam_angle) & (theta < next_beam_angle))
        pts = pts[beam_drop_mask]

    elif reduce_beams_to in [16, 4, 1]:
        mask = torch.zeros(len(pts), dtype=torch.bool)
        if reduce_beams_to == 16:
            selected_beams = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
        elif reduce_beams_to == 4:
            selected_beams = [7, 9, 11, 13]
        elif reduce_beams_to == 1:
            selected_beams = [9]  # Selecting the 9th beam as an example

        for beam_id in selected_beams:
            beam_angle = beam_range[beam_id]
            prev_beam_angle = beam_range[beam_id - 1] if beam_id - 1 >= 0 else top_ang
            beam_mask = (theta >= beam_angle) & (theta < prev_beam_angle)
            mask |= beam_mask

        pts = pts[mask]

    else:
        raise NotImplementedError("Supported 'reduce_beams_to' values are 16, 4, 1, or 'corruption'.")

    return pts.numpy() if is_numpy else pts


