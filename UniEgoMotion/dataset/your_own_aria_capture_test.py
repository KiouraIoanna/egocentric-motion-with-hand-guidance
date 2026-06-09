import os
import json
import numpy as np
import torch
from loguru import logger
from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core import mps
from projectaria_tools.core.mps.utils import get_nearest_pose

from dataset.ee4d_motion_dataset import EE4D_Motion_Dataset
from dataset.egoexo4d_utils import ARIA_INTRINSICS, ego_extri_to_egoego_head_traj
from dataset.canonicalization import saved_sequence_to_full_sequence
from dataset.smpl_utils import evaluate_smpl
from utils.torch_utils import to_tensor
from utils.rotation_conversions import matrix_to_rotation_6d




def find_floor_height_1(pointcloud):
    """Find floor height from pointcloud using heuristic method of histogram and filtering.

    Use as follows:

    from dataset.egoexo4d_utils import get_pointcloud
    pointcloud_path = f"<mps_folder>/slam/semidense_points.csv.gz"
    pointcloud = get_pointcloud(pointcloud_path)
    floor_height = find_floor_height_1(pointcloud)
    """

    # First, filter out outlier points before doing histogram. All points should be within 4m height of the camera.
    # Since the camera is z=0, the floor will have negative z value. Filter out points below the camera level.
    # Also, camera won't be more than 6m above the floor.
    x = pointcloud.copy()
    x = x[(x[:, 2] < 0) & (x[:, 2] > -4)]

    # Now, do a histogram of the z values. The floor will have many points and hence one of the peaks in histogram.
    # Find 10 most frequent bins and find the bin with the lowest z value. This bin should be near the floor.
    freq, bin_boundaries = np.histogram(x[:, 2], bins=100)

    # find lowest of max 10 most frequent bins. May be need to do NMS.
    # bin_idx = np.argsort(freq)[::-1][:10]
    # bin_idx = np.min(bin_idx)

    # Find the lowest bin with significant frequency.
    th = np.max(freq) * 0.25
    bin_idx = np.min(np.where(freq > th))
    z = bin_boundaries[bin_idx : bin_idx + 2].mean()

    # Now, filter out points within a small range of the floor.
    range_thresh = 0.2
    x = pointcloud.copy()
    x = x[(x[:, 2] > z - range_thresh) & (x[:, 2] < z + range_thresh)]

    # Do a finer histogram of the z values to find the floor more accurately.
    freq, bin_boundaries = np.histogram(x[:, 2], bins=40)
    bin_idx = np.argmax(freq)
    z = bin_boundaries[bin_idx : bin_idx + 2].mean()

    return z


def get_calib_from_vrs(vrs_file):
    rgb_stream_label, calib_fix = "camera-rgb", True

    # Get VRS file and calibration object
    assert os.path.exists(vrs_file), f"VRS file not found: {vrs_file}"
    vrs_data_provider = data_provider.create_vrs_data_provider(vrs_file)
    # rgb_stream_id = StreamId("214-1")
    # rgb_stream_label = vrs_data_provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = vrs_data_provider.get_device_calibration()
    src_calib = device_calibration.get_camera_calib(rgb_stream_label)

    # Fix calibration for Aria camera
    # https://github.com/EGO4D/ego-exo4d-egopose/blob/2494c0d192b92784525342df6c6828e3aad04c2e/handpose/data_preparation/main.py#L295
    if calib_fix:
        proj_params = src_calib.projection_params()
        proj_params[0] /= 2
        proj_params[1] = (proj_params[1] - 0.5 - 32) / 2
        proj_params[2] = (proj_params[2] - 0.5 - 32) / 2

        src_calib = calibration.CameraCalibration(
            src_calib.get_label(),
            src_calib.model_name(),
            proj_params,
            src_calib.get_transform_device_camera(),
            src_calib.get_image_size()[0],
            src_calib.get_image_size()[1],
            src_calib.get_valid_radius(),
            src_calib.get_max_solid_angle(),
            src_calib.get_serial_number(),
        )

    return src_calib


def process_aria_data(vrs_file, closed_loop_path):

    # Fixed Aria intrinsics for undistorted image
    intri = np.array(ARIA_INTRINSICS)

    # Per frame extrinsics
    stored_extri_path = os.path.join(os.path.dirname(closed_loop_path), "aria_extrinsics.json")
    if os.path.exists(stored_extri_path):
        # If stored extrinsics exist, load them
        with open(stored_extri_path) as f:
            extri = json.load(f)
        extri = {int(k): np.array(v) for k, v in extri.items()}
    else:
        # If stored extrinsics do not exist, calculate them and store them
        src_calib = get_calib_from_vrs(vrs_file)
        T_device_rgb_camera = src_calib.get_transform_device_camera()
        closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)
        start_ns = int(closed_loop_traj[0].tracking_timestamp.total_seconds() * 1e9)
        end_ns = int(closed_loop_traj[-1].tracking_timestamp.total_seconds() * 1e9)

        extri = {}
        for fidx, cur_ns in enumerate(np.arange(start_ns + 1, end_ns + 1, 1e9 / 30.005)):
            pose_info = get_nearest_pose(closed_loop_traj, cur_ns)
            T_world_device = pose_info.transform_world_device
            T_world_rgb_camera = T_world_device @ T_device_rgb_camera
            extri[fidx] = np.linalg.inv(T_world_rgb_camera.to_matrix())[:3, :4]

        extri_to_store = {k: np.round(v, 6).tolist() for k, v in extri.items()}
        with open(stored_extri_path, "w") as f:
            json.dump(extri_to_store, f)
        logger.info(f"Stored Aria extrinsics in {stored_extri_path}")

    return intri, extri


class MyAriaCapture_Motion_Dataset(EE4D_Motion_Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = "egogait"

    def load_motion_data(self):

        seq_idx = 0
        self.seq_names = []
        self.motion_data = {}
        self.idx_to_sidx_fidx = []

        for vrs_file, closed_loop_path, floor_height in ["PUT_YOUR_VRS_FILE_HERE", "PUT_YOUR_MPS_CLOSED_LOOP_PATH_HERE", "PUT_YOUR_FLOOR_HEIGHT_HERE"]:

                # Get Aria 3D trajectory
                ego_intri, ego_extri = process_aria_data(vrs_file, closed_loop_path)
                aria_traj = ego_extri_to_egoego_head_traj(ego_extri)
                aria_traj = to_tensor(aria_traj)

                # Downsample and convert to rotation 6d and translation (for UniEgoMotion compatibility)
                frame_idxs = sorted(list(aria_traj.keys()))
                frame_idxs = frame_idxs[::3]  # 30fps to 10fps
                aria_traj = torch.stack([aria_traj[fid] for fid in frame_idxs])
                aria_traj_rot6d = matrix_to_rotation_6d(aria_traj[:, :3, :3])  # T x 6
                aria_traj = torch.cat([aria_traj_rot6d, aria_traj[:, :3, 3]], dim=1)  # T x (6 + 3)
                T = aria_traj.shape[0]

                # ------------------------------------------------------------------------------------------------
                # Create dummy SMPL params and other dummy assets because we don't have any ground truth data
                # ------------------------------------------------------------------------------------------------
                betas = torch.zeros(10).float()[None]  # 1 x 10
                smpl_params = {
                    "global_orient": matrix_to_rotation_6d(torch.eye(3)[None].repeat(T, 1, 1)),  # T x 6
                    "body_pose": matrix_to_rotation_6d(torch.eye(3)[None, None].repeat(T, 21, 1, 1)),  # T x 21 x 6
                    "betas": betas,
                    "transl": torch.zeros(T, 3) - 100,  # move far away
                    "left_hand_pose": torch.zeros(T, 12),
                    "right_hand_pose": torch.zeros(T, 12),
                }
                _, smpl_params_full = saved_sequence_to_full_sequence(aria_traj, smpl_params, self.smpl)
                kp3d, verts, full_pose = evaluate_smpl(self.smpl, smpl_params_full)
                body_root_offset = kp3d[0, 0] - smpl_params["transl"][0]  # 3
                # ------------------------------------------------------------------------------------------------

                # floor height
                # pointcloud_path = f"<mps_folder>/slam/semidense_points.csv.gz"
                # pointcloud = get_pointcloud(pointcloud_path)
                # floor_height = find_floor_height_1(pointcloud)

                seq_name = f"seq_name"
                self.motion_data[seq_name] = {
                    "aria_traj": aria_traj,
                    "start_idx": frame_idxs[0],
                    "end_idx": frame_idxs[-1],
                    "floor_height": floor_height,
                    "num_frames": T,
                    "smpl_params": smpl_params,
                    "kp3d": kp3d[:, :76],  # only body and hands
                    "body_root_offset": body_root_offset,
                }

                # Segment indices
                for fidx in range(0, len(frame_idxs) - self.segment_stride, self.segment_stride):
                    self.idx_to_sidx_fidx.append((seq_idx, fidx))
                logger.info(f"Loaded {seq_name} with {len(self.idx_to_sidx_fidx)} segments")

                self.seq_names.append(seq_name)
                seq_idx += 1


if __name__ == "__main__":

    from utils.vis_utils import save_video
    from utils.vis_utils import visualize_sequence

    dataset = MyAriaCapture_Motion_Dataset(
        data_dir="/vision/u/chpatel/data/egoexo4d_ee4d_motion",
        split="val",
        repre_type="v4_beta",
        cond_traj=True,
        cond_img_feat=False,
        cond_betas=False,
        window=80,
        img_feat_type="clip_all",
    )
    idx = 0

    sample = dataset[idx]
    imgs = dataset.visualize_sample(sample)
    save_video(imgs[..., ::-1], f"check_capture", "/vision/u/chpatel/test", fps=10)

    import IPython

    IPython.embed()