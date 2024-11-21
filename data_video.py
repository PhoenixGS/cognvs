import io
import os
import sys
import json
from functools import partial
import math
import torchvision.transforms as TT
from sgm.webds import MetaDistributedWebDataset
import random
from fractions import Fraction
from typing import Union, Optional, Dict, Any, Tuple
from torchvision.io.video import av
import numpy as np
import torch
from torchvision.io import _video_opt
from torchvision.io.video import _check_av_available, _read_from_stream, _align_audio_frames
from torchvision.transforms.functional import center_crop, resize
from torchvision.transforms import InterpolationMode
import decord
from decord import VideoReader
from torch.utils.data import Dataset
from packaging import version as pver
import torch.nn as nn
import torchvision.transforms.functional as F
import cv2


def read_video(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """

    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")

    _check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")

    info = {}
    audio_frames = []
    audio_timebase = _video_opt.default_timebase

    with av.open(filename, metadata_errors="ignore") as container:
        if container.streams.audio:
            audio_timebase = container.streams.audio[0].time_base
        if container.streams.video:
            video_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.video[0],
                {"video": 0},
            )
            video_fps = container.streams.video[0].average_rate
            # guard against potentially corrupted files
            if video_fps is not None:
                info["video_fps"] = float(video_fps)

        if container.streams.audio:
            audio_frames = _read_from_stream(
                container,
                start_pts,
                end_pts,
                pts_unit,
                container.streams.audio[0],
                {"audio": 0},
            )
            info["audio_fps"] = container.streams.audio[0].rate

    aframes_list = [frame.to_ndarray() for frame in audio_frames]

    vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    if aframes_list:
        aframes = np.concatenate(aframes_list, 1)
        aframes = torch.as_tensor(aframes)
        if pts_unit == "sec":
            start_pts = int(math.floor(start_pts * (1 / audio_timebase)))
            if end_pts != float("inf"):
                end_pts = int(math.ceil(end_pts * (1 / audio_timebase)))
        aframes = _align_audio_frames(aframes, audio_frames, start_pts, end_pts)
    else:
        aframes = torch.empty((1, 0), dtype=torch.float32)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes, aframes, info


def resize_for_rectangle_crop(arr, image_size, reshape_mode="random"):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]
    arr = arr.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
    return arr


def pad_last_frame(tensor, num_frames):
    # T, H, W, C
    if len(tensor) < num_frames:
        pad_length = num_frames - len(tensor)
        # Use the last frame to pad instead of zero
        last_frame = tensor[-1]
        pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        return padded_tensor
    else:
        return tensor[:num_frames]


def load_video(
    video_data,
    sampling="uniform",
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    if nb_read_frames is not None:
        ori_vlen = nb_read_frames
    else:
        ori_vlen = min(int(duration * actual_fps) - 1, len(vr))

    max_seek = int(ori_vlen - skip_frms_num - num_frames / wanted_fps * actual_fps)
    start = random.randint(skip_frms_num, max_seek + 1)
    end = int(start + num_frames / wanted_fps * actual_fps)
    n_frms = num_frames

    if sampling == "uniform":
        indices = np.arange(start, end, (end - start) / n_frms).astype(int)
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(np.arange(start, end))
    assert temp_frms is not None
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]

    return pad_last_frame(tensor_frms, num_frames)


import threading


def load_video_with_timeout(*args, **kwargs):
    video_container = {}

    def target_function():
        video = load_video(*args, **kwargs)
        video_container["video"] = video

    thread = threading.Thread(target=target_function)
    thread.start()
    timeout = 20
    thread.join(timeout)

    if thread.is_alive():
        print("Loading video timed out")
        raise TimeoutError
    return video_container.get("video", None).contiguous()


def process_video(
    video_path,
    image_size=None,
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    nb_read_frames=None,
):
    """
    video_path: str or io.BytesIO
    image_size: .
    duration: preknow the duration to speed up by seeking to sampled start. TODO by_pass if unknown.
    num_frames: wanted num_frames.
    wanted_fps: .
    skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
    """

    video = load_video_with_timeout(
        video_path,
        duration=duration,
        num_frames=num_frames,
        wanted_fps=wanted_fps,
        actual_fps=actual_fps,
        skip_frms_num=skip_frms_num,
        nb_read_frames=nb_read_frames,
    )

    # --- copy and modify the image process ---
    video = video.permute(0, 3, 1, 2)  # [T, C, H, W]

    # resize
    if image_size is not None:
        video = resize_for_rectangle_crop(video, image_size, reshape_mode="center")

    return video


def process_fn_video(src, image_size, fps, num_frames, skip_frms_num=0.0, txt_key="caption"):
    while True:
        r = next(src)
        if "mp4" in r:
            video_data = r["mp4"]
        elif "avi" in r:
            video_data = r["avi"]
        else:
            print("No video data found")
            continue

        if txt_key not in r:
            txt = ""
        else:
            txt = r[txt_key]

        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")
        else:
            txt = str(txt)

        duration = r.get("duration", None)
        if duration is not None:
            duration = float(duration)
        else:
            continue

        actual_fps = r.get("fps", None)
        if actual_fps is not None:
            actual_fps = float(actual_fps)
        else:
            continue

        required_frames = num_frames / fps * actual_fps + 2 * skip_frms_num
        required_duration = num_frames / fps + 2 * skip_frms_num / actual_fps

        if duration is not None and duration < required_duration:
            continue

        try:
            frames = process_video(
                io.BytesIO(video_data),
                num_frames=num_frames,
                wanted_fps=fps,
                image_size=image_size,
                duration=duration,
                actual_fps=actual_fps,
                skip_frms_num=skip_frms_num,
            )
            frames = (frames - 127.5) / 127.5
        except Exception as e:
            print(e)
            continue

        item = {
            "mp4": frames,
            "txt": txt,
            "num_frames": num_frames,
            "fps": fps,
        }

        yield item


class VideoDataset(MetaDistributedWebDataset):
    def __init__(
        self,
        path,
        image_size,
        num_frames,
        fps,
        skip_frms_num=0.0,
        nshards=sys.maxsize,
        seed=1,
        meta_names=None,
        shuffle_buffer=1000,
        include_dirs=None,
        txt_key="caption",
        **kwargs,
    ):
        if seed == -1:
            seed = random.randint(0, 1000000)
        if meta_names is None:
            meta_names = []

        if path.startswith(";"):
            path, include_dirs = path.split(";", 1)
        super().__init__(
            path,
            partial(
                process_fn_video, num_frames=num_frames, image_size=image_size, fps=fps, skip_frms_num=skip_frms_num
            ),
            seed,
            meta_names=meta_names,
            shuffle_buffer=shuffle_buffer,
            nshards=nshards,
            include_dirs=include_dirs,
        )

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(path, **kwargs)


class SFTDataset(Dataset):
    def __init__(self, data_dir, video_size, fps, max_num_frames, skip_frms_num=3):
        """
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
        """
        super(SFTDataset, self).__init__()

        self.video_size = video_size
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frms_num = skip_frms_num

        self.video_paths = []
        self.captions = []

        for root, dirnames, filenames in os.walk(data_dir):
            for filename in filenames:
                if filename.endswith(".mp4"):
                    video_path = os.path.join(root, filename)
                    self.video_paths.append(video_path)

                    caption_path = video_path.replace(".mp4", ".txt").replace("videos", "labels")
                    if os.path.exists(caption_path):
                        caption = open(caption_path, "r").read().splitlines()[0]
                    else:
                        caption = ""
                    self.captions.append(caption)

    def __getitem__(self, index):
        decord.bridge.set_bridge("torch")

        video_path = self.video_paths[index]
        vr = VideoReader(uri=video_path, height=-1, width=-1)
        actual_fps = vr.get_avg_fps()
        ori_vlen = len(vr)

        if ori_vlen / actual_fps * self.fps > self.max_num_frames:
            num_frames = self.max_num_frames
            start = int(self.skip_frms_num)
            end = int(start + num_frames / self.fps * actual_fps)
            end_safty = min(int(start + num_frames / self.fps * actual_fps), int(ori_vlen))
            indices = np.arange(start, end, (end - start) // num_frames).astype(int)
            temp_frms = vr.get_batch(np.arange(start, end_safty))
            assert temp_frms is not None
            tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
            tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
        else:
            if ori_vlen > self.max_num_frames:
                num_frames = self.max_num_frames
                start = int(self.skip_frms_num)
                end = int(ori_vlen - self.skip_frms_num)
                indices = np.arange(start, end, max((end - start) // num_frames, 1)).astype(int)
                temp_frms = vr.get_batch(np.arange(start, end))
                assert temp_frms is not None
                tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
                tensor_frms = tensor_frms[torch.tensor((indices - start).tolist())]
            else:

                def nearest_smaller_4k_plus_1(n):
                    remainder = n % 4
                    if remainder == 0:
                        return n - 3
                    else:
                        return n - remainder + 1

                start = int(self.skip_frms_num)
                end = int(ori_vlen - self.skip_frms_num)
                num_frames = nearest_smaller_4k_plus_1(end - start)  # 3D VAE requires the number of frames to be 4k+1
                end = int(start + num_frames)
                temp_frms = vr.get_batch(np.arange(start, end))
                assert temp_frms is not None
                tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms

        tensor_frms = pad_last_frame(
            tensor_frms, self.max_num_frames
        )  # the len of indices may be less than num_frames, due to round error
        tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        tensor_frms = resize_for_rectangle_crop(tensor_frms, self.video_size, reshape_mode="center")
        tensor_frms = (tensor_frms - 127.5) / 127.5

        item = {
            "mp4": tensor_frms,
            "txt": self.captions[index],
            "num_frames": num_frames,
            "fps": self.fps,
        }
        return item

    def __len__(self):
        return len(self.video_paths)

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)

# modified from CameraCtrl https://github.com/hehao13/CameraCtrl

class RandomHorizontalFlipWithPose(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        return torch.rand(n_image) < self.p

    def forward(self, image, flip_flag=None):
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)


class Camera(object):
    def __init__(self, entry):
        assert len(entry) == 19
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

class RealEstate10K(Dataset):
    def __init__(
            self,
            root_path,
            annotation_json,
            sample_stride=4,
            sample_n_frames=16,
            sample_size=[256, 384],
            is_image=False,
    ):
        self.root_path = root_path
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image

        self.dataset = json.load(open(os.path.join(root_path, annotation_json), 'r'))
        self.length = len(self.dataset)

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        pixel_transforms = [TT.Resize(sample_size),
                            TT.RandomHorizontalFlip(),
                            TT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]

        self.pixel_transforms = TT.Compose(pixel_transforms)

    def load_video_reader(self, idx):
        video_dict = self.dataset[idx]

        video_path = os.path.join(self.root_path, video_dict['clip_path'])
        video_reader = VideoReader(video_path)
        return video_reader, video_dict['caption']

    def get_batch(self, idx):
        video_reader, video_caption = self.load_video_reader(idx)
        fps = video_reader.get_avg_fps() 
        total_frames = len(video_reader)

        if self.is_image:
            frame_indice = [random.randint(0, total_frames - 1)]
        else:
            # if isinstance(self.sample_stride, int):
            #     current_sample_stride = self.sample_stride
            # else:
            #     assert len(self.sample_stride) == 2
            #     assert (self.sample_stride[0] >= 1) and (self.sample_stride[1] >= self.sample_stride[0])
            #     current_sample_stride = random.randint(self.sample_stride[0], self.sample_stride[1])
            current_sample_stride = int((fps + 7) / 8)

            cropped_length = self.sample_n_frames * current_sample_stride
            start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
            end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

            assert end_frame_ind - start_frame_ind >= self.sample_n_frames
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)

        pixel_values = torch.from_numpy(video_reader.get_batch(frame_indice).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.

        if self.is_image:
            pixel_values = pixel_values[0]

        return pixel_values, video_caption

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                video, video_caption = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)

        video = self.pixel_transforms(video)
        sample = dict(pixel_values=video, caption=video_caption)

        return sample


class RealEstate10KPose(Dataset):
    def __init__(
            self,
            data_dir,
            video_size,
            fps, # not considered yet
            max_num_frames,
            annotation_json="train.json",
            sample_stride=4,
            minimum_sample_stride=1,
            relative_pose=False,
            zero_t_first_frame=False,
            rescale_fxy=False,
            shuffle_frames=False,
            use_flip=False,
            # return_clip_name=False,
    ):
        self.root_path = data_dir
        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        self.sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = max_num_frames
        # self.return_clip_name = return_clip_name
        self.fps = fps

        self.dataset = json.load(open(os.path.join(data_dir, annotation_json), 'r'))
        self.length = len(self.dataset)

        sample_size = tuple(video_size) if not isinstance(video_size, int) else (video_size, video_size)
        self.sample_size = sample_size
        if use_flip:
            pixel_transforms = [TT.Resize(sample_size),
                                RandomHorizontalFlipWithPose(),
                                TT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        else:
            pixel_transforms = [TT.Resize(sample_size),
                                TT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = sample_size[1] / sample_size[0]

        self.pixel_transforms = pixel_transforms
        self.shuffle_frames = shuffle_frames
        self.use_flip = use_flip

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]
        if self.zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def load_video_reader(self, idx):
        video_dict = self.dataset[idx]

        video_path = os.path.join(self.root_path, video_dict['clip_path'])
        video_reader = VideoReader(video_path)
        return video_dict['clip_name'], video_reader, video_dict['caption']

    def load_cameras(self, idx):
        video_dict = self.dataset[idx]
        pose_file = os.path.join(self.root_path, video_dict['pose_file'])
        with open(pose_file, 'r') as f:
            poses = f.readlines()
        poses = [pose.strip().split(' ') for pose in poses[1:]]
        cam_params = [[float(x) for x in pose] for pose in poses]
        cam_params = [Camera(cam_param) for cam_param in cam_params]
        return cam_params

    def get_batch(self, idx):
        clip_name, video_reader, video_caption = self.load_video_reader(idx)
        cam_params = self.load_cameras(idx)
        assert len(cam_params) >= self.sample_n_frames
        total_frames = len(cam_params)

        current_sample_stride = self.sample_stride

        if total_frames < self.sample_n_frames * current_sample_stride:
            maximum_sample_stride = int(total_frames // self.sample_n_frames)
            current_sample_stride = random.randint(self.minimum_sample_stride, maximum_sample_stride)

        cropped_length = self.sample_n_frames * current_sample_stride
        start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        assert end_frame_ind - start_frame_ind >= self.sample_n_frames
        # frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)
        frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)

        if self.shuffle_frames:
            perm = np.random.permutation(self.sample_n_frames)
            frame_indices = frame_indices[perm]
        # print(f"?frame_indices: {frame_indices}")

        pixel_values = torch.from_numpy(video_reader.get_batch(frame_indices).numpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.

        cam_params = [cam_params[indice] for indice in frame_indices]
        if self.rescale_fxy:
            ori_h, ori_w = pixel_values.shape[-2:]
            ori_wh_ratio = ori_w / ori_h
            if ori_wh_ratio > self.sample_wh_ratio:       # rescale fx
                resized_ori_w = self.sample_size[0] * ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
            else:                                          # rescale fy
                resized_ori_h = self.sample_size[1] / ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]
        intrinsics = np.asarray([[cam_param.fx * self.sample_size[1],
                                  cam_param.fy * self.sample_size[0],
                                  cam_param.cx * self.sample_size[1],
                                  cam_param.cy * self.sample_size[0]]
                                 for cam_param in cam_params], dtype=np.float32)
        intrinsics = torch.as_tensor(intrinsics)[None]                  # [1, n_frame, 4]
        if self.relative_pose:
            c2w_poses = self.get_relative_pose(cam_params)
        else:
            c2w_poses = np.array([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
        c2w = torch.as_tensor(c2w_poses)[None]                          # [1, n_frame, 4, 4]
        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(self.sample_n_frames, dtype=torch.bool, device=c2w.device)
        plucker_embedding = ray_condition(intrinsics, c2w, self.sample_size[0], self.sample_size[1], device='cpu',
                                          flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()

        return pixel_values, video_caption, plucker_embedding, flip_flag, clip_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        decord.bridge.set_bridge("torch")

        while True:
            try:
                video, video_caption, plucker_embedding, flip_flag, clip_name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)

        if self.use_flip:
            video = self.pixel_transforms[0](video)
            video = self.pixel_transforms[1](video, flip_flag)
            video = self.pixel_transforms[2](video)
        else:
            for transform in self.pixel_transforms:
                video = transform(video)
        # if self.return_clip_name:
        #     sample = dict(pixel_values=video, text=video_caption, plucker_embedding=plucker_embedding, clip_name=clip_name)
        # else:
        #     sample = dict(pixel_values=video, text=video_caption, plucker_embedding=plucker_embedding)
        
        # plucker_embedding to bfloat16
        plucker_embedding = plucker_embedding.to(torch.bfloat16)
        # print(f"?????shape of video and plucker_embedding: {video.shape}, {plucker_embedding.shape}")
        sample = dict(mp4=video, txt=video_caption, num_frames=self.sample_n_frames, fps=self.fps, plucker_embedding=plucker_embedding)
        return sample

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)

class VidReader: # directly to numpy
    def __init__(self, frame_list):
        self.frame_list = frame_list
        self.length = len(frame_list)
        self.images = [cv2.imread(frame) for frame in frame_list]
        self.images = np.array(self.images)

    def get_batch(self, frame_indices):
        return self.images[frame_indices]
    
class DL3DV10KPose(Dataset):
    def __init__(
            self,
            data_dir,
            video_size,
            fps, # not considered yet
            max_num_frames,
            annotation_json="train.json",
            sample_stride=4,
            minimum_sample_stride=1,
            relative_pose=False,
            zero_t_first_frame=False,
            rescale_fxy=False,
            shuffle_frames=False,
            use_flip=False,
            # return_clip_name=False,
    ):
        self.root_path = data_dir
        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        self.sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = max_num_frames
        # self.return_clip_name = return_clip_name
        self.fps = fps

        self.dataset = json.load(open(os.path.join(data_dir, annotation_json), 'r'))
        self.length = len(self.dataset)

        sample_size = tuple(video_size) if not isinstance(video_size, int) else (video_size, video_size)
        self.sample_size = sample_size
        if use_flip:
            pixel_transforms = [TT.Resize(sample_size),
                                RandomHorizontalFlipWithPose(),
                                TT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        else:
            pixel_transforms = [TT.Resize(sample_size),
                                TT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = sample_size[1] / sample_size[0]

        self.pixel_transforms = pixel_transforms
        self.shuffle_frames = shuffle_frames
        self.use_flip = use_flip

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]
        if self.zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    # def load_cameras(self, idx):
    #     video_dict = self.dataset[idx]
    #     pose_file = os.path.join(self.root_path, video_dict['pose_file'])
    #     with open(pose_file, 'r') as f:
    #         poses = f.readlines()
    #     poses = [pose.strip().split(' ') for pose in poses[1:]]
    #     cam_params = [[float(x) for x in pose] for pose in poses]
    #     cam_params = [Camera(cam_param) for cam_param in cam_params]
    #     return cam_params


    def load_video(self, idx):
        video_dict = self.datasest[idx]
        clip_name = video_dict["clip_name"]
        video_caption = video_dict["caption"]
        frame_idx = video_dict["frame_idx"]

        # pose_file = os.path.join(self.root_path, video_dict['pose_file'])
        pose_file = video_dict['pose_file']
        with open(pose_file, 'r') as f:
            transforms = json.load(f) # data in nerfStudio format
        fl_x = transforms['fl_x']
        fl_y = transforms['fl_y']
        cx = transforms['cx'] / transforms['w']
        cy = transforms['cy'] / transforms['h']
        # transforms[']
        cam_params = []
        frames = [transforms['frames'][x] for x in frame_idx]
        for transform in frames:
            transform_matrix = transform['transform_matrix']
            cam_params.append(Camera([0, fl_x, fl_y, cx, cy] + transform_matrix[0] + transform_matrix[1] + transform_matrix[2]))

        frame_list = []
        for transform in frames:
            frame_list.append(os.path.join(video_dict['root_path'], transform['file_path']))
        video_reader = VidReader(frame_list)

        return clip_name, cam_params, video_reader, video_caption

    def get_batch(self, idx):
        clip_name, cam_params, video_reader, video_caption = self.load_video(idx)
        assert len(cam_params) >= self.sample_n_frames
        total_frames = len(cam_params)

        current_sample_stride = self.sample_stride

        if total_frames < self.sample_n_frames * current_sample_stride:
            maximum_sample_stride = int(total_frames // self.sample_n_frames)
            current_sample_stride = random.randint(self.minimum_sample_stride, maximum_sample_stride)

        cropped_length = self.sample_n_frames * current_sample_stride
        start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        assert end_frame_ind - start_frame_ind >= self.sample_n_frames
        # frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)
        frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)

        if self.shuffle_frames:
            perm = np.random.permutation(self.sample_n_frames)
            frame_indices = frame_indices[perm]
        # print(f"?frame_indices: {frame_indices}")

        pixel_values = torch.from_numpy(video_reader.get_batch(frame_indices)).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.

        cam_params = [cam_params[indice] for indice in frame_indices]
        if self.rescale_fxy:
            ori_h, ori_w = pixel_values.shape[-2:]
            ori_wh_ratio = ori_w / ori_h
            if ori_wh_ratio > self.sample_wh_ratio:       # rescale fx
                resized_ori_w = self.sample_size[0] * ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
            else:                                          # rescale fy
                resized_ori_h = self.sample_size[1] / ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]
        intrinsics = np.asarray([[cam_param.fx * self.sample_size[1],
                                  cam_param.fy * self.sample_size[0],
                                  cam_param.cx * self.sample_size[1],
                                  cam_param.cy * self.sample_size[0]]
                                 for cam_param in cam_params], dtype=np.float32)
        intrinsics = torch.as_tensor(intrinsics)[None]                  # [1, n_frame, 4]
        if self.relative_pose:
            c2w_poses = self.get_relative_pose(cam_params)
        else:
            c2w_poses = np.array([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
        c2w = torch.as_tensor(c2w_poses)[None]                          # [1, n_frame, 4, 4]
        if self.use_flip:
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(self.sample_n_frames, dtype=torch.bool, device=c2w.device)
        plucker_embedding = ray_condition(intrinsics, c2w, self.sample_size[0], self.sample_size[1], device='cpu',
                                          flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()

        return pixel_values, video_caption, plucker_embedding, flip_flag, clip_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        decord.bridge.set_bridge("torch")

        while True:
            try:
                video, video_caption, plucker_embedding, flip_flag, clip_name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)

        if self.use_flip:
            video = self.pixel_transforms[0](video)
            video = self.pixel_transforms[1](video, flip_flag)
            video = self.pixel_transforms[2](video)
        else:
            for transform in self.pixel_transforms:
                video = transform(video)
        # if self.return_clip_name:
        #     sample = dict(pixel_values=video, text=video_caption, plucker_embedding=plucker_embedding, clip_name=clip_name)
        # else:
        #     sample = dict(pixel_values=video, text=video_caption, plucker_embedding=plucker_embedding)
        
        # plucker_embedding to bfloat16
        plucker_embedding = plucker_embedding.to(torch.bfloat16)
        # print(f"?????shape of video and plucker_embedding: {video.shape}, {plucker_embedding.shape}")
        sample = dict(mp4=video, txt=video_caption, num_frames=self.sample_n_frames, fps=self.fps, plucker_embedding=plucker_embedding)
        return sample

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)
    