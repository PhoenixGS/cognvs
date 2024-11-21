import os
import json
import numpy as np

def step_one(root_path):
    # os.walk
    # each video:
    # ./
    #     transforms.json
    #     images_4/

    data = []
    for root, dirs, files in os.walk(root_path):
        if "transforms.json" in files:
            uid = os.path.basename(root)
            dat = {
                "video_name": uid,
                "path": root,
            }
            data.append(dat)
    return data

def step_two(origin):
    data = []
    for video in origin:
        uid = video["video_name"]
        path = video["path"]
        try:
            with open(os.path.join(path, "transforms.json"), "r") as f:
                transforms = json.load(f)
        except:
            print(f"Error reading {uid}")
            continue

        frames = transforms["frames"]
        
        for idx, i in enumerate(range(0, len(frames), 96)):
            dat = {
                "clip_name": f"{uid}_{idx}",
                "video_name": uid,
                "frame_idx": [],
                "caption": "",
                "root_path": path,
                "pose_file": os.path.join(path, "transforms.json"),
            }
            for j in range(i, i+96, 2):
                dat["frame_idx"].append(j)
            data.append(dat)
    return data

if __name__ == "__main__":
    root_path = "/mnt/petrelfs/zhaohang.p/data/dl3dv-10k"
    output_json_one = "dl3dv-10k.json"
    output_json_two = "train.json"

    data = step_one(root_path)
    output_json_one = os.path.join(root_path, output_json_one)
    with open(output_json_one, "w") as f:
        json.dump(data, f, indent=4)

    data2 = step_two(data)
    output_json_two = os.path.join(root_path, output_json_two)
    print(f"Writing to {output_json_two}")
    with open(output_json_two, "w") as f:
        json.dump(data2, f, indent=4)
