import os
import cv2

if __name__ == "__main__":
    root = "/mnt/petrelfs/zhaohang.p/data/dl3dv-10k/main/7K/7f460b01cbb51c3a579e8acbcf2dff7aa611f784dbd85148ea6327b3c3bfbe3c/images_4/"
    # merge all the images in the folder to video mp4
    img_names = os.listdir(root)
    img_names.sort()
    img_paths = [os.path.join(root, img_name) for img_name in img_names]
    img = cv2.imread(img_paths[0])
    h, w, _ = img.shape
    fps = 12
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(root, 'output.mp4'), fourcc, fps, (w, h))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        video.write(img)
    video.release()
    print("Done")