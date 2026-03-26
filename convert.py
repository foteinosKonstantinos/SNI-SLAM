import os
import shutil

base = "/home/fotkon/VSLAM/SNI-SLAM/recon"
idx = 0
os.mkdir("tornado/rgb")
os.mkdir("tornado/depth")
os.mkdir("tornado/semantic_class")
# for frame in sorted(os.listdir(base)):
for idx in range(len(os.listdir(base))):
    frame = f"frame{idx:03d}"
    print(frame)
    shutil.copy(os.path.join(base, frame ,"color.png"),os.path.join("tornado", "rgb", f"rgb_{idx}.png"))
    shutil.copy(os.path.join(base, frame ,"depth.png"),os.path.join("tornado", "depth", f"depth_{idx}.png"))
    idx += 1