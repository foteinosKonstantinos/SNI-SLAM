import torch
import argparse

from src import config
from src.SNI_SLAM import SNI_SLAM

from src.tools.cull_mesh import cull_mesh

WEIGHTS = torch.load("/home/fotkon/VSLAM/SNI-SLAM/output/Replica/room1/test/ckpts/01999.tar")

parser = argparse.ArgumentParser(
        description='Arguments for running SNI_SLAM.'
    )
parser.add_argument('--config', type=str, default='/home/fotkon/VSLAM/SNI-SLAM/configs/Replica/room1.yaml', help='Path to config file.')
parser.add_argument('--input_folder', type=str,
                    help='input folder, this have higher priority, can overwrite the one in config file')
parser.add_argument('--output', type=str,
                    help='output folder, this have higher priority, can overwrite the one in config file')
args = parser.parse_args()

cfg = config.load_config(args.config, 'configs/SNI-SLAM.yaml')


sni_slam = SNI_SLAM(cfg, args)
sni_slam.shared_decoders.load_state_dict(WEIGHTS["decoder_state_dict"])

sni_slam.shared_s_planes_xy = WEIGHTS["s_planes_xy"]
sni_slam.shared_s_planes_xz = WEIGHTS["s_planes_xz"]
sni_slam.shared_s_planes_yz = WEIGHTS["s_planes_yz"]

sni_slam.shared_planes_xy = WEIGHTS["planes_xy"]
sni_slam.shared_planes_xz = WEIGHTS["planes_xz"]
sni_slam.shared_planes_yz = WEIGHTS["planes_yz"]

sni_slam.shared_c_planes_xy = WEIGHTS["c_planes_xy"]
sni_slam.shared_c_planes_yz = WEIGHTS["c_planes_xz"]
sni_slam.shared_c_planes_yz = WEIGHTS["c_planes_yz"]

sni_slam.estimate_c2w_list = WEIGHTS["estimate_c2w_list"]

sni_slam.mapper.keyframe_list = WEIGHTS["keyframe_list"]
sni_slam.idx = WEIGHTS["idx"]
sni_slam.gt_c2w_list = WEIGHTS["gt_c2w_list"]

mesh_out_semantic = f'output/mesh/mesh_sem.ply'
mesh_out_color = f'output/mesh/mesh_rgb.ply'
all_planes = (
        sni_slam.shared_planes_xy, sni_slam.shared_planes_xz, sni_slam.shared_planes_yz,
        sni_slam.shared_c_planes_xy, sni_slam.shared_c_planes_xz, sni_slam.shared_c_planes_yz,
        sni_slam.shared_s_planes_xy, sni_slam.shared_s_planes_xz, sni_slam.shared_s_planes_yz)
sni_slam.mesher.get_mesh(mesh_out_color, all_planes, sni_slam.mapper.decoders, sni_slam.mapper.keyframe_dict, sni_slam.device, mesh_out_semantic=mesh_out_semantic, semantic=False)
cull_mesh(mesh_out_color, sni_slam.cfg, sni_slam.args, sni_slam.device, estimate_c2w_list=sni_slam.estimate_c2w_list)