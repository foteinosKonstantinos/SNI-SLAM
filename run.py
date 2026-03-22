# This file is a part of SNI-SLAM
#
import argparse

from src import config
from src.SNI_SLAM import SNI_SLAM
from src.tools.cull_mesh import cull_mesh

def main():
    parser = argparse.ArgumentParser(
        description='Arguments for running SNI_SLAM.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/SNI-SLAM.yaml')
    sni_slam = SNI_SLAM(cfg, args)

    sni_slam.run()
    
    mesh_out_semantic = f'output/Replica/dokimi/test/mesh/mesh_sem.ply'
    mesh_out_color = f'output/Replica/dokimi/test/mesh/mesh_rgb.ply'
    all_planes = (
            sni_slam.shared_planes_xy, sni_slam.shared_planes_xz, sni_slam.shared_planes_yz,
            sni_slam.shared_c_planes_xy, sni_slam.shared_c_planes_xz, sni_slam.shared_c_planes_yz,
            sni_slam.shared_s_planes_xy, sni_slam.shared_s_planes_xz, sni_slam.shared_s_planes_yz)
    sni_slam.mesher.get_mesh(mesh_out_color, all_planes, sni_slam.mapper.decoders, sni_slam.mapper.keyframe_dict, sni_slam.device, mesh_out_semantic=mesh_out_semantic, semantic=False)
    cull_mesh(mesh_out_color, sni_slam.cfg, sni_slam.args, sni_slam.device, estimate_c2w_list=sni_slam.estimate_c2w_list)

if __name__ == '__main__':
    main()
