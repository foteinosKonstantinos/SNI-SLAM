import argparse
from src import config
from src.SNI_SLAM import SNI_SLAM
from src.tools.cull_mesh import cull_mesh

MESH_SEMANTIC = f'mesh_sem.ply'
MESH_COLOR = f'mesh_rgb.ply'

def construct_mesh(sni_slam:SNI_SLAM, color_mesh:str, sem_mesh:str):
    all_planes = (
            sni_slam.shared_planes_xy, sni_slam.shared_planes_xz, sni_slam.shared_planes_yz,
            sni_slam.shared_c_planes_xy, sni_slam.shared_c_planes_xz, sni_slam.shared_c_planes_yz,
            sni_slam.shared_s_planes_xy, sni_slam.shared_s_planes_xz, sni_slam.shared_s_planes_yz)
    sni_slam.mesher.get_mesh(color_mesh, all_planes, sni_slam.mapper.decoders, sni_slam.mapper.keyframe_dict, sni_slam.device, mesh_out_semantic=sem_mesh, semantic=True)
    cull_mesh(color_mesh, sni_slam.cfg, sni_slam.args, sni_slam.device, estimate_c2w_list=sni_slam.estimate_c2w_list)
    cull_mesh(sem_mesh, sni_slam.cfg, sni_slam.args, sni_slam.device, estimate_c2w_list=sni_slam.estimate_c2w_list)

def main():
    parser = argparse.ArgumentParser(
        description='Arguments for running SNI_SLAM.'
    )
    parser.add_argument("config", type=str, help='Path to config file.')
    parser.add_argument("--color-mesh", type=str, help="Path to the color mesh output (.ply)", default=MESH_COLOR)
    parser.add_argument("--semantic-mesh", type=str, help="Path to the semantic mesh output (.ply)", default=MESH_SEMANTIC)
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/SNI-SLAM.yaml')
    sni_slam = SNI_SLAM(cfg, args)

    print("\nTraining model ...\n")
    sni_slam.run()

    mesh_out_color = args.color_mesh
    mesh_out_semantic = args.semantic_mesh

    print("\nConstructing meshes ...\n")
    construct_mesh(sni_slam=sni_slam, color_mesh=mesh_out_color, sem_mesh=mesh_out_semantic)

    print("\nFinished\n")
    

if __name__ == '__main__':
    main()
