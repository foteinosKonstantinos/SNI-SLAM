import argparse
from src import config
from src.SNI_SLAM import SNI_SLAM


MESH_SEMANTIC = f'mesh_sem.ply'
MESH_COLOR = f'mesh_rgb.ply'


def main():
    parser = argparse.ArgumentParser(
        description='Arguments for running SNI_SLAM.'
    )
    parser.add_argument("config", type=str, help='Path to config file.')
    # parser.add_argument("--color-mesh", type=str, help="Path to the color mesh output (.ply)", default=MESH_COLOR)
    # parser.add_argument("--semantic-mesh", type=str, help="Path to the semantic mesh output (.ply)", default=MESH_SEMANTIC)
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/SNI-SLAM.yaml')
    sni_slam = SNI_SLAM(cfg, args)

    print("\nTraining model ...\n")
    sni_slam.run()

    # mesh_out_color = args.color_mesh
    # mesh_out_semantic = args.semantic_mesh

    # print("\nConstructing meshes ...\n")
    # construct_mesh(sni_slam=sni_slam, color_mesh=mesh_out_color, sem_mesh=mesh_out_semantic)

    print("\nFinished\n")
    

if __name__ == '__main__':
    main()
