import argparse
import sys
sys.path.append('DIL')

from DIL_exp import exp
from configs.DIP import cfg as DIP_cfg
from configs.DIP_Vid import cfg as DIP_Vid_cfg
from configs.DIP_Vid_3DCN import cfg as DIP_Vid_3DCN_cfg
from configs.DIP_Vid_Flow import cfg as DIP_Vid_Flow_cfg


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_mode', type=str, default='DIP_Vid_Flow', help='mode of the experiment: (DIP|DIP_Vid|DIP_Vid_3DCN|DIP_Vid_Flow)', metavar='')
    parser.add_argument('--resize', nargs='+', type=int, default=None, help='height and width of the output', metavar='')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size', metavar='')
    parser.add_argument('--data_path', type=str, 
                        default='data/Narrabeen/S1_Landsat', 
                        help='Input data', metavar='')
    parser.add_argument('--input_type', type=str, 
                        default='S1', 
                        help='Input noise data', metavar='')
    parser.add_argument('--cloud_ratio', type=float, 
                        default=0.5, 
                        help='Ratio of simulated cloud', metavar='')
    parser.add_argument('--res_dir', type=str, 
                        default='result', help='path to save the result', metavar='')
    parser.add_argument('--num_pass', type=int, default=10, help='number of passes for training', metavar='')
    parser.add_argument('--paired', action='store_true', help='Only use paired images for training')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def main(args):
    if args.train_mode == 'DIP':
        cfg = DIP_cfg
    elif args.train_mode == 'DIP-Vid':
        cfg = DIP_Vid_cfg
    elif args.train_mode == 'DIP-Vid-3DCN':
        cfg = DIP_Vid_3DCN_cfg
    elif args.train_mode == 'DIP-Vid-Flow':
        cfg = DIP_Vid_Flow_cfg
    else:
        raise Exception("Train mode {} not implemented!".format(args.train_mode))

    # resizes = {
    #     'Narrabeen': (384, 192),
    #     'Coolangatta': (512, 512),
    #     'OceanBeach': (512, 512),
    #     'Castelldefels': (448, 512)
    # }

    cfg['site'] = args.site
    cfg['resize'] = tuple(args.resize)
    cfg['resize'] = resizes[args.site]
    cfg['batch_size'] = args.batch_size
    if len(cfg['resize']) != 2: raise Exception("Resize must be a tuple of length 2!")
    cfg['data_path'] = args.data_path
    cfg['res_dir'] = args.res_dir
    cfg['optical'] = args.optical
    cfg['input_type'] = args.input_type
    cfg['cloudratio'] = args.cloudratio
    cfg['num_pass'] = args.num_pass
    cfg['paired'] = args.paired

    model_exp = exp(cfg)
    model_exp.create_data_loader()
    model_exp.visualize_single_batch()
    model_exp.create_model()
    model_exp.create_optimizer()
    model_exp.create_loss_function()
    model_exp.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)