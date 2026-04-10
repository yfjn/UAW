import os
import sys
sys.path.append("/media/dongli911/Documents/Workflow/YanFangjun/UAW/")
from UAWUP.uaw_attack import RepeatAdvPatch_Attack
import logging
from AllConfig.GConfig import abspath
import warnings
import argparse
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Train adversarial watermark with UAWUP')
    parser.add_argument('--eps', type=int, default=100, help='Perturbation budget')
    parser.add_argument('--evaluate', type=bool, default=False, help='Whether to freeze STD parameters')
    parser.add_argument('--decay', type=float, default=0.1, help='Momentum decay rate')
    parser.add_argument('--step_alpha', type=int, default=3, help='Step size (will be divided by 255)')
    parser.add_argument('--lm_mui_thre', type=float, default=0.06, help='Threshold for mui')
    parser.add_argument('--size', type=int, default=30, help='Adversarial patch size')
    parser.add_argument('--lambdaw', type=float, default=0.01, help='Weight for mmloss')
    parser.add_argument('--lambdax', type=float, default=0, help='Weight for l1loss, recommand 1e-5')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--gap', type=int, default=5, help='Gap for diversity module')
    parser.add_argument('--T', type=int, default=100, help='Total number of iterations')
    parser.add_argument('--model_name', type=str, default='CRAFT', choices=['CRAFT', 'EasyOCR'], help='Model name for training')
    parser.add_argument('--save_mui', type=float, nargs='+', default=[0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12], help='MUI thresholds for saving checkpoints')
    parser.add_argument('--debug', type=bool, default=False, help='Enable detailed debug output')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    log_dir = os.path.join(abspath, f'results/AllData_Mylog/{args.model_name}')
    result_dir = os.path.join(abspath, f'results/AllData_results/{args.model_name}')

    log_name = os.path.join(log_dir, f's{args.size}_eps{args.eps}_step{args.step_alpha}_w{args.lambdaw}_x{args.lambdax}.log')
    savedir = os.path.join(result_dir, f's{args.size}_eps{args.eps}_step{args.step_alpha}_w{args.lambdaw}_x{args.lambdax}')

    RAT = RepeatAdvPatch_Attack(
        data_root=os.path.join(abspath, "AllData"),
        savedir=savedir,
        log_name=log_name,
        alpha=args.step_alpha / 255,
        batch_size=args.batch_size,
        gap=args.gap,
        T=args.T,
        lambdaw=args.lambdaw,
        lambdax=args.lambdax,
        eps=args.eps / 255,
        decay=args.decay,
        adv_patch_size=(1, 3, args.size, args.size),
        model_name=args.model_name,
        save_mui=args.save_mui,
        lm_mui_thre=args.lm_mui_thre,
        evaluate=args.evaluate,
        debug=args.debug
    )
    RAT.train()
    del(RAT)
    logging.shutdown()
