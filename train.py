import os 
import argparse

from utils import set_seed, get_results_path

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--save_folder', type=str, default='save', help='save folder path for trained results')

    return parser.parse_args()

def main(): 
    args = parse_args()
    
    set_seed(args.seed)
    results_path = get_results_path(args.save_folder)
