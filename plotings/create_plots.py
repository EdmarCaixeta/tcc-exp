from os import path
from argparse import ArgumentParser
from utils import *

def make_parse() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('exp_dir', type=str, help='directory with all experiment directories')
    parser.add_argument('graph', type=str, help='boxplot, histogram, loss, scatter, all')
    return parser

if __name__ == '__main__':
    parser = make_parse()
    args = parser.parse_args()

    if args.graph == 'boxplot' or args.graph== 'all':
        boxplot(args.exp_dir)
        print("Boxplot OK")

    if args.graph == 'histogram' or args.graph== 'all':
        histogram(args.exp_dir)
        print("Histograms OK")

    if args.graph == 'loss' or args.graph== 'all':
        loss(args.exp_dir)
        print("Loss OK")
    
    if args.graph == 'scatter' or args.graph== 'all':
        scatter(args.exp_dir)
        print("Scatter OK")

    
    
