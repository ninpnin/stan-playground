import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def main(args):
    df = pd.read_csv(args.datapath)
    variables = args.variables
    if variables is None:
        for col in df.columns:
            print(col, col[:-2])
        variables = list(filter(lambda col: col[-2:] != "__", df.columns))
    
    sns.set_theme()
    for var in variables:
        sns.histplot(data=df, x=var, kde=True)
        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--variables", nargs='+', type=str, default=None, help="Which variables to plot. Defaults to all.")
    args = parser.parse_args()
    main(args)