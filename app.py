import stan
import json

def fit_posterior(args):
    model = args.model
    dataset = args.model
    if args.dataset is not None:
        dataset = args.dataset
    
    # Load data
    with open(f'data/{args.model}.json') as f:
        data = json.load(f)
    with open(f'models/{args.model}.stan') as f:
        stan_code = f.read()

    print(type(data), data.keys(), data.values())

    posterior = stan.build(stan_code, data=data)
    fit = posterior.sample(num_chains=4, num_samples=args.samples)
    
    return fit

def main(args):
    fit = fit_posterior(args)
    fit_df = fit.to_frame()
    print(fit_df)
    if args.savepath is not None:
        fit_df.to_csv(args.savepath, index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="schools")
    parser.add_argument("--dataset", type=str, default=None,
        help="Name of the dataset. If not provided, use model default.")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--savepath", type=str, default=None)
    args = parser.parse_args()
    main(args)
