import stan
import json

def main(args):
    with open('data/' + args.model + '.json') as f:
        data = json.load(f)

    with open('models/' + args.model + '.stan') as f:
        stan_code = f.read()

    print(type(data), data.keys(), data.values())

    posterior = stan.build(stan_code, data=data, random_seed=1)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    fit_df = fit.to_frame()
    
    return fit_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="schools")
    args = parser.parse_args()
    fit = main(args)
    print(fit)