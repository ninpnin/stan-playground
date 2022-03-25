import stan
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from bidict import bidict
import random

def get_vocab(s, limit=1):
    vocab = {}
    for wd in s:
        vocab[wd] = vocab.get(wd, 0) + 1
    vocab = [wd for wd, v in vocab.items() if v >= limit]
    vocab = bidict({wd: ix + 1 for ix, wd in enumerate(vocab)})
    return vocab

def preprocess(s, vocab):
    for wd in s:
        if wd in vocab:
            yield vocab[wd]

def fit_posterior(args):
    with open('data/' + args.data) as f:
        data = f.read().split()

    vocab = get_vocab(data)
    data = list(preprocess(data, vocab))
    ps_w_0 = data[2:] + data[1:]
    ps_c_0 = data[:-2] + data[:-1]
    ps_w = ps_w_0 + ps_c_0
    ps_c = ps_c_0 + ps_w_0
    print(len(ps_w), len(ps_c))
    ns_w = ps_w[:]
    ns_c = ps_c[:]
    random.shuffle(ns_c)

    w = ps_w + ns_w
    c = ps_c + ns_c

    x = [1] * len(ps_w) + [0] * len(ns_w)
    N = len(x)

    assert N == len(w)
    assert N == len(c)
    V = max(data)
    D = 25
    print(N, V, D)
    data = dict(N=N,
                V=V,
                D=D,
                w=w,
                c=c,
                x=x,
                lambda0=1.0
        )
    with open('models/' + args.model + '.stan') as f:
        stan_code = f.read()

    #print(type(data), data.keys(), data.values())

    posterior = stan.build(stan_code, data=data, random_seed=1)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    fit_df = fit.to_frame()
    
    return fit_df

def main(args):
    fit = fit_posterior(args)
    print(fit)

    # TODO: Plot
    print(fit.columns)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="nietzsche.txt")
    parser.add_argument("--model", type=str, default="bernoulli_embeddings")
    args = parser.parse_args()
    main(args)
