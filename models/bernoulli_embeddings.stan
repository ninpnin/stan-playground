data {
  // training data
  int<lower=1> V;               // num words
  int<lower=1> D;               // dimensions
  int<lower=0> N;               // total word instances
  int<lower=1,upper=V> w[N];    // center word at n
  int<lower=1,upper=V> c[N];    // context word at n
  int<lower=0,upper=1> x[N];    // x | x_c true or not
  // hyperparameters
  real<lower=0> lambda0;         // word vector
}
parameters {
  real rho[V, D];                  // word vector
  real alpha[V, D];                // context vector
}
transformed parameters {
  real theta[N];
  for (n in 1:N) {
    //rho_n = rho[w[n]];
    //alpha_n = alpha[c[n]];
    //eta = dot_product(rho_n, alpha_n);
    //theta[n] = inv_logit(eta);
    theta[n] = inv_logit(dot_product(rho[w[n]], alpha[c[n]]));
  }
}
model {
  for (v in 1:V) {
    for (d in 1:D) {
      target += normal_lpdf(rho[v, d] | 0, lambda0);
      target += normal_lpdf(alpha[v, d] | 0, lambda0);
    }
  }
  //rho ~ normal(0, lambda);
  //alpha ~ normal(0, lambda);
  target += bernoulli_lpmf(x | theta);
}