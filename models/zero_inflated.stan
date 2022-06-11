data {
  int<lower=0> N;
  array[N] int<lower=0> y;         // Samples
}
parameters {
  real<lower=0, upper=1> tau;      // standard deviation in treatment effects
  real<lower=0> lambda;          // unscaled deviation from mu by school
}

model {
  for (n in 1:N) {
    if (y[n] == 0) {
      target += log_sum_exp(bernoulli_lpmf(1 | tau),
                            bernoulli_lpmf(0 | tau) + poisson_lpmf(y[n] | lambda));
    } else {
      target += bernoulli_lpmf(0 | tau) + poisson_lpmf(y[n] | lambda);
    }
  }
}