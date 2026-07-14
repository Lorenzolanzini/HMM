# HMM

NumPy implementation of scaled multi-sequence Hidden Markov Models.

- `hmm` class: Forward-Backward, Viterbi, Expectation-Maximization (with ECME inner loop)
- `emission_models`: Discrete, Gaussian, Student-t
- Brent's method for degrees-of-freedom optimization (Student-t)
- Scaled forward-backward for numerical stability on long sequences
- Tests on synthetic data, ready for real data (BTC, WTI, SPY)
