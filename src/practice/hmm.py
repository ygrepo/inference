# https: // en.wikipedia.org / wiki / Hidden_Markov_model  # A_concrete_example
import numpy as np


def normalize(u):
    z = np.sum(u)
    return u / z, z


def forward(obs, transition_probability, emission_probability, start_probability):
    T = len(obs)
    N = transition_probability.shape[0]
    alpha = np.zeros((T, N))
    alpha[0] = emission_probability[:, obs[0]] * start_probability
    for i in range(1, T):
        alpha[i] = emission_probability[:, obs[i]] * np.inner(alpha[i - 1], transition_probability)
    return alpha


def likelihood(alpha):
    return alpha[-1].sum()


def backward(obs, transition_probability, emission_probability):
    T = len(obs)
    N = transition_probability.shape[0]
    beta = np.zeros((N, T))
    beta[:, -1] = 1
    for t in range(T - 2, -1, -1):
        for n in range(N):
            beta[n, t] = np.sum(transition_probability[n, :] * beta[:, t + 1] * emission_probability[:, obs[t + 1]])
    return beta


def gamma(alpha, beta):
    obs_prob = likelihood(alpha)
    return np.multiply(alpha.T, beta) / obs_prob


def viterbi(obs, transition_probability, emission_probability, start_probability):
    T = len(obs)
    N = transition_probability.shape[0]
    delta = np.zeros((T, N))
    psi = np.zeros((T, N))
    delta[0] = start_probability * emission_probability[:, obs[0]]
    for t in range(1, T):
        for n in range(N):
            delta[t, n] = np.max(delta[t - 1] * transition_probability[:, n]) * emission_probability[n, obs[t]]
            psi[t, n] = np.argmax(delta[t - 1] * transition_probability[:, n])
    # Now we backtyrack
    states = np.zeros(T, dtype=int)
    states[T - 1] = np.argmax(delta[T - 1])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


if __name__ == "__main__":
    states = ("Rainy", "Sunny")
    observations = ("walk", "shop", "clean")
    start_probability = np.array([0.6, 0.4])
    transition_probability = np.array([[0.7, 0.3], [0.4, 0.6]])
    emission_probability = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
    bob_says = np.array([0, 2, 1, 1, 2, 0])  # walk, clean, shop, shop, clean, walk
    alpha = forward(bob_says, transition_probability, emission_probability, start_probability)
    print("Probability rainy={:.4f}, sunny={:.4f}".format(alpha[-1, 0], alpha[-1, 1]))
    alice_hears = viterbi(bob_says, transition_probability, emission_probability, start_probability)
    print("Bob says:", ", ", list(map(lambda y: observations[y], bob_says)))
    print("Alice hears:", ", ", list(map(lambda s: states[s], alice_hears)))

    # ===============
    # start_probability = np.array([.8, .2])
    # transition_probability = np.array([[.6, .4], [.3, .7]])
    # emission_probability = np.array([[.3, .4, .3], [.4, .3, .3]])
    # observations = ("R", "W", "B")
    # obs = np.array([0, 1, 2, 2])
    # alpha = forward(obs, transition_probability, emission_probability, start_probability)
    # print(alpha)
    # print(likelihood(alpha))
    # beta = backward(obs, transition_probability, emission_probability)
    # g = gamma(alpha, beta)
    # print(g)

    states = ("Happy", "Angry")
    observations = ("smile", "frown", "laugh", "yell")
    start_probability = np.array([1., 0])
    transition_probability = np.array([[0.9, 0.1], [0.9, 0.1]])
    emission_probability = np.array([[0.6, 0.1, 0.2, 0.1], [0.1, 0.6, 0.1, 0.2]])
    observed = np.array([1, 1, 1, 1, 1])
    inferred_states = viterbi(observed, transition_probability, emission_probability, start_probability)
    print("States:", list(map(lambda s: states[s], inferred_states)))
