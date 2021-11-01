import numpy as np


def generate_toy_data(n_samples, n_samples_low, n_dimensions):
    np.random.seed(0)
    varA = 0.8
    aveApos = [-1.0] * n_dimensions
    aveAneg = [1.0] * n_dimensions
    varB = 0.5
    aveBpos = [0.5] * int(n_dimensions / 2) + [-0.5] * int(
        n_dimensions / 2 + n_dimensions % 2
    )
    aveBneg = [0.5] * n_dimensions

    X = np.random.multivariate_normal(
        aveApos, np.diag([varA] * n_dimensions), n_samples
    )
    X = np.vstack(
        [
            X,
            np.random.multivariate_normal(
                aveAneg, np.diag([varA] * n_dimensions), n_samples
            ),
        ]
    )
    X = np.vstack(
        [
            X,
            np.random.multivariate_normal(
                aveBpos, np.diag([varB] * n_dimensions), n_samples_low
            ),
        ]
    )
    X = np.vstack(
        [
            X,
            np.random.multivariate_normal(
                aveBneg, np.diag([varB] * n_dimensions), n_samples
            ),
        ]
    )
    sensible_feature = [1] * (n_samples * 2) + [-1] * (n_samples + n_samples_low)
    sensible_feature = np.array(sensible_feature)
    sensible_feature.shape = (len(sensible_feature), 1)
    X = np.hstack([X, sensible_feature])
    y = [1] * n_samples + [-1] * n_samples + [1] * n_samples_low + [-1] * n_samples
    y = np.array(y)
    sensible_feature_id = len(X[1, :]) - 1
    idx_A = list(range(0, n_samples * 2))
    idx_B = list(range(n_samples * 2, n_samples * 3 + n_samples_low))

    # print('(y, sensible_feature):')
    # for el in zip(y, sensible_feature):
    #     print(el)
    return X, y, sensible_feature_id, idx_A, idx_B
