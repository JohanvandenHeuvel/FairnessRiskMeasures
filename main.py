import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize, LinearConstraint
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold

from load_data import load_adult, load_toy_test


#################
# loss function #
#################
def hinge_loss(actual, predicted):
    loss = np.array(1 - np.multiply(actual, predicted)).clip(0)
    return loss


def square_hinge_loss(actual, predicted):
    return hinge_loss(actual, predicted) ** 2


####################
# loss aggregators #
####################
def CVaR(
    losses: np.ndarray, subgroups: list, p: float, alpha: float, clip: bool = True
) -> float:
    """Conditional Value at Risk / superquantile
    Loss aggregator, analogous to E[X | X > alpha-quantile]

    :param subgroup_losses: losses per subgroup to aggregate over
    :param p: discard all subgroup risks lower then p
    :param alpha: degree of fairness
    :param clip: clip negative values to zero, this makes the function non-continuous
    :return: aggregated losses
    """
    subgroup_losses = [losses[mask] for mask in subgroups]

    # compute the subgroup mean losses
    subgroup_mean_losses = np.zeros(len(subgroup_losses))
    for i, subgroup_loss in enumerate(subgroup_losses):
        # calculate the expected loss over the subgroup
        mean = np.mean(subgroup_loss - p) ** 2
        subgroup_mean_losses[i] = mean

    if clip:
        # NOTE this operation is not continuous!
        subgroup_mean_losses = subgroup_mean_losses.clip(min=0)

    return p + 1 / (1 - alpha) * np.mean(subgroup_mean_losses)


#####################
# linear classifier #
#####################
def predict(w, x, threshold=False):
    y_hat = x @ np.expand_dims(w, 1)

    if threshold:
        y_hat[y_hat < 0] = -1.0
        y_hat[y_hat >= 0] = 1.0

    return y_hat


def regularised_linear_scorer(w, x, y, loss, lmbd=0.1):
    """loss + L2 regularization

    :param w: weights
    :param x: data points
    :param y: true values
    :param lmdb: weight regularization
    :return: loss(y, x*w) + lmbda * ||w||_2^2
    """
    y_pred = predict(w, x)  # linear scorer
    regularisation = norm(w, 2)  # L2 regularization
    return loss(y, y_pred) + lmbd * regularisation


########
# data #
########
def get_data():
    # load the data
    train_data, test_data = load_adult()

    x_train = train_data.data
    y_train = train_data.target
    y_train = np.expand_dims(y_train, 1)

    x_test = test_data.data
    y_test = test_data.target
    y_test = np.expand_dims(y_test, 1)

    return x_train, y_train, x_test, y_test


# def constraint():
#     # constraints = (LinearConstraint(np.eye(len(mask), 0, np.inf)))
#     params[-1] + 1 / (1 - a) * np.mean(subgroup_loss_aggregation(
#         regularised_linear_scorer(
#             params[:-1], x=x_train, y=y_train, loss=hinge_loss
#         ),
#         subgroups=mask,
#         p=params[-1],
#     pass
#     f = lambda params: np.sum(params)
#     # set constraints
#     cons = (
#         {"type": "ineq", "fun": lambda w: w[:-1]},
#         {
#             "type": "ineq",
#             "fun": lambda w: CVaR(
#                 regularised_linear_scorer(w[:-1], x_train, y_train, square_hinge_loss),
#                 subgroups=mask,
#                 p=w[-1],
#                 alpha=a
#             ),
#         },
#     )


def optimize(x, y, alpha, lmbda, verbose=False):
    # get the indices of the groups induced by the sensitive feature
    sensitive_feature = 9
    sensitive_feature_values = np.unique(x[:, sensitive_feature])
    mask = []
    for i, value in enumerate(sensitive_feature_values):
        mask.append(x[:, sensitive_feature] == value)

    x0 = np.zeros(x.shape[1] + 1)
    f = lambda params: CVaR(
        regularised_linear_scorer(params[:-1], x, y, square_hinge_loss, lmbd=lmbda),
        subgroups=mask,
        p=params[-1],
        alpha=alpha,
    )
    if verbose:
        print(f"--> Minimizing the objective for alpha={alpha} and lambda={lmbda} ...")
    return minimize(f, x0, options={"disp": verbose})


def main(lambdas=None):
    if lambdas is None:
        lambdas = []

    x_train, y_train, x_test, y_test = get_data()

    # find optimal solution
    optimal_lambdas = []
    for i, a in enumerate(np.arange(1, 10) / 10):

        if len(lambdas) == i:
            best_lambda = 0
            best_score = 0
            for l in np.arange(0, 11) / 10:
                kf = KFold(n_splits=5)
                scores = []
                for train, validation in kf.split(x_train):
                    x = x_train[train]
                    y = y_train[train]
                    x_val = x_train[validation]
                    y_val = y_train[validation]

                    result = optimize(x, y, a, l)

                    w = result.x[:-1]
                    scores.append(
                        balanced_accuracy_score(
                            y_val, predict(w, x_val, threshold=True)
                        )
                    )
                if np.mean(scores) > best_score:
                    best_score = np.mean(scores)
                    best_lambda = l
            lambdas.append(best_lambda)

        print(f"Found optimal lambdas: {lambdas} for alphas {np.arange(1, 10)/10}")

        l = lambdas[i]
        result = optimize(x_train, y_train, alpha=a, lmbda=l, verbose=True)

        w = result.x[:-1]
        p = result.x[-1]

        # get the indices of the groups induced by the sensitive feature
        sensitive_feature = 9
        sensitive_feature_values = np.unique(x_train[:, sensitive_feature])
        mask_train = []
        mask_test = []
        for value in sensitive_feature_values:
            mask_train.append(
                np.logical_and(
                    x_train[:, sensitive_feature] == value, y_train.flatten() == 1
                )
            )
            mask_test.append(
                np.logical_and(
                    x_test[:, sensitive_feature] == value, y_test.flatten() == 1
                )
            )

        print(f"p: {p}")
        y_predict = predict(w, x_train, threshold=True)
        print(
            f"Training data balanced accuracy: {balanced_accuracy_score(y_train, y_predict):0.2f}"
        )
        print(
            f"Training data equality of opportunity: {np.abs(np.mean(y_predict[mask_train[0]]) - np.mean(y_predict[mask_train[1]]))}"
        )
        y_predict = predict(w, x_test, threshold=True)
        print(
            f"Test data balanced accuracy: {balanced_accuracy_score(y_test, y_predict):0.2f}"
        )
        print(
            f"Test data equality of opportunity: {np.abs(np.mean(y_predict[mask_test[0]]) - np.mean(y_predict[mask_test[1]]))}"
        )
        print("<-- Done!")


if __name__ == "__main__":
    main(lambdas=[0.1] * 9)
