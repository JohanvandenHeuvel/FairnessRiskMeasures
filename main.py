from scipy.optimize import minimize

from sklearn.metrics import accuracy_score

from load_data import load_adult, load_toy_test
from losses import *


def Expectation(losses: np.ndarray, subgroups: list) -> float:
    """Expectation
    Loss aggregator, E[X]

    :param losses: losses to aggregate over
    :param subgroups: list of list filled with indices of the subgroup
    :return: aggregated losses
    """

    subgroup_losses = [losses[mask] for mask in subgroups]

    # compute the subgroup mean losses
    subgroup_mean_losses = np.zeros(len(subgroup_losses))
    for i, subgroup_loss in enumerate(subgroup_losses):
        # calculate the expected loss over the subgroup
        # and take the square (squared hinge loss)
        mean = np.mean(subgroup_loss) ** 2
        subgroup_mean_losses[i] = mean

    return np.mean(subgroup_mean_losses)


def CVaR(
    subgroup_losses: np.ndarray, p: float, alpha: float, clip: bool = True
) -> float:
    """Conditional Value at Risk / superquantile
    Loss aggregator, analogous to E[X | X > alpha-quantile]

    :param subgroup_losses: losses per subgroup to aggregate over
    :param p: discard all subgroup risks lower then p
    :param alpha: degree of fairness
    :param clip: clip negative values to zero, this makes the function non-continuous
    :return: aggregated losses
    """

    # compute the subgroup mean losses
    subgroup_mean_losses = np.zeros(len(subgroup_losses))
    for i, subgroup_loss in enumerate(subgroup_losses):
        # calculate the expected loss over the subgroup
        mean = np.mean(subgroup_loss - p)
        subgroup_mean_losses[i] = mean

    if clip:
        # NOTE this operation is not continuous!
        subgroup_mean_losses = subgroup_mean_losses.clip(min=0)

    return p + 1 / (1 - alpha) * np.mean(subgroup_mean_losses)


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
    return loss(y, y_pred)


def main():

    # load the data
    train_data, test_data = load_adult()

    x_train = train_data.data
    y_train = train_data.target
    y_train = np.expand_dims(y_train, 1)

    x_test = test_data.data
    y_test = test_data.target
    y_test = np.expand_dims(y_test, 1)

    sensitive_feature = 9
    sensitive_feature_values = np.unique(x_train[:, sensitive_feature])
    mask = []
    for i, value in enumerate(sensitive_feature_values):
        mask.append(x_train[:, sensitive_feature] == value)

    # find optimal solution
    x0 = np.zeros(x_train.shape[1])
    f = lambda w: Expectation(
        regularised_linear_scorer(w, x=x_train, y=y_train, loss=hinge_loss), subgroups=mask
    )
    print("Minimizing the objective...")
    result = minimize(
        f,
        x0,
        options={"disp": True, "return_all": True},
    )

    print(accuracy_score(y_train, predict(result.x, x_train, threshold=True)))
    print(accuracy_score(y_test, predict(result.x, x_test, threshold=True)))

    print("Done!")


if __name__ == "__main__":
    main()
