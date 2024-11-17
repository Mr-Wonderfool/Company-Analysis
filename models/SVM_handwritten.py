import matplotlib.pyplot as plt
from models.utils.get_data import get_data
from sklearn.model_selection import train_test_split
from models.selector import Selector
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import cvxpy as cp

def select_feature():
    data_path = "../data/data.csv"
    data_X, data_Y = get_data(path=data_path, threshold=0.7)
    data_Y = np.array([1 if 1 == each else -1 for each in data_Y]) # convert 0 to -1
    selector = Selector(data_X=data_X, data_Y=data_Y)
    features_f, _ = selector.select_univar(percent=30, method="mutual_info_classif")
    _select_data_X = data_X[..., features_f]
    return _select_data_X, data_Y

def gaussianKernel(trainData_, tau=1):
    """Gaussian Kernel Matrix

    Parameters
    ----------
    trainData_: List of NDArray containing data vector
    tau: bandwidth parameter

    Returns
    -------
    kernel: kernel matrix with K[i,j] = <phi(x_i), phi(x_j)>
    """
    trainNum = len(trainData_)
    kernel = np.ones((trainNum, trainNum))
    for i in range(trainNum):  # traverse column
        for j in range(i):  # diagonal entries already suffices(1)
            diff = trainData_[i] - trainData_[j]
            kernel[i][j] = np.exp(-np.sum(diff**2, axis=-1) / (2 * tau * tau))
            kernel[j][i] = kernel[i][j]
    return kernel


# Check if the kernel matrix is positive semi-definite
def is_psd(matrix):
    """Check if a matrix is positive semi-definite."""
    eigvals = np.linalg.eigvals(matrix)
    return np.all(eigvals >= 0)

def test():
    # random data for testing
    TRAINNUM = 200
    samples = np.random.rand(TRAINNUM, 2)
    labels = np.array([1 if x1 < 2 * x2 - 0.3 else -1 for x1, x2 in samples])

    # Prepare training data
    train_data = []
    for i in range(TRAINNUM):
        train_data.append(samples[i])
        if labels[i] == 1:
            plt.plot(samples[i][0], samples[i][1], "rx")
        else:
            plt.plot(samples[i][0], samples[i][1], "b.")

    _train_data = np.array(train_data)

    tau = 0.1
    n = len(_train_data)

    # Compute Gaussian Kernel Matrix
    kernelMat = cp.psd_wrap(gaussianKernel(_train_data, tau))

    # # Check if the kernel matrix is PSD
    # if not is_psd(kernelMat):
    #     print("Warning: Kernel matrix is not positive semi-definite.")
    #     # Add small regularization to make it PSD
    #     kernelMat += np.eye(kernelMat.shape[0]) * 1e-5

    # Penalty factor
    C = 1
    alpha = cp.Variable(n)
    obj = cp.Maximize(cp.sum(cp.multiply(labels, alpha)) - (1/2) * cp.quad_form(alpha, kernelMat))

    # Constraints
    constraints = [0 <= cp.multiply(labels, alpha), cp.sum(alpha) == 0]

    prob = cp.Problem(obj, constraints)
    prob.solve()  # Solving quadratic programming

    # Check the solution
    alpha_with_label = np.array([i if abs(i) > 1e-5 else 0 for i in alpha.value])

    w = np.sum(np.multiply(alpha_with_label, _train_data.T), axis=1)

    # Calculate intercept
    positive_set = _train_data[labels == 1]
    negative_set = _train_data[labels == -1]
    max_neg = max([w @ neg for neg in negative_set])
    min_pos = min([w @ pos for pos in positive_set])
    b = -(max_neg + min_pos) / 2

    # Plot decision boundary
    m = -w[0] / w[1]
    c = -b / w[1]
    x = np.linspace(0, 1, 100)
    y = m * x + c
    plt.plot(x, y, label='Linear Boundary')
    plt.legend()
    plt.show()

def main():
    _select_data_X, _data_Y = select_feature()
    X_train, X_test, y_train, y_test = train_test_split(
        _select_data_X, _data_Y, test_size=0.2
    )
    tau = 0.1
    # smote = BorderlineSMOTE(random_state=42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)
    n = len(X_train)

    # Compute Gaussian Kernel Matrix
    kernelMat = cp.psd_wrap(gaussianKernel(X_train, tau))

    # Penalty factor
    C = 1
    alpha = cp.Variable(n)
    obj = cp.Maximize(cp.sum(cp.multiply(y_train, alpha)) - (1/2) * cp.quad_form(alpha, kernelMat))

    # Constraints
    constraints = [0 <= cp.multiply(y_train, alpha), 
                cp.multiply(y_train, alpha) <= C, 
                cp.sum(alpha) == 0]
    prob = cp.Problem(obj, constraints)
    prob.solve()  # Solving quadratic programming

    # Check the solution
    alpha_with_label = np.array([i if abs(i) > 1e-5 else 0 for i in alpha.value])

    w = np.sum(np.multiply(alpha_with_label, X_train.T), axis=1)

    # Calculate intercept
    positive_set = X_train[y_train == 1]
    negative_set = X_train[y_train == -1]
    max_neg = max([w @ neg for neg in negative_set])
    min_pos = min([w @ pos for pos in positive_set])
    b = -(max_neg + min_pos) / 2
    
    boundary = X_test @ w + b
    y_pred = [1 if each >=0 else -1 for each in boundary]
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()