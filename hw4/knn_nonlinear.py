import numpy as np
import matplotlib.pyplot as plt

def least_square(y:np.ndarray, x_mat: np.matrix):
    a = np.matmul(x_mat.T, x_mat)
    try:
        inv_a = np.linalg.inv(a)
    except np.linalg.LinAlgError:
        inv_a = np.linalg.pinv(a)

    y = np.squeeze(y)
    b = x_mat.T.dot(y)
    return np.matmul(inv_a, b)

def normalize_axis(a: np.ndarray):
    """
    Make sure the matrix has two dimension
    """
    if len(a.shape) == 1:
        return np.expand_dims(a, axis=1)
    else:
        return a
def load_npz(path: str):
    data = np.load(path)
    x, y  = data['X'], data['y']
    x = normalize_axis(x)
    y = normalize_axis(y)
    return x, y
def build_matrix(x: np.ndarray):
    return np.c_[np.ones((len(x), 1)), x]
def mse_loss(a, b):
    return np.mean((a - b) ** 2)


def split_data(x, y, split_ratio = 0.9, seed=456):
    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(seed)
    np.random.shuffle(y)

    num_sample = len(x)
    interval = int(num_sample * split_ratio)
    x_train, y_train = x[:interval,...], y[:interval,...]
    x_test, y_test = x[interval:,...], y[interval:,...]

    return (x_train, y_train), (x_test, y_test)

class KNN_nonlinear_regressor(object):
    def __init__(self, n_neighbors:int=10):
        self.n_neighbors = n_neighbors
        self.x = None
        self.y = None
    def fit(self, x:np.ndarray, y:np.ndarray):
        self.x = normalize_axis(x)
        self.y = normalize_axis(y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(self.x, np.ndarray):
            raise ValueError("Please call .fit() first!")
        x = normalize_axis(x)

        distance = np.sqrt(np.sum((x[:,np.newaxis,:] - self.x[np.newaxis,:,:]) ** 2, axis=-1))
        min_indices = np.argsort(distance, axis=-1)
        neighbors_indices = min_indices[:,:self.n_neighbors]
        result = []
        for i, indices in enumerate(neighbors_indices):
            neighbors_y = self.y[indices,...]
            result.append(np.mean(neighbors_y))


        return normalize_axis(np.squeeze(np.array(result)))
    
    def _plot2d(self):
        plt.figure()
        plt.xlabel('X')
        plt.ylabel('y')
        plt.scatter(self.x, self.y, s=2)

        new_x = np.arange(min(self.x), max(self.x), 0.001)
        new_y = self.predict(new_x)
        plt.scatter(new_x, new_y, c='r',s=0.5)
        plt.show()
    def _plot3d(self):
        ax = plt.axes(projection='3d')
        ax.set_xlabel('$x_0$')
        ax.set_ylabel('$x_1$')
        ax.set_zlabel('$y$')
        x0 = self.x[:, 0]
        x1 = self.x[:, 1]

        ax.scatter(x0, x1, self.y, c=self.y, cmap='Reds', marker='o')
        x0_range = np.arange(min(x0), max(x0), 0.1)
        x1_range = np.arange(min(x1), max(x1), 0.1)

        new_x = []
        for x0 in x0_range:
            for x1 in x1_range:
                new_x.append([x0, x1])
        new_x = np.array(new_x)
        new_y = self.predict(new_x)
        ax.scatter(new_x[:,0], new_x[:,1], new_y, c=new_y, cmap='Blues', marker='o', s=1)
        plt.show()
    def plot_fig(self):
        if not isinstance(self.x, np.ndarray):
            raise ValueError("Please call .fit() first!")
        if self.x.shape[1] == 1:
            self._plot2d()
        else:
            self._plot3d()

if __name__ == "__main__":
    x, y = load_npz('./data1.npz')
    (x_train, y_train), (x_test, y_test) = split_data(x, y)

    start_n = 5
    end_n = 30
    losses = []
    for n in range(start_n, end_n + 1):
        knn_reg = KNN_nonlinear_regressor(n)
        knn_reg.fit(x_train, y_train)
        y_predict = knn_reg.predict(x_test)
        losses.append(mse_loss(y_predict, y_test))
    plt.figure()
    plt.plot(range(start_n, end_n + 1), losses)
    plt.ylabel('MSE Error')
    plt.xlabel('neighbor')
    plt.show()
    knn_reg = KNN_nonlinear_regressor(20)
    knn_reg.fit(x, y)
    knn_reg.plot_fig()