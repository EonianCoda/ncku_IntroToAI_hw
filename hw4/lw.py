import numpy as np
import matplotlib.pyplot as plt
def normalize_axis(a: np.ndarray):
    """
    Make sure the matrix has two dimension
    """
    if len(a.shape) == 1:
        return np.expand_dims(a, axis=1)
    else:
        return a

def build_matrix(x: np.ndarray):
    return np.c_[np.ones((len(x), 1)), x]
def mse_loss(a, b):
    return np.mean((a - b) ** 2)

def load_npz(path: str):
    data = np.load(path)
    x, y  = data['X'], data['y']
    x = normalize_axis(x)
    y = normalize_axis(y)
    return x, y
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

class Local_weighted_linear_regressor(object):
    def __init__(self, kernel_width:int=10):
        self.x = None
        self.y = None
        self.kernel_width = kernel_width
    def weighted(self, query_x:np.ndarray):
        k_coff = 4 / (self.kernel_width ** 2)

        query_x = normalize_axis(query_x)
        num_x = len(self.x)
        distance = np.sqrt(np.sum((query_x[:,np.newaxis,:] - self.x[np.newaxis,:,:]) ** 2, axis=-1))
        identity_mask = (np.identity(num_x) == 1)

        for dist in distance:
            weight = np.maximum(0, 1 - k_coff * dist * dist)
            result = np.zeros((len(self.x), len(self.x)))
            result[identity_mask] = weight
            yield result
    def fit(self, x:np.ndarray, y:np.ndarray):
        self.x = normalize_axis(x)
        self.y = normalize_axis(y)
    @staticmethod
    def solve_LWLR(x:np.ndarray, y:np.ndarray, w:np.ndarray):
        a = x.T.dot(w).dot(x)
        try:
            inv_a = np.linalg.inv(a)
        except np.linalg.LinAlgError:
            inv_a = np.linalg.pinv(a)
        return inv_a.dot(x.T).dot(w).dot(y)

    def predict(self, query_xs: np.ndarray) -> np.ndarray:
        if not isinstance(self.x, np.ndarray):
            raise ValueError("Please call .fit() first!")
        query_xs = normalize_axis(query_xs)
        result = []
        x = build_matrix(normalize_axis(self.x))
        for q_x, weight in zip(query_xs, self.weighted(query_xs)):
            w = self.solve_LWLR(x, self.y, weight)
            result.append(build_matrix(np.expand_dims(q_x, axis=0)).dot(w))

        return normalize_axis(np.squeeze(np.array(result)))
    
    def _plot2d(self):
        plt.figure()
        plt.xlabel('X')
        plt.ylabel('y')
        plt.scatter(self.x, self.y, s=2)

        new_x = np.arange(min(self.x), max(self.x), 0.01)
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
        x0_range = np.arange(min(x0), max(x0), 0.2)
        x1_range = np.arange(min(x1), max(x1), 0.2)

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

    start_n = 2
    end_n = 15
    losses = []
    for w in range(start_n, end_n + 1):
        lw_reg = Local_weighted_linear_regressor(w)
        lw_reg.fit(x_train, y_train)
        y_predict = lw_reg.predict(x_test)
        losses.append(mse_loss(y_predict, y_test))
    plt.figure()
    plt.plot(range(start_n, end_n + 1), losses)
    plt.ylabel('MSE Error')
    plt.xlabel('kernel width')
    plt.show()
    lw_reg = Local_weighted_linear_regressor(10)
    lw_reg.fit(x, y)
    lw_reg.plot_fig()
