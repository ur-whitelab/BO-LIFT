import numpy as np
from .asktellGPR import AskTellGPR
from .llm_model import GaussDist


class AskTellRidgeKernelRegression(AskTellGPR):
    def __init__(self, alpha=1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _set_regressor(self):
        self.coefficients = None
        self.train_x = None
        self.train_y = None
        self.mean_x = None
        self.std_x = None
        self.mean_y = None
        self.std_y = None

    def _dot_product_kernel(self, X1, X2):
        return X1.dot(X2.T)

    def _normalize(self, X, mean, std):
        if mean is None or std is None:
            raise ValueError(
                "Mean and standard deviation must be set for normalization"
            )
        return (X - mean) / std

    def _train(self, X, y):
        self.train_x = np.array(self._query_cache(X), dtype=np.float64)
        self.mean_x = np.mean(self.train_x, axis=0)
        self.std_x = np.std(self.train_x, axis=0)

        self.train_y = np.array(y, dtype=np.float64)
        self.mean_y = np.mean(self.train_y, axis=0)
        self.std_y = np.std(self.train_y, axis=0)

        self.train_x = self._normalize(self.train_x, self.mean_x, self.std_x)
        self.train_y = self._normalize(self.train_y, self.mean_y, self.std_y)

        kernel_train = self._dot_product_kernel(self.train_x, self.train_x)
        ridge_matrix = self.alpha * np.eye(kernel_train.shape[0])

        # Add a small diagonal matrix to the kernel matrix to ensure it's PSD
        eps = 1e-8
        kernel_train += eps * np.eye(kernel_train.shape[0])

        self.coefficients = np.linalg.solve(kernel_train + ridge_matrix, self.train_y)

    def _predict(self, X):
        if len(X) == 0:
            raise ValueError("X is empty")
        embedding = np.array(self._query_cache(X), dtype=np.float64)

        # Normalize the input features
        embedding = self._normalize(embedding, self.mean_x, self.std_x)

        kernel_test = self._dot_product_kernel(embedding, self.train_x)
        mean = np.dot(kernel_test, self.coefficients) * self.std_y + self.mean_y
        results = [GaussDist(m, s) for m, s in zip(mean, np.zeros_like(mean))]
        return results, 0
