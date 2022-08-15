import numpy as np


class linear_reg:
    def __init__(self, x_train, y_train) -> None:
        self.alpha = 0.3
        self.num_iter = 10000
        self.w = 0
        self.b = 0
        self.gradient_descent(x_train, y_train)

    def compute_cost(self, x, y):
        m = x.shape[0]
        cost = 0

        for i in range(m):
            f_wb = self.w * x[i] + self.b
            cost = cost + (f_wb - y[i]) ** 2
        total_cost = 1 / (2 * m) * cost
        return total_cost

    def compute_gradient(self, x, y):
        m = x.shape[0]
        dj_dw = 0
        dj_db = 0

        for i in range(m):
            f_wb = self.w * x[i] + self.b
            dj_dw_i = (f_wb - y[i]) * x[i]
            dj_db_i = f_wb - y[i]
            dj_dw = dj_dw + dj_dw_i
            dj_db = dj_db + dj_db_i
        dj_dw = dj_dw / m
        dj_db = dj_db / m
        return dj_dw, dj_db

    def gradient_descent(self, x, y):

        for i in range(self.num_iter):
            dj_dw, dj_db = self.compute_gradient(x, y)
            self.b = self.b - self.alpha * dj_db
            self.w = self.w - self.alpha * dj_dw

        return self.w, self.b

    def predict(self, x):
        return (self.w * x) + self.b


x_train = np.array([1.0, 2.0])
y_train = np.array([300, 500])

model = linear_reg(x_train, y_train)
print(model.predict(1.2))
