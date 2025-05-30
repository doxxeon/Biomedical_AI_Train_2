import numpy as np

def MSE(y, y_hat):
    """
    MSE 함수 구현을 채우세요.
    """
    error = None
    squared = None
    mean = None

    # 소수점 둘째 자리에서 반올림하세요.
    mean = round(mean, 2)
    return mean
    
# 테스트1
y     = np.array([0,   0,   1,   0,   0])
y_hat = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
mse = MSE(y, y_hat)
print(mse)

# 테스트2
y     = np.array([0,   0,   1,   0,   0])
y_hat = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
mse = MSE(y, y_hat)
print(mse)

# 테스트3
y     = np.array([43,  7, 53, 86, -44])
y_hat = np.array([54, 67, 23, 96, -50])
mse = MSE(y, y_hat)
print(mse)
