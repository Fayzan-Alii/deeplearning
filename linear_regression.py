import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(X)
    test_indices = np.random.choice(n, size=int(n * test_size), replace=False)
    train_indices = np.array(list(set(range(n)) - set(test_indices)))
    
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    
    return X_train, X_test, y_train, y_test

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.alpha = None
        self.max_iters = None
        self.threshold = 1e-6

    def train(self, X, y, alpha=0.01, max_iters=None, print_loss_iter=100):
        self.alpha = alpha
        self.max_iters = max_iters
              
        X = np.c_[np.ones((X.shape[0], 1)), X]     
        
        self.weights = np.zeros(X.shape[1])
        
        prev_loss = float('inf')
        constant_loss_count = 0
        
        for iteration in range(1, max_iters + 1 if max_iters else 1000000):
            
            y_pred = np.dot(X, self.weights)            
            
            loss = np.mean((y_pred - y) ** 2)
                       
            gradients = 2 * np.dot(X.T, (y_pred - y)) / X.shape[0]
                       
            self.weights -= self.alpha * gradients            
            
            if iteration % print_loss_iter == 0:
                print(f"Iteration {iteration}, Loss: {loss}")            
            
            if max_iters is None:
                if abs(prev_loss - loss) < self.threshold:
                    constant_loss_count += 1
                    if constant_loss_count >= 3:
                        print(f"Converged after {iteration} iterations")
                        break
                else:
                    constant_loss_count = 0
                prev_loss = loss
            
            
            if max_iters and iteration >= max_iters:
                print(f"Reached maximum iterations: {max_iters}")
                break

    def predict(self, X):
       
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return np.dot(X, self.weights)

    def get_weights(self):
        return self.weights


data = pd.read_csv('D:\Work\Python\dl\Advertising.csv')
X = data[['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']]
y = data['Sales ($)']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train_scaled, train_mean, train_std = standardize(X_train)
X_test_scaled = (X_test - train_mean) / train_std


X_train_scaled = X_train_scaled.values
X_test_scaled = X_test_scaled.values
y_train = y_train.values
y_test = y_test.values


model = LinearRegression()
model.train(X_train_scaled, y_train, alpha=0.01, max_iters=1000, print_loss_iter=100)


y_pred = model.predict(X_test_scaled)


mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error: {mse:.4f}")


weights = model.get_weights()
print("Model Weights:")
print(f"Bias: {weights[0]:.4f}")
print(f"TV: {weights[1]:.4f}")
print(f"Radio: {weights[2]:.4f}")
print(f"Newspaper: {weights[3]:.4f}")


print("\nActual vs Predicted Sales:")   
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted:.4f}")    
   

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.tight_layout()
plt.show()