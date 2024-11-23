import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.alpha = None
        self.max_iters = None
        self.threshold = 1e-6

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self, X, Y, alpha=0.01, max_iters=None, print_loss_iter=1000):
        self.alpha = alpha
        self.max_iters = max_iters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        iter_count = 0
        prev_loss = float('inf')
        constant_loss_count = 0

        while True:
            z = np.dot(X, self.weights)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - Y)) / n_samples
            self.weights -= self.alpha * gradient
            
            loss = -np.mean(Y * np.log(h + 1e-15) + (1 - Y) * np.log(1 - h + 1e-15))
            
            if iter_count % print_loss_iter == 0:
                print(f"Iteration {iter_count}, Loss: {loss:.6f}")
            
            if max_iters is not None:
                if iter_count >= max_iters:
                    break
            else:
                if abs(prev_loss - loss) < self.threshold:
                    constant_loss_count += 1
                    if constant_loss_count >= 3:
                        break
                else:
                    constant_loss_count = 0
            
            prev_loss = loss
            iter_count += 1

    def predict_class(self, X):
        z = np.dot(X, self.weights)
        probabilities = self.sigmoid(z)
        return (probabilities >= 0.5).astype(int)

    def predict_confidence(self, X):
        z = np.dot(X, self.weights)
        return self.sigmoid(z)

    def get_weights(self):
        return self.weights

def one_vs_all_train(X, Y, num_classes):
    models = []
    for i in range(num_classes):
        model = LogisticRegression()
        binary_y = (Y == i).astype(int)
        model.train(X, binary_y)
        models.append(model)
    return models

def one_vs_all_predict(X, models):
    predictions = np.array([model.predict_confidence(X) for model in models])
    return np.argmax(predictions, axis=0)

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std, mean, std

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def classification_report(y_true, y_pred, target_names):
    report = {}
    for i, class_name in enumerate(target_names):
        true_positive = np.sum((y_true == i) & (y_pred == i))
        false_positive = np.sum((y_true != i) & (y_pred == i))
        false_negative = np.sum((y_true == i) & (y_pred != i))
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        report[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': np.sum(y_true == i)
        }
    return report


data = pd.read_csv('D:\Work\Python\dl\\vehicle.csv')


data = data.replace('?', np.nan).dropna()


X = data.iloc[:, :-1].astype(float).values
y = data.iloc[:, -1].values


class_mapping = {class_name: i for i, class_name in enumerate(np.unique(y))}
y = np.array([class_mapping[class_name] for class_name in y])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train_scaled, train_mean, train_std = standardize(X_train)
X_test_scaled = (X_test - train_mean) / train_std


num_classes = len(class_mapping)
models = one_vs_all_train(X_train_scaled, y_train, num_classes)


y_pred = one_vs_all_predict(X_test_scaled, models)


print("\nActual vs Predicted Vehicles:")   
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted:.4f}") 


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100}")


print("\nClassification Report:")
report = classification_report(y_test, y_pred, list(class_mapping.keys()))
for class_name, metrics in report.items():
    print(f"{class_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.2f}")


feature_names = data.columns[:-1]
for i, model in enumerate(models):
    print(f"\nFeature importance for class {list(class_mapping.keys())[i]}:")
    for j, weight in enumerate(model.get_weights()):
        print(f"{feature_names[j]}: {weight:.4f}")