import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.colors import ListedColormap

df = pd.read_csv("Iris.csv")
# df.drop('ID', axis=1, inplace=True)

df['Species'] = df['Species'].astype('category').cat.codes

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=69)

k_values = range(1, 11)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"K={k}, Accuracy={acc:.2f}")

best_k = k_values[np.argmax(accuracies)]
print(f"\nBest K: {best_k}")

best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(x_train, y_train)
y_best_pred = best_model.predict(x_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_best_pred))

def plot_decision_boundary(x, y, model, title):
    h = 0.2
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    plt.contourf(xx, yy, z, cmap=cmap_light)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_bold, edgecolor='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig("decision_boundary_plot.png")

x_plot = x_scaled[:, :2]
model_2d = KNeighborsClassifier(n_neighbors=best_k)
model_2d.fit(x_plot, y)
plot_decision_boundary(x_plot, y, model_2d, f"Decision Boundary (K = {best_k})")


