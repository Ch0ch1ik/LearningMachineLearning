import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# tutaj ładujemy dane do obiektu data frame z biblioteki pandas
# plik CSV nie posiada nagłówka dlatego header=None
# kolumnom nadajemy nazwy korzystając z parametru names
# W skryptach ML dane trzeba skądś pobrać, stad znajomość polecenia
# read_csv jest super przydatna

iris = pd.read_csv('iris.data',
                   header=None,
                   names=['sepal length', 'sepal width',
                          'petal length', 'petal width', 'class'])
print(iris.head())
# print(iris.shape)
# print(iris.shape[0])
# print(iris.shape[1])

x_min, x_max = iris['petal length'].min() - .5, iris['petal length'].max() + .5
y_min, y_max = iris['petal width'].min() - .5, iris['petal width'].max() + .5

colors = {'Iris-setosa':'red', 'Iris-versicolor':'blue', 'Iris-virginica':'green'}

fig, ax = plt.subplots(figsize=(8, 6))

for key, group in iris.groupby('class'):
    plt.scatter(group['petal length'], group['petal width'],
                c=colors[key], label=key)

ax.legend()
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
ax.set_title('Iris dataset')

plt.show()




x_min, x_max = iris['sepal length'].min() - .5, iris['sepal length'].max() + .5
y_min, y_max = iris['sepal width'].min() - .5, iris['sepal width'].max() + .5

colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'blue', 'Iris-virginica': 'green'}

fig, ax = plt.subplots(figsize=(8, 6))

for key, group in iris.groupby(by='class'):
    # funkcja scatter przyjmuje argumenty - współrzędne X punktów, współrzędne Y punktów,
    # kolor i nazwę rysowanej grupy
    plt.scatter(group['sepal length'], group['sepal width'],
                c=colors[key], label=key)

ax.legend()
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
ax.set_title("IRIS DATASET CATEGORIZED")

plt.show()




# utwórz wykres składający się z 4 małych wykresów
fig, ax = plt.subplots(2, 2, figsize=(10, 6))

# aktualnie rysowanie odbędzie się w określonym pod-wykresie
plt_position = 1

# obrazujemy zależność miedzy tą zmienną, a pozostałymi cechami próbek
feature_x = 'petal width'

# dla każdej cechy opisującej kwiaty
for feature_y in iris.columns[:4]:

    # wybierz kolejny pod wykres
    plt.subplot(2, 2, plt_position)

    # i rysuj osobne wykresy dla każdego gatunku (te 3 rysowane tu wykresy
    # nakładają sie na siebie, co pozwala automatycznie generować legendę)
    for species, color in colors.items():
        # podczas rysowanie należy odfiltrować tylko kwiaty jednego gatunku
        # zobacz jak filtrować dane. Służy do tego funkcja loc wywoływana dla data frame
        # wyrażenie w nawiasie kwadratowym ma zwracać True/False. Zwrócone będą wiersze,
        # gdzie wyrażenie ma wartość True. Po przecinku znajduje się nazwa kolumny, która ma być zwrócona
        plt.scatter(iris.loc[iris['class'] == species, feature_x],
                    iris.loc[iris['class'] == species, feature_y],
                    label=species,
                    alpha=0.45,  # transparency
                    color=color)

    # opisujemy wykres
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt_position += 1

plt.show()
