import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve



#Carregando o data
data = 'wdbc.data'
colunas = ['classe', 'instance1','raio1','textura1','perímetro1', 'área1','suavidade1','compacidade1','concavidade1','concave_points1','simetria1','fractal_dimension1','raio2','textura2','perímetro2','área2', 'suavidade2','compacidade2','concave_points2', 'simetria2','fractal_dimension2	','raio3','textura3','perímetro3','área3','suavidade3','compacidade3','concavidade3','concave_points3','simetria3','fractal_dimension3']
df = pd.read_csv(data, header=None, names=colunas, skiprows=1)
X = df.drop(columns=['classe'])
y = df['classe']
####################################
#Transformaçaõ para binário
y = y.replace({'M':1, 'B':0})
#Normalizar o X
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
print(df.info())
###############################

#Aplicaçaõ do KNN
#Treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
# Inicialize o classificador
knn_classifier = KNeighborsClassifier(n_neighbors=3)
# Treine o modelo
knn_classifier.fit(X_train, y_train)
# previsões teste
y_pred = knn_classifier.predict(X_test)

# Desempenho do modelo
print('\nDesempenho DO KNN')
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
# Exibindo as métricas
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
########################################################
#Aplicação Random forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=10)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)
#Desempenho do modelo
accuracy_rf = accuracy_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)
print('\nDesempenho do Random Forest:')
print(f'Acurácia (Random Forest): {accuracy_rf:.2f}')
print(f'Mean Squared Error (Random Forest): {mse_rf:.4f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print('Matriz de Confusão (Random Forest):')
print(conf_matrix_rf)
############################################################
#Aplicação MLP
#classificador MLP                                #camadas ocultas
mlp_classifier = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=500, random_state=40)
# Treino
mlp_classifier.fit(X_train, y_train)
# Previsões
y_pred_mlp = mlp_classifier.predict(X_test)

# Desempenho
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
precision = precision_score(y_test, y_pred_mlp)
recall = recall_score(y_test, y_pred_mlp)
f1 = f1_score(y_test, y_pred_mlp)

# Imprima os resultados
print('\nDesempenho Rede Neural MLP:')
print(f'Acurácia (MLP): {accuracy_mlp:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'Mean Squared Error (MLP): {mse_mlp:.4f}')
print('Matriz de Confusão (MLP):')
print(conf_matrix_mlp)
######################################################################

#VERIFICANDO overfitting (CURVA DE APRENDIZAGEM)
#KNN
sizes, training_scores, testing_scores = learning_curve(KNeighborsClassifier(), X, y, cv=10, scoring='accuracy',
                                                        train_sizes=np.linspace(0.01, 1.0, 50))

# Mean and Standard Deviation of training scores
mean_training = np.mean(training_scores, axis=1)
Standard_Deviation_training = np.std(training_scores, axis=1)

# Mean and Standard Deviation of testing scores
mean_testing = np.mean(testing_scores, axis=1)
Standard_Deviation_testing = np.std(testing_scores, axis=1)

# dotted blue line is for training scores and green line is for cross-validation score
plt.plot(sizes, mean_training, '--', color="b", label="Training score")
plt.plot(sizes, mean_testing, color="g", label="Cross-validation score")

# Drawing plot
plt.title("LEARNING CURVE FOR KNN Classifier")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()
#########################
#random_forest
sizes, training_scores, testing_scores = learning_curve(RandomForestClassifier(), X, y, cv=10, scoring='accuracy',
                                                        train_sizes=np.linspace(0.01, 1.0, 50))
# Mean and Standard Deviation of training scores
mean_training = np.mean(training_scores, axis=1)
Standard_Deviation_training = np.std(training_scores, axis=1)

# Mean and Standard Deviation of testing scores
mean_testing = np.mean(testing_scores, axis=1)
Standard_Deviation_testing = np.std(testing_scores, axis=1)

# dotted blue line is for training scores and green line is for cross-validation score
plt.plot(sizes, mean_training, '--', color="b", label="Training score")
plt.plot(sizes, mean_testing, color="g", label="Cross-validation score")

# Drawing plot
plt.title("LEARNING CURVE FOR KNN Classifier")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

#MLP
sizes, training_scores, testing_scores = learning_curve(MLPClassifier(), X, y, cv=10, scoring='accuracy',
                                                        train_sizes=np.linspace(0.01, 1.0, 50))
# Mean and Standard Deviation of training scores
mean_training = np.mean(training_scores, axis=1)
Standard_Deviation_training = np.std(training_scores, axis=1)

# Mean and Standard Deviation of testing scores
mean_testing = np.mean(testing_scores, axis=1)
Standard_Deviation_testing = np.std(testing_scores, axis=1)

# dotted blue line is for training scores and green line is for cross-validation score
plt.plot(sizes, mean_training, '--', color="b", label="Training score")
plt.plot(sizes, mean_testing, color="g", label="Cross-validation score")

# Drawing plot
plt.title("LEARNING CURVE FOR MLP Classifier")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()






#ANÁLISE GRÁFICA
modelos = ['Random Forest', 'k-NN', 'MLP']
acuracia = [accuracy, accuracy_rf, accuracy_mlp]
msedf = [mse, mse_rf, mse_mlp]

fig, axs = plt.subplots(figsize=(10, 5))
bar_width = 0.4
bar_acuracia = axs.bar(modelos, acuracia, width=bar_width, color=['skyblue', 'black', 'lightgreen'])
for index, value in enumerate(acuracia):
    axs.text(index-0.2, value + 0.03, f'{value}', va='center')

# Barra para MSE
bar_mse = axs.bar(modelos, msedf, width=bar_width, color=['red', 'red', 'red'], label='MSE')
for index, value in enumerate(msedf):
    axs.text(index , value - 0.15, f'{value:.2f}', va='center')
axs.set_title('Acurácia e MSE dos Modelos')
axs.set_ylabel('Acurácia')
axs.legend()
plt.show()