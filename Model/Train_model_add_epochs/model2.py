import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 




def plot_accuracy(all_reports):
    # Extraer accuracy de cada fold
    accuracies = [report['accuracy'] for report in all_reports]
    
    # Crear gráfica
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', label='Accuracy', color='b')
    plt.ylim(0, 1)
    plt.xticks(range(1, len(accuracies) + 1))
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy por Fold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()


def plot_precision_recall(all_reports):

    # Inicializar listas para almacenar precisión y recall por clase
    precision_class_0 = []
    recall_class_0 = []
    precision_class_1 = []
    recall_class_1 = []

    # Extraer precisión y recall de cada reporte
    for report in all_reports:
        precision_class_0.append(report['0']['precision'])
        recall_class_0.append(report['0']['recall'])
        precision_class_1.append(report['1']['precision'])
        recall_class_1.append(report['1']['recall'])

    # Crear índices para los folds
    folds = range(1, len(all_reports) + 1)

    # Crear la figura y los subplots
    plt.figure(figsize=(12, 6))

    # Gráfica de precisión
    plt.subplot(1, 2, 1)
    plt.plot(folds, precision_class_0, label='Clase 0', marker='o')
    plt.plot(folds, precision_class_1, label='Clase 1', marker='o')
    plt.title('Precisión por Fold')
    plt.xlabel('Fold')
    plt.ylabel('Precisión')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    # Gráfica de recall
    plt.subplot(1, 2, 2)
    plt.plot(folds, recall_class_0, label='Clase 0', marker='o')
    plt.plot(folds, recall_class_1, label='Clase 1', marker='o')
    plt.title('Recall por Fold')
    plt.xlabel('Fold')
    plt.ylabel('Recall')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    # Mostrar las gráficas
    plt.tight_layout()
    plt.show()



# Definir el modelo MLP
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


# Verificar si hay una GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando {device}")

X_train_embeddings = torch.load('Embeddings_train_and_test/X_train_emebeddings_big_data.pth').to(device)
y_train_tensor = torch.load('Embeddings_train_and_test/y_train_big_data.pth').to(device)
X_test_embeddings = torch.load('Embeddings_train_and_test/X_test_emebeddings_big_data.pth').to(device)
y_test_tensor = torch.load('Embeddings_train_and_test/y_test_big_data.pth').to(device)

# Configurar parámetros del modelo
input_dim = X_train_embeddings.shape[1]
hidden_dim = 256
output_dim = 2

# Configurar KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Inicializar listas para almacenar resultados
all_losses = []
all_reports = []

# Loop de cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X_train_embeddings)):
    print(f"Fold {fold+1}")

    # Dividir datos para el fold actual
    X_train_fold = X_train_embeddings[train_index]
    y_train_fold = y_train_tensor[train_index]
    X_test_fold = X_train_embeddings[test_index]
    y_test_fold = y_train_tensor[test_index]

    # Inicializar modelo, pérdida y optimizador
    model = MLPClassifier(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Entrenar modelo
    epochs = 2000  # Reducir el número de épocas
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_fold)
        loss = criterion(outputs, y_train_fold)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:  # Mostrar progreso cada 5 épocas
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Evaluar modelo
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_fold).argmax(dim=1)
        report = classification_report(y_test_fold.cpu().numpy(), y_pred.cpu().numpy(), output_dict=True)
        print(f'Report fold {fold}:\n {report}')
        all_reports.append(report)
        all_losses.append(loss.item())


print("Classification Reports:")
plot_precision_recall(all_reports)
plot_accuracy(all_reports)

# Guardar modelo final
#torch.save(model.state_dict(), 'modelo_entrenado_final.pth')
#print("Modelo guardado correctamente.")
