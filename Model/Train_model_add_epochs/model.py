import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report

# 1. Cargar modelo preentrenado
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name)

def get_embeddings(texts, tokenizer, model, device="cpu"):
    """
    Genera embeddings promedio para una lista de textos usando un modelo Transformer.
    """
    # Tokenizar y mover los tensores al dispositivo
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Mover al dispositivo

    # Asegurarse de que el modelo esté en el mismo dispositivo
    model.to(device)

    with torch.no_grad():
        outputs = model(**inputs)  # Calcular los embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Promediar las representaciones
    return embeddings

# 3. Verificar si hay una GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# 4. Cargar embeddings preprocesados
try:
    X_train_embeddings = torch.load('Embeddings_train_and_test/X_train_emebeddings_big_data.pth').to(device)
    y_train_tensor = torch.load('Embeddings_train_and_test/y_train_big_data.pth').to(device)
    X_test_embeddings = torch.load('Embeddings_train_and_test/X_test_emebeddings_big_data.pth').to(device)
    y_test_tensor = torch.load('Embeddings_train_and_test/y_test_big_data.pth').to(device)
    print("Embeddings cargados correctamente.")
except FileNotFoundError as e:
    print(f"Error al cargar los archivos: {e}")
    exit()

# 5. Definir el modelo MLP
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Crear el modelo y moverlo al dispositivo
input_dim = X_train_embeddings.shape[1]
hidden_dim = 256
output_dim = 2
model = MLPClassifier(input_dim, hidden_dim, output_dim).to(device)
print("Modelo creado.")

# 6. Configurar pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 7. Entrenar el modelo
epochs = 2000  # Ajustar según sea necesario
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_embeddings)  # Asegúrate de que X_train_embeddings esté en el dispositivo
    loss = criterion(outputs, y_train_tensor)  # Asegúrate de que y_train_tensor esté en el dispositivo
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# 8. Evaluar el modelo
model.eval()
with torch.no_grad():
    y_pred = model(X_test_embeddings).argmax(dim=1)
    print("Reporte de clasificación:")
    print(classification_report(y_test_tensor.cpu().numpy(), y_pred.cpu().numpy()))

def predict_texts(texts, model, tokenizer, transformer, device):
    """
    Predice la clase de una lista de textos.
    """
    model.eval()
    model.to(device)  # Asegurarse de que el modelo esté en el dispositivo

    # Generar embeddings y mover al dispositivo
    embeddings = torch.cat([get_embeddings([text], tokenizer, transformer, device) for text in texts]).to(device)
    
    with torch.no_grad():
        predictions = model(embeddings).argmax(dim=1)
    return predictions.cpu().numpy()

new_texts = [
    "The implementation of natural language processing algorithms in diverse industries has demonstrated significant potential.",
    "In starlit nights, I saw you so cruelly, you kissed me. Your lips, a magic world."
]
predictions = predict_texts(new_texts, model, tokenizer, transformer, device)
print(f"Predicciones: {predictions}")

# 10. Guardar el modelo entrenado
torch.save(model.state_dict(), 'modelo_entrenado_big_data.pth')
torch.save(optimizer.state_dict(), 'optimizador_big_data.pth')
print("Modelo y optimizador guardados correctamente.")
