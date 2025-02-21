import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

######################## This script is used tp generate embeddings for all train and tet data ########################

# 1. Cargar datos
data = pd.read_csv('Data and exploratory analysis/balanced_data_big.csv', quotechar='"', delimiter=',', on_bad_lines='skip', encoding='utf-8', doublequote=True)
print(data.head())  # Verifica las primeras filas del dataframe


X = data['text']
y = data['label']

# 2. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

# 3. Cargar modelo preentrenado
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name)

# 4. Función para generar embeddings
def get_embeddings(texts):
    #print(f'Generating embedding for {i}...')
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = transformer(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    #print(f"Embeddings generados para los textos.")
    return embeddings

# 5. Verificar si hay una GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando {device}")

# Generar embeddings para los conjuntos de datos y moverlos al dispositivo adecuado
try:
    ''' 
    embeddings_list = []
    for i, text in enumerate(X_test.astype(str).tolist()):
        embeddings_list.append(get_embeddings([text]))  # Usar un solo texto por vez
        if i % 1000 == 0:
            print(f'Progreso: {i}/{len(X_test)} ejemplos procesados')

    X_embeddings = torch.cat(embeddings_list).to(device)
    '''
    # Convertir las etiquetas a tensores
    y_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

    print('Embeddings generados')

except Exception as e:
    print(f"Error generando embeddings: {e}")

### Now save test data 
##torch.save(X_embeddings, 'Embeddings_train_and_test/X_test_emebeddings_big_data.pth')
print('Embeddings test saved')
torch.save(y_tensor, 'Embeddings_train_and_test/y_test_big_data.pth')
