from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Multi-Layer Perceptron for binary classification.
        """
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)



def get_embeddings(text):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    transformer = AutoModel.from_pretrained('bert-base-uncased')
    # Use autotokeniker, then generate embeddings 
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = transformer(**inputs)  # Calcular los embeddings
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Promediar las representaciones
    return embeddings


def use_model(embedding, model):
    model.eval()
    with torch.no_grad():
        prediction = model(embedding)
    #print(f'Prediction arrary : {prediction.cpu().numpy()}')
    return prediction.cpu().numpy()[0] # Return as a Python scalar

def load_model():
    input_dim = 768  # BERT embedding size
    hidden_dim = 256
    output_dim = 2  # Binary classification: [0: human, 1: machine]

    # Load the model architecture and weights
    model = MLPClassifier(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load('code/modelo_entrenado_big_data.pth', map_location=torch.device('cpu'), weights_only=True))
    return model 
