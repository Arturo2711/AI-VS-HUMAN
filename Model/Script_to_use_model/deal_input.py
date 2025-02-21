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


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
transformer = AutoModel.from_pretrained('bert-base-uncased')


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


def use_model(embedding, model):
    """
    Use the trained model to predict whether the input is machine-generated or human-generated.
    """
    model.eval()
    with torch.no_grad():
        prediction = model(embedding).argmax(dim=1)
    return prediction.cpu().numpy() # Return as a Python scalar


# Model configuration
input_dim = 768  # BERT embedding size
hidden_dim = 256
output_dim = 2  # Binary classification: [0: human, 1: machine]

# Load the model architecture and weights
model = MLPClassifier(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('modelo_entrenado_big_data.pth', map_location=torch.device('cpu'), weights_only=True))

# Example input texts
input_text_machine = ''' 
Red Dead Redemption 2 is widely regarded as one of the best open-world games ever created. Its detailed world-building, immersive storytelling, and complex characters set it apart from many other games. The narrative, centered around Arthur Morgan and the Van der Linde gang, is emotionally rich and compelling, offering a deep exploration of themes like loyalty, morality, and the inevitability of change.

The open world is breathtakingly beautiful, with diverse landscapes ranging from snowy mountains to dense forests, and the attention to detail is remarkable. The game features a dynamic weather system and realistic animal behavior, making the environment feel alive and constantly evolving.

The gameplay is varied, with elements of exploration, combat, hunting, and interaction with NPCs. The gunplay and horseback riding mechanics are particularly praised for their realism, though some players might find the slow pace and focus on realism a bit tedious at times.

Overall, Red Dead Redemption 2 is a masterpiece in terms of its world design, storytelling, and character development, though its slower pace and emphasis on realism may not be for everyone. It's a game that rewards patience and attention to detail, making it an unforgettable experience for those who appreciate immersive, narrative-driven gameplay.
'''

input_text_human =  ''' 
Red dead redemption 2 is one of the best pieces of entertainment media ever made. Masterpiece '''

# Generate embeddings
embedding_machine = get_embeddings(input_text_machine, tokenizer, transformer)
embedding_human = get_embeddings(input_text_human, tokenizer, transformer)

# Predict using the model
prediction_machine = use_model(embedding_machine, model)
prediction_human = use_model(embedding_human, model)

# Output the predictions
print(f'The prediction for the machine text is: {prediction_machine[0]} (1: Machine, 0: Human)')
print(f'The prediction for the human text is: {prediction_human[0]} (1: Machine, 0: Human)')
