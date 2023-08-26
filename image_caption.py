import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json

# Load vocabulary
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

# Load models
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs

encoder = EncoderCNN()
encoder.eval()
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab), num_layers=1)
decoder.eval()

# Load image and preprocess
image_path = 'image.jpg'
image = Image.open(image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
image = transform(image).unsqueeze(0)

# Generate caption
with torch.no_grad():
    features = encoder(image)
    sampled_ids = []
    inputs = torch.tensor([[vocab['<start>']]])
    for _ in range(20):  # Maximum caption length
        lstm_out = decoder.lstm(features.unsqueeze(1), inputs)
        outputs = decoder.linear(lstm_out[0])
        _, predicted = outputs.max(2)
        sampled_ids.append(predicted.item())
        inputs = predicted

# Convert IDs to words
caption = []
for word_id in sampled_ids:
    word = [word for word, idx in vocab.items() if idx == word_id][0]
    caption.append(word)
    if word == '<end>':
        break

final_caption = ' '.join(caption)
print(final_caption)
