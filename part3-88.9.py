import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB
import re


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        """
        TODO:
        Create and initialise weights and biases for the layers.
        """
        hidden_dim = 128
        hidden_dim2 = 128
        input_dim = 50
        self.lstm = tnn.LSTM(input_dim, hidden_dim, num_layers = 2, bidirectional = True, dropout = 0.2, batch_first = True)
        #self.lstm2 = tnn.LSTM(input_dim, hidden_dim2, batch_first = True)
        #self.ih1 = tnn.Linear(hidden_dim, hidden_dim2)
        self.ih2 = tnn.Linear(hidden_dim2, 64)
        self.ih3 = tnn.Linear(64, 1)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        TODO:
        Create the forward pass through the network.
        """
        
        #print(input.size())
        #print(length.size())
        
        batch_size = length.size()[0]

        pack_input = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True, enforce_sorted=True)

        output, (h_n, c_n) = self.lstm(pack_input)
        
        new_h = h_n
        #new_h = tnn.functional.relu(self.ih1(h_n))
        #new_h = new_h.permute(1,0,2)
        #output, (h_n, c_n) = self.lstm2(new_h)
        #new_h = h_n
        new_h = tnn.functional.relu(self.ih2(new_h))
        #print(new_h.size())
        new_h = self.ih3(new_h)
        #new_h = torch.sum(new_h, 0)
        #print(new_h.size())
        new_h = new_h[-1, :, :]
        new_h = new_h.reshape(-1)
        return new_h
        
        '''
    def __init__(self):
        super(Network, self).__init__()
        self.conv = tnn.Conv1d(in_channels=50, out_channels = 50, kernel_size=8, padding=5)
        self.maxpool = tnn.MaxPool1d(kernel_size=4)
        self.maxglobal = tnn.AdaptiveMaxPool1d(1)
        self.linear = tnn.Linear(50, 1)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        """
        input = input.permute(0,2,1)
        output = self.conv(input)
        output = tnn.functional.relu(output)
        output = self.maxpool(output)
        output = tnn.functional.relu(self.conv(output))
        output = self.maxpool(output)
        output = tnn.functional.relu(self.conv(output))
        output = self.maxglobal(output)
        output = output.permute(0,2,1)
        output = self.linear(output)
        output = output.reshape(-1)
        return output
        '''



class PreProcessing():
    def pre(x):
        #from Spacy library
        stopwords =  ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', \
                      'ourselves', 'you', "you're", "you've", "you'll", \
                      "you'd", 'your', 'yours', 'yourself', 'yourselves', \
                      'he', 'him', 'his', 'himself', 'she', "she's", 'her',\
                      'hers', 'herself', 'it', "it's", 'its', 'itself', 'they',\
                      'them', 'their', 'theirs', 'themselves', 'what', 'which',\
                      'who', 'whom', 'this', 'that', "that'll", 'these', \
                      'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',\
                      'being', 'have', 'has', 'had', 'having', 'do', 'does',\
                      'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', \
                      'or', 'because', 'as', 'until', 'while', 'of', 'at', \
                      'by', 'for', 'with', 'about', 'against', 'between', \
                      'into', 'through', 'during', 'before', 'after', 'above',\
                      'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',\
                      'off', 'over', 'under', 'again', 'further', 'then', \
                      'once', 'here', 'there', 'when', 'where', 'why', 'how',\
                      'all', 'any', 'both', 'each', 'few', 'more', 'most', \
                      'other', 'some', 'such', 'no', 'nor', 'not', 'only',\
                      'own', 'same', 'so', 'than', 'too', 'very', 's', 't', \
                      'can', 'will', 'just', 'don', "don't", 'should', \
                      "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', \
                      'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", \
                      'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",\
                      'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", \
                      'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',\
                      "needn't", 'shan', "shan't", 'shouldn', "shouldn't", \
                      'wasn', "wasn't", 'weren', "weren't", 'won', "won't",\
                      'wouldn', "wouldn't"]
        pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）'
        new = [] 
        for w in x:
            n = re.split(pattern, w)
            new += n
        new = [w for w in new if w not in stopwords]
        return new

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        #print(1)
        return batch

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    loss_fn = tnn.BCEWithLogitsLoss()
    return loss_fn

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)
    #print(1234)
    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")
    #print(1234)
    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(10):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")
    
    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")

if __name__ == '__main__':
    main()
