import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(7)
#torch.autograd.detect_anomaly()

train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU!')
else:
    print('No GPU available, training on CPU; consider making n_epochs very small.')

class CharRNN(nn.Module):

    def __init__(self, tokens, n_hidden=512, n_layers=8,
                 drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens

        # define the layers of the model
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))

    def forward(self, x, hidden):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''

        # Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        # Initializes hidden state
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

# Replace a character string with a standard encoding
def encode(in_string, chars):
    output = ''
    encode_dict = {}
    i = 0
    for c in in_string:
        if c not in encode_dict.keys():
            encode_dict[c] = chars[i]
            i = i + 1
        output = output + encode_dict[c]
    return output

# Take a single batch string and return a numpy array
def make_numpy(in_string, chars, shape):
    arr = np.array([[chars.index(i) for i in in_string]])
    #arr = np.array([[random.randrange(26) for i in in_string]])
    arr = np.reshape(arr, (-1,shape))
    return arr

def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot

def process_batch(file, batch_size, seq_length, chars):
    # Get next batch
    arr = file.read(batch_size * seq_length)
    # End when not enough text for a batch
    if len(arr) < batch_size * seq_length:
        return None
    # Cyphertext, re-encode with standard encoding
    x = make_numpy(encode(arr, chars), chars, seq_length)
    x = one_hot_encode(x, len(chars))
    # Plaintext
    y = make_numpy(arr, chars, seq_length)
    return x, y

def get_batches(file, batch_size, seq_length, chars):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.

       Arguments
       ---------
       file: Source of data
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
       chars: list of features
    '''
    return iter(lambda: process_batch(file, batch_size, seq_length, chars), None)

def train(net, data, val_data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, print_every=10):
    ''' Training a network

        Arguments
        ---------

        net: CharRNN network
        data: text data to train the network
        val_data: text of validation data for the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        print_every: Number of steps for printing training and validation loss

    '''
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if train_on_gpu:
        net.cuda()

    counter = 0
    chars = net.chars
    with open(data, 'r') as fopen, open(val_data, 'r') as fopen_val:
        for e in range(epochs):
            # initialize hidden state
            h = net.init_hidden(batch_size)

            for x, y in get_batches(fopen, batch_size, seq_length, chars):
                counter += 1

                # Make them Torch tensors
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y).long()
                del x, y

                if train_on_gpu:
                    inputs, targets = inputs.cuda(), targets.cuda()

                # Creating new variables for the hidden state, otherwise
                # we'd back-prop through the entire training history
                h = tuple([each.data for each in h])

                # zero accumulated gradients
                net.zero_grad()

                # get the output from the model
                output, h = net(inputs, h)

                # calculate the loss and perform back-prop
                loss = criterion(output, targets.view(batch_size * seq_length))
                loss.backward()
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                nn.utils.clip_grad_norm_(net.parameters(), clip)
                opt.step()

                # loss stats
                if counter % print_every == 0:
                    # Get validation loss
                    val_h = net.init_hidden(batch_size)
                    val_losses = []
                    net.eval()
                    for x, y in get_batches(fopen_val, batch_size, seq_length, chars):
                        # Make them Torch tensors
                        inputs, targets = torch.from_numpy(x), torch.from_numpy(y).long()
                        del x, y

                        # Creating new variables for the hidden state, otherwise
                        # we'd back-prop through the entire training history
                        val_h = tuple([each.data for each in val_h])

                        if train_on_gpu:
                            inputs, targets = inputs.cuda(), targets.cuda()

                        output, val_h = net(inputs, val_h)
                        val_loss = criterion(output, targets.view(batch_size * seq_length))
                        #print("Current loss: " + str(val_loss))

                        val_losses.append(val_loss.item())

                    net.train()  # reset to train mode after iterating through validation data
                    fopen_val.seek(0) # reset validation data

                    print("Epoch: {}/{}...".format(e + 1, epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.4f}...".format(loss.item()),
                          "Val Loss: {:.4f}".format(np.mean(val_losses)))

def predict(net, char, freq, freq_prop=0.5, h=None, top_k=5):
    ''' Given a character, predict the next character.
        Returns the predicted character and the hidden state.
    '''

    # tensor inputs
    x = np.array([[net.chars.index(char)]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)

    if train_on_gpu:
        inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])
    # get the output of the model
    out, h = net(inputs, h)

    # get the character probabilities
    p = F.softmax(out, dim=1).data
    if train_on_gpu:
        p = p.cpu()  # move to cpu

    # get top characters
    p, top_ch = p.topk(top_k)
    #print(p)
    top_ch = top_ch.numpy().squeeze()
    p = p.numpy().squeeze()

    # Factor in frequency analysis
    char_freq = np.array([freq[ch] for ch in top_ch])
    char_freq = char_freq/sum(char_freq)
    p = p*(1-freq_prop) + char_freq*freq_prop

    # select the likely next character with some element of randomness
    char = np.random.choice(top_ch, p=p / p.sum())

    # return the encoded value of the predicted char and the hidden state
    return net.chars[char], h

def decrypt(net, ciphertext, freq_prop=0.5, top_k=5):
    if train_on_gpu:
        net.cuda()
    else:
        net.cpu()

    # Clean inputs
    cipher = ''.join([c.lower() for c in ciphertext if c.isalpha()])
    cipher = encode(cipher, net.chars)
    # Frequency analysis
    freq = [cipher.count(c) for c in net.chars]
    # Prepare network
    net.eval()  # eval mode
    output = ''
    h = net.init_hidden(1)
    #for ch in prime:
    #    char, h = predict(net, ch, freq, freq_prop=freq_prop, h=h, top_k=top_k)
    #    output = output + ch
    #cipher = cipher[len(prime):]

    # Attempt to decrypt each character
    for ch in cipher:
        char, h = predict(net, ch, freq, freq_prop=freq_prop, h=h, top_k=top_k)
        output = output + char

    return output