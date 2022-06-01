from caesar_network import *
import string

text_train = 'train_clean.txt'
text_valid = 'valid_clean.txt'
chars = string.ascii_lowercase #'etaoinsrhdlucmfywgpbvkxqjz'

n_hidden = 512
n_layers = 2
batch_size = 512
seq_length = 500
learning_rate = 0.0014
clip = 5
n_epochs = 5  # start small if you are just testing initial behavior

# train the model
net = CharRNN(chars, n_hidden, n_layers)
train(net, text_train, text_valid, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=learning_rate, clip=clip, print_every=10)

model_name = 'rnn_x_epoch.net'
checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)