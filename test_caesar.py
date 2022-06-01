from caesar_network import *

with open('rnn_x_epoch.net', 'rb') as f:
    checkpoint = torch.load(f)

loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])

ciphertext = 'myxqbkdevkdsyxc sd gkc sxnoon usvbyi dro wyeco gry cdyvo dro pokcd led ro rkn rovz yxo rexnbon kxn ovofox yp usvbyic pebbi bovkdsfoc rsn drowcovfoc sx dro ryeco kxn kbyexn dro qkbnox kxn kd ovofox wsxedoc dy ovofox droi kvv cmkwzobon sxdy dro lkxaeod rkvv gsdr usvbyi kxn kdo ez kvv yp rybkmoc lokedspev pyyn kxn xyg tecd dy cryg ryg mvofob iye kbo mkx iye psxn yxo rexnbon kxn ovofox wsmo rsnnox sx dro zsmdeboc rkzzi rexdsxq'

plaintext = decrypt(loaded, ciphertext, freq_prop=0.5, top_k=5)

ciphertext = ciphertext.replace(' ','')
accuracy = sum([ciphertext[i] == plaintext[i] for i in range(len(ciphertext))])/len(ciphertext)
print(plaintext)
print()
print("Accuracy: " + str(accuracy))