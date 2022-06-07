from caesar_network import *

with open('rnn_x_epoch.net', 'rb') as f:
    checkpoint = torch.load(f)

loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])

ciphertext = 'myxqbkdevkdsyxc sd gkc sxnoon usvbyi dro wyeco gry cdyvo dro pokcd led ro rkn rovz yxo rexnbon kxn ovofox yp usvbyic pebbi bovkdsfoc rsn drowcovfoc sx dro ryeco kxn kbyexn dro qkbnox kxn kd ovofox wsxedoc dy ovofox droi kvv cmkwzobon sxdy dro lkxaeod rkvv gsdr usvbyi kxn kdo ez kvv yp rybkmoc lokedspev pyyn kxn xyg tecd dy cryg ryg mvofob iye kbo mkx iye psxn yxo rexnbon kxn ovofox wsmo rsnnox sx dro zsmdeboc rkzzi rexdsxq'

valid_plain = 'congratulations it was indeed kilroy the mouse who stole the feast but he had help one hundred and eleven of kilroys furry relatives hid themselves in the house and around the garden and at eleven minutes to eleven they all scampered into the banquet hall with kilroy and ate up all of horaces beautiful food and now just to show how clever you are can you find one hundred and eleven mice hidden in the pictures happy hunting'

plaintext = decrypt(loaded, ciphertext, freq_prop=0.5, top_k=5)

ciphertext = ciphertext.replace(' ','')
valid_plain = valid_plain.replace(' ','')
accuracy = sum([valid_plain[i] == plaintext[i] for i in range(len(valid_plain))])/len(valid_plain)
print(plaintext)
print()
print("Accuracy: " + str(accuracy))