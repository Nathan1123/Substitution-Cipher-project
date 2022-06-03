# Substitution-Cipher-project
Attempting to use a Recurrent-Neural Network (RNN) to solve a substitution cipher

# Requirements
Requires pytorch, numpy, and Python 3.7

# Theory
Substitution ciphers are the easiest kind of cipher to solve, because it is historically the oldest (dating back to at least the first century BC) and the techniques to solving it have been thoroughly documented for centuries. Despite this being the case, a substitution cipher is not quite straight-forward enough for a computer to solve algorithmically (as opposed to, say, a Viginere cipher, for which explicit algorithms exist). Typically, a substitution cipher is solved by frequency analysis to identify the most likely common letters (usually E, S, and T), followed by guesses at fairly common words following human intuition. Intuition, of course, is not something a computer is designed to simulate, but with some analysis we can quantify the problem enough to the point where a neural network can be useful.

## Brute force approach

Theoretically, a substitution cipher can only have a finite number of possibilities. Given that every letter of the alphabet has to be represented by a unique character, then there are only a finite number of combinations that a cipher can possibly be. Given we have 26 letters in play, then the total number of combinations is 26!, which is about 4e26. Of course, despite this number being finite, it is much too large for any computer to handle, so solving by brute force in this method would not work. 

Of course, in practice the scope of the problem can be significantly reduced, as it is highly unlikely all 26 letters are in play. Given a text short enough with fewer unique letters, it is likely that letters like Z, X, or Q may not appear in the cipher at all. Thus, even though we don't accurately guess what these letters are, it doesn't matter for the problem we are solving. Even with this limitation, however, the complexity of the problem can be quite severe, as anything with ten unique letters or more will already be too large to handle (10! = 3.6 million). 

We could improve the problem even better by only examining one word at a time, as it is likely a single word will have fewer unique characters. While this might seem to work for a brute force method, this immediately runs into a problem of ambiguity. For example, given the ciphertext "ABBC", we could guess this as either "meet" or "book" (or any number of other possibilities) and they would both fit the pattern. Furthermore, this assumes that we have a cipher with spaces between words, while most substitution ciphers use no spaces at all. 

## Hamming distance

Given that we attempt each possible combination of cipher solutions, the obvious question becomes "how can we tell whether this solution is correct?" Intuitively, a human can try to read a series of text, and if they are able to understand it's meaning then they can assume that the cipher has been solved. A computer, however, cannot get that kind of meaning out of text, so this system is not going to work. 

Typically, computers are able to tell the difference between random letters and plain English using a spelling distance metric. That is, the characters of one word are compared to the characters of the most-similar word using some formula. Most spellcheck software uses Levenshtein distance, which accounts for character replacements, deletions and insertions. However, a substitution cipher is always the same length, so we should discount the latter two and only count replacements. This refined distance measurement is known as Hamming distance. 

Now, we have a way to quantify a given decryption guess as a single number: the average Hamming distance of all words in the text. The smaller the average Hamming distance, the closer the text is to plain English. A Hamming distance of zero indicates the cipher has been fully cracked. Having created this distance measurement, it is self-evident that the problem has been reduced to an optimization problem, which is where machine learning becomes useful. 

The brute force method of decryptian can be refined using Hamming distance, to create a backtracking algorithm:
1) Initialize the key guess with frequency analysis
2) Select the first word in the ciphertext
3) Find all words that could satisfy this enciphered word given the current key guess
3a) If no such word exists, the current key guess is wrong. Go back to the previous word and previous key guess
4) Take the candidate word as the minimum Hamming distance from this list
5) Assume this is the correct word, update the decryption key guess accordingly
6) Attempt to decrypt the rest of the ciphertext with the updated key
7) If the average Hamming distance is zero, break
8) Repeat 2-6
9) If the last word is processed and the Hamming distance is not zero, the decryptian failed

This algorithm can be sped up by preprocessing the corpus of words, arranging them in a dictionary based on the size of each word. The file caesar2.py uses this method, which results in cracking the test cipher in about three minutes. That implementation uses the default Linux dictionary consisting of 700,000 words. 

## Recurrent Neural Network

A recurrent neural network (RNN) is able to process a text one character at a time, retaining memory of processing all previous characters using a Long Short-Term Memory cells. For example, an auto-generative text examines the most recent characters to guess the most likely character that should follow. In autogenerative text, the RNN learns the patterns of the English language to understand what letters are most likely supposed to follow in readable sentences. 

It is a safe assumption that the solution to a substitution cipher is supposed to be plain, readable English. Therefore, the dcryptian RNN should learn the patterns of the English language just like an autogenrative text, and have a good way of predicting what the next most-likely character should be. However, instead of taking the _previous_ character as input, the decryption RNN takes in the corresponding encrypted character as input. Essentially, the RNN combines two pieces of information to guess the solution of the current encrypted character: 1) what the encrypted character is, and 2) the pattern of decrypted English characters up to this point. 

Now, if we left the RNN at that, then the network would still have a hard time learning a good decryptian method, because it is going to pick up an implied level of information that isn't actually useful. For example, let's say the network scans the next letter in the cipher and finds it to be the letter Q. The fact that this particular cipher chose Q to represent this letter, as opposed to using any other glyph, is essentially irrelevant information. What truly matters in deciphering the text is the position and context that glyph is in, and not exactly what that glyph is. For example, if it was paired together like QQ then we should deduce it must be a common English pair of letters, like vowels oo or ee, or paired constenants like tt or ss. 

In order to fix this, we need to preprocess the ciphertext such that the __pattern__ of letters is being used as input. I refer to this kind of preprocessing as a __standard encryption__. First, we know that any single set of plain text can be represented by any combination of ciphertexts. For example, the word "foot" can be represented as XYYZ, QRRS, PLLH, etc., and these would all be equally valid encryptions that follow the same pattern. So when we preprocess the ciphertext, we simply replace the current cipher with an isometric one, following a standard pattern. The key property of a standard ecryption is that __all ciphers that use the same pattern will be read as identical by the neural network__. The standard encryptian will assign A as the first glyph in the ciphertext, then assign B to the second unique glyph, then C to the third unique glyph, etc. As there are an equal number of letters in the alphabet for both the plaintext and ciphertext, then this will always match perfectly. 

One clear advantage of using an RNN instead of brute force is that it doesn't rely on the ciphertext having spaces. 

## Augmenting frequency analysis

Even if the recurrent neural network is trained perfectly, it will still run into some issues because it doesn't know what the solution to the cipher is _a priori_, so the first character it examines will be random guessing. And if this guess is wrong, that will pollute the hidden state that will affect future guesses. As an attempt to improve our guesses, the RNN should be helped with the simple frequency analysis in order to push the results in the right direction. So given P is the probability distribution returned by the neural network, then P is updated using the following formula:

P = P*(1-K) + F*K

Where F is the probability distribution determined by basic frequency analysis, and K is a scaling constant called the "Frequency Proportion". By default, the frequency proportion in this test is set to 0.5. 

# File descriptions
## Substitution Cipher Project
* caesar_network.py - Defines a Recurrent Neural Network, able to train from characters from a given text file. Also defines functions for decrypting a substitution cipher with a given network model
* train_caesar.py - Run this script to train the network. Hard coded values for the network training parameters and training file names. Saves trained network to an output file
* rnn_x_epoch.net - Saved network model after training
* test_caesar.py - Run this script to test the model after training. Hard coded values for the decryptian functions and output. Prints to the console the best guess for the hard-coded cipher and the accuracy rating

## Other files not used in project
* Character_Level_RNN_Exercise.ipynb - Udacity given Jupyter notebook from which the project was inspired, using an RNN to auto generate a passage of the novel Anna Katerina
* char_generator.py - Python script based on the Jupyter Notebook above, training a network based on the corpus of text from Anna Katerina
* anna.txt - Training text used to train network. Validation text is selected randomly from 10% of this file
* caesar2.py - Python script to solve a substitution cipher with brute force, instead of a neural network, to use as a benchmark. This script solves the same cipher from the test_caesar.py script in about three minutes. **Runs in Linux only**

# Implementation

## Training and Testing parameters

Training parameters:
* LSTM layer size = 512
* Number of LSTM layers = 2
* Sequence length = 500
* Learning Rate = 0.0014
* Clip = 5
* Number of epochs = 1

Testing parameters:
* Frequency proportion = 0.5
* Top K = 5

## Training data

The corpus of text used to train the neural network comes from the Gigaword dataset. After stripping out spaces, punctuation, and non-English characters the training set became a single file of 540 MB, and the validation set was a separate corpus of 26 MB. All remaining characters were set to lowercase. 

## Testing cipher

The cipher used for testing this program is this passage from the children's book _The Eleventh Hour_:

myxqbkdevkdsyxc sd gkc sxnoon usvbyi dro wyeco gry cdyvo dro pokcd led ro rkn rovz yxo rexnbon kxn ovofox yp usvbyic pebbi bovkdsfoc rsn drowcovfoc sx dro ryeco kxn kbyexn dro qkbnox kxn kd ovofox wsxedoc dy ovofox droi kvv cmkwzobon sxdy dro lkxaeod rkvv gsdr usvbyi kxn kdo ez kvv yp rybkmoc lokedspev pyyn kxn xyg tecd dy cryg ryg mvofob iye kbo mkx iye psxn yxo rexnbon kxn ovofox wsmo rsnnox sx dro zsmdeboc rkzzi rexdsxq

This is supposed to translate to:

congratulations it was indeed kilroy the mouse who stole the feast but he had help one hundred and eleven of kilroys furry relatives hid themselves in the house and around the garden and at eleven minutes to eleven they all scampered into the banquet hall with kilroy and ate up all of horaces beautiful food and now just to show how clever you are can you find one hundred and eleven mice hidden in the pictures happy hunting

## Results

The network trained over the course of a couple of days, completing a single epoch. The final value of the validation set was about 2.6, which was not much smaller than it started out at 2.9. The decryption test resulted in an accuracy of 0.06, which isn't much better than randomly guessing (0.03). It seems that the probability distributions were also never fully certain, with the highest probability ranging from 10-30% at most. It is unclear if this was the best possible result, or if something was wrong in the implementation. 
