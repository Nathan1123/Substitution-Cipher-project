import re
import time
from copy import copy
from queue import LifoQueue

# Global constants
freq = 'etaoinsrhdlucmfywgpbvkxqjz' # Order of alphabet by letter frequency
dictionary = {} # All words arranged by size

# Prepare dictionary
with open('/usr/share/dict/words') as inFile:
     lines = list(inFile)
     # Format words to remove special characters
     lines = [lin.replace('\'','').rstrip() for lin in lines]
     # Remove words that don't use the Latin alphabet
     lines = [lin.lower() for lin in lines if re.fullmatch(r'[a-zA-Z]+', lin) is not None]
     # Remove duplicate words
     lines = list(set(lines))
     lines.sort()
     for i in range(1,23):
          dictionary[i] = [lin for lin in lines if len(lin) == i]

# The state of the algorithm is our current knowledge at any given time
class state:
     def __init__(self):
          self.key = {}
          self.missing = ''
          self.prev_missing = freq
          self.prev_key = {}
          self.candidates = LifoQueue()
          self.unknown = []
          self.new = True
          
class cracker:
     def __init__(self, ciphertext):
          self.cipher = ciphertext
          self.plain = ''
          self.states = LifoQueue()
          self.curr = state()
          self.init_key = {}
          self.dist = self.get_dist(self.cipher)
          
     # Attempt to decipher with current knowledge
     def decrypt(self, text):
          plaintext = ''
          self.curr.unknown = []
          for i in range(len(text)):
               # Ignore spaces
               if text[i] == ' ':
                    plaintext = plaintext + ' '
                    continue
               # If we haven't translated this character yet, use frequency analysis
               if text[i] not in self.curr.key.keys():
                    plaintext = plaintext + self.init_key[text[i]]
                    # Keep track of which letters are not translated yet
                    self.curr.unknown.append(i)
                    continue
               # Translate using key
               plaintext = plaintext + self.curr.key[text[i]]
          return plaintext
              
     # Hamming distance is the number of characters not in common
     def hamming(self, text, compare):
          return sum([a != b for a, b in zip(text, compare)])
          
     # Transform a string into an int list representing a unique pattern
     def encode(self, text):
          index = 0
          enc = {}
          for letter in text:
               if letter not in enc.keys():
                    enc[letter] = index
                    index = index + 1
          return [enc[i] for i in text]
     
     # Get all words that could be translated
     def get_cands(self, text, unknown, get_all=False):
          candidates = LifoQueue()
          # From dictionary
          dict_cand = dictionary[len(text)]
          # Sort by *largest* Hamming distance (because stack inverts) 
          hammed = [self.hamming(text, c) for c in dict_cand]
          dict_cand = [x for _,x in sorted(zip(hammed,dict_cand), reverse=True)]
          # Filter each word
          for cand in dict_cand:
               text_known = ''
               cand_known = ''
               text_unknown = ''
               cand_unknown = ''
               check_arr = []
               for i in range(len(cand)):
                    if i not in unknown:
                         # Known letters should be identical
                         text_known = text_known + text[i]
                         cand_known = cand_known + cand[i]
                         check_arr.append(text[i] == cand[i])
                    else:
                         # Unknown letters should be in the missing list
                         text_unknown = text_unknown + text[i]
                         cand_unknown = cand_unknown + cand[i]
                         check_arr.append(cand[i] in self.curr.missing)
               # Match pattern exactly
               check = (self.encode(text_known)+self.encode(text_unknown))==(self.encode(cand_known)+self.encode(cand_unknown))
               if (all(check_arr) and check) or get_all:
                    candidates.put(cand)
          return candidates
          
     # Get average hamming distance of entire text
     def get_dist(self, text):
          text_list = text.split(' ')
          total = 0
          for word in text_list:
               cands = self.get_cands(word, [], True)
               total = total + self.hamming(word, cands.get())
          return total/len(text_list)
          
     # Print current plaintext guess and distance
     def report(self):
          self.plain = ''.join([str(x) for x in self.decrypt(self.cipher)])
          self.dist = self.get_dist(self.plain)
          print(self.plain)
          print()
          print("Current distance: " + str(self.dist))
          print("Unmapped letters: " + self.curr.missing)
          print()
     
     def crack(self):
          start = time.time()
          # Frequency analysis
          let_freq = [[letter, self.cipher.count(letter)] for letter in freq]
          let_freq = sorted(let_freq, key=lambda l:l[1],reverse=True)
          for i in range(26):
               self.init_key[let_freq[i][0]] = freq[i]
          # Initialize
          self.report()
          self.states.put(copy(self.curr))
          word_list = self.cipher.split(' ')
          i = 0
          count = 0
          # Main loop
          while not self.states.empty() and i < len(word_list):
               # Initialize
               print('Word '+str(i)+'|'+str(len(word_list)))
               self.curr.missing = self.curr.prev_missing
               self.curr.key = copy(self.curr.prev_key)
               word = word_list[i]
               count = count + 1
               # Attempt to decipher
               plain_word = self.decrypt(word)
               # Get all words that could be the solution
               if self.curr.new:
                    self.curr.candidates = self.get_cands(plain_word, self.curr.unknown)
               # If at least one word is a solution, update the keys
               if not self.curr.candidates.empty():
                    cand_word = self.curr.candidates.get()
                    print("Next word guess: " + cand_word)
                    #print(self.curr.candidates.qsize())
                    for k in range(len(cand_word)):
                         # Update key and missing list for each letter
                         if cand_word[k] in self.curr.missing:
                              self.curr.key[word[k]] = cand_word[k]
                              self.curr.missing = self.curr.missing.replace(cand_word[k],'')
                    # Advanced to next word
                    i = i + 1
                    self.curr.new = False
                    self.states.put(copy(self.curr))
                    # Re initialize for next loop
                    self.curr.candidates = LifoQueue()
                    self.curr.new = True
                    self.curr.prev_missing = self.curr.missing
                    self.curr.prev_key = copy(self.curr.key)
                    # If fully translated, end
                    self.report()
                    if self.dist == 0:
                         break
               # If no candidates found, backtrack and try again
               else:
                    i = i - 1
                    self.curr = self.states.get()
          # Check if successful
          if self.dist == 0:
               #self.report()
               print("Decryption successful")
          else:
               print("Decryption failed")
          print("Total time: " + str(count) + " rounds (" + str(time.time()-start) + " s)") 
               
ciphertext = "myxqbkdevkdsyxc sd gkc sxnoon usvbyi dro wyeco gry cdyvo dro pokcd led ro rkn rovz yxo rexnbon kxn ovofox yp usvbyic pebbi bovkdsfoc rsn drowcovfoc sx dro ryeco kxn kbyexn dro qkbnox kxn kd ovofox wsxedoc dy ovofox droi kvv cmkwzobon sxdy dro lkxaeod rkvv gsdr usvbyi kxn kdo ez kvv yp rybkmoc lokedspev pyyn kxn xyg tecd dy cryg ryg mvofob iye kbo mkx iye psxn yxo rexnbon kxn ovofox wsmo rsnnox sx dro zsmdeboc rkzzi rexdsxq"
crac = cracker(ciphertext)
crac.crack()
