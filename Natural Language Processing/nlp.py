import string
import sys
import math

def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)
    
def shred(filename):
    '''
    This function parses a .txt file into a dict of character counts, where
    the input file may contain any printable ASCII characters and can be short 
    (e.g. a single word) or long (e.g. an article). Ignores case, i.e. merges 
    'A' and 'a' counts together, and so on (this is known as case-folding). 
    Only counts characters A to Z (after case-folding), ignoring all other 
    characters such as space, punctuations, etc.

    Sample Input/Output functionality:
    Input: "Hi! I'll go :-)"
    Output: {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0,
             'G': 1, 'H': 1, 'I': 2, 'J': 0, 'K': 0, 'L': 2,
             'M': 0, 'N': 0, 'O': 1, 'P': 0, 'Q': 0, 'R': 0,
             'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0,
             'Y': 0, 'Z': 0}

    Returns: dict of character counts
    '''

    boc=dict()
    with open (filename,encoding='utf-8') as f:
        corpus=f.read()
    f.close()
    #convert all lowercase alphabets to uppercase
    corpus=corpus.upper()
    #initialize X
    for a in string.ascii_uppercase:
        boc[a]=0
    for c in corpus:
        if c in string.ascii_uppercase:
            boc[c]+=1
    return boc

#the mystery letter
letter_file = sys.argv[1]

if(len(sys.argv) >= 3):
    ePrior = float(sys.argv[2])
    sPrior = float(sys.argv[3])
else:
    ePrior = 0.6
    sPrior = 0.4

alpha_dict = shred(letter_file)

#usage of letters in english and spanish parameter vectors
parameter_vectors = get_parameter_vectors()

# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!

def Q1():
    

    
    alpha_vector = int(alpha_dict['A'])

    english = parameter_vectors[0]
    spanish = parameter_vectors[1]

    englishOut = round(alpha_vector * math.log(english[0]), 4)
    spanishOut = round(alpha_vector * math.log(spanish[0]), 4)

    print("Q1")
    print(englishOut)
    print(spanishOut)

def F(language):
    funcE = 0

    for index,(key,value) in enumerate(alpha_dict.items()):
        value = alpha_dict[key]

        #Multiplies itself by the value of the parameter to the power of Xi
        funcE += value * math.log((parameter_vectors[language][index]))
    
    if(language == 0):
        funcE = funcE + math.log(ePrior)
    if(language == 1):
        funcE = funcE + math.log(sPrior)

    return funcE

def Q2():
    print("Q2")
    print(round(F(0),4))
    print(round(F(1),4))

def Q3():
    FofE = F(0)
    FofS = F(1)

    print("Q3")

    #control overflows for extreme FofS and FofE
    if(FofS - FofE >= 100):
        print(round(0,4))
        return
    if(FofS - FofE <= -100):
        print(round(1,4))
        return
    
    #output probability of language being english, given all of the letters
    PEgivenX = 1 / (1 + (math.e ** (FofS - FofE)))
    print(round(PEgivenX,4))

#call Q1, Q2, and Q3
Q1()
Q2()
Q3()