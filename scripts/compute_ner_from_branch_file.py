import re, string, codecs, os
import nltk
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer

isSentenceWritten = {}

def main():
    
    
    listing = os.listdir('./final_data/CNET/Basma_final_train/')
    listing = filter(lambda y: y[-4:] == '.ner', listing)
    
    for l in listing:
        
        print l
            
        branch_reader = open("./final_data/CNET/Basma_final_train/" + l +".branch" , "r")
        
        global isSentenceWritten
        isSentenceWritten = {}
        branchID = 1
        for branch in branch_reader:
            
            branch_writer = open("./final_data/CNET/Basma_final_train/" + l + ".branch-"+str(branchID), "w")
            branchID += 1
            
            branch = branch.strip()
            branch = branch.split(",") # we have a table with sentence numbers
            
            #l_ = l.split(".")
            #l_ = ".".join(l_[:-1])
            
            for Id in branch:
                sentence = getSentenceById("./final_data/CNET/Basma_final_train/" + l, int(Id))
                #print sentence
                if sentence != None:
                    branch_writer.write(sentence)
                
            #print isSentenceWritten

"""

def main():
    
    
    listing = os.listdir('../cnn_coherence/dataset/CNET/test_trees/')
    #listing = filter(lambda y: y[-4:] == '.ner', listing)
    
    for l in listing:
        
        print l
            
        branch_reader = open("../cnn_coherence/dataset/CNET/test_trees/" + l , "r")
        
        global isSentenceWritten
        isSentenceWritten = {}
        branchID = 1
        for branch in branch_reader:
            
            branch_writer = open("../cnn_coherence/dataset/CNET/test_trees/" + l + ".branch-"+str(branchID), "w")
            branchID += 1
            
            branch = branch.strip()
            branch = branch.split(",") # we have a table with sentence numbers
            
            l_ = l.split(".")
            l_ = ".".join(l_[:-1])
            
            for Id in branch:
                sentence = getSentenceById("../cnn_coherence/dataset/CNET/threads/" + l_, int(Id))
                #print sentence
                if sentence != None:
                    branch_writer.write(sentence)
                
            #print isSentenceWritten
"""

def getSentenceById(file_name, Id):
    global isSentenceWritten
    reader = open(file_name,'r')
    i = 0
    for line in reader:
        i += 1
        if i == Id:
            #if i not in isSentenceWritten: # sentence never written (to avoid doubles)
                #print "yess"
                #isSentenceWritten.update({i:True})
                return line
            #else: #sentence already written
                #return None
    return "! "
            
    
if __name__=='__main__':
    main()