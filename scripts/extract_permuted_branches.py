import re, string, codecs, os
import nltk
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer


def main():
    
    
    listing = os.listdir('./final_data/CNET/Basma_final_test_branch_level')
    listing = filter(lambda y: y[-7:] == 'sed.ner', listing)
    for l in listing:
        
        for i in range(20): # for each permuted doc compute branches
            
            permutation_file = "./final_data/CNET/Basma_final_test_branch_level/"+l+".perm-" + str(i+1)
            
            print permutation_file
            
            branch_id = 1
            
            branch_reader = open("./final_data/CNET/Basma_final_test_branch_level/" + l + ".branch", "r")
            
            for branch in branch_reader:
                
                branch_writer = open(permutation_file + ".branch-" + str(branch_id), "w")
                #branch_Idwriter = open(permutation_file + ".branchId-" + str(branch_id), "w")
                
                
                
                branch = branch.strip()
                branch = branch.split(",") # we have a table with sentence numbers
                
                for Id in branch:
                    sentence = getSentenceById(permutation_file, int(Id))
                    branch_writer.write(sentence)
                    #branch_Idwriter.write(str(getIdBySentence("../cnn_coherence/dataset/CNET/test_folder/"+ l + ".branch-" + str(branch_id), sentence)) + "\n")
                
                branch_id += 1

def getSentenceById(file_name, Id):
    reader = open(file_name,'r')
    i = 0
    for line in reader:
        i += 1
        if i == Id:
            return line
    return -1

def getIdBySentence(file_name, sentence):
    reader = open(file_name,'r')
    i = 0
    for line in reader:
        
        if line.strip() == sentence.strip():
            return i
        i += 1
    return "!!!!!"           
    
if __name__=='__main__':
    main()