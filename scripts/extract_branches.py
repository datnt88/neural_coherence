"""
import re, string, codecs, os
import nltk
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer


def main():
    
    
    listing = os.listdir('../cnn_coherence/dataset/CNET/threads')
    
    k = 0
    listing = filter(lambda y: y[-6:] == '.depth', listing)
    
    boolean = False
    for l in listing:
        #l = "Thread_4591.txt.text.all.parsed.ner.depth"
        boolean = True
        print l
        
        listOfLines = {}
        listOfBranches = {}
        thread_dict = {}
        line_dict = {}
        
        reader = open("../cnn_coherence/dataset/CNET/threads/" + l, "r")
        thread_reader = open("../cnn_coherence/dataset/CNET/threads/" + l[:-6], "r")
        
        branch_writer = open("../cnn_coherence/dataset/CNET/threads/"+l[:-6]+".branch", "w")
        
        line_id = 0
        
        # first fill the dictioanry with line_id:depth and line_dict with line_id:line
        
        for line in reader:
            
            th_line = thread_reader.next()
            
            line_id += 1
            line = line.strip()
            th_line = th_line.strip()
            
            thread_dict.update({line_id:int(line)})
            line_dict.update({line_id:th_line})
        
        # once the dict contains all information start observing the increasing of detpths
        
        line_to_print = ""
        branch_line_to_print = ""
        previous_ = -1
        branch_counter = 0
        
        clean = True
        
        for line_id in thread_dict:

            if thread_dict[line_id] == previous_ + 1: # we keep increasing means we are in the same branch
                
                if previous_ == -1: # first value
                    line_to_print = str(line_id)
                    branch_line_to_print = line_dict[line_id]
                else:
                    line_to_print += ", " + str(line_id)
                    branch_line_to_print += "\n" + line_dict[line_id]

                previous_ = thread_dict[line_id]
                previous = previous_
                
                #print "++ ", line_to_print
                
                #if line_id == len(thread_dict):
                    #clean = True

            else: # we stop increasing this means the end of the previous branch ==> we write it in file and start looking for the beginning of the new branch
                line_id_parc = line_id
                clean = False
                while boolean == True:
                    
                    while line_id_parc <= len(thread_dict) and thread_dict[line_id_parc] != previous + 1:
                        line_id_parc += 1
                    
                    if line_id_parc <= len(thread_dict) and thread_dict[line_id_parc] != thread_dict[line_id_parc - 1] + 1:
                        
                        while line_id_parc <= len(thread_dict) and thread_dict[line_id_parc] == previous + 1:
                            line_to_print += ", " + str(line_id_parc)
                            branch_line_to_print += "\n" + line_dict[line_id_parc]
                            previous = thread_dict[line_id_parc]
                            
                            line_id_parc += 1
                    
                    elif line_id_parc > len(thread_dict):
                        boolean = False
                    else:
                        line_id_parc += 1
                
                
                if line_to_print not in listOfLines:
                    branch_counter += 1
                    listOfLines.update({line_to_print:branch_counter})
                    listOfBranches.update({branch_counter:branch_line_to_print})
                    
                    #print "-- ", line_to_print
                    
                    line_to_print = ""
                    branch_line_to_print = ""
                    
                    
                    clean = True
                
                line_id_ = line_id - 1
                previously = thread_dict[line_id_ + 1]
                
                while line_id_ > 0:
                    #print line_id_
                    if thread_dict[line_id_] == previously -1:
                        if line_to_print == "":
                            line_to_print =  str(line_id_) 
                        else:
                            line_to_print =  str(line_id_) + ", " + line_to_print
                        branch_line_to_print =  line_dict[line_id_] + "\n" + branch_line_to_print
                    
                        previously = thread_dict[line_id_]
                    line_id_ -= 1
                    boolean = True
                    
                ##line_id_ += 1
                #previously = -1
                #while (thread_dict[line_id_] < thread_dict[line_id] and line_id_ <= len(thread_dict)) :#thread_dict[line_id_] == previously + 1 #  start from the beginning of the dict and get all the sentences with depth < current_depth
                    #if thread_dict[line_id_] == previously + 1:
                        #if line_id_ == 1: # first value
                            #line_to_print = str(line_id_)
                            #branch_line_to_print = line_dict[line_id_]
                        #else:
                            #line_to_print += ", " + str(line_id_)
                            #branch_line_to_print += "\n" + line_dict[line_id_]
                        #print "// ", line_to_print
                        #previously = thread_dict[line_id_]
                        #line_id_ += 1
                        
                        #boolean = True
                    #else:
                        #line_id_ += 1
                
                    #print line_id_, "while"
                ## stop when we achieve the current sentence thread_dict[line_id_] == thread_dict[line_id]
                
                line_to_print += ", " + str(line_id)
                branch_line_to_print += "\n" + line_dict[line_id]
                #print "** ", line_to_print
                previous_ = thread_dict[line_id]
        
        if clean:
            branch_counter += 1
            if line_to_print not in listOfLines:
                listOfLines.update({line_to_print:branch_counter})
                listOfBranches.update({branch_counter:branch_line_to_print})        
        for line_to_print in listOfLines:
            branch_writer.write(line_to_print + "\n")
            branch_counter = listOfLines[line_to_print]
            branch_line_writer = open("../cnn_coherence/dataset/CNET/subthreads/"+l[:-6]+".branch-"+str(branch_counter), "w")
            branch_line_writer.write(listOfBranches[branch_counter] + "\n")
if __name__=='__main__':
    main()
    
"""

import os


 # merge the grid of positive document 
def get_branch_EGrid(e_trans="", idxs=""):
    x = e_trans.split()
    entity = x[0]
    x = x[1:] # remove the first 

    role = []
    for idx in idxs:
        role.append(x[idx])  #id in file starts at 1
    role = ' '.join(role)

    n = len(entity)
    ent =  " " * (20-n) + entity + " "

    return ent + role

    


list_of_files = [line.rstrip('\n') for line in open("final_data/CNET/x_cnet.train")]


for file_id, file in enumerate(list_of_files):
        print "------------------------------------"
        
        filename = file.split("/")[-1]
        print(filename) 
        #loading commentIDs
        cmtIDs  = [line.rstrip('\n') for line in open(file + ".commentIDs")]
        cmtIDs = [int(i) for i in cmtIDs] 

        org_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]

        branchID = 1
        with open("./final_data/CNET/Basma_final_train/" + filename + ".branch", "w") as f: 
            for branch in org_tree:   #reading branch, each branch is considered as a document
                
                line = ""
                branch = [int(id) for id in branch.split('.')] # convert string to integer, branch
                idxs = [ idx+1 for idx, cmtID  in enumerate(cmtIDs) if cmtID in branch]

                for idx in idxs:
                    line = line  + str(idx) + ", "
                f.write(line[:-2] + "\n") 
        branchID +=1