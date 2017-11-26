from __future__ import absolute_import
from six.moves import cPickle
import gzip
import random
import numpy as np

import glob, os, csv, re
from collections import Counter
from keras.preprocessing import sequence

import itertools
from utilities import gen_trees
#from zss import simple_distance, Node


#initilize basic vocabulary for cnn, this will change when using features
def init_vocab(emb_size):
    vocabs =['0','S','O','X','-']
    
    roles = ['S','O','X']
    f1 = ['F01','F02','F03','F04']
    f3 = ['F30','F31']
    f4 = ['F40','F41','F42','F43','F44','F45','F46','F47','F48','F49']

    #vocabs += [x+y+z+k for x in roles for y in f1 for z in f3 for k in f4]

    #print vocabs
    
    v2s = list(itertools.product('SOX-', repeat=2))
    for tupl in v2s:
        vocabs.append(''.join(tupl))

    v3s = list(itertools.product('SOX-', repeat=3))
    for tupl in v3s:
        vocabs.append(''.join(tupl))

    v4s = list(itertools.product('SOX-', repeat=4))
    for tupl in v4s:
        vocabs.append(''.join(tupl))

    v5s = list(itertools.product('SOX-', repeat=5))
    for tupl in v5s:
        vocabs.append(''.join(tupl))
    
    v6s = list(itertools.product('SOX-', repeat=6))
    for tupl in v6s:
        vocabs.append(''.join(tupl))
    

    np.random.seed(2017)
    E    = 0.01 * np.random.uniform( -1.0, 1.0, (len(vocabs), emb_size))
    E[0] = 0

    return vocabs, E

# compute tree edit distance
def compute_tree_edit_dist(tree1, tree2):

    x = ''.join(tree1)
    
    tree1_nodes = {}
    tree2_nodes = {}

    for i in x:
        tree1_nodes[i] = Node(i)
        tree2_nodes[i] = Node(i)

    # extract edges for each tree
    def get_edges(tree):
        edges = []
        
        for br in tree:
            for i in range(0,len(br)-1):
                edges.append(br[i:i+2])
        edges = sorted(list(set(edges)))
        return edges

    
    tree1_edges = get_edges(tree1)
    tree2_edges = get_edges(tree2)

    tree11 = tree1_nodes['1'].addkid(tree1_nodes['2']) 
    tree22 = tree2_nodes['1'].addkid(tree2_nodes['2']) 

    for edge in tree1_edges:
        n1 = edge[0]
        n2 = edge[1]
        tree1_nodes[n1].addkid(tree1_nodes[n2])
    #adding children for each tree

    for edge in tree2_edges:
        n1 = edge[0]
        n2 = edge[1]
        tree2_nodes[n1].addkid(tree2_nodes[n2])
    
    #print simple_distance(tree11,tree22)
    return simple_distance(tree11,tree22)


def load_one_tree_only(file="", sent_levels=[], maxlen=15000, w_size=2, vocabs=None, emb_size=300, fn=None):
    #load data with tree representation
    
    sentences_1 = []
    lines = [line.rstrip('\n') for line in open(file + ".EGrid")]
    grid_1 = "0 "* window_size
    for idx, line in enumerate(lines):
        e_trans = get_eTrans_with_Tree_Structure(sent=line, sent_levels=sent_levels) # merge the grid of positive document 
        if len(e_trans) !=0:
            grid_1 = grid_1 + e_trans + " " + "0 "* window_size

    sentences_1.append(grid_1)    

    #print len(sentences_1)
    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=window_size)
    X_1 = sequence.pad_sequences(X_1, maxlen)

    #need to return edit distance
    dists = np.array([0], dtype='int32').ravel()

    return X_1, dists


#load tree pair to train the tree level model
def load_tree_pairs(filelist="list_of_grid.txt", perm_num = 20, maxlen=15000, w_size=3, E=None, vocabs=None, emb_size=300, fn=None):
    if vocabs is None:
        print("Please input vocab list")
        return None

    list_of_files = [line.rstrip('\n') for line in open(filelist)]

    sentences_1 = []
    sentences_0 = []
    dists = []
    max_l = 0

    for file in list_of_files:  
        #print "---------------------------------------"
        #print file
        cmtIDs  = [line.rstrip('\n') for line in open(file + ".commentIDs")]
        cmtIDs = [int(i) for i in cmtIDs] 
        x_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]
        org_tree = []
        #print x_tree

        cmtIDs02 = []
        for i in x_tree:
            org_tree += [''.join(i.split("."))]
            cmtIDs02 += [int(j) for j in i.split(".")]

        sentDepths = get_sentences_depth(cmtIDs=cmtIDs,tree=org_tree) #load sentence depths
        lines = [line.rstrip('\n') for line in open(file + ".EGrid")] #load entity grid
        
        grid_1 = "0 "* w_size
        for idx, line in enumerate(lines):
            e_trans = get_eTrans_with_Tree_Structure(sent=line, sent_levels=sentDepths) # merge the grid of positive document 
            if len(e_trans) !=0:
                grid_1 = grid_1 + e_trans + " " + "0 "* w_size

        #loading possible tree
        nPost = max(cmtIDs02)
        #print nPost

        p_trees = gen_trees.gen_tree_branches(n=nPost)
        
        for p_tree in p_trees:
            p_sentDepths = get_sentences_depth(cmtIDs=cmtIDs,tree=p_tree)
            #print p_tree
            #print p_sentDepths

            grid_0 = "0 "* w_size
            for idx, line in enumerate(lines):
                e_trans_0 = get_eTrans_with_Tree_Structure(sent=line, sent_levels=p_sentDepths)
                if len(e_trans_0) !=0:
                    grid_0 = grid_0 + e_trans_0  + " " + "0 "* w_size
            
            if grid_0 != grid_1: #check the duplication
                #addding more pos/neg data        
                sentences_0.append(grid_0)
                sentences_1.append(grid_1)

                dists.append(compute_tree_edit_dist(org_tree,p_tree)) #compute tree-edit distance here

                if len(grid_1)/2 > max_l:
                    max_l = len(grid_1)/2

                if len(grid_0)/2 > max_l:
                    max_l = len(grid_0)/2
            
    assert len(sentences_0) == len(sentences_1)
    assert len(dists) == len(sentences_1)

    vocab_idmap = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)    

    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=w_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)
    
    dists = np.array(dists, dtype='int32').ravel()

    return X_1, X_0 , dists


def get_eTrans_with_Tree_Structure(sent="",sent_levels=None):
    x = sent.split()
    length = len(x) - 1 
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of occurrence of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""
            
    x = x[1:]
    #print x
    #print sent_levels

    sent_idxs = range(len(x))
    final_sent = [] #final sentence

    for lv in range(max(sent_levels)+1):
        indexes = [i for i,idx in enumerate(sent_levels) if idx == lv]
        tmp = ""
        for idx in indexes:
            tmp = tmp + x[idx]

        #TODO; maybe cut off the token latter on
        #if len(tmp) > 3:
        #    # pick up the highest grammarical role
        #    tmp = get_right_encode(vb=tmp)

        final_sent.append(tmp)

    return ' '.join(final_sent)
   
def get_sentences_depth(cmtIDs=[],tree=[]):
    branches = get_tree_struct(cmtIDs=cmtIDs,tree=tree) # get branches with sentID
    level_dict = {}
    
    for branch in branches:
        for i,j in enumerate(branch):
            level_dict[j] = i

    sentDepths = level_dict.values()
        
    return sentDepths

def get_tree_struct(cmtIDs=[],tree=[]):
    x_tree = []
    for branch in tree:
        sentIDs = []
        for cmtID in branch:   
            sentIDs += [ i for i, id_ in enumerate(cmtIDs) if id_ == int(cmtID)]
        x_tree.append(sentIDs)    

    return x_tree


#===================================================================
# this part for compute local coherence model
# consider coherence at edge level
#===================================================================

def load_one_edge_only(file="", pair="", maxlen=1000, w_size=5, vocabs=[], emb_size=50):


    postIDs = [line.rstrip('\n') for line in open(file + ".commentIDs")]
    postIDs = [int(i) for i in postIDs]         

    lines = [line.rstrip('\n') for line in open(file + ".EGrid")]  # Entity grid tranmistion 

    grid_1 = "0 "* w_size
    for e_trans in lines:
        x1, x2 = get_entity_trans(e_trans,postIDs,pair)
        if len(x1) !=0:
            #print e_trans
            grid_1 += x1 + " " + "0 "* w_size
    
    sentences_1 = []
    sentences_1.append(grid_1) 
    vocab_idmap = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)
    X_1 = sequence.pad_sequences(X_1, maxlen)
    
    #need to return edit distance
    dists = np.array([0], dtype='int32').ravel()

    return X_1, dists

#==================================================================
#loading training for pair of post
def load_edge_pairs_data(filelist="list_of_file.txt", maxlen=15000, w_size=3, E=None, vocabs=None, emb_size=300):
    # loading entiry-grid data for each pair of post in a thread
    list_of_files = [line.rstrip('\n') for line in open(filelist)]
        
    sentences_0 = []
    sentences_1 = []

    for file in list_of_files:
        #print "-------------------------------"
        #print file        
        postIDs = [line.rstrip('\n') for line in open(file + ".commentIDs")]
        postIDs = [int(i) for i in postIDs]         
        o_pairs, p_pairs, nPost = get_pos_and_neg_pairs(file)

        #print postIDs
        #print o_pairs
        #print p_pairs

        #loading additional data
        lines = [line.rstrip('\n') for line in open(file + ".EGrid")]  # Entity grid tranmistion 

        for p in o_pairs:
            pa_grid_1 = "0 "* w_size
            ch_grid_1 = "0 "* w_size

            for e_trans in lines:
                x1, x2 = get_entity_trans(e_trans,postIDs,p)
                if len(x1) !=0:
                    pa_grid_1 += x1 + " " + "0 "* w_size
                if len(x2) !=0:
                    ch_grid_1 += x2 + " " + "0 "* w_size
                
            same_parent, same_child  = get_incoherent_pairs(p,p_pairs) # get neg pair for corrent pair
            #print p, " - ", same_parent, same_child 

            if len(same_parent) > 0:
                for neg_p in same_parent:
                    grid_0 = "0 "* w_size
                
                    for e_trans in lines: #get trainsition for permutead papr
                        x1, x2 = get_entity_trans(e_trans,postIDs, neg_p)
                        if len(x1) !=0:
                            grid_0 += x1 + " " + "0 "* w_size
                    
                    if grid_0 != pa_grid_1:
                        sentences_0.append(grid_0)
                        sentences_1.append(pa_grid_1) 

            '''
            if len(same_child) > 0: # consider same parent, same child
                for neg_p in same_child:
                    grid_0 = "0 "* w_size
                
                    for e_trans in lines: #get trainsition for permutead papr
                        x1, x2 = get_entity_trans(e_trans,postIDs, neg_p)
                        if len(x2) !=0:
                            grid_0 += x2 + " " + "0 "* w_size
                    
                    if grid_0 != ch_grid_1:
                        sentences_0.append(grid_0)
                        sentences_1.append(ch_grid_1) 
            '''

    assert len(sentences_0) == len(sentences_1)
    
    vocab_idmap = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=w_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)

    #dists = np.array(dists, dtype='int32').ravel()

    return X_1, X_0


def get_incoherent_pairs(pair,p_pairs):

    parent = pair[0]
    child  = pair[2]
    
    same_parent  = []
    same_child = []

    for p in p_pairs:
        if parent == p[0]:
            same_parent.append(p) 
        if child == p[2]:
            same_child.append(p) 

    return same_parent, same_child

def get_entity_trans(e_trans,postIDs,p):
    #consider entity just appears in the parent    
    p1_sentIDs = [i for i,j in enumerate(postIDs) if str(j) == p[0]]
    p2_sentIDs = [i for i,j in enumerate(postIDs) if str(j) == p[2]]

    x = e_trans.split()
    x = x[1:] # remove the first 
    
    e_p1 = [x[i] for i in p1_sentIDs]
    e_p2 = [x[i] for i in p2_sentIDs]
    
    e_occur_p1 = e_p1.count('X') + e_p1.count('S') + e_p1.count('O') #counting occur in parent post
    e_occur_p2 = e_p2.count('X') + e_p2.count('S') + e_p2.count('O') #counting occur in children post

    X1 = ""
    X2 = ""
    if e_occur_p1 > 0:
        X1 = ' '.join(e_p1 + e_p2)

    if e_occur_p2 > 0:
        X2 = ' '.join(e_p1 + e_p2)

    return X1, X2

    
def get_pos_and_neg_pairs(file): # get every pair from a thread
    tree = [line.rstrip('\n') for line in open(file+".orgTree")]
    #print tree
    o_pairs = []
    nPost = 0
    for branch in tree:
        branch = branch.split('.')
        n=  len(branch)        
        for i in range(0,n-1):
            o_pairs.append(branch[i] + "." +  branch[i+1])
            tmp = int(branch[i+1])
            if tmp > nPost:
                nPost = tmp

    o_pairs = sorted(list(set(o_pairs)))

    all_pairs = []
    for i in range(1,nPost):
        all_pairs += [str(i)+ "." +str(j) for j in range(i+1,nPost+1) ]

    p_pairs = sorted([i for i in all_pairs if i not in o_pairs])

    return o_pairs, p_pairs, nPost



#=====================================================================
#loading data by branch
#
#=====================================================================
def load_data_by_branch(filelist="list_of_grid.txt", perm_num = 20, maxlen=15000, w_size=3, vocabs=None, emb_size=300, fn=None):
    # loading entiry-grid data from list of pos document and list of neg document
    list_of_files = [line.rstrip('\n') for line in open(filelist)]
    
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []
    f_track = [] # tracking branch
    pair_id = 0
    
    for file_id, file in enumerate(list_of_files):
        print(file) 

        #loading commentIDs
        cmtIDs  = [line.rstrip('\n') for line in open(file + ".commentIDs")]
        cmtIDs = [int(i) for i in cmtIDs] 

        org_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]
        egrids = [line.rstrip('\n') for line in open(file + ".EGrid")]

        f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]
        f_list ={}
        for line in f_lines:
            x = line.split()
            entity = x[0]
            x = x[1:] # remnove the entity name
            f_list[entity] = x
    
    
        #f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]

        for branch in org_tree:   #reading branch, each branch is considered as a document
            f_track.append(pair_id) #keep track branch for each thread

            branch = [int(id) for id in branch.split('.')] # convert string to integer, branch
            idxs = [ idx for idx, cmtID  in enumerate(cmtIDs) if cmtID in branch]

            grid_1 = "0 "* w_size
            for idx, line in enumerate(egrids):
                e_trans = get_eTrans_by_Index(e_trans=line, idxs=idxs, f_list=f_list, feats=fn) # merge the grid of positive document 
                if len(e_trans) !=0:
                    #print e_trans
                    grid_1 = grid_1 + e_trans + " " + "0 "* w_size

            #print grid_1
                
            p_count = 0
            for i in range(1,perm_num+1): # reading the permuted docs
                permuted_lines = [p_line.rstrip('\n') for p_line in open(file+ ".EGrid" +"-"+str(i))]    
                grid_0 = "0 "* w_size

                for idx, p_line in enumerate(permuted_lines):
                    e_trans_0 = get_eTrans_by_Index(e_trans=p_line, idxs=idxs, f_list=f_list, feats=fn)
                    if len(e_trans_0) !=0:
                        grid_0 = grid_0 + e_trans_0  + " " + "0 "* w_size

                if grid_0 != grid_1: #check the duplication
                    p_count = p_count + 1
                    sentences_0.append(grid_0)        
            
            for i in range (0, p_count): #stupid code
                sentences_1.append(grid_1)

        pair_id +=1

    assert len(sentences_0) == len(sentences_1)


    vocab_idmap = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=w_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)

    return X_1, X_0, f_track


def load_branch_POS_EGrid(filename="", w_size=3, maxlen=1000, vocabs=None , feats=None ):
    cmtIDs  = [line.rstrip('\n') for line in open(filename + ".commentIDs")]
    cmtIDs = [int(i) for i in cmtIDs] 

    org_tree = [line.rstrip('\n') for line in open(filename + ".orgTree")]
    egrids = [line.rstrip('\n') for line in open(filename + ".EGrid")]

    f_lines = [line.rstrip('\n') for line in open(filename + ".Feats")]
    f_list ={}
    for line in f_lines:
        x = line.split()
        entity = x[0]
        x = x[1:] # remove the entity name
        f_list[entity] = x
    
    sentences_1 = []
    for branch in org_tree:   #reading branch, each branch is considered as a document
        branch = [int(id) for id in branch.split('.')] # convert string to integer, branch
        idxs = [ idx for idx, cmtID  in enumerate(cmtIDs) if cmtID in branch]

        grid_1 = "0 "* w_size
        for idx, line in enumerate(egrids):
            e_trans = get_eTrans_by_Index(e_trans=line, idxs=idxs, f_list=f_list, feats=feats) # merge the grid of positive document 
            if len(e_trans) !=0:
                    #print e_trans
                grid_1 = grid_1 + e_trans + " " + "0 "* w_size

        sentences_1.append(grid_1)


    #print(grid_1)
    vocab_idmap = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i

    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)
    X_1 =  sequence.pad_sequences(X_1, maxlen)

    return X_1

def load_branch_NEG_EGrid(filename="", w_size=3, maxlen=1000, vocabs=None , feats=None, perm=None):
    cmtIDs  = [line.rstrip('\n') for line in open(filename + ".commentIDs")]
    cmtIDs = [int(i) for i in cmtIDs] 

    org_tree = [line.rstrip('\n') for line in open(filename + ".orgTree")]
    egrids = [line.rstrip('\n') for line in open(filename + ".EGrid")]

    f_lines = [line.rstrip('\n') for line in open(filename + ".Feats")]
    f_list ={}
    for line in f_lines:
        x = line.split()
        entity = x[0]
        x = x[1:] # remove the entity name
        f_list[entity] = x
    

    sentences_1 = []
    for branch in org_tree:   #reading branch, each branch is considered as a document
        branch = [int(id) for id in branch.split('.')] # convert string to integer, branch
        idxs = [ idx for idx, cmtID  in enumerate(cmtIDs) if cmtID in branch]

        grid_1 = "0 "* w_size
        for idx, line in enumerate(egrids):
            e_trans = get_eTrans_by_Index_4Insert(e_trans=line, idxs=idxs, f_list=f_list, feats=feats, perm=perm) # merge the grid of positive document 
            if len(e_trans) !=0:
                    #print e_trans
                grid_1 = grid_1 + e_trans + " " + "0 "* w_size

        sentences_1.append(grid_1)


    #print(grid_1)
    vocab_idmap = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i

    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)
    X_1 =  sequence.pad_sequences(X_1, maxlen)

    return X_1

def get_eTrans_by_Index_4Insert(e_trans="", idxs=None, f_list={}, feats=None, perm=None):
    x = e_trans.split()
    entity = x[0]
    x = x[1:] # remove the first 

    #need to re-order the sentence, but keep the same tree structure 
    p_x = []
    for i in perm:
        p_x.append(x[i])
    x = p_x

    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of occurrence of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""

    final_sent = []

    for idx in idxs:
        final_sent.append(x[idx])  #id in file starts at 1
    if feats==None: #coherence model without features    
        return ' '.join(final_sent)

    f = f_list[entity]
    x_f = []
    for sem_role in x:
        new_role = sem_role;
        if new_role != '-':
            for i in feats:
                if i ==0 : #adding salience
                    if e_occur == 1:
                        new_role = new_role + "F01"
                    elif e_occur == 2:
                        new_role = new_role + "F02"
                    elif e_occur == 3:
                        new_role = new_role + "F03"
                    elif e_occur >3 :
                        new_role = new_role + "F04"
                else:
                    new_role = new_role + "F" + str(i) + f[i-1] # num feat = idx + 1
  
        x_f.append(new_role)

    return ' '.join(x_f)


def get_eTrans_by_Index(e_trans="", idxs=None, f_list={}, feats=None):

    x = e_trans.split()
    entity = x[0]
    x = x[1:] # remove the first 

    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of occurrence of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""

    final_sent = []

    for idx in idxs:
        final_sent.append(x[idx])  #id in file starts at 1
    if feats==None: #coherence model without features    
        return ' '.join(final_sent)

    f = f_list[entity]
    x_f = []
    for sem_role in x:
        new_role = sem_role;
        if new_role != '-':
            for i in feats:
                if i ==0 : #adding salience
                    if e_occur == 1:
                        new_role = new_role + "F01"
                    elif e_occur == 2:
                        new_role = new_role + "F02"
                    elif e_occur == 3:
                        new_role = new_role + "F03"
                    elif e_occur >3 :
                        new_role = new_role + "F04"
                else:
                    new_role = new_role + "F" + str(i) + f[i-1] # num feat = idx + 1
  
        x_f.append(new_role)

    return ' '.join(x_f)





#=====================================================================
# load data by temporal order
#
#
#=====================================================================
def load_data_by_temporal(filelist="list_of_grid.txt", maxlen=15000, w_size=3, E=None, vocabs=None, emb_size=300, perm_num=20, feats=None):
    # loading entiry-grid data from list of pos document and list of neg document
    if vocabs is None:
        print("Please input vocab list")
        return None

    list_of_files = [line.rstrip('\n') for line in open(filelist)]

    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []

    for file in list_of_files:
        #print(file) 

        e_lines = [line.rstrip('\n') for line in open(file + ".EGrid")]
        f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]

        f_list = {}
        for line in f_lines:
            x = line.split()
            entity = x[0]
            x = x[1:] # remnove the entity name
            f_list[entity] = x


        grid_1 = "0 "* w_size

        for idx, line in enumerate(e_lines):
            e_trans = get_eTrans_with_Feats(sent=line,f_list=f_list,feats=feats) # merge the grid of positive document  
            if len(e_trans) !=0:
                #print e_trans
                grid_1 = grid_1 + e_trans + " " + "0 "* w_size


        p_count = 0
        for i in range(1,perm_num+1): # reading the permuted docs
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file+ ".EGrid" +"-"+str(i))]    
            grid_0 = "0 "* w_size

            for idx, p_line in enumerate(permuted_lines):
                e_trans_0 = get_eTrans_with_Feats(sent=p_line,f_list=f_list,feats=feats)
                if len(e_trans_0) !=0:
                    grid_0 = grid_0 + e_trans_0  + " " + "0 "* w_size

            if grid_0 != grid_1: #check the duplication
                p_count = p_count + 1
                sentences_0.append(grid_0)
            #else:
            #    print(file+ ".EGrid" +"-"+str(i)) // print duplicates permuted docs with original
        
        for i in range (0, p_count): #stupid code
            sentences_1.append(grid_1)

    
    assert len(sentences_0) == len(sentences_1)

    vocab_idmap = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=w_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)

    return X_1, X_0



def load_temporal_POS_EGrid(filename="", w_size=3, maxlen=1000, vocabs=None , feats=None ):
    lines = [line.rstrip('\n') for line in open(filename + ".EGrid")]
    f_lines = [line.rstrip('\n') for line in open(filename + ".Feats")]
    #get feats information
    f_list = {}
    for line in f_lines:
        x = line.split()
        entity = x[0]
        x = x[1:] # remnove the entity name
        f_list[entity] = x



    grid_1 = "0 "* w_size
    for idx, line in enumerate(lines):
        # merge the grid of positive document 
        e_trans = get_eTrans_with_Feats(sent=line, f_list=f_list,feats=feats)
        if len(e_trans) !=0:
            grid_1 = grid_1 + e_trans + " " + "0 "* w_size
            #print(e_trans)
    #print(grid_1)
    vocab_idmap = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i

    X_1 = numberize_sentences([grid_1], vocab_idmap)
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)
    X_1 =  sequence.pad_sequences(X_1, maxlen)

    return X_1

def load_temporal_NEG_EGrid(filename="", w_size=3, maxlen=1000,  vocabs=None, feats=None, perm=None):
    #print(perm)
    if perm != None:
        lines = [line.rstrip('\n') for line in open(filename + ".EGrid")]
        f_lines = [line.rstrip('\n') for line in open(filename + ".Feats")]
        #get feats information
        f_list = {}
        for line in f_lines:
            x = line.split()
            entity = x[0]
            x = x[1:] # remnove the entity name
            f_list[entity] = x


        grid_0 = "0 "* w_size
        for idx, line in enumerate(lines):
            #print(line)
            e_trans_0 = get_eTrans_With_Feats_4Insert(sent=line,f_list=f_list,feats=feats, perm=perm) # need to include features
            if len(e_trans_0) !=0:
                grid_0 = grid_0 + e_trans_0  + " " + "0 "* w_size
            #print(e_trans_0)
        
        #print(grid_0)
        vocab_idmap = {}
        for i in range(len(vocabs)):
            vocab_idmap[vocabs[i]] = i

        X_0 = numberize_sentences([grid_0], vocab_idmap)
        X_0 = adjust_index(X_0, maxlen=maxlen, window_size=w_size)
        X_0 = sequence.pad_sequences(X_0, maxlen)

        return X_0

    else:
        print("no permuted list")
        return None

def get_eTrans_With_Feats_4Insert(sent="",f_list={},feats=None, perm=[]):
    x = sent.split()
    entity = x[0]
    x = x[1:] # remnove the entity name
    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of coocurence of the entity
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""
    #need to re-order the entity meaning re-order sentence in a document
    p_x = []
    for i in perm:
        p_x.append(x[i])

    if feats==None: #coherence model without features    
       return ' '.join(p_x)

    f = f_list[entity]
    p_x_f = []
    for sem_role in p_x: # extended coherence model without features
        new_role = sem_role;
        if new_role != '-':
            for i in feats:
                if i ==0 : #adding salience
                    if e_occur == 1:
                        new_role = new_role + "F01"
                    elif e_occur == 2:
                        new_role = new_role + "F02"
                    elif e_occur == 3:
                        new_role = new_role + "F03"
                    elif e_occur >3 :
                        new_role = new_role + "F04"
                else:
                    new_role = new_role + "F" + str(i) + f[i-1] # num feat = idx + 1
  
        p_x_f.append(new_role)

    return ' '.join(p_x_f)


#get entity transition from a row of Entity Grid
def get_eTrans_with_Feats(sent="",f_list={},feats=None):
    x = sent.split()
    entity = x[0]
    x = x[1:] # remnove the entity name
    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of occurrence of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""
    
    if feats==None: #coherence model without features    
        return ' '.join(x)     

    f = f_list[entity]
    x_f = []
    for sem_role in x:
        new_role = sem_role;
        if new_role != '-':
            for i in feats:
                if i ==0 : #adding salience
                    if e_occur == 1:
                        new_role = new_role + "F01"
                    elif e_occur == 2:
                        new_role = new_role + "F02"
                    elif e_occur == 3:
                        new_role = new_role + "F03"
                    elif e_occur >3 :
                        new_role = new_role + "F04"
                else:
                    new_role = new_role + "F" + str(i) + f[i-1] # num feat = idx + 1
  
        x_f.append(new_role)

    return ' '.join(x_f)



def find_doc_size(filename=""):
    lines = [line.rstrip('\n') for line in open(filename)]
    doc_size = find_len(sent=lines[1])
    return doc_size



def find_len(sent=""):
    x = sent.split()
    #print(x)
    return len(x) -1








#=====================================================================
#loading data by tree/sentecne structure
#
#=====================================================================
def load_data_by_tree(filelist="list_of_grid.txt", perm_num = 20, maxlen=15000, w_size=3, vocabs=None, emb_size=300, fn=None):
    # loading entiry-grid data from list of pos document and list of neg document
    list_of_files = [line.rstrip('\n') for line in open(filelist)]
    
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []
    
    for file_id, file in enumerate(list_of_files):
        #print(file) 

        #loading commentIDs
        depths  = [int(line) for line in open(file + ".depth")]
        egrids = [line.rstrip('\n') for line in open(file + ".EGrid")]
        f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]
        f_list ={}

        #print depths

        for line in f_lines:
            x = line.split()
            entity = x[0]
            x = x[1:] # remnove the entity name
            f_list[entity] = x

        grid_1 = "0 "* w_size
        for idx, line in enumerate(egrids):
            e_trans = get_eTrans_by_Depth(e_trans=line, depths=depths, f_list=f_list, feats=fn) # merge the grid of positive document 
            if len(e_trans) !=0:
                 #print e_trans
                grid_1 = grid_1 + e_trans + " " + "0 "* w_size

            #print grid_1
                
        p_count = 0
        for i in range(1,perm_num+1): # reading the permuted docs
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file+ ".EGrid" +"-"+str(i))]    
            grid_0 = "0 "* w_size

            for idx, p_line in enumerate(permuted_lines):
                e_trans_0 = get_eTrans_by_Depth(e_trans=p_line, depths=depths, f_list=f_list, feats=fn)
                if len(e_trans_0) !=0:
                    grid_0 = grid_0 + e_trans_0  + " " + "0 "* w_size

            if grid_0 != grid_1: #check the duplication
                p_count = p_count + 1
                sentences_0.append(grid_0)        
            
        for i in range (0, p_count): #stupid code
            sentences_1.append(grid_1)

    assert len(sentences_0) == len(sentences_1)


    vocab_idmap = {}
    for i in range(len(vocabs)):
        vocab_idmap[vocabs[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=w_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=w_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)

    return X_1, X_0


def get_eTrans_by_Depth(e_trans="", depths=None, f_list={}, feats=None):

    x = e_trans.split()
    entity = x[0]
    x = x[1:] # remove the first 

    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of occurrence of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""

    final_sent = []

    for lv in range(max(depths)+1):
        idxs = [i for i,idx in enumerate(depths) if idx == lv]
        tmp = ""
        for j in idxs[0:6]:  # pick the first 6 role in a level
            tmp = tmp + x[j]

        #TODO; maybe cut off the token latter on
        #if len(tmp) > 3:
        #    # pick up the highest grammarical role
        #    tmp = get_right_encode(vb=tmp)

        final_sent.append(tmp)

    return ' '.join(final_sent)



































#=====================================================================================

def load_permuted_tree(file=[], tree=[], maxlen=15000, window_size=2, vocab_list=None, emb_size=300, fn=None):
    #load data with tree representation
    sentences_1 = []
    lines = [line.rstrip('\n') for line in open(file + ".EGrid")]

    for branch in tree:  
        #print branch
        grid_1 = "0 "* window_size

        for idx, line in enumerate(lines):
            e_trans = get_eTrans_with_Branch_New(sent=line, idxs=branch) # merge the grid of positive document 
            if len(e_trans) !=0:
                #print e_trans
                grid_1 = grid_1 + e_trans + " " + "0 "* window_size

        sentences_1.append(grid_1) 

    #print len(sentences_1)

    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=window_size)
    X_1 = sequence.pad_sequences(X_1, maxlen)
    
    return X_1

def get_eTrans_with_Branch_New(sent="p_line", idxs=[]):
    x = sent.split()
    
    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of occurrence of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""

    x = x[1:]
    final_sent = []

    for idx in idxs:
        final_sent.append(x[idx])  #sentence id in file starts at 0

    return ' '.join(final_sent)


def load_original_tree(file="list_of_grid.txt", maxlen=15000, window_size=3, vocab_list=None, emb_size=300, fn=None):
    
    #loading original entity grid
    sentences_1 = []
    branches = [line.rstrip('\n') for line in open(file + ".branch")]
    lines = [line.rstrip('\n') for line in open(file + ".EGrid")]

    for branch in branches:  
        grid_1 = "0 "* window_size
        for idx, line in enumerate(lines):
            e_trans = get_eTrans_with_Branch(sent=line, branch=branch) # merge the grid of positive document 
            if len(e_trans) !=0:
                        #print e_trans
                grid_1 = grid_1 + e_trans + " " + "0 "* window_size

        sentences_1.append(grid_1) 


    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=window_size)
    X_1 = sequence.pad_sequences(X_1, maxlen)
    
    return X_1 


def load_testing_data(filelist="list_of_grid.txt", perm_num = 20, maxlen=15000, window_size=3, E=None, vocab_list=None, emb_size=300, fn=None):
    # loading entiry-grid data from list of pos document and list of neg document
    list_of_files = [line.rstrip('\n') for line in open(filelist)]
    
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []
    f_track = [] # tracking branch
    pair_id = 0

    for file_id, file in enumerate(list_of_files):
        print(file) 
        branches = [line.rstrip('\n') for line in open(file + ".branch")]
        lines = [line.rstrip('\n') for line in open(file + ".EGrid")]
        #f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]

        for i in range(1,perm_num+1): # reading the permuted docs
            
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file+ ".EGrid" +"-"+str(i))]    

            
            for branch in branches:   #reading branch, each branch is considered as a document
                f_track.append(pair_id)

                grid_1 = "0 "* window_size
                for idx, line in enumerate(lines):
                    e_trans = get_eTrans_with_Branch(sent=line, branch=branch) # merge the grid of positive document 
                    if len(e_trans) !=0:
                        #print e_trans
                        grid_1 = grid_1 + e_trans + " " + "0 "* window_size

                grid_0 = "0 "* window_size
                for idx, p_line in enumerate(permuted_lines):
                    e_trans_0 = get_eTrans_with_Branch(sent=p_line, branch=branch)
                    if len(e_trans_0) !=0:
                        grid_0 = grid_0 + e_trans_0  + " " + "0 "* window_size
                
                sentences_0.append(grid_0)
                sentences_1.append(grid_1) 

            pair_id +=1

    assert len(sentences_0) == len(sentences_1)

    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=window_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=window_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)

    if E is None:
        E      = 0.01 * np.random.uniform( -1.0, 1.0, (len(vocab_list), emb_size))
        E[len(vocab_list)-1] = 0

    return X_1, X_0, E , f_track






#=====================================================================


def load_task_X(filelist="list_of_grid.txt", perm_num = 20, maxlen=15000, window_size=3, E=None, vocab_list=None, emb_size=300, fn=None):
    if vocab_list is None:
        print("Please input vocab list")
        return None

    list_of_files = [line.rstrip('\n') for line in open(filelist)]

    sentences_1 = []
    sentences_0 = []
    
    for file in list_of_files:  
        #print "---------------------------------------"
        #print file
    
        cmtIDs  = [line.rstrip('\n') for line in open(file + ".commentIDs")]
        cmtIDs = [int(i) for i in cmtIDs] 
        x_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]
        org_tree = []
        for i in x_tree:
            org_tree += [''.join(i.split("."))]
        #print org_tree


        sentDepths = get_sentences_depth(cmtIDs=cmtIDs,tree=org_tree)
        #print sentDepths
        #print "---------"

        lines = [line.rstrip('\n') for line in open(file + ".EGrid")]
        #f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]

        grid_1 = "0 "* window_size
        for idx, line in enumerate(lines):
            e_trans = get_eTrans_with_Tree_Structure(sent=line, sent_levels=sentDepths) # merge the grid of positive document 
            if len(e_trans) !=0:
                grid_1 = grid_1 + e_trans + " " + "0 "* window_size


        #loading possible tree

        nPost = max(cmtIDs)
        if nPost > 5:
            nPost = 5

        p_trees = gen_trees.gen_tree_branches(n=nPost)
        p_count = 0
        for p_tree in p_trees:
            p_sentDepths = get_sentences_depth(cmtIDs=cmtIDs,tree=p_tree)
            #print p_tree
            #print p_sentDepths

            grid_0 = "0 "* window_size
            for idx, line in enumerate(lines):
                e_trans_0 = get_eTrans_with_Tree_Structure(sent=line, sent_levels=p_sentDepths)
                if len(e_trans_0) !=0:
                    grid_0 = grid_0 + e_trans_0  + " " + "0 "* window_size
            #print grid_0

            if grid_0 != grid_1: #check the duplication
                p_count = p_count + 1
                sentences_0.append(grid_0)
        
    
        #addding more positive data
        for i in range (0, p_count): #stupid code
            sentences_1.append(grid_1)

    assert len(sentences_0) == len(sentences_1)


    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=window_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=window_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)

    if E is None:
        E      = 0.01 * np.random.uniform( -1.0, 1.0, (len(vocab_list), emb_size))
    E[0] = 0

    return X_1, X_0, E 



def load_and_numberize_with_Tree_Structure(filelist="list_of_grid.txt", perm_num = 20, maxlen=15000, window_size=3, E=None, vocab_list=None, emb_size=300, fn=None):
    # loading entiry-grid data from list of pos document and list of neg document
    if vocab_list is None:
        print("Please input vocab list")
        return None

    list_of_files = [line.rstrip('\n') for line in open(filelist)]
    
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []
    
    for file in list_of_files:
        #print "--------------------------------"
        #print(file) 
        
        #loading tree level in diffent ways
        cmtIDs  = [line.rstrip('\n') for line in open(file + ".commentIDs")]
        cmtIDs = [int(i) for i in cmtIDs] 
        org_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]
        x_tree = []
        for i in org_tree:
            x_tree += [''.join(i.split("."))]

        branches = get_tree_struct(cmtIDs=cmtIDs,tree=x_tree) # get branches with sentID

        level_dict = {}
        for branch in branches:
            for i,j in enumerate(branch):
                level_dict[j] = i

        sentDepths = level_dict.values()

        #print sentDepths
        
        #loading entity
        lines = [line.rstrip('\n') for line in open(file + ".EGrid")]
        #f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]

        grid_1 = "0 "* window_size
        for idx, line in enumerate(lines):
            e_trans = get_eTrans_with_Tree_Structure(sent=line, sent_levels=sentDepths) # merge the grid of positive document 
            if len(e_trans) !=0:
                grid_1 = grid_1 + e_trans + " " + "0 "* window_size
        #print(grid_1)
                
        p_count = 0
        for i in range(1,perm_num+1): # reading the permuted docs
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file+ ".EGrid" +"-"+str(i))]    
            grid_0 = "0 "* window_size

            for idx, p_line in enumerate(permuted_lines):
                e_trans_0 = get_eTrans_with_Tree_Structure(sent=p_line, sent_levels=sentDepths)
                if len(e_trans_0) !=0:
                    grid_0 = grid_0 + e_trans_0  + " " + "0 "* window_size

            if grid_0 != grid_1: #check the duplication
                p_count = p_count + 1
                sentences_0.append(grid_0)
            #else:
            #    print(file+ ".EGrid" +"-"+str(i)) // print duplicates permuted docs with original
        
        for i in range (0, p_count): #stupid code
            sentences_1.append(grid_1)


    assert len(sentences_0) == len(sentences_1)


    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=window_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=window_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)

    if E is None:
        E      = 0.01 * np.random.uniform( -1.0, 1.0, (len(vocab_list), emb_size))
        E[len(vocab_list)-1] = 0

    return X_1, X_0, E 




def get_right_encode(vb="S---XOS"):
    
    priority_list = []
    for ch in vb:
        if ch == 'S':
            priority_list.append("4")
        elif ch =='O':
            priority_list.append("3")
        elif ch =='X':
            priority_list.append("2")
        else:
            priority_list.append("1")

    x = list(itertools.combinations(priority_list, 3))
    combs = []
    for tupl in x:
        combs.append(int(''.join(tupl)))

    x_vb = str(max(combs))
    
    right_vb = ""
    for ch in x_vb:
        if ch == '4':
            right_vb = right_vb + "S" 
        elif ch =='3':
            right_vb = right_vb + "O" 
        elif ch =='2':
            right_vb = right_vb + "X" 
        else:
            right_vb = right_vb + "-" 

    return right_vb




#=================================================================================


def remove_entity(sent=""):
    x = sent.split()
    count = x.count('X') + x.count('S') + x.count('O') #counting the number of entities
    if count <3: #remove lesss ferequent entities
        return ""
    x = x[1:]
    return ' '.join(x)

#get entity transition from a row of Entity Grid
def get_eTrans(sent=""):
    x = sent.split()
    x = x[1:]
    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of entities
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""
    return ' '.join(x)





#get entity transition from a row of Entity Grid for insertion experiment
def get_eTrans_With_Perm(sent="",feats="",fn=None, perm=[]):
    #print(feats)
    #print(perm)

    x = sent.split()
    x = x[1:]
    length = len(x)
    e_occur = x.count('X') + x.count('S') + x.count('O') #counting the number of coocurence of the entity
    if length > 80:
        if e_occur < 3:
            return ""
    elif length > 20:
        if e_occur < 2:
            return ""
    #need to re-order the entity meaning re-order sentence in a document
    p_x = []
    for i in perm:
        p_x.append(x[i])

    if fn==None: #coherence model without features
       return ' '.join(p_x)


    f = feats.split()
    f = f[1:]

    p_x_f = []
    for sem_role in p_x: # extended coherence model without features
        new_role = sem_role;
        if new_role != '-':
            for i in fn:
                if i ==0 : #adding salience
                    if e_occur == 1:
                        new_role = new_role + "F01"
                    elif e_occur == 2:
                        new_role = new_role + "F02"
                    elif e_occur == 3:
                        new_role = new_role + "F03"
                    elif e_occur >3 :
                        new_role = new_role + "F04"
                else:
                    new_role = new_role + "F" + str(i) + f[i-1] # num feat = idx + 1
  
        p_x_f.append(new_role)

    return ' '.join(p_x_f)







#loading grid with features
def load_and_numberize_Egrid_with_Feats(filelist="list_of_grid.txt", perm_num = 20, maxlen=15000, window_size=3, E=None, vocab_list=None, emb_size=300, fn=None):
    # loading entiry-grid data from list of pos document and list of neg document
    if vocab_list is None:
        print("Please input vocab list")
        return None

    list_of_files = [line.rstrip('\n') for line in open(filelist)]
    
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []
    
    for file in list_of_files:
        #print(file) 

        lines = [line.rstrip('\n') for line in open(file + ".EGrid")]
        f_lines = [line.rstrip('\n') for line in open(file + ".Feats")]

        grid_1 = "0 "* window_size

        for idx, line in enumerate(lines):
            e_trans = get_eTrans_with_Feats(sent=line,feats=f_lines[idx],fn=fn) # merge the grid of positive document 
            if len(e_trans) !=0:
                grid_1 = grid_1 + e_trans + " " + "0 "* window_size
        #print(grid_1)
                
        p_count = 0
        for i in range(1,perm_num+1): # reading the permuted docs
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file+ ".EGrid" +"-"+str(i))]    
            grid_0 = "0 "* window_size

            for idx, p_line in enumerate(permuted_lines):
                e_trans_0 = get_eTrans_with_Feats(sent=p_line, feats=f_lines[idx],fn=fn)
                if len(e_trans_0) !=0:
                    grid_0 = grid_0 + e_trans_0  + " " + "0 "* window_size

            if grid_0 != grid_1: #check the duplication
                p_count = p_count + 1
                sentences_0.append(grid_0)
            #else:
            #    print(file+ ".EGrid" +"-"+str(i)) // print duplicates permuted docs with original
        
        for i in range (0, p_count): #stupid code
            sentences_1.append(grid_1)


    assert len(sentences_0) == len(sentences_1)

    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=window_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=window_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)

    if E is None:
        E      = 0.01 * np.random.uniform( -1.0, 1.0, (len(vocab_list), emb_size))
        E[len(vocab_list)-1] = 0

    return X_1, X_0, E 

#===================================================
#loading data for summary experiment task 
def load_summary_data(filelist="list_of_paris", perm_num = 20, maxlen=15000, window_size=3, E=None, vocab_list=None, emb_size=300, fn=None):
    # loading entiry-grid data from list of pos document and list of neg document
    if vocab_list is None:
        print("Please input vocab list")
        return None

    list_of_pairs = [line.rstrip('\n') for line in open(filelist)]
    
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []
    
    maxLEN_ = 0

    for pair in list_of_pairs:
        #print(pair) 
        #loading pos document
        lines = [line.rstrip('\n') for line in open("./final_data/summary/" +  pair.split()[0] + ".parsed.ner.EGrid")]
        f_lines = [line.rstrip('\n') for line in open("./final_data/summary/" +  pair.split()[0] + ".parsed.ner.Feats")]
        grid_1 = "0 "* window_size
        for idx, line in enumerate(lines):
            e_trans = get_eTrans_with_Feats(sent=line,feats=f_lines[idx],fn=fn) # merge the grid of positive document 
            if len(e_trans) !=0:
                grid_1 = grid_1 + e_trans + " " + "0 "* window_size
        

        #loading neg document
        neg_lines = [line.rstrip('\n') for line in open("./final_data/summary/" + pair.split()[1] + ".parsed.ner.EGrid")]
        neg_f_lines = [line.rstrip('\n') for line in open("./final_data/summary/"  +pair.split()[1] + ".parsed.ner.Feats")]
        grid_0 = "0 "* window_size
        for idx, p_line in enumerate(neg_lines):
                #print(p_line)
                e_trans_0 = get_eTrans_with_Feats(sent=p_line, feats=neg_f_lines[idx],fn=fn)
                if len(e_trans_0) !=0:
                    grid_0 = grid_0 + e_trans_0  + " " + "0 "* window_size

        #find max length for the copus
        if len(grid_1) > maxLEN_:
            maxLEN_ = len(grid_1)

        if len(grid_0) > maxLEN_:
            maxLEN_ = len(grid_0)


        sentences_1.append(grid_1)        
        sentences_0.append(grid_0)
        

    assert len(sentences_0) == len(sentences_1)

    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

    # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, window_size=window_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, window_size=window_size)

    X_1 = sequence.pad_sequences(X_1, maxlen)
    X_0 = sequence.pad_sequences(X_0, maxlen)

    if E is None:
        E      = 0.01 * np.random.uniform( -1.0, 1.0, (len(vocab_list), emb_size))
        E[len(vocab_list)-1] = 0

    return X_1, X_0, E, maxLEN_
#===================================================


#loading the grid with normal CNN
def load_and_numberize_Egrid(filelist="list_of_grid.txt", perm_num = 3, maxlen=None, window_size=3, ignore=0):
    # loading entiry-grid data from list of pos document and list of neg document
    list_of_files = [line.rstrip('\n') for line in open(filelist)]
    
    # process postive gird, convert each file to be a sentence
    sentences_1 = []
    sentences_0 = []

    e_count_list = []

    for file in list_of_files:
        #print(file)
        lines = [line.rstrip('\n') for line in open(file)]

        grid_1 = "0 "* window_size
        e_count = 0
        for line in lines:
            # merge the grid of positive document 
            e_trans = get_eTrans(sent=line)
            if len(e_trans) !=0:
                grid_1 = grid_1 + e_trans + " " + "0 "* window_size
                e_count = e_count + 1
        	
        p_count = 0
        for i in range(1,perm_num+1): # reading the permuted docs
            permuted_lines = [p_line.rstrip('\n') for p_line in open(file+"-"+str(i))]    
            grid_0 = "0 "* window_size
            for p_line in permuted_lines:
                e_trans_0 = get_eTrans(sent=p_line)
                if len(e_trans_0) !=0:
                    grid_0 = grid_0 + e_trans_0  + " " + "0 "* window_size

            if grid_0 != grid_1:
                p_count = p_count + 1
                sentences_0.append(grid_0)
        
        for i in range (0, p_count): #stupid code
            sentences_1.append(grid_1)
            
       #update new number of entity
        e_count_list.append(e_count)

    assert len(sentences_0) == len(sentences_1)
#    with open('e_count_list_after.txt','w') as f:
#        f.write('\n'.join([str(n) for n in e_count_list])+'\n')

    # numberize_data
    vocab_list = ['0','S','O','X','-']
    vocab_idmap = {}
    for i in range(len(vocab_list)):
        vocab_idmap[vocab_list[i]] = i

     # Numberize the sentences
    X_1 = numberize_sentences(sentences_1, vocab_idmap)
    X_0  = numberize_sentences(sentences_0,  vocab_idmap)
    
    X_1 = adjust_index(X_1, maxlen=maxlen, ignore=ignore, window_size=window_size)
    X_0  = adjust_index(X_0,  maxlen=maxlen, ignore=ignore, window_size=window_size)

    return X_1, X_0


def load_embeddings(emb_size=300):
    # maybe we have to load a fixed embeddeings for each S,O,X,- the representation of 0 is zeros vector
    np.random.seed(2016)
    E      = 0.01 * np.random.uniform( -1.0, 1.0, (5, emb_size))
    E[0] = 0
    return E   
 

def numberize_sentences(sentences, vocab_idmap):  

    sentences_id=[]  

    for sid, sent in enumerate (sentences):
        tmp_list = []
        #print(sid)
        for wrd in sent.split():
            if wrd in vocab_idmap:
                wrd_id = vocab_idmap[wrd]  
            else:
                wrd_id = 0
            tmp_list.append(wrd_id)

        sentences_id.append(tmp_list)

    return sentences_id  

def adjust_index(X, maxlen=None, window_size=3):

    if maxlen: # exclude tweets that are larger than maxlen
        new_X = []
        for x in X:

            if len(x) > maxlen:
                #print("************* Maxlen of whole dataset: " + str(len(x)) )
            	tmp = x[0:maxlen]
            	tmp[maxlen-window_size:maxlen] = ['0'] * window_size
            	new_X.append(tmp)
            else:
                new_X.append(x)

        X = new_X

    return X





