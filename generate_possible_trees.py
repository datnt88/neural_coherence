from utilities import email_helper
from utilities import my_callbacks
from utilities import gen_trees

def main():
    
    #parameter for data_helper
    p_num = 20
    w_size = 5
    maxlen=14000
    emb_size = 50
    fn = [0,3,4]    #fn = range(0,10) #using feature

    vocab = email_helper.init_vocab()
    
    #listing = os.listdir('../cnn_coherence/dataset/CNET/test/')
    #listing = filter(lambda y: y[-6:] == 'ed.ner', listing)
    
    list_of_files = [line.rstrip('\n') for line in open("./dataset/CNET/less_posts.test")]
    count = 0
    
    
    for file in list_of_files:
        
        
        
        #process each test
        file_name = "./dataset/CNET/threads/" + file  + ".txt.text.all.parsed.ner"
        print file_name
        X_org = email_helper.load_original_tree(file=file_name, maxlen=maxlen, window_size=w_size, vocab_list=vocab, emb_size=emb_size, fn=fn)

        #processing each possible tree candidate
        cmtIDs  = [line.rstrip('\n') for line in open(file_name + ".commentIDs")]
        cmtIDs = [int(i) for i in cmtIDs] 
        p_trees =  gen_trees.gen_tree_branches(max(cmtIDs))
        
        max_score = 0.0
        
        k = 1
        
        for p_tree in p_trees:

            x_tree = [] # tree with sentence ID
            #print cmtIDs
            for branch in p_tree:
                
                #then compare to the original one
                #print list(branch)
                sentIDs = []
                for cmtID in list(branch):   
                    sentIDs += [ i for i, id_ in enumerate(cmtIDs) if id_ == int(cmtID)]
                x_tree.append(sentIDs)
            
            tree_file_name = "./dataset/CNET/test_trees/" + file  + ".txt.text.all.parsed.ner.tree-"
            
            
            f = open(tree_file_name + str(k), "w")
            
            for branch in x_tree:
                
                line = ""
                
                for id in branch:
                    
                    line += str(id + 1) + ", "
            
                f.write(line[:-2] + "\n")
            
            k += 1
            f.close()

if __name__=='__main__':
    main()