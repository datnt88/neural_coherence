import os


def generate_depth(filename="./final_data/CNET/x_cnet.ALL"):
    print "-------------------"
    files = [line.rstrip('\n') for line in open(filename)]
    for i,file in enumerate(files):
        #if i == 2:
        #    return False  
        
        print file
        cmtIDs  = [line.rstrip('\n') for line in open(file + ".commentIDs")]
        cmtIDs = [int(i) for i in cmtIDs] 
        org_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]
        
        print "Cmts:\t", cmtIDs
        print "Tree:\t", org_tree
        print "-------------------"

        depth = {}
        num_cmts = 0


        for branch in org_tree:              
            branch = [int(id) for id in branch.split('.')] # convert string to integer
            if max(branch) > num_cmts:
                num_cmts = max(branch)
            #print branch
            for j, cmtID in enumerate(branch):
                if cmtID not in depth:
                    if cmtID == 1:
                        depth[cmtID] = range(0,cmtIDs.count(cmtID))
                    else:
                        parent = branch[j-1]
                        start = max(depth[parent]) + 1
                        depth[cmtID] = range( start, start + cmtIDs.count(cmtID) )
                    

        # write to file
        
        
        depth_list = []
        for i in range(1, num_cmts+1):
            depth_list +=  depth[i]
            print "Comment:", i, " with sentence depth:", depth[i]

        with open(file + ".depth","w") as f:
            f.write("\n".join([str(d) for d in depth_list]))
        
        print depth_list

    return True

print generate_depth()






















def check_data(filename,path1,path2):
    files = [line.rstrip('\n') for line in open(filename)]
    print "check in: ", path1, path2

    for file in files:
        #print file
        #chekc check files
        basma_f = path1 + file

        lines1 = [line.rstrip('\n') for line in open(basma_f)]
        my_f = path2 + file
        lines2 = [line.rstrip('\n') for line in open(my_f)]


        print basma_f
        print my_f

        for i in range(0,len(lines1)):
            if lines1[i] != lines2[i]:
                return False

    return True


#print check_data("./final_data/testing/list.train","./final_data/CNET/Basma_final_train/","./final_data/CNET/threads/")
#print check_data("./final_data/testing/list.train.branch","./final_data/CNET/Basma_final_train/","./final_data/testing/train-branch/")
#print check_data("./final_data/testing/list.test.branch","./final_data/CNET/Basma_final_test_branch_level/","./final_data/testing/test-branch/")


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

def get_branch_NER(e_trans="", idxs=""):
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
    

def gen_permutation():
    list_of_files = [line.rstrip('\n') for line in open("final_data/CNET/x_cnet.test")]
    for file_id, file in enumerate(list_of_files):
            print "------------------------------------"
            
            filename = file.split("/")[-1]
            print(filename) 
            #loading commentIDs
            cmtIDs  = [line.rstrip('\n') for line in open(file + ".commentIDs")]
            cmtIDs = [int(i) for i in cmtIDs] 
            org_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]
            ners = [line.rstrip('\n') for line in open(file)]

            for i in range(1,21):
                #print each permutation file
                p_ners = [line.rstrip('\n') for line in open( file + ".perm-" + str(i))]
                
                sort_ners = sorted(ners)
                sort_p_ners = sorted(p_ners)

                assert sort_ners == sort_p_ners # check perm is the same sentence with orginal tree

                branchID = 1
            
                for branch in org_tree:              
                    branch = [int(id) for id in branch.split('.')] # convert string to integer 
                    idxs = [ idx for idx, cmtID  in enumerate(cmtIDs) if cmtID in branch]

                    branch_ner = []
                    for idx in idxs:
                        branch_ner.append(p_ners[idx])
                    branch_ner = "\n".join(branch_ner)

                    with open("./final_data/testing/test-branch/" + filename + ".perm-" + str(i) + ".branch-" + str(branchID), "w") as f: 
                        f.write(branch_ner) 

                    branchID +=1



def gen_orignal():
    list_of_files = [line.rstrip('\n') for line in open("final_data/CNET/x_cnet.test")]
    for file_id, file in enumerate(list_of_files):
            print "------------------------------------"
            
            filename = file.split("/")[-1]
            print(filename) 
            #loading commentIDs
            cmtIDs  = [line.rstrip('\n') for line in open(file + ".commentIDs")]
            cmtIDs = [int(i) for i in cmtIDs] 

            org_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]
            #egrids = [line.rstrip('\n') for line in open(file + ".EGrid")]
            ners = [line.rstrip('\n') for line in open(file)]

            branchID = 1
            
            for branch in org_tree:              
                branch = [int(id) for id in branch.split('.')] # convert string to integer 
                idxs = [ idx for idx, cmtID  in enumerate(cmtIDs) if cmtID in branch]


                branch_ner = []
                for idx in idxs:
                    branch_ner.append(ners[idx])
                branch_ner = "\n".join(branch_ner)

                #branch_Egrid = []
                #for idx, line in enumerate(egrids): #print each branch EGRID
                #    branch_Egrid.append(get_branch_EGrid(e_trans=line, idxs=idxs)) # merge the grid of positive document 
                #branch_Egrid = "\n".join(branch_Egrid)
                #print branch_Egrid
                with open("./final_data/testing/test-branch/" + filename + ".branch-" + str(branchID), "w") as f: 
                    f.write(branch_ner) 

                branchID +=1

