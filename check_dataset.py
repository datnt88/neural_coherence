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



gen_permutation()

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

