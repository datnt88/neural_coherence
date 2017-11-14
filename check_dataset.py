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

    


list_of_files = [line.rstrip('\n') for line in open("final_data/CNET/x_cnet.test")]


for file_id, file in enumerate(list_of_files):
        print "------------------------------------"
        
        filename = file.split("/")[-1]
        print(filename) 
        #loading commentIDs
        cmtIDs  = [line.rstrip('\n') for line in open(file + ".commentIDs")]
        cmtIDs = [int(i) for i in cmtIDs] 

        org_tree = [line.rstrip('\n') for line in open(file + ".orgTree")]
        egrids = [line.rstrip('\n') for line in open(file + ".EGrid")]

        branchID = 1
        for branch in org_tree:   #reading branch, each branch is considered as a document
            

            branch = [int(id) for id in branch.split('.')] # convert string to integer, branch
            idxs = [ idx for idx, cmtID  in enumerate(cmtIDs) if cmtID in branch]

            branch_Egrid = []

            for idx, line in enumerate(egrids): #print each branch EGRID

                branch_Egrid.append(get_branch_EGrid(e_trans=line, idxs=idxs)) # merge the grid of positive document 
            branch_Egrid = "\n".join(branch_Egrid)

            #print branch_Egrid
            with open("./final_data/testing/genEGridBranches/" + filename + ".EGrid.branch-" + str(branchID), "w") as f: 
                f.write(branch_Egrid) 

            branchID +=1

