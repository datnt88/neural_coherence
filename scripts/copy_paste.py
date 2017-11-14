import os

f = open("final_data/CNET/x_cnet.train", 'r')

for line in f:
    line = line.strip()
    print line
    cmd = " cp " + line  + " final_data/CNET/Basma_final_train_temporal_order" 
    os.system(cmd)