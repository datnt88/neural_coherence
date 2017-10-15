import os

f = open("/home/basma/neural_coherence/final_data/CNET/CL_journal.all", 'r')

for line in f:
    line = line.strip()
    #line = line.split(".")
    #line = ".".join(line[:2]) + ".parsed.ner"
    print line
    cmd = " cp /home/basma/neural_coherence/final_data/CNET/threads/" + line + " /home/basma/neural_coherence/final_data/CNET/subthreads" 
    os.system(cmd)
