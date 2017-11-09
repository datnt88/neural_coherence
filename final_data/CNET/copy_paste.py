import os

f = open("/home/basma/neural_coherence/final_data/CNET/x_cnet.test", 'r')

for line in f:
    line = line.strip()
    #line = line.split(".")
    #line = ".".join(line[:2]) + ".parsed.ner"
    print line
    cmd = " cp /home/basma/neural_coherence/" + line + " /home/basma/neural_coherence/final_data/CNET/test" 
    os.system(cmd)
