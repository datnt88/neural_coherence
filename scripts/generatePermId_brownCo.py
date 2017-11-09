import os

def getIdBySentence(file_name, sentence):
    reader = open(file_name,'r')
    i = 0
    for line in reader:
        if line == sentence:
            return i
        i += 1
    return -1



listing = os.listdir('../final_data/CNET/test_temporal_order/')
    
k = 0
listing = filter(lambda y: y[-4:] == '.ner', listing)

for l in listing:
    print l
    for perm_id in range(20):
        permID_writer = open('../final_data/CNET/test_temporal_order/' + l + ".permId-"+ str(perm_id+1), "w")
        perm_file = '../final_data/CNET/test_temporal_order/' + l + ".perm-"+ str(perm_id+1)
        perm_reader = open(perm_file, 'r')
        
        for line in perm_reader:
            permID_writer.write(str(getIdBySentence('../final_data/CNET/test_temporal_order/' + l, line)) + "\n")
        