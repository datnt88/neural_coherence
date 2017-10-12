from __future__ import division
import sys
import difflib
from sklearn.metrics import f1_score
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer


def insert(x=[],last=10,new_node=11):
	#print last
	
	res = []
	if last ==1:
		res =x
		res.append([1,new_node])
		return res
	else:
		tmp1 = []
		tmp2 = []

		for i in x:
			if last in i:
				idx = i.index(last) + 1
				#print i[0:idx]
				tmp2 = i[0:idx] + [new_node]
				tmp1 = i
				break
		
		if len(tmp2) > len(tmp1):
			res = [k for k in x if k!=tmp1]
			res.append(tmp2)
		else:
			res = [k for k in x]
			res.append(tmp2)

	return res


#get tree given string
def toString(branches=[]):
	tmp = []
	for i in branches:
		tmp.append('.'.join([str(x) for x in i]))
	return "-".join(tmp)

#get edge given trees
def get_edges(branches=[]):
	res = []
	for i in branches:
		branch = ''.join([str(x) for x in i])
		n = len(branch)
		for i in range(0,n-1):
			res.append(branch[i:i+2])
	#return res
	return sorted(list(set(res)))

#=============================================================
#generating baseline1: temporal order, ALL-Previous
def getBaseline1(n=3):

	bl1 = [range(1,n+1)]
	bl1_tree = toString(branches=bl1)
	bl1_edge = get_edges(branches=bl1)
	#print bl1_edge
	#print bl1_tree

	return bl1_tree , bl1_edge

#generating baseline2; every comment reply to the first comment
def getBaseline2(n=3):
	bl2 = []
	for i in range(1,n):
		bl2.append([1,i+1])

	bl2_tree = toString(branches=bl2)
	bl2_edge = get_edges(branches=bl2)
	
	return bl2_tree, bl2_edge


#for computing baseline 3
stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


#get baseline 3
def getBaseline3(filename="",n=3):

	def compute_diff(s1="",s2=""):
		seq=difflib.SequenceMatcher(None, s1,s2)
		return seq.ratio()

	#text_cmms = []
	#for i in range(1,n+1):
	#	lines = [line.rstrip('\n') for line in open("exp.fragments/" + filename + ".txt.text.frag_" + str(i) + ".sent")]
	#	text_cmms.append(' '.join(lines))

	#get Text
	cmtIDs  = [line.rstrip('\n') for line in open(file + ".commentIDs")]
	cmtIDs = [int(i) for i in cmtIDs]
	texts = [line.rstrip('\n') for line in open(file + ".TEXT")]
	assert len(cmtIDs) == len(texts)

	nPOST = max(cmtIDs)
	text_cmms = []
	for pID in range(1,nPOST +1):
		idx = [ i for i, item in enumerate(cmtIDs) if item==pID ]
		post = [ j for i,j in enumerate(texts) if i in idx ]
		text_cmms.append(' '.join(post))



	branches = [[1,2]]
	veryLast = [1,2]


	for i in range(3,n+1):
	 	current_cmm = text_cmms[i-1]
	 	#compute the distance with those last
	 	maxD = -1
	 	parent = 0
	 	#print veryLast
	 	for last in veryLast:
	 		diff = cosine_sim(current_cmm,text_cmms[last-1])
	 		#print diff
	 		if diff > maxD:
	 			maxD = diff
	 			parent = last

	 	veryLast.append(i)
	 	#inserther the parent 
	 	branches = insert(x=branches,last=parent,new_node=i)



	bl3_tree = toString(branches=branches)
	bl3_edge = get_edges(branches=branches)
	#print bl1_edge
	#print bl1_tree

	return bl3_tree , bl3_edge

	

def getOriginalInfo(tree=""):
	#print s
	
	branches = []
	nPost = 0
	for branch in tree:
		tmp = [int(x) for x in branch.split('.')]
		branches += [tmp]
		if max(tmp) > nPost:
			nPost = max(tmp)

	return nPost, get_edges(branches=branches)

def compute_edge_acc(edge1=[],edge2=[]):
	n = len(edge1)
	k = 0
	for i in edge2:
		if i in edge1:
			k+=1
	return k/n

#=============================================================
def remove_post(tree=[]):

	x = []

	for branch in tree:
		branch = [int(i) for i in branch.split('.')]

		branch = [i for i in branch if i <=4]

		branch = ''.join([str(i) for i in branch])

		if len(branch) > 1:
			x.append(branch)

	x	= list(set(x))
	x = remove_duplication(tree=x)

	x = sorted(x)
	x = ['.'.join(i) for i in x]

	return x

def remove_duplication(tree=[]):

	tree = sorted([int(i) for i in tree])

	tree = [str(i) for i in tree]

	n = len(tree)

	res = []
	for i in range(0,n):
		#print "--------"
		current_bracnh = tree[i]
		tmp = '-'.join(tree[i+1:n])
		#print tmp
		if current_bracnh not in tmp:
			res.append(current_bracnh)


	return res


count_bl1 = 0
count_bl2 = 0
count_bl3 = 0

acc_bl1 = 0.0
acc_bl2 = 0.0
acc_bl3 = 0.0

f1_bl1 = 0.0
f1_bl2 = 0.0
f1_bl3 = 0.0


fileList = [line.rstrip('\n') for line in open(sys.argv[1])]
#fileList = ['threads/Thread_9291.txt.text.all.parsed.ner']

for file in fileList:

	org_tree = [line.rstrip('\n') for line in open(file+".orgTree")] 
	#print org_tree 

	n = len(org_tree)
	'''
	if n==1:
		print file + "\t: ALL-previous" + " num branches: ", n, org_tree
	else:
		#find max
		maxPost = 0
		for branch in org_tree:
			nPost = max([int(i) for i in branch.split(".")])
			if nPost > maxPost:
				maxPost= nPost
		
		if maxPost == n+1:
			print file + "\t: ALL-first" + " num branches: ", n, org_tree
		else:
			
			#remove those post from 7
			#x = remove_post(tree=org_tree)

			#k = "\n".join(x)

			#with open(file + ".tree.cutOFF4",'w') as f_:
			#	f_.write(k)

			print file + "\t: X" + " num branches: ", n, org_tree

	'''
	#print for some baseline	
	n, org_edge = getOriginalInfo(tree=org_tree)

	bl1 , bl1_edge = getBaseline1(n=n)
	bl2 , bl2_edge = getBaseline2(n=n)
	bl3 , bl3_edge = getBaseline3(filename=file,n=n)

	print "-----------------------"
	print org_edge
	print bl3_edge
	
	org_tree = '-'.join(org_tree)
	#print bl2

	if org_tree == bl1:
		count_bl1 +=1

	if org_tree == bl2:
		count_bl2 +=1

	if org_tree == bl3:
		count_bl3 +=1


	f1_bl1 += f1_score(org_edge,bl1_edge, average='micro')  
	f1_bl2 += f1_score(org_edge,bl2_edge, average='micro')  
	f1_bl3 += f1_score(org_edge,bl3_edge, average='micro')  

	acc_bl1 += compute_edge_acc(org_edge,bl1_edge)
	acc_bl2 += compute_edge_acc(org_edge,bl2_edge)
	acc_bl3 += compute_edge_acc(org_edge,bl3_edge)


k = len(fileList)
print(k)
print "------------------"
print "Tree level: " , count_bl1/k*100, count_bl2/k*100, count_bl3/k*100
print "Edge level: " , acc_bl1/k*100, acc_bl2/k*100, acc_bl3/k*100
print "F Edge level: " , f1_bl1/k, f1_bl2/k, f1_bl3/k









	


















