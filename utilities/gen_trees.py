from __future__ import absolute_import
import glob, os, csv, re
import sys

def insert_node(x=[],node="100"):
	tree_4 = []

	#addin new branch
	for item in x:
		tmp = [k for k in item] 
		tmp.append("1" + node)
		tree_4.append(tmp)
	
	for item in x:
		for i in item:
			#keep old_branches
			old_branches = [branch for branch in item if branch != i]
			#compute new sub trees
			new_subTrees = adding(x=i, new_node=node)
			
			for tree in new_subTrees:
				tmp = [branch for branch in tree if branch not in old_branches]
				#print tmp
				# each new_braches, adding to the old branches 
				tree_4.append(old_branches + tmp )

	return tree_4


def adding(x="123",new_node="4"):
	res = []
	res.append([x + new_node])
	n = len(x)

	tmp = []
	for i in range(1,n):
		tmp = []
		tmp.append(x)
		tmp.append(x[0:i] + new_node)

		res.append(tmp)


	return res

#res = adding(x="123",new_node="4")
#for i in res:
#	print i
#print "-----------------\n"


def gen_Tree3():
	return [["123"],["13","12"]]

def gen_Tree4():
	x3 = gen_Tree3() 
	tree_4 = insert_node(x=x3,node="4")
	return tree_4

def gen_Tree5():

	x4 = gen_Tree4() 
	tree_5 = insert_node(x=x4,node="5")

	#print ['1234','1235'] in tree_5
	return tree_5

def gen_Tree6():
	x = gen_Tree5() 
	tree_6 = insert_node(x=x,node="6")
	
	#print ['1246', '1245','1234'] in tree_6

	return tree_6

def gen_tree_branches(n=3):
	if n == 3:
		return [["123"],["12","13"]]
	if n == 4:
		return remove_duplication(x=gen_Tree4())
	if n == 5:
		return remove_duplication(x=gen_Tree5())
	if n == 6:
		return remove_duplication(x=gen_Tree6())
	if n >6:
		print "Have not supported larger threads (num of commnet > 6) yet...!"
		return None

def remove_duplication(x=[]):
	sort_x = []
	for i in x:
		sort_x.append('.'.join(sorted(i)))
	
	uniq_x = list(set(sort_x))

	res = []
	for i in uniq_x:
		res.append(i.split("."))

	return res



x = gen_tree_branches(int(sys.argv[1]))
for i in x:
	print i
#print "-------------------------------"
print len(x)

print "-------------------------------"




	


