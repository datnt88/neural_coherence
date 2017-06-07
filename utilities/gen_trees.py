from __future__ import absolute_import
import glob, os, csv, re


def insert_node(x=[],node="100"):
	tree_4 = []
	for item in x:
		tmp = [k for k in item] 
		tmp.append("1" + node)
		tree_4.append(tmp)
		for i in item:
			#adding 1-4
			new_branch = i + node
			#print "new branch: " + new_branch
			old_branches = [branch for branch in item if branch != i]
			#print "old branch: " + str(old_branches)
			old_branches.append(new_branch)
			tree_4.append(old_branches)
	return tree_4


def swap(x=[],n1="3",n2="4"):
	res = []
	for tree in x:
		new_tree = []
		check = 0	
		for branch in tree:
			#find if there is n1, n2 in the branch
			if find(branch,n1,n2) == True:
				#swap then add
				tmp = branch.replace(n1,"x")
				tmp = tmp.replace(n2,n1)
				tmp = tmp.replace("x",n2)
				new_tree.append(tmp)
				#print tmp
				check = 1
			else:
				new_tree.append(branch)
		if check == 1:
			res.append(tree)
		res.append(new_tree)
	return res



def find(s="",n1="3", n2="4"):
	if s.find(n1) <0:
		return False
	if s.find(n2) < 0:
		return False
	return True
	

def gen_Tree3():
	return [["123"],["12","13"]]

def gen_Tree4():
	x = gen_Tree3() 
	tree_4 = insert_node(x=x,node="4")
	#tree_4 = swap(x=tree_4,n1="3",n2="4")

	return tree_4

def gen_Tree5():
	x = gen_Tree4() 

	tree_5 = insert_node(x=x,node="5")
	#tree_5 = swap(x=tree_5,n1="3",n2="5")
	#tree_5 = swap(x=tree_5,n1="4",n2="5")

	return tree_5

def gen_Tree6():
	x = gen_Tree5() 

	tree_6 = insert_node(x=x,node="6")
	#tree_6 = swap(x=tree_6,n1="3",n2="6")
	#tree_6 = swap(x=tree_6,n1="4",n2="6")
	#tree_6 = swap(x=tree_6,n1="5",n2="6")

	return tree_6

def gen_tree_branches(n=3):
	if n == 3:
		return [["123"],["12","13"]]
	if n == 4:
		return gen_Tree4()
	if n == 5:
		return gen_Tree5()
	if n == 6:
		return gen_Tree6()
	if n >6:
		print "Have not supported larger threads (num of commnet > 6) yet...!"
		return None

#x = gen_tree_branches(n=6)

#for i in x:
#	print i

	


