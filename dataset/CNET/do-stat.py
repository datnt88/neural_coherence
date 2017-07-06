from __future__ import absolute_import
import glob, os, csv, re
import sys
#import nltk, string
#from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


list_of_files = [line.rstrip('\n') for line in open("list.threads")]

for file in list_of_files:
	posts = [line.rstrip('\n') for line in open(file)]

	n = len(posts)
	p_ids = []
	parent_ids = []

	if n>2 and n < 16:
		print file, n

	for post in posts:
		p_ids.append(post.split('\t')[3])
		parent_ids.append(post.split('\t')[4])
			#t_stamp = post.split('\t')[4]
			#print p_id
	#print p_ids
	#print parent_ids

	for i in range(6,n+1):

		parent = parent_ids[i-1]

		idx = p_ids.index(parent) 

		#print "post: ", i, "reply to: ", idx +1
	


