import numpy as np
import math
import sys
import json
import csv
import time
import jieba # TODO
from argparse import ArgumentParser
from operator import itemgetter
import xml.etree.cElementTree as ET
#http://www.cnblogs.com/ifantastic/archive/2013/04/12/3017110.html

# jieba.set_dictionary('dict.txt.big')

#Okipi/BM25 Parmeters
BM25_K = 2
SLOPE = 0.6

#Rocchio Relevance Feedback parameters
ALPHA = 0.90
BETA = 0.20
GAMMA = 0.10
RELATED_RATIO = 0.2

global OUTPUTFILE
OUTPUTFILE = 'S_' + str(SLOPE) + 'A_' + str(ALPHA) + 'B_' + str(BETA) + 'C_' + str(GAMMA)+'_Jieba'

def Read_FileList(Model_Dir):
	File_List = []
	Avg_Doc_Len = 0
	print('Reading File List ...')
	with open(str(Model_Dir)+'/file-list') as f :
		Doc_Path = [line.rstrip('\n') for line in f]

		# print(Doc_Path)
		for i, path in enumerate(Doc_Path) :
			# print('File list No. '+ str(i+1))
			Doc_Info = {}
			tree = ET.parse(path)
			root = tree.getroot()
			# print(root)
			Doc_Info['id'] = root.find('./doc/id').text.lower()
			# print(Doc_Info['id'])
			Doc_Info['date'] = root.find('./doc/date').text.lower()
			try:
				root.find('./doc/title').text.strip()
			except :
				# print('No title')
				Doc_Info['title'] = ''
			else:
				Doc_Info['title'] = root.find('./doc/title').text.strip()

			textInDoc = ""
			for p in root.iter('p'):
				textInDoc += p.text.strip()
			Doc_Info['content'] = textInDoc
			File_List.append(Doc_Info)

	print('Calculating Average Doc Length ...')
	for doc in File_List:
		Avg_Doc_Len += len(doc['content'])
	Avg_Doc_Len = float(Avg_Doc_Len/len(File_List))
	print('Average Doc Length: ' + str(Avg_Doc_Len))

	return File_List, Avg_Doc_Len

def Read_VocabList(Model_Dir):
	Vocab_List = []
	print('Reading Vocab List ...')
	with open(str(Model_Dir)+'/vocab.all') as f :
		Vocab = [line.rstrip('\n') for line in f]
		for i, word in enumerate(Vocab):
			# print('Vocab List No. ' + str(i+1))
			Vocab_List.append(word)

	return Vocab_List

def Read_InvertedList(Model_Dir):

	InvertedFileDict = {}
	currentIndex = ""
	t1 = time.time()
	print('Reading Inverted List ...')
	with open(str(Model_Dir)+'/inverted-file') as f :
		Invert = [line.rstrip('\n') for line in f]
		for i, info in enumerate(Invert):
			fields = info.split(" ")
			if len(fields)==3: # if this field is describing vocab unigram/bigram
				if fields[1] == "-1": # unigram
					currentIndex = fields[0]+',-1'
					for l in range(int(fields[2])):

						(docid, countindoc) = Invert[i+l+1].split(' ')

						try :
							InvertedFileDict[currentIndex]
						except:
							# not exist index
							InvertedFileDict[currentIndex]={'Doc_Freq': fields[2], 'Docs':[]}
						else:
							pass
						finally:
							InvertedFileDict[currentIndex]['Docs'].append({'docID':docid, 'countInDoc':countindoc})
				else:  # bigram
					currentIndex = fields[0]+','+fields[1]
					for l in range(int(fields[2])):

						(docid,countindoc) = Invert[i+l+1].split(' ')

						try :
							InvertedFileDict[currentIndex]
						except:
							# not exist index
							InvertedFileDict[currentIndex]={'Doc_Freq': fields[2], 'Docs':[]}
						else:
							pass
						finally:
							InvertedFileDict[currentIndex]['Docs'].append({'docID':docid, 'countInDoc':countindoc})
			else:# if this field is describing doc id and its feq
				pass
			# print(len(fields))
	# print(InvertedFileDict)
	t2 = time.time()
	print('Reading Inverted List time : ' + str(t2-t1))
	return InvertedFileDict


#https://www.cnblogs.com/wenBlog/p/8441231.html
def read_in_chunks(filePath, chunk_size=1024*1024):
	file_object = open(filePath,'r')
	while True:
		chunk_data = file_object.readlines(chunk_size)
		if not chunk_data:
			break
		yield chunk_data

def Query(query, Feedback, OutputFile):
	Text = "query_id,retrieved_docs\n"
	with open(query) as f:
		# print(query)
		tree = ET.parse(query)
		root = tree.getroot()
		# print(root)
		for topic in root.iter('topic'):
			query_id = topic.find('./number').text[-3:].strip()
			# print(query_id)
			title = topic.find('./title').text.strip()
			concepts = topic.find('./concepts').text.strip()

			puncs = [u'，', u'?', u'@', u'!', u'$', u'%', u'『', u'』', u'「', u'」', u'＼', u'｜', u'？', u' ', u'*', u'(', u')', u'~', u'.', u'[', u']', 'u\n',u'1',u'2',u'3',u'4',u'5',u'6',u'7',u'8',u'9',u'0', u'。']
			for punc in puncs:
				concepts = concepts.replace(punc,'')

			queryTermsDict = {}
			for term in concepts.split(u'、'):
				term = jieba.cut(term, cut_all=False)
				for item in term:
					if len(item) % 2 == 0: # if length of term is even, sliding windows shift distance 2
						for i in range(0,len(item),2):
							try:
								queryTermsDict[item[i]+item[i+1]]
							except:
								# first
								queryTermsDict[item[i]+item[i+1]]=1
							else:
								queryTermsDict[item[i]+item[i+1]]+=1
					else:
						if len(item)== 3: # trigram
							try:
								queryTermsDict[item[0]+item[1]+item[2]]
							except:
								queryTermsDict[item[0]+item[1]+item[2]]=1
							else:
								queryTermsDict[item[0]+item[1]+item[2]]+=1

						else: # if length of term is odd, sliding windows shift dictance 1
							for i in range(0,len(item)-1):
								try:
									queryTermsDict[item[i]+item[i+1]]
								except:
									queryTermsDict[item[i]+item[i+1]]=1
								else:
									queryTermsDict[item[i]+item[i+1]]+=1
			# print(queryTermsDict)
			# Bigram(queryTermsDict, query_id, Feedback)
			Text += Bigram(queryTermsDict, query_id, Feedback)
	f.close()
	OutputFile = OUTPUTFILE + '.csv'
	ouputData = open(OutputFile,"w")
	ouputData.write(Text)
	ouputData.close()

def Bigram(queryTermsDict, query_id, Feedback):
	t1 = time.time()
	queryVector=[]
	rankingDict = {}
	queryTermsIndex = 0
	print('len(queryTermsDict)',len(queryTermsDict))
	for key, value in queryTermsDict.items():

		print(key,value)

		if len(key)<=2:

			if key[0] in Vocab_List:

				if key[1] in Vocab_List: #bigram

					# print(key[0]+key[1]+'in Vocab_List')
					Inverted_Index = str(Vocab_List.index(key[0]))+','+str(Vocab_List.index(key[1])) # find out the bigram term index

					# print('Inverted Index:' + Inverted_Index)

					if Inverted_Index in InvertedFile_Dict:
						# Use Okapi/BM25 to normalize Document Length

						print('bigram: '+str(Inverted_Index))
						queryVector.append(value) # queryVector
						BigramInDoc = InvertedFile_Dict[Inverted_Index]
						doc_freq = int(BigramInDoc['Doc_Freq'])
						Docs = BigramInDoc['Docs']

						# Formula: IDF(w) = log(m+1/k)
						# m – total number of docs
						# k – numbers of docs with term t (doc freq)
						IDF = math.log( (len(File_List)+1) / doc_freq)

						for doc in Docs:
							docID = int(doc['docID'])
							countindoc = int(doc['countInDoc'])
							Bigram_TF = ( BM25_K + 1 ) * countindoc / ( countindoc + BM25_K * ( 1 - SLOPE + SLOPE * doc_freq / Avg_Doc_Len) )
							# print(Bigram_TF)
							if docID not in rankingDict:
								rankingDict[docID] = [0] * len(queryTermsDict)
							rankingDict[docID][queryTermsIndex] = float(Bigram_TF * IDF * value)
						queryTermsIndex += 1
				else: # unigram:key[0]
					Inverted_Index = str(Vocab_List.index(key[0]))+',-1' # find out the unigram term index

					if Inverted_Index in InvertedFile_Dict:
						print('unigram key[0]: '+str(Inverted_Index))
						queryVector.append(value) # queryVector
						UnigramInDoc = InvertedFile_Dict[Inverted_Index]
						doc_freq = int(UnigramInDoc['Doc_Freq'])
						Docs = UnigramInDoc['Docs']

						IDF = math.log( (len(File_List)+1) / doc_freq)

						for doc in Docs:
							docID = int(doc['docID'])
							countindoc = int(doc['countInDoc'])
							Bigram_TF = ( BM25_K + 1 ) * countindoc / ( countindoc + BM25_K * ( 1 - SLOPE + SLOPE * doc_freq / Avg_Doc_Len) )
							# print(Bigram_TF)
							if docID not in rankingDict:
								rankingDict[docID] = [0] * len(queryTermsDict)
							rankingDict[docID][queryTermsIndex] = float(Bigram_TF * IDF * value)
						queryTermsIndex += 1
			else: # unigram:key[1]
				if key[1] in Vocab_List:# unigram:key[1]
					Inverted_Index = str(Vocab_List.index(key[1]))+',-1' # find out the unigram term index

					if Inverted_Index in InvertedFile_Dict:
						print('unigram key[1]: '+str(Inverted_Index))
						queryVector.append(value) # queryVector
						UnigramInDoc = InvertedFile_Dict[Inverted_Index]
						doc_freq = int(UnigramInDoc['Doc_Freq'])
						Docs = UnigramInDoc['Docs']

						IDF = math.log( (len(File_List)+1) / doc_freq)

						for doc in Docs:
							docID = int(doc['docID'])
							countindoc = int(doc['countInDoc'])
							Bigram_TF = ( BM25_K + 1 ) * countindoc / ( countindoc + BM25_K * ( 1 - SLOPE + SLOPE * doc_freq / Avg_Doc_Len) )
							# print(Bigram_TF)
							if docID not in rankingDict:
								rankingDict[docID] = [0] * len(queryTermsDict)
							rankingDict[docID][queryTermsIndex] = float(Bigram_TF * IDF * value)
						queryTermsIndex += 1
		else: # len(key)>2
			if (key[0] in Vocab_List) and (key[1] in Vocab_List):
				print('first two word in Vocab_List')
				if key[2] in Vocab_List:
					Inverted_Index = str(Vocab_List.index(key[0]))+','+str(Vocab_List.index(key[1]))+','+str(Vocab_List.index(key[2]))
					if Inverted_Index in InvertedFile_Dict:
						# Use Okapi/BM25 to normalize Document Length
						print('in trigram & exist ' + Inverted_Index)
						queryVector.append(value) # queryVector
						BigramInDoc = InvertedFile_Dict[Inverted_Index]
						doc_freq = int(BigramInDoc['Doc_Freq'])
						Docs = BigramInDoc['Docs']

						# Formula: IDF(w) = log(m+1/k)
						# m – total number of docs
						# k – numbers of docs with term t (doc freq)
						IDF = math.log( (len(File_List)+1) / doc_freq)

						for doc in Docs:
							docID = int(doc['docID'])
							countindoc = int(doc['countInDoc'])
							Bigram_TF = ( BM25_K + 1 ) * countindoc / ( countindoc + BM25_K * ( 1 - SLOPE + SLOPE * doc_freq / Avg_Doc_Len) )
							# print(Bigram_TF)
							if docID not in rankingDict:
								rankingDict[docID] = [0] * len(queryTermsDict)
							rankingDict[docID][queryTermsIndex] = float(Bigram_TF * IDF * value)
						queryTermsIndex += 1
				if key[2] not in Vocab_List:
					Inverted_Index = str(Vocab_List.index(key[0]))+','+str(Vocab_List.index(key[1]))+',-1' # find out the bigram term index

					# print('Inverted Index:' + Inverted_Index)

					if Inverted_Index in InvertedFile_Dict:
						# Use Okapi/BM25 to normalize Document Length
						print('trigram first two word in InvertedFile_Dict '+str(Inverted_Index))
						queryVector.append(value) # queryVector
						BigramInDoc = InvertedFile_Dict[Inverted_Index]
						doc_freq = int(BigramInDoc['Doc_Freq'])
						Docs = BigramInDoc['Docs']

						# Formula: IDF(w) = log(m+1/k)
						# m – total number of docs
						# k – numbers of docs with term t (doc freq)
						IDF = math.log( (len(File_List)+1) / doc_freq)

						for doc in Docs:
							docID = int(doc['docID'])
							countindoc = int(doc['countInDoc'])
							Bigram_TF = ( BM25_K + 1 ) * countindoc / ( countindoc + BM25_K * ( 1 - SLOPE + SLOPE * doc_freq / Avg_Doc_Len) )
							# print(Bigram_TF)
							if docID not in rankingDict:
								rankingDict[docID] = [0] * len(queryTermsDict)
							rankingDict[docID][queryTermsIndex] = float(Bigram_TF * IDF * value)
						queryTermsIndex += 1

	if Feedback:
		queryVector = Rocchio_Relevance_Feedback(queryVector, rankingDict)

	bestResultList = Score_N_Sort(queryVector, rankingDict)
	t2 = time.time()
	print('IN Bigram time: '+str(t2-t1))
	Text = query_id + ","
	for docid in bestResultList:
		Text += File_List[int(docid)]['id']
		Text += ' '
	Text += '\n'
	return Text


# https://blog.csdn.net/baimafujinji/article/details/50930260
def Rocchio_Relevance_Feedback(queryVector, rankingDict):
	print('Rocchio Relevance Feedback...')
	rankListLength = len(rankingDict)
	RelevanceCount = int(rankListLength * RELATED_RATIO) # assume RELATED_RATIO% docs is related
	NonRelevanceCount = int(rankListLength * (1 - RELATED_RATIO))

	Score_Dict = {}

	for docid in rankingDict:
		Score_Dict[docid] = sum(rankingDict[docid])
	Score_Dict = list(sorted(Score_Dict.items(), key = itemgetter(1), reverse = True))

	# Counting Related Docs
	RelatedSum = np.array([0] * len(queryVector))
	for docid in Score_Dict[ 0:RelevanceCount ]:
		RelatedSum += np.array(rankingDict[docid[0]])

	# Counting Non-related Docs
	NonRealtedSum = np.array([0] * len(queryVector))
	for docid in Score_Dict[ -RelevanceCount: ]:
		NonRealtedSum += np.array(rankingDict[docid[0]])

	Origial = np.array(queryVector)
	RelatedSum = np.array(RelatedSum)
	NonRealtedSum = np.array(NonRealtedSum)

	NewqueryVector = (ALPHA * Origial) + (BETA*RelatedSum/RelevanceCount) - (GAMMA*NonRealtedSum/NonRelevanceCount)

	return NewqueryVector

def Score_N_Sort(queryVector, rankingDict):
	Score_Dict = {}
	index = 0
	print('len(rankingDict)',len(rankingDict))
	for docid, vector in rankingDict.items():

		Score_Dict[docid]=np.inner(np.asarray(queryVector), np.asarray(rankingDict[docid]))
		

	# sort score from large to small
	sortedlistOfDoc = list(sorted(Score_Dict.items(), key = itemgetter(1), reverse = True))

	# with open('sortedlistOfDoc.csv',"w") as f:
	# 	writer = csv.writer(f)
	# 	writer.writerows(sortedlistOfDoc)
	# f.close()

	returnList = []
	for index in range(len(sortedlistOfDoc)):
		returnList.append(sortedlistOfDoc[index][0]);
		if index == 99:
			break
	return returnList

def Cut_By_Jieba():

	print('Cut By Jieba ...')

	Doc_freq={}
	# File_List_Index = build_dict(File_List, key = 'id')
	File_List_Index = 0
	for Doc_Info in File_List:
		textInDoc = Doc_Info['content']
		docid = Doc_Info['id']

		currentIndex = ''

		textInDoc = list(jieba.cut(textInDoc))

		puncs = [u'，', u'?', u'@', u'!', u'$', u'%', u'『', u'』', u'「', u'」', u'＼', u'｜', u'？', u' ', u'*', u'(', u')', u'~', u'.', u'[', u']', 'u\n',u'1',u'2',u'3',u'4',u'5',u'6',u'7',u'8',u'9',u'0', u'。']

		for item in textInDoc:
			if item in puncs:
				textInDoc.remove(item)

		item_freq = {}
		for item in textInDoc:
			if len(item) == 3: # for trigram
				if item in item_freq:
					item_freq[item]+=1
				else:
					item_freq[item] = 1

					if item in Doc_freq:
						Doc_freq[item]+=1
					else:
						Doc_freq[item]=1

				index_of_item0 = str(Vocab_List.index(item[0]))
				index_of_item1 = str(Vocab_List.index(item[1]))
				index_of_item2 = str(Vocab_List.index(item[2]))

				if (item[0] in Vocab_List) and (item[1] in Vocab_List):
					if item[2] in Vocab_List:
						currentIndex = index_of_item0+','+index_of_item1+','+index_of_item2
					else:
						currentIndex = index_of_item0+','+index_of_item1+',-1'
				# print('in jieba:'+currentIndex)
				try:
					InvertedFile_Dict[currentIndex]
				except:
					InvertedFile_Dict[currentIndex] = {'Doc_Freq': Doc_freq[item], 'Docs':[]}
				else:
					pass
				finally:

					# print(docid_index)
					InvertedFile_Dict[currentIndex]['Docs'].append({'docID':str(File_List_Index), 'countInDoc':item_freq[item]})
		File_List_Index += 1


def main():
	parser = ArgumentParser()
	parser.add_argument('-r', help = " if specified, turn on relevance feedback", type = bool, default = False)
	parser.add_argument('-i', help = " query-file", type = str)
	parser.add_argument('-b', help = " best-result", type = bool, default = False)
	parser.add_argument('-o', help = " output ranked list : *.csv", type = str)
	parser.add_argument('-m', help = " model-dir", type = str)
	parser.add_argument('-d', help = " NTCIR-dir", type = str)
	args = parser.parse_args()

	global File_List, Avg_Doc_Len, InvertedFile_Dict, Vocab_List

	if(args.r):
		OUTPUTFILE = OUTPUTFILE + 'R_' + str(RELATED_RATIO)

	if(args.o):
		OUTPUTFILE = str(args.o)

	File_List , Avg_Doc_Len = Read_FileList(args.m)
	Vocab_List = Read_VocabList(args.m)
	InvertedFile_Dict = Read_InvertedList(args.m)

	if(args.b):
		Cut_By_Jieba() # trigram or above
	Query(args.i, args.r, args.o)

if __name__ == '__main__':
	main()