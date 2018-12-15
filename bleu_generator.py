from sacreBLEU.sacrebleu import *
from nltk.translate import bleu_score
import argparse
try:
	parser = argparse.ArgumentParser(description='Bleu score generator')
	parser.add_argument('--input', type=str, default='', metavar='P',
                    help="File name to get bleu score")
	parser.add_argument('--bleu',type=str,default="nltk",metavar='P',help="Which bleu score to use")
	args = parser.parse_args()
	in_file=open(args.input,'r')
except:
	print("This program takes a file with translations in the following manner:\nTranslation: ...\nTarget: ...\n\nTranslation: ...\n...\n..\n\nPlease give a valid file.")
else:
	system_sentences=[]
	ref_sentences=[]
	in_string=in_file.read()
	##nltk bleu takes tokenized complete string and not sentences
	in_list=in_string.split("\n")
	if args.bleu=="nltk":
		for line in in_list:
			if line.startswith("Translation: "):
			#system_sentences.append(line[13:])
				system_sentences+=line[13:].split(" ")
			elif line.startswith("Target: "):
			#ref_sentences.append(line[8:])
				ref_sentences+=line[8:].split(" ")
	else:
		for line in in_list:
			if line.startswith("Translation: "):
				system_sentences.append(line[13:])
				#system_sentences+=line[13:].split(" ")
			elif line.startswith("Target: "):
				ref_sentences.append(line[8:])
				#ref_sentences+=line[8:].split(" ")
	ref_stream=[ref_sentences]
	#print(len(system_sentences),len(ref_stream[0]))
	#print(ref_stream)
	if args.bleu=="nltk":
		print("NLTK bleu score is:",bleu_score.corpus_bleu([ref_stream],[system_sentences]))	
	else:
		print(corpus_bleu(system_sentences,ref_stream,tokenize="13a"))
	
