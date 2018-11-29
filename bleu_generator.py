from sacreBLEU.sacrebleu import *
import argparse
try:
	parser = argparse.ArgumentParser(description='Bleu score generator')
	parser.add_argument('--input', type=str, default='', metavar='P',
                    help="File name to get bleu score")
	args = parser.parse_args()
	in_file=open(args.input,'r')
except:
	print("This program takes a file with translations in the following manner:\nTranslation: ...\nTarget: ...\n\nTranslation: ...\n...\n..\n\nPlease give a valid file.")
else:
	system_sentences=[]
	ref_sentences=[]
	in_string=in_file.read()
	in_list=in_string.split("\n")
	for line in in_list:
		if line.startswith("Translation: "):
			system_sentences.append(line[13:])
		elif line.startswith("Target: "):
			ref_sentences.append(line[8:])
	ref_stream=[ref_sentences]

	print(corpus_bleu(system_sentences,ref_stream,tokenize="split_tokenizer"))
	
