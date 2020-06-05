import re
import string
import sys
import math
import pprint
import random
import Queue
import time
import argparse

argumentparser = argparse.ArgumentParser()
argumentparser.add_argument('-infile', help="filename of input log")
argumentparser.add_argument('--outf', help="filename output log. if not add .out after in")
argumentparser.add_argument('-maxline', type=int, help="Number of line to perform the operation")
argumentparser.add_argument('--m', help="get the max instead of the sum")
def main(args):
	inFileName = str(args.infile)
	outFileName = inFileName + ".out"
	maxline = int(args.maxline) #VGG: 13 in forward (allreduce), 12 in backward
	opt = "SUM"
	if args.m is not None:
		opt = str(args.m)
		
	print 'Read log from ', inFileName
	
	f = open(inFileName, 'r')
	f2 = open(outFileName,'w+')

	gpu_sum = 0
	cpu_sum = 0
	lineIdx = 1
	colNum = 0
	if opt == "MIN":
		gpu_sum = 1e9
	for line in f:
		splitLine = line.replace('\r','')
		splitLine = splitLine.replace('\n','')
		splitLine = splitLine.split('\t')
		if lineIdx ==1:
			colNum = len(splitLine)
		
		if len(splitLine) == colNum:
			if len(splitLine) >= 2:
				gpu_time = float(splitLine[0])
				cpu_time = 0
				if (not splitLine[1]) and (splitLine[1] != ""):
					cpu_time = float(splitLine[1])
				
				if opt == "SUM":
					gpu_sum = gpu_sum + gpu_time
					cpu_sum = cpu_sum + cpu_time
				elif opt == "MAX":
					gpu_sum = max(gpu_sum, gpu_time)
					cpu_sum = cpu_time
				elif opt == "MIN":
					gpu_sum = min(gpu_sum, gpu_time)
					cpu_sum = cpu_time				

			elif len(splitLine) >= 1:
				gpu_time = float(splitLine[0])
				if opt == "SUM":
					gpu_sum = gpu_sum + gpu_time
				elif opt == "MAX":
					gpu_sum = max(gpu_sum, gpu_time)
				elif opt == "MIN":
					gpu_sum = min(gpu_sum, gpu_time)	
				cpu_sum = ""
		else:
			print "Lack of item at line", lineIdx
		
		#print lineIdx, val
	
		#print lineIdx,maxline
		if (lineIdx % maxline) == 0:
			#write line:
			#print lineIdx, sum
			if cpu_sum is not "":
				f2.write(str(gpu_sum) + "\t" + str(cpu_sum) + "\n")
			else:
				f2.write(str(gpu_sum) + "\n")
			gpu_sum = 0
			if opt == "MIN":
				gpu_sum = 1e9
			cpu_sum = 0
		lineIdx = lineIdx + 1
	f.close()
	f2.close()
	
if __name__ == '__main__':
	main(argumentparser.parse_args())	