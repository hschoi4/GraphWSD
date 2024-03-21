import os, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

'''
Objective : 
Required splitting: 80-10-10
Verb: 7901-987-987
Noun: 22129-2766-2766

Preprocess the discontinuity and mwe  

for each lemma:
	if count_sense(lemma) == 1:
		temp.append(~instance)

for each quotation:
	if count_instance(quotation) > 1:
	For each instance in quotation :
If instance not in temp:
				train.append(instance)


for each lemma whose instances are not in train and temp:
	split based on inverse population of splits.
	rank = population decreasing importance
	if sum(split) > samples:
		with rank distribute
	else:
		with reverse rank distribute
	Do full check
	Do append to train test dev 


train.append(temp)

		
'''
def reformat(s):
	return s.split("__")[1].split('_')[-1]

def importance(e):
	v = np.exp(e)
	iv = sum(v)/v
	return iv/sum(iv)

def status(tr,dv,ts, lim= (22129,2766,2766)):
	st = [tr/lim[0], dv/lim[1], ts/lim[2]]
	#print('Current:',st)
	return st

def getrank(l, reverse=False):
	array = np.array(l)
	temp = array.argsort()
	ranks = np.empty_like(temp)
	ranks[temp] = np.arange(len(array))

	#ranks = [r-1 for r in rankdata(l).astype(int)]
	if reverse:
		#print(ranks)
		ranks = [r-1 for r in len(l)- ranks]

	return list(ranks)

def check_full(split, stat):
	#print(split, stat)
	fidx = set()
	nidx = set()
	c = 0
	for i, s in enumerate(stat):
		if s >= 1:
			fidx.add(i)
		else:
			nidx.add(i)
	if len(fidx):
		for f in fidx:
			c += split[f]
			split[f] = 0

	if len(nidx) == 1:
		for n in nidx:
			split[n] +=c
	#print(split)
	return split


def strategy_split(args):

	#nountrain = pd.read_csv(f'/home/amansinha/Downloads/data/train_{args.pos}s.gold.key.txt', sep='\t', names=['instance_id', 'label'])
	#nountest = pd.read_csv(f'/home/amansinha/Downloads/data/test_{args.pos}s.gold.key.txt', sep='\t', names=['instance_id', 'label'])
	#noun = pd.concat([nountrain, nountest])

	noun = pd.read_csv(args.save_dir / f'{args.pos}s{args.version}.gold.key.txt', sep='\t', names=['instance_id', 'label'])


	#if name == 'noun':
	n = len(noun)

	noundict = dict(zip(noun['instance_id'], noun['label']))

	train, dev, test = set(), set(), set()


	lemma2sense = dict()
	#
	# 'dissoudre': {'__ws_1_dissoudre__verb__1', '__ws_2_dissoudre__verb__1'}
	#
	for inst, lab in zip(noun['instance_id'], noun['label']):

		rlab= reformat(lab)
		if rlab not in lemma2sense:
			lemma2sense[rlab] = {lab}
		else:
			lemma2sense[rlab].add(lab)

	cit2instance = dict()
	#
	# 'cit58': {'d000.cit58.t0', 'd000.cit58.t1'}
	#
	for inst, lab in zip(noun['instance_id'], noun['label']):
		cit = inst.split('.')[1]
		if cit not in cit2instance:
			cit2instance[cit] = {inst}
		else:
			cit2instance[cit].add(inst)

	label2instance = dict()
	#
	# '__ws_1_naturaliser__verb__1': {'d000.cit7022.t1', 'd000.cit7022.t2', 'd000.cit7022.t0'}
	#
	for inst, lab in zip(noun['instance_id'], noun['label']):
		#print(instance, label)
		if lab not in label2instance:
			label2instance[lab] = {inst}

		else:
			label2instance[lab].add(inst)

	lemma2inst = dict()
	#
	# 'peindre': {'d000.cit15093.t0', 'd000.cit15090.t0'}
	#
	for inst, lab in zip(noun['instance_id'], noun['label']):
		r = reformat(lab)
		#print(inst, r)
		if r not in lemma2inst:
			lemma2inst[r] = {inst}
		else:
			lemma2inst[r].add(inst)

	#
	#  'd000.cit33537.t1': '__ws_1_hésiter__verb__1'
	#
	inst2label = {inst: lab for inst, lab in zip(noun['instance_id'], noun['label'])}
	
	##############################################################

	count = 0

	temp = set()
	for k,values in lemma2sense.items():
		#print(k, len(values))
		if len(values) == 1:
			count +=1
			for v in values:
				temp |= label2instance[v]

	#print(count, len(temp))

	if not args.keep_temp:
		print('**** Temp is not included ****')
		tmpname = 'notmp'
		n -= len(temp)
	else:
		tmpname = 'tmp'
		# uncomment for unbiased splitting with tmp
		#tmpname = 'tmp2'
		#n -= len(temp)

	print(f'Number of samples - {args.pos} corpora: ', n)
	print('\nRequired percentage splitting: 80-10-10')
	lim = (int(n*.8),int(n*.1),int(n*.1))
	print(f'{args.pos}: {int(n*.8)}-{int(n*.1)}-{int(n*.1)}')
	
	for k, vs in cit2instance.items():
		#print(k,vs)
		if len(vs) > 1:
			for v in vs:
				if v not in temp:
					train.add(v)

	stat = status(len(train), len(dev), len(test), lim=lim)

	#print('instances where lemma has only one sense: ', count)

	trainlemmas = {reformat(inst2label[inst]) for inst in train}
	#print(trainlemmas)
	

	for k, v in lemma2inst.items():
		added = train | temp
		if len(v - added)>0:
			#print(k, k in trainlemmas,len(trainlemmas), len(v - added))
			#print((v-added) &  temp)
			samples = list(v - added)
			stat = status(len(train), len(dev), len(test), lim=lim)
			# stat = .33 0 0 
			#print('Current: ',status(len(train), len(dev), len(test), lim=nlim))
			split = importance(stat)
			split *= len(samples)
			split = np.rint(split)
			split = list(map(int, split))
			
			#print('==>', len(samples), split)
			ranks = getrank(list(stat))

			diff = abs(len(samples) - sum(split))

			if len(samples) > sum(split):
				
				# if all non zeros 
				# get rank incr order indexes based on value

				for i in range(len(split)):
					idx = ranks.index(i)
					if stat[idx] < 1:
						split[idx] += 1
						diff -= 1
					if diff ==0:
						break
					elif i ==2:
						i =0
				
			elif len(samples) < sum(split):
				
				ranks = getrank(list(stat), reverse=True)
				#print(ranks)
				for i in range(len(split)):
					idx = ranks.index(i)
					if split[idx] > 0:
						split[idx] -= 1
						diff -= 1
					if diff ==0:
						break
					elif i ==2:
						i =0

				#print('**',len(samples), split)

			split = check_full(split, stat)
			if not split[0]:
				# if split_orig doesnot allow any lemma into train e.g. [0, 1, 1] then permute the split until we have [1, .. , ..]
				while (split[0] == 0):
					split = np.random.permutation(split)

			#print(k, split)

			train |= set(samples[0:split[0]]); [trainlemmas.add(reformat(inst2label[s])) for s in samples[0:split[0]]]
			dev |= set(samples[split[0]:split[0]+split[1]])
			test |= set(samples[split[0]+split[1]:])

			if args.logs:
				print(stat)

	if args.keep_temp:
		train1 = train | temp
		stat= status(len(train1), len(dev), len(test), lim=lim)


	print(len(train), len(dev), len(test),'=',len(train)+len(dev)+len(test))

	if args.write:

		with open(args.save_dir / args.pos /f'train-{args.pos}.gold.key.txt', 'w') as fp:
			for d in train:
				print(d, noundict[d],sep='\t', file=fp)

		if args.keep_temp:
			with open(args.save_dir / args.pos /f'train-{args.pos}-{tmpname}.gold.key.txt', 'w') as fp:
				for d in train1:
					print(d, noundict[d],sep='\t', file=fp)

		with open(args.save_dir / args.pos /f'dev-{args.pos}-{tmpname}.gold.key.txt', 'w') as fp:
			for d in dev:
				print(d, noundict[d],sep='\t', file=fp)

		with open(args.save_dir / args.pos /f'test-{args.pos}-{tmpname}.gold.key.txt', 'w') as fp:
			for d in test:
				print(d, noundict[d],sep='\t', file=fp)

		print('Files saved ...')

	print('Splitting done ...')
	#print(len(temp & train ))
	#print(len(temp & dev ))
	#print(len(temp & test ))
	#return train, dev, test
		




if __name__ == '__main__':

	import argparse, pathlib

	parser = argparse.ArgumentParser(description="Strategy Splitting...")
	#parser.add_argument('--dataframe', required=True, help="Dataset dataframe")
	parser.add_argument('--pos', required=True, help='name of corpora')
	parser.add_argument('--version', help='corpora version required')
	parser.add_argument('--logs', action='store_true',help='Print logs')
	parser.add_argument('--write',  action='store_true',help='whether or not to write into the files')
	parser.add_argument('--keep_temp',  action='store_true',help='whether or not to keep temp')
	parser.add_argument('--save_dir', type=pathlib.Path, help='location to save the split files')

	args = parser.parse_args()

	strategy_split(args)