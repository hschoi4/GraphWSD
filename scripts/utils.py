import pandas as pd
import numpy as np
from tqdm import tqdm
import os, pathlib, random

from lxml import etree
from xml.dom import minidom
import xml.etree.ElementTree as ET

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tools import belfr_to_semeval, strategy_splitting3


def seed_everything(seed=1234):
    """random seed control"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def prettify(elem):
		"""Return a pretty-printed XML string for the Element.
		"""
		rough_string = ET.tostring(elem, 'utf-8')
		reparsed = minidom.parseString(rough_string)
		return reparsed.toprettyxml(indent="  ")

def check_xml(infile, keep_mwe=False):

	tree = etree.parse(str(infile))
	corpus = tree.getroot()

	for text in corpus:
		for i, sentence in enumerate(text):
			if keep_mwe:
				sent = [tok.text.replace(' ', '_') for tok in sentence]
			else:
				sent= [subtok for tok in sentence for subtok in tok.text.split(' ')]
	print('Test passed!!')

def construct_lemma(source, nodes, entries):

	srcid = 'ls:fr:node:' +source
	entry = nodes[nodes['id'] == srcid]['entry'].values[0]
	ent = entries[entries['id'] == entry]
	add_to = ent['addtoname'].values[0]
	subscript = ent['subscript'].values[0]
	supscript = ent['superscript'].values[0]
	if pd.isna(supscript): supscript = '*'
	if isinstance(supscript, float): supscript = str(int(supscript))
	lemma = ent['name'].values[0]
	if add_to is not np.nan:
		lemma  = add_to + lemma
	if subscript is not np.nan:
		lemma = '.'.join([lemma, subscript])
	if supscript is not np.nan and supscript != '*':
		lemma = '.'.join([lemma, supscript])
	return entry,lemma

def make_sensefile(save_dir, nodes, entries, name=''):
	"""makes global sense files using nodes and entry files from RL-fr"""
	entrydict = dict()
	f = open(save_dir / f'nodeid_senselabel_{name}.dat', 'w')

	print('nodeid\tsense_label', file=f)
	for i,info in tqdm(enumerate(nodes['id'])):
			nodeid = info.split(':')[-1]
			entry, lemma = construct_lemma(nodeid, nodes, entries)
			lemma = lemma.replace(' ', '')
			lemma = lemma.replace(',', '.')
			#lemma = lemma.replace('-', '.')

			if entry not in entrydict:
					entrydict[entry] = {nodeid:1}
			else:
					if nodeid not in entrydict[entry]:
							entrydict[entry][nodeid] = len(entrydict[entry]) + 1
			print(nodeid+'\t'+'__ws_{}_{}__'.format(entrydict[entry][nodeid],lemma), file=f)
			
	f.close()
	print('File is saved!')


def report_split_distribution(traindata):
	'''
	traindata is class instance of WSDDataReader
	'''
	distr = defaultdict(dict)
	labels = set()
	for instance in traindata.instances:
		
		#print(instance.__dict__)
		
		if instance.first_label not in labels:
			distr[instance.lemma][instance.first_label] = 1
		else:
			distr[instance.lemma][instance.first_label] += 1
			
		labels.add(instance.first_label)    
	
	return distr


def resolve_discontinuity(p, version, xml_path, save_dir, senses):
	#p = 'verb'

	#xml_path = '/home/amansinha/wsd-master/copy/wsd/logs/'
	tree = ET.parse(xml_path/ f'{p}s{version}.xml')
	root = tree.getroot()

	corpus = ET.Element('corpus')
	corpus.set('lang','fr')
	corpus.set('source','BEL-RL-fr')
	corpus.set('version',root.attrib['version'])#extra annotation
	text = ET.SubElement(corpus, 'text')
	text.set('id',"d000")

	#save_dir = '/home/amansinha/wsd-master/copy/wsd/logs/'
	f = open(save_dir / f'{p}s{version}.xml','w')
	g = open(save_dir / f'{p}s{version}.gold.key.txt','w')


	for j,s in tqdm(enumerate(root.iter('sentence'))):
			#print(s)
			#'''    
			name = s.attrib['name']
			#if name != 'cit12219':
					#continue
			sent = ET.SubElement(text, 'sentence', s.attrib)
			
			instances = list(s.iter('instance'))
			sdict = dict()
			for i, seg in enumerate(instances):
					src = seg.attrib['source']
					sdict[src] = seg
					#print(seg, seg.text)
			
			#print(j, sdict)
			c = 0
			for e in s:
					if e.tag == 'instance':
							src = e.attrib['source']
							#print(src)
							slabel = senses[senses['nodeid'] == int(src.split('/')[-1])]['sense_label'].values[0]
							if sdict[src] == e:
									tag = e.tag
									e.attrib['id'] = f"d000.{name}.t{str(c)}"
									c += 1
									attrib = e.attrib
									print(e.attrib['id'], slabel+f'{p}__1', sep='\t', file=g)
							else:
									tag = 'wf'
									attrib = {'lemma':e.attrib['lemma'], 'pos':e.attrib['pos']}
							
							w = ET.SubElement(sent, tag, attrib)
							if e.text == None:
									w.text = attrib['lemma']
							w.text = e.text
					else:
							w = ET.SubElement(sent, e.tag, e.attrib)
							w.text = e.text
							#print(e.tag, e.attrib)
			
			#if name =='cit12219':
					#break

	print(prettify(corpus), file=f)

	f.close()
	g.close()

	print('new version file saved...')


def create_data(p, root, lfile, save_dir, tmp, name='train'):
	
	lfile = str(lfile)
	outfile = lfile.split('/')[-1].split('.')[0]

	label_file = pd.read_csv(lfile, sep='\t', names=['instanceId', 'label'])
	#print(root.attrib['version'])
	sentences = root.iter('sentence')
	samples = list({'.'.join(instance.split('.')[:2]) for instance in label_file['instanceId']})
	#print(samples[-4:])
	sampleInstancesfromRoot = 0
	sampleInstancedict = {s.attrib['id']:s for s in sentences}
	#print(sampleInstancedict)
	
	for s in samples:
		if not (s in sampleInstancedict.keys()):
			print(s)
	
	f = open(save_dir / f'{outfile}.data.xml', 'w')
	
	corpus = ET.Element('corpus')
	corpus.set('lang','fr')
	corpus.set('source','BEL-RL-fr')
	corpus.set('version',root.attrib['version'])#extra annotation
	text = ET.SubElement(corpus, 'text')
	text.set('id',"d000")
	   
	c =1
	for s in samples:
		sen = sampleInstancedict[s]
		#print('==',sen.attrib['name'], s, sen)
		
		sent = ET.SubElement(text, 'sentence')
		sent.set('name',sen.attrib['name'])
		sent.set('idx',str(c))
		sent.set('id', s)
		sent.set('source', sen.attrib['source'])

		for w in sen.iter():
			if w.tag != 'sentence':
				word = ET.SubElement(sent, w.tag, w.attrib)
				word.text = w.text
			
		c+=1
		
		
	
	print(prettify(corpus), file=f)
	f.close()
	print(label_file.head(), '\nFile loaded...')
	print('file saved...')


def load_sense_mat(senses: set):

	b = np.load('/home/amansinha/Downloads/lexical_e50_embeddings.npz')
	semb = []
	embs = b['embeddings'] 
	embs = embs.reshape(len(embs), -1)
	dim = b['embeddings'].shape[-1]
	pretrained_dict = {k:e for k,e in zip(b['synsets'], embs)}

	for s in senses:
		if s not in pretrained_dict:
			semb.append(pretrained_dict['OOV'])
		else:
			semb.append(pretrained_dict[s])

	semb = np.asarray(semb)

	return torch.Tensor(semb)


def load_adjacency_mat(senses: set, semantics=False, fragment=False):
	try:
		reldf = pd.read_csv('/home/amansinha/wsdmaster/copy/wsd/data/version_31iii21/RL-fr/ls-fr-spiderlex/15-lslf-rel_boost.csv', sep='\t')
	except:
		reldf = pd.read_csv('~/wsd-grid/wsd/data/version_31iii21/RL-fr/ls-fr-spiderlex/15-lslf-rel_boost.csv', sep='\t')

	ndict = {n:i for i, n in enumerate(senses)} # node dict
	#print('Number of senses for A construction:', len(senses), len(ndict))
	adjmat = np.zeros((len(ndict), len(ndict)))

	if not fragment:
	 	# fragment = False
	 	# construct adjmat -> sparse components
		for src, tar, sem in zip(reldf['source'], reldf['target'], reldf['semantics']):
			if (src in senses) and (tar in senses):
				isrc = ndict[src]
				itar = ndict[tar]
				if semantics:
					adjmat[isrc][itar] += sem + 1 #(to avoid 0 class strength)
				else:
					adjmat[isrc][itar] += 1

		from scipy import sparse

		adjs = sparse.csr_matrix(adjmat, shape=(len(senses),len(senses)))
		Acoo = adjs.tocoo()
		adj_sparse = torch.sparse.LongTensor(torch.LongTensor([Acoo.row.tolist(), Acoo.col.tolist()]),
									  torch.FloatTensor(Acoo.data.astype(np.float64)), (len(senses),len(senses)))

	else:
		# fragment = True
		# directly construct sparse components (fragments)
		rdict, r2e, e2r = {}, {}, {}

		for i, (src, tar, sem, r) in enumerate(zip(reldf['source'], reldf['target'], reldf['semantics'], reldf['lf'])):
			if (src in senses) and (tar in senses):
				key = sem if semantics else r

				if key not in rdict:
					rdict[key] = len(rdict)

				isrc, itar = ndict[src], ndict[tar]

				if rdict[key] not in r2e:
					r2e[rdict[key]] = [(isrc, itar)]
				else:
					r2e[rdict[key]].append((isrc, itar))

				if (isrc, itar) not in e2r:
					e2r[(isrc, itar)] = [rdict[key]]
				else:
					e2r[(isrc, itar)].append(rdict[key])

		# setting rel/sem values
		rval = [1 for i in range(len(rdict))]

		#print(r2e.items(), e2r.values())
		indices, values = [], []
		for r,es in r2e.items():

			x,y,vals = [], [], []
			for e in es:
				x.append(int(e[0])); y.append(int(e[1]))
				if semantics:
					vals.append(r)
				else:
					vals.append(1)
			indices.append([x,y])
			values.append(vals)


		adj_sparse = [torch.sparse_coo_tensor(torch.LongTensor(indices[i]),torch.FloatTensor(values[i]), (len(ndict),len(ndict)),requires_grad=False) for i in range(len(indices))]

	return adj_sparse


def make_batches(dataset, encoder, batch_size, padding):
	""" Make batches based from dataset instances
		:param dataset: wsd dataset
		:type dataset: WSDDataset
		:param encoder: encoder to obtain contextual vector
		:type encoder: WSDEncoder
		:param batchsize: size of batches
		:type batchsize: int
		:param padding: padding length
		:type padding: int
	"""

	batches = []
	batch = []

	for sent_id, instances in tqdm(dataset.sent_id2instances.items(), "Making batches", leave=False):

		# if sentence with no instances continue
		if len(instances) == 0:
			continue

		sent = dataset.sent_id2sent[sent_id]

		sent_lst = sent
		tok_ids, att_mask, span = encoder.encode(sent_lst, padding=padding)
		length = len(tok_ids)

		# if length of tokenizer encodings > padding
		if length > padding and padding > 0:
			batch_ = [((tok_ids, att_mask,span), instances)]
			inputs_batch, instances_batch = list(zip(*batch_))
			inputs = encoder.collate_fn(inputs_batch)
			wsd_data = list(zip(*[(i.id,i.tok_id,  i.key.replace(".","<COMMA>"), i.first_label, i.labels, j) for j,x in enumerate(instances_batch) for i in x]))
			batches.append((inputs, wsd_data))
			continue

		batch.append(((tok_ids, att_mask,span), instances))

		if len(batch) == batch_size:
			inputs_batch, instances_batch = list(zip(*batch))
			inputs = encoder.collate_fn(inputs_batch)
			wsd_data = list(zip(*[(i.id,i.tok_id,  i.key.replace(".","<COMMA>"), i.first_label, i.labels, j) for j,x in enumerate(instances_batch) for i in x]))
			batches.append((inputs, wsd_data))
			batch = []

	# last batch
	if len(batch) > 0:
		inputs_batch, instances_batch = list(zip(*batch))
		inputs = encoder.collate_fn(inputs_batch)
		wsd_data = list(zip(*[(i.id,i.tok_id,  i.key.replace(".","<COMMA>"), i.first_label, i.labels, j) for j,x in enumerate(instances_batch) for i in x]))
		batches.append((inputs, wsd_data))

	return batches


def make_batches2(devdata, encoder, batch_size, shuffle=False):
    
    db = DataLoader(devdata, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    batches = []

    for bat in tqdm(db):

        b = bat['id']
        padding = -1
        batch_instances = [devdata.instances[i] for i in b]
        wsd_data = list(zip(*[(i.id,i.position, i.key, i.first_label, i.labels, bi) for bi, i in enumerate(batch_instances)]))
        cbatch = [devdata.instances[i].context for i in b]
        batch = []

        for cbi in cbatch:
            tok_ids, att_mask, spans = encoder.encode(cbi)

            padding = max(padding, len(tok_ids))
            batch.append( (tok_ids, att_mask, spans) )

        nbatch = []
        for t,a,s in batch:
            s = s.numpy().tolist()

            if len(t) < padding:
                diff = padding -len(t)
                t = F.pad(t, (0,diff),value=0)
                a = F.pad(a, (0,diff),value=False)
                s.extend([s[-1] for i in range(diff)])
            s = torch.tensor(s)
            nbatch.append((t,a,s))

        ebatch = encoder.collate_fn(nbatch)
        word_id = wsd_data[1]
        batch_id = wsd_data[-1]
        batches.append((ebatch, wsd_data))

    return batches

class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 7
			verbose (bool): If True, prints a message for each validation loss improvement.
							Default: False
			delta (float): Minimum change in the monitored quantity to qualify as an improvement.
							Default: 0
		"""
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.inf
		self.delta = delta
		self.save_path = save_path
		os.makedirs(pathlib.Path(self.save_path).parent, exist_ok=True)

	def __call__(self, val_loss, model):

		score = -val_loss

		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
		elif score < self.best_score - self.delta:
			self.counter += 1
			print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
			self.counter = 0

	def save_checkpoint(self, val_loss, model):
		"""Saves model when validation loss decrease."""
		if self.verbose:
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		torch.save(model.state_dict(), self.save_path)
		self.val_loss_min = val_loss



if __name__== '__main__':

	# recheck saving paths to make sure

	import argparse, pathlib

	parser = argparse.ArgumentParser(description="Running utils...")
	parser.add_argument('--name', required=True, help='name required')
	parser.add_argument('--version', help='corpora version required')
	parser.add_argument('--pos', help='pos name required')
	parser.add_argument('--save_dir', type=pathlib.Path, help='location to save files')
	parser.add_argument('--home_dir', type=pathlib.Path, help='location of wsd directory')
	parser.add_argument('--xml_path', type=pathlib.Path, help='location to xml files')
	parser.add_argument('--logs', action='store_true',help='Print logs')
	parser.add_argument('--write',  action='store_true',help='whether or not to write into the files')
	parser.add_argument('--keep_temp',  action='store_true',help='whether or not to keep temp')


	args = parser.parse_args()

	#home_dir = '/home/amansinha/wsd-master/copy/wsd/'
	#xml_path = home_dir + 'data/BEL-RL-fr/version_12.18.20/xmlPOS/'
	#save_dir = '/home/amansinha/Downloads/'#xml_path + 'meta/'

	nodes = pd.read_csv(args.home_dir / 'data/version_31iii21/RL-fr/ls-fr-spiderlex/01-lsnodes.csv', sep='\t')
	entries = pd.read_csv(args.home_dir / 'data/version_31iii21/RL-fr/ls-fr-spiderlex/02-lsentries.csv', sep='\t')
	
	try:
	    os.mkdir(args.save_dir / args.pos)
	except OSError:
	    print ("Creation of the directory failed!!")
	else:
	    print (f"Successfully created the directory - {args.pos} ")


	# command 1 
	# to create sense files - check name of file to be saved otherwise previous will be overwritten"
	make_sensefile(args.save_dir, nodes, entries, name=args.name)
	print('step1 done....')
	senses = pd.read_csv(args.save_dir/f'nodeid_senselabel_{args.name}.dat', sep='\t')
	
	#command 2
	# convert bel-rl to conll format verb and noun
	# refer belrl-to-semeval file
	belfr_to_semeval.convert_format(args.pos, args.version, args.xml_path, args.save_dir, nodes, entries, senses)
	print('step2 done....')
	# command 3
	# to check empty tags in xml format
	check_xml(args.save_dir / f'{args.pos}s{args.version}.xml')
	print('step3 done....')
	# command 4
	# resolveing discontinuity and making newer version of corpora

	resolve_discontinuity(args.pos, args.version, args.save_dir, args.save_dir, senses)
	print('step4 done....')
	# command 5
	# go do spliiting train test dev - using strategy_splitting2
	strategy_splitting3.strategy_split(args)
	print('step5 done....')
	# command 6
	# after doing splitting get corresponding xml files
	
	p = args.pos
	infile = args.save_dir / f'{p}s{args.version}.xml'
	tree = ET.parse( str(infile))
	root = tree.getroot()
	# add notmp if needed
	for t in ['tmp']:
		for s in ['train', 'dev', 'test']:

			if s == 'train':

				if t != 'notmp':
					lfile = args.save_dir / f'{p}/{s}-{p}.gold.key.txt'
					create_data(p, root, lfile, args.save_dir / f'{p}/', t, name=s)

					lfile = args.save_dir / f'{p}/{s}-{p}-{t}.gold.key.txt'
					create_data(p, root, lfile, args.save_dir / f'{p}/', t, name=s)
				else:
					lfile = args.save_dir / f'{p}/{s}-{p}.gold.key.txt'
					create_data(p, root, lfile, args.save_dir / f'{p}/', t, name=s)

			else:
				lfile = args.save_dir / f'{p}/{s}-{p}-{t}.gold.key.txt'
				create_data(p, root, lfile, 	args.save_dir / f'{p}/', t, name=s)
	
	print('step6 done....')
	


	
