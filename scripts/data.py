import os
import torch
from collections import defaultdict

from lxml import etree

# conll format data (usfac format)
# evaluation metric

class WSDData():
    "Class containing instances"
    def __init__(self, name=None, sent_id2sent=None, sent_id2instances=None,  id2instance=None, key2instances=None):
        self.name = name
        self.sent_id2sent, self.sent_id2instances, self.id2instance, self.key2instances = sent_id2sent, sent_id2instances, id2instance, key2instances
        self.instances = list(self.id2instance.values())
        
        
    def get_target_pos(self):
        """Return Pos contained in dataset"""
        return {x.split('.')[1] for x in self.key2instances}
        
    def get_target_words(self):
        """Returns vocab contained in dataset"""
        return {x.split('.')[0] for x in self.key2instances}
    
    def get_target_keys(self):
        """ return keys contained in dataset"""
        return self.key2instances.keys()
    
    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        #return {'id':idx, 'sample':self.instances[idx], 'length':len(self.instances[idx].context)}
        return {'id':torch.tensor(idx,dtype=torch.long), 
        #'sample':torch.tensor(self.instances[idx], dtype=Instance), 
        'length':torch.tensor(len(self.instances[idx].context),dtype=torch.long)}

    def get_instances(self):
        yield from self.id2instance.values()
            
    def get_labels(self):
        labels = defaultdict(bool)
        for i, instance in self.id2instance.items():
            labels[instance.first_label] = True
            
        return list(labels.keys())
      
        
class Instance():
    "Class for instances to be dismabiguated"
    def __init__(self,id, sent_id=None, word_form=None, key=None, lemma=None, pos=None, pos_BELRL=None, tok_id=None, labels=None, first_label=None,
                 source=None, instance_src=None, is_mwe=False, context=None, position=None):
        
        self.id = id
        self.sent_id = sent_id
        self.lemma = lemma
        self.key= key
        self.pos = pos
        self.labels = labels
        self.pos_BELRL = pos_BELRL
        self.source = source
        self.instance_src = instance_src
        self.word_form = word_form
        self.is_mwe = is_mwe
        self.context = context
        self.first_label = first_label
        self.tok_id = tok_id
        self.position = position
        #self.complete_text = complete_text#' '.join(complete_text)
        
    def __str__(self):
        string = []
        
        id_str = f"ID = {self.id}"
        string.append(id_str)
        
        key = f"Key = {self.key}"
        string.append(key)
        
        label = f"First label = {self.first_label}"
        string.append(label)
        
        labels = f"Labels = {self.labels}"
        string.append(labels)
        
        tok_id = f"Tok ID = {self.tok_id}"
        string.append(tok_id)
        
        context = f"\nContext\n{self.context}"
        string.append(context)
        
        #text = f"\nText\n{self.complete_text}"
        #string.append(text)
        
        position = f"Position = {self.position}"
        string.append(position)
        
        return string
        
class WSDDataReader():
    """Class to read a WSD Directory"""
    
    def __init__(self, name):
        self.name = name
    
    def count_token(self, data_dir):
        """ Coun number of tokens in WSD dir"""
        count = 0
        xml_fpath, gold_fpath = self.get_data_paths(data_dir)
        
        tree = etree.parse(xml_fpath)
        corpus = tree.getroot()
        
        for text in corpus:
            for sent in text:
                count += len(list(sent))
                
        return count
                     
        
    def get_data_paths(self, indir):
        """ Get files from WSD folder"""
        xml_fpath, gold_fpath = None, None
        
        for f in os.listdir(indir):
            if f.endswith('tmp.data.xml') and f.startswith(self.name):
                xml_fpath = indir / f
            if f.endswith('tmp.gold.key.txt') and f.startswith(self.name):
                gold_fpath = indir / f
        print(xml_fpath, gold_fpath)
        return xml_fpath, gold_fpath
        
                
    def read_gold(self, infile):
        return {line.split()[0]: line.rstrip('\n').split('\t')[1:] for line in open(infile).readlines()}
    
    def read_sentences(self, data_dir, keep_mwe=True):
        
        xml_fpath, _ = self.get_data_paths(data_dir)
        return self.read_sentences_from_xml(xml_fpath, keep_mwe=keep_mwe)
    
    def read_sentences_from_xml(self, infile, keep_mwe=False):
        
        tree = etree.parse(str(infile))
        corpus = tree.getroot()
        
        for text in corpus:
            for sentence in text:
                if keep_mwe:
                    sent= [tok.text.replace(' ', '-') for tok in sentence]
                else:
                    sent = [subtok for tok in sentence for subtok in tok.text.split(' ')]
                yield sent
                    
    def read_target_keys(self, infile):
        return [x.rstrip('\n') for x in open(infile).readlines()]
    
        
    def read_from_data_dirs(self, data_dirs, target_pos=None, target_words=None, target_keys=None, ignore_source=[], keep_mwe=False, add_context_to_instance=False):
        
        id2sent = {}
        id2instance = {}
        key2instances = defaultdict(list)
        sent_id2instances = defaultdict(list)
        
        for d in data_dirs:
            #print(d)
            xml_fpath, gold_fpath = self.get_data_paths(d)
            target_pos, target_words, target_keys = target_pos, target_words, target_keys
            #print(xml_fpath, gold_fpath)
            id2gold = self.read_gold(str(gold_fpath))
            
            sentences = self.read_sentences(d, keep_mwe=keep_mwe)
            
            # parse xml
            tree = etree.parse(str(xml_fpath))
            corpus = tree.getroot()
            
            # process data
            
            for text in corpus:
                text_id = text.get('id')
                if len(text.get('id').split('.'))>1:
                    source = text.get('id').split('.')[0]
                else:
                    source = corpus.get('source')
                
                
                
                for sentence in text:
                    #print(sentence)
                    if source in ignore_source:
                        continue
                    
                    sent_id = sentence.get('id')
                    #print(sent_id)
                    sent_id_with_source = source + "." + sent_id if len(sent_id.split('.')) == 2 else source + ' '.join(sent_id.split('.')[1:])
                    
                    flag = False
                    
                    sent = next(sentences)
                    #print(sent_id, sent)
                    tok_idx = 0
                    
                    complete_text = []
                    instance=None
                    for i, tok in enumerate(sentence):
                        
                        lemma, pos, pos_BELRL = tok.get('lemma'), tok.get('pos'), tok.get('pos_BELRL')
                        key = lemma + '__' + pos
                        wf = tok.text
                        complete_text.append(wf)
                        subtokens = wf.split(' ')
                        
                        is_mwe = True if len(subtokens) > 1 else False
                        
                        if tok.tag == "instance":
                            
                            if len(lemma.split())>1: lemma = '-'.join(lemma.split())
                            if target_pos and pos not in target_pos:
                                pass
                            elif target_words and lemma not in target_words:
                                pass
                            elif target_keys and key not in target_keys:
                                pass
                            else:
                                
                                id =  tok.get("id")
                                id_with_source = source + '.' + id if len(id.split('.')) == 3 else source + '.' + '.'.join(id.split('.')[1:])
                                
                                target_labels = id2gold[id]
                                target_first_label = target_labels[0]

                                nodeid = tok.get("source").split('/')[-1]
                                isrc = f"ls:fr:node:{nodeid}"
                                
                                if keep_mwe:
                                    tgt_idx = tok_idx
                                    
                                else:
                                    if pos == "VERB":
                                        tgt_idx = tok_idx
                                    else:
                                        tgt_idx = tok_idx + len(subtokens)-1
                                        
                                instance = Instance(id_with_source, sent_id_with_source, wf, key, lemma=lemma, pos=pos, pos_BELRL=pos_BELRL,
                                                   tok_id = tgt_idx, labels=target_labels,
                                                   first_label=target_first_label,
                                                   source = source, instance_src=isrc, is_mwe=is_mwe, position=i)
                                
                                if add_context_to_instance:
                                    instance.context = sent
                                    
                                key2instances[key].append(instance)
                                sent_id2instances[sent_id_with_source].append(instance)
                                id2instance[id_with_source] = instance
                                flag = True
                                
                            off_set = 1 if keep_mwe else len(subtokens)
                            tok_idx += off_set
                                
                            if flag:
                                id2sent[sent_id_with_source] = sent
                    
                    #instance.complete_text = complete_text            

        return WSDData(self.name, id2sent, sent_id2instances, id2instance, key2instances)                        
