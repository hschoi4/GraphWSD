#belfr-to-semeval.py

# version 7

"""
VERSION7

Constructing ver6 for verb xml w/ example src, udpipe annotaion pos, source seg, with hi tag text
Constructing target file for the experiment

"""
from xml.dom import minidom
import xml.etree.ElementTree as ET
import pandas as pd
import time
import numpy as np
import spacy_udpipe
from xml.dom import minidom
from tqdm import tqdm

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


'''
def construct_lemma(source):

  srcid = 'ls:fr:node:' +source
  #print('constructed node', srcid)
  entry = nodes[nodes['id'] == srcid]['entry'].values[0]
  #print('obtained entry', entry)
  lemma = entries[entries['id'] == entry]['name'].values[0]
  return entry, lemma
'''

def construct_lemma(source, nodes, entries):

  srcid = 'ls:fr:node:' +source
  try:
    entry = nodes[nodes['id'] == srcid]['entry'].values[0]
  except:
    raise IndexError(f'{source} is not found in nodes file')
  ent = entries[entries['id'] == entry]
  add_to = ent['addtoname'].values[0]
  subscript = ent['subscript'].values[0]
  lemma = ent['name'].values[0]
  if add_to is not np.nan:
    lemma  = add_to + lemma
  if subscript is not np.nan:
    lemma = '-'.join([lemma, subscript])
  return entry,lemma


other_pos = []

p= 'verb'



def convert_format(p, version,xml_path, save_dir, nodes, entries, senses):

  spacy_udpipe.download("fr") 
  nlp = spacy_udpipe.load("fr")

  if p == 'verb':
    poslist = ['ls:fr:gc:23','ls:fr:gc:24']
    poslabel = 'VERB'
  else:
    poslist = ['ls:fr:gc:20']
    poslabel = 'NOUN'

  tree = ET.parse(xml_path/ f'BEL-RL-fr_AllSources_{p}s.xml')
  root = tree.getroot()

  #=====
  corpus = ET.Element('corpus')
  corpus.set('lang','fr')
  corpus.set('source','BEL-RL-fr')
  corpus.set('version',f'V{version}')#extra annotation
  text = ET.SubElement(corpus, 'text')
  text.set('id',"d000")
  #=====

  flag = 'i'
  nid,c = 0,0
  ids = set()

  fg=0
  
  f = open(save_dir / f'{p}s{version}.xml','w')
  g = open(save_dir / f'{p}s{version}.gold.key.txt','w')
  fp = open(save_dir / f'cit_examples_{p}s.dat', 'w')
  #gp = open(save_dir / f'cit_pos_seg_{p}s.dat', 'w')



  for i,r in enumerate(tqdm(root.iter('{http://www.tei-c.org/ns/1.0}cit'))):
    cid = r.attrib['{http://www.w3.org/XML/1998/namespace}id']
    #print(cid)
    #if cid != citid:
    #  continue

    src = r.attrib['source']
    #print('#'*20, cid)        
    if cid not in ids:
      sents = ET.SubElement(text, 'sentence')
      nid += 1
      sid = "d000."+cid
      sents.set('name', cid)
      sents.set('idx', str(nid))
      sents.set('id',sid)
      sents.set('source', src)
      ids.add(cid)
      
    
    full_text = []
    c = 0

    for quote in r.findall('./{http://www.tei-c.org/ns/1.0}quote'):
      #print('==', quote, quote.text != None)
      if quote.text != None:
        #print('found head==', quote.text)
        doc_head = nlp(quote.text)
        for d in doc_head:
          w = ET.SubElement(sents,'wf')
          w.set('pos', d.pos_)
          w.set('lemma', d.lemma_)
          w.text = d.text
          #print(d.text, d.lemma_, d.pos_)
        full_text.append( quote.text)

      for seg in quote.findall('./'):
        #print(cid,seg.attrib)
        if seg.tag in ['{http://www.tei-c.org/ns/1.0}hi', '{http://www.tei-c.org/ns/1.0}lb']:
            if not seg.text and not seg.tail:
              continue
            
            segtext = seg.text
            if segtext == None:
              segtext = ''
            if seg.tail != None:
              segtext = segtext + ' '+seg.tail
            #print('===', segtext , 'is found')
            full_text.append(segtext)
            htagtext = nlp(segtext)
            for d in htagtext:
              w = ET.SubElement(sents,'wf')
              w.set('pos', d.pos_)
              w.set('lemma', d.lemma_)
              w.text = d.text
        else:
            #print(seg.tag, src, seg.text, seg.tail)
            src = seg.attrib['source'].split('/')[-1]
            #print(seg, src, seg.text)
            e,l = construct_lemma(src, nodes, entries)
            pos = seg.attrib['ana']
            
            ssrc = seg.attrib['source']
                  
            if pos in poslist:
              w = ET.SubElement(sents,'instance')
              slabel = senses[senses['nodeid'] == int(src)]['sense_label'].values[0]
              w.set('id',sid+".t"+str(c))
              w.set('pos_BELRL', pos)
              w.set('pos', poslabel)
              w.set('lemma', l)
              w.set('source', ssrc)
              #w.text = seg.text
              print(sid+".t"+str(c)+'\t'+slabel+'{}__1'.format(p), file=g)
              c += 1
            else:
              w = ET.SubElement(sents,'wf')
              w.set('pos_BELRL', pos)
              w.set('pos', pos)
              w.set('lemma', l)
              w.set('source', ssrc)
              #w.text = seg.text
              other_pos.append(pos)
              
            if seg.text != None:
              #print("*************************")
              full_text.append(seg.text.strip())
              if seg.text != ' ':
                  sdoc = nlp(seg.text)
                  w.text = seg.text
                  if sdoc[0].pos_ != None:
                    if w.tag != 'instance':
                      #print(w.tag)
                      w.set('pos', sdoc[0].pos_)
          

            
            #iseg = ''
            for iseg in seg.findall('./{http://www.tei-c.org/ns/1.0}seg'):
              isrc = iseg.attrib['source'].split('/')[-1]
              ie,il = construct_lemma(isrc, nodes, entries)
              ipos = iseg.attrib['ana']
              
              issrc = iseg.attrib['source']
              
              if ipos in poslist:
                w1 = ET.SubElement(sents,'instance')
                slabel = senses[senses['nodeid'] == int(isrc)]['sense_label'].values[0]
                w1.set('id',sid+".t"+str(c))
                w1.set('pos_BELRL', ipos)
                w1.set('pos',poslabel)
                w1.set('lemma', il)
                w1.text = iseg.text
                w1.set('source', issrc)
                print(sid+".t"+str(c)+'\t'+slabel+'{}__1'.format(p), file=g)
                c += 1
              else:
                w1 = ET.SubElement(sents,'wf')
                w1.set('pos_BELRL', ipos)
                w1.set('pos', ipos)
                w1.set('lemma', il)
                w1.set('source', issrc)
                other_pos.append(pos)
                w1.text = iseg.text
                  
              if iseg.text != None:
                  full_text.append(iseg.text)
                  isdoc = nlp(iseg.text)
                  if isdoc[0].pos_ != None:
                    if w1.tag != 'instance':
                      #print(w1.tag)
                      w1.set('pos', isdoc[0].pos_)     
              
              if iseg.tail != None:
                  full_text.append(iseg.tail)
                  w.text = iseg.tail
              else:
                  if not seg.text:
                      w.text = iseg.text
                  #print("found tail", iseg.tail)
            
            #if not seg.text and not iseg
            
            if seg.tail != None:
              full_text.append(seg.tail.strip())
              doc_tail = nlp(seg.tail)
              for d in doc_tail:
                w = ET.SubElement(sents,'wf')
                w.set('pos', d.pos_)
                w.set('lemma', d.lemma_)
                w.text = d.text
      
      print(cid, r.attrib['source'], ' '.join(full_text),sep='\t', file=fp)            
    
    c +=1
    #if i == 10:
      #print(cid, full_text)
      #break
    #if cid == citid:
    #  break

  #print()
  print(prettify(corpus), file=f)
  print('corpus formatted and saved...')
  f.close()
  g.close()
  fp.close()

  print('files saved...')

if __name__ == '__main__':

  import argparse, pathlib

  parser = argparse.ArgumentParser(description="convert belrl to semeval...")
  parser.add_argument('--pos', required=True, help='pos name required')
  parser.add_argument('--save_dir', type=pathlib.Path, help='location to save files')
  parser.add_argument('--home_dir', type=pathlib.Path, help='location of wsd directory')
  parser.add_argument('--xml_path', type=pathlib.Path, help='location to xml files')
  parser.add_argument('--sense_dir', type=pathlib.Path, help='location to senses file')
  parser.add_argument('--version', help='corpora version required')
  

  args = parser.parse_args()

  #wsd_path = '/home/amansinha/wsd-master/copy/wsd/'
  #xml_path = wsd_path + 'data/BEL-RL-fr/version_12.18.20/xmlPOS/'
  #save_dir = '/home/amansinha/Downloads/'#
  #sense_dir = xml_path + 'meta/'
  
  nodes = pd.read_csv(args.home_dir / 'data/version_31iii21/RL-fr/ls-fr-spiderlex/01-lsnodes.csv', sep='\t')
  entries = pd.read_csv(args.home_dir / 'data/version_31iii21/RL-fr/ls-fr-spiderlex/02-lsentries.csv', sep='\t')
  senses = pd.read_csv(args.sense_dir / 'nodeid_senselabel_31iii21.dat', sep='\t')



  convert_format(args.pos, args.version, args.xml_path, args.save_dir, nodes, entries, senses)
