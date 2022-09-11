#encoder.py
from abc import abstractmethod
import torch
import torch.nn.functional as F
import torch.nn as nn

class WSDEncoder(nn.Module):
	""" Generic Class to run a model on WSD dataset """

	def __init__(self, model, tokenizer):
		super(WSDEncoder, self).__init__()
		self.model = model
		self.tokenizer = tokenizer

	@abstractmethod
	def encode(self, seq, padding=None):
		""" This method should use the tokenizer to encode sequence and add padding if specified"""
		pass

	@abstractmethod
	def forward(self, inputs):
		""" This method should run the model on inputs and outputs context vectors """
		pass

	@abstractmethod
	def collate_fn(self, inputs):
		""" This method should collate multiple inputs into a single a batch """
		pass

class WSDEncoder1():
	def __init__(self, model, tokenizer):
		self.tokenizer = tokenizer
		self.model = model
		
	def encode(self, instance, padding=50):
		
		# ids, span [start, end]
		tok2span = {}
		ids = []
		spans = []
		start, end = 0,0
		
		
		ids.append(self.tokenizer.cls_token_id)
		end +=1
		tok2span['[CLS]'] = [start, end]
		spans.append([start, end])
		
		start = end
		
		for idx, token in enumerate(instance.context):
			#print(token)
			tokenized = self.tokenizer.encode(token, add_special_tokens=False)
			end += len(tokenized)
			tok2span[token] = [start, end]
			spans.append([start, end])
			ids.extend(tokenized)
			
			start = end
			
		ids.append(self.tokenizer.sep_token_id)
		end +=1
		tok2span['[SEP]'] = [start, end]
		spans.append([start, end])
			
		ids = torch.tensor(ids).to(self.model.device)
			
		# padding check
		if padding and len(ids)< padding:
			ids = F.pad(ids, (0,padding -len(ids)),value=self.tokenizer.pad_token_id)
			
		att_mask = torch.ne(ids, self.tokenizer.pad_token_id).to(self.model.device)
		spans.extend([[end, end] for i in range(len(ids)-len(spans))])
		
		spans = torch.tensor(spans).to(self.model.device)
		
		#instance = 'creusé'
		
		return ids, att_mask, spans, tok2span, instance
	
	def collate_fn(self, inputs):
		""" Collate inputs into a single batch """

		tok_ids, att_mask, span, tok2span, instance = [torch.stack(x) for x in list(zip(*inputs))]

		return (tok_ids, att_mask, span, tok2span, instance)


	def forward(self, inputs, rtype ='word'):
		
		# 1x complete_len , 1x complete_len, 1x2x complete_len, unique_lemmas
		tok_ids, att_mask, spans, tk2span, instance = inputs

		# batch x padded_len x 768
		with torch.autograd.no_grad():
			output = self.model(tok_ids.unsqueeze(0), attention_mask = att_mask.unsqueeze(0))[0]

			target = []
			if rtype == 'word':
				#print('Looking for :',instance.word_form)
				output = output.squeeze(0)
				s  = instance.word_form.split(' ')
				for k in tk2span.keys():
					if s[-1] in k:
						target.append(tk2span[k])
					#    print('found: ',k)
						break
				#print('got:',target)
				target = [list(range(ll[0], ll[1]+1)) for ll in target]
				target = [l for ll in target for l in ll]
				target = list(set(target))
				#print('final:', target)
				vecs = torch.mean(output[target].unsqueeze(0), axis=1)
				output = vecs

			else:
				# rtype == 'sentence'
				# 1x1xcomplete_len
				nbpe = spans[:,1] - spans[:,0]
				# 1x1xcomplete_len
				mask = nbpe.ne(0)
				# 1x1x(complete_len-padding)
				nbpe = nbpe[mask]


				indices = torch.arange(nbpe.size(0), device=output.device).repeat_interleave(nbpe)
				avg_vec = torch.zeros(nbpe.size(0), output.size(2), device=output.device)
				avg_vec.index_add_(0, indices, output[att_mask.unsqueeze(0)])
				avg_vec.div_(nbpe.view(nbpe.size(0),1))

				output_ = torch.zeros_like(output)
				output_[mask.unsqueeze(0)] = avg_vec
				output = torch.mean(output_[:,1:(nbpe.size(-1)-1),:], axis=1)
			
			return output#2, output_, avg_vec, mask





class WSDEncoder2(WSDEncoder):
	def __init__(self, *args, **kwargs):
		super(WSDEncoder2, self).__init__(*args, **kwargs)
		#self.model = model
		#self.tokenizer = tokenizer


	def encode(self, seq, padding=200):

		tok2span = {}
		tok_ids = []
		spans = []
		start, end = 0,0

		# cls token
		tok_ids.append(self.tokenizer.cls_token_id)
		end += 1


		tok2span = [(start, end)]
		start = end

		# iterate over sequence and encode on token ids
		for tok_id, tok in enumerate(seq):
			tok_encoding = self.tokenizer.encode(tok, add_special_tokens=False)
			end += len(tok_encoding)
			tok2span.append([start, end])
			start = end
			tok_ids.extend(tok_encoding)

		# add sep token
		tok_ids.append(self.tokenizer.sep_token_id)
		end += 1
		tok2span.append([start, end])

		# token ids to tensor
		tok_ids = torch.tensor(tok_ids)

		# padding
		if padding and len(tok_ids) < padding:
			tok_ids = F.pad(tok_ids, (0,padding-len(tok_ids)),value=self.tokenizer.pad_token_id)

		# attention mask on pad tokens
		att_mask = torch.ne(tok_ids, self.tokenizer.pad_token_id)
		tok2span.extend([[end,end] for x in range(len(tok_ids)-len(tok2span))])

		# span to tensor
		span = torch.tensor(tok2span)

		return tok_ids, att_mask, span


	def collate_fn(self, inputs):

		tok_ids, att_mask, spans =  [torch.stack(x) for x in list(zip(*inputs))]

		return (tok_ids, att_mask, spans)


	def forward(self, inputs, encoder_type='linear'):
		"""Run transformer model on inputs. Average bpes per token and remove cls and sep vectors"""

		tok_ids, att_mask, span = inputs
		#print('***********',tok_ids.shape, att_mask.shape, span.shape)
		with torch.autograd.no_grad():

			outputs  = self.model(tok_ids, attention_mask=att_mask, output_hidden_states=True) # mask is used for pad tokens
			#if encoder_type == 'linear':
			output = outputs[0]
			#else: # h1
			#output = torch.sum(torch.Tensor([outputs['hidden_states'][-i].detach().cpu().numpy() for i in range(1,5)]), dim=0)
			#print("output from encoder:",output.shape)

			# compute number of bpe per token
			first_bpe = span[:,:,0] # first bpe indice
			last_bpe = span[:,:,1] # last bpe indice
			n_bpe = last_bpe-first_bpe # number of bpe by token = first - last bpe from span

			# mask pad tokens
			mask = n_bpe.ne(0)
			n_bpe = n_bpe[mask] # get only actual token bpe

			# compute mean : sum up corresponding bpe then divide by number of bpe
			indices = torch.arange(n_bpe.size(0), device=self.model.device).repeat_interleave(n_bpe) # indices for index_add
			average_vectors = torch.zeros(n_bpe.size(0), output.size(2), device=self.model.device) # starts from zeros vector
			#print(f"checking indiices:{indices.shape}, mask_op: {output[att_mask].shape}, mask:{att_mask.shape}")
			average_vectors.index_add_(0, indices, output[att_mask].to(device=self.model.device)) # sum of bpe based in indices
			average_vectors.div_(n_bpe.view(n_bpe.size(0),1)) # divide by number of bpe

			output_ = torch.zeros_like(output, device=self.model.device) # new output vector to match outputsize
			output_[mask] = average_vectors

			output = output_[:,1:-1,:] # get rid of cls and sep

			return output


