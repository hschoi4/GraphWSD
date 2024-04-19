# wsd_ewiser.py
import numpy as np
import pandas as pd
import os
# import logging; logger = logging.getLogger(); logging.basicConfig(level=logging.INFO)
from tqdm import tqdm
import time

import torch
import torch_sparse, torch_scatter
from torch.utils import data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import utils
from activations import *

from transformers import AutoTokenizer, AutoModel
# encoder additions 
from encoder import WSDEncoder2
from utils import make_batches2

# from wsd_baseline_mlp import WSDData, Instance, WSDDataReader
from data import WSDData, Instance, WSDDataReader


class Ewiser(torch.nn.Module):
    def __init__(self, model_params, A=None, O=None, adjacency_trainable=False, renormalize=False):
        super(Ewiser, self).__init__()

        self.embed_dim = model_params['embed_dim']
        self.hidden_size = model_params['hidden_size']
        self.dropout = model_params['dropout']
        self.device = model_params['device']
        self.lemma2ids = dict()
        # model type: mtype
        self.mtype = model_params['mtype']
        self.renormalize = renormalize

        # self.n_tunable_layers = model_params.get('fine_tune_layers', None)
        # bert-base-multilingual-cased
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_params['lm'])  # camembert-base)#flaubert/flaubert_large_cased)#camembert-base)
        self.bert = AutoModel.from_pretrained(
            model_params['lm'])  # camembert-base#flaubert/flaubert_large_cased)#camembert-base)
        # encoder addition
        self.encoder = WSDEncoder2(self.bert, self.tokenizer)
        # ewiser implementation naive
        #######################################
        self.batchnorm = nn.BatchNorm1d(self.embed_dim, affine=False)
        # self.pretrained_O = False
        self.wt1 = nn.Linear(self.embed_dim, self.embed_dim, bias=True)  # ewiser
        self.wt2 = nn.Linear(self.embed_dim, model_params['nclasses'], bias=True)  # adaptation
        # definition information
        if not O:
            self.O = nn.Embedding(self.embed_dim, model_params['nclasses'])
        else:
            self.O = O
        self.swish = swish
        self.adjacency_trainable = adjacency_trainable
        # if self.pretrained_O:
        #   self.embed.weight.data.copy_(self.pretrained_weight)
        # else:
        # init_embedding(self.embed.weight)

        # relation information
        if A is not None:
            ######### ewiser A trainable ######
            # print('A',A)
            if not model_params['fragment']:
                ii, vv, size = self.unpack_sparse_tensor(A)
                ii = torch.nn.Parameter(ii, requires_grad=False)
                vv = torch.nn.Parameter(vv, requires_grad=self.adjacency_trainable)
                size = torch.nn.Parameter(size, requires_grad=False)
                print('===>', size)
                self.A = torch.nn.ParameterList([ii, vv, size])
                self._coalesce(self.A)

                if not self.renormalize:
                    self._initialize_to_1_over_n(self.A)
            else:

                self.pars = torch.nn.ParameterList([])
                self.A = []
                for each in A:
                    ii, vv, size = self.unpack_sparse_tensor(each)
                    ii = torch.nn.Parameter(ii, requires_grad=False)
                    vv = torch.nn.Parameter(vv, requires_grad=False)
                    size = torch.nn.Parameter(size, requires_grad=False)

                    each = [ii, vv, size]
                    self.pars.extend(each)
                    self._coalesce(each)
                    self.A.append(each)

                self.vec = torch.nn.Parameter(torch.tensor([1.0 for _ in range(len(self.A))]),
                                              requires_grad=self.adjacency_trainable)
                self.pars.append(self.vec)

            ###################################
            # self.A = self.unpack_sparse_tensor(A)
            print("A loaded ...")
            # print(self.A)
            # print("*****",self.A[1], self.A[-1])
        else:
            self.A = torch.ones(model_params['nclasses'], model_params['nclasses'])
        self.sm = nn.Softmax(dim=1)
        #######################################

        self.linear = nn.Sequential(nn.Linear(self.embed_dim, self.hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(p=self.dropout),
                                    nn.Linear(self.hidden_size, model_params['nclasses']))
        # uncomment for other berts
        # self.bert.pooler.dense.weight.requires_grad = False
        # self.bert.pooler.dense.bias.requires_grad = False

        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input):
        # print(self.tokenizer.tokenize(input))

        #### encoder addition ####
        inputs, wsd_data = input
        inputs = [x.to(self.device) for x in inputs]
        output = self.encoder(inputs, encoder_type=self.mtype)
        instance_ids = wsd_data[0]
        tgt = wsd_data[1]  # token indices
        batch_idx = wsd_data[-1]
        # print(batch_idx, tgt)
        output = output[batch_idx, tgt]
        # print("new_output_shape:", output.shape)
        ##########################

        # before encoder 
        # input = self.tokenizer(input, padding=True)
        # ids = torch.tensor(input['input_ids'], dtype=torch.long).to(self.device)
        # att_mask = torch.tensor(input['attention_mask'], dtype=torch.long).to(self.device)
        # output = self.bert(ids, attention_mask=att_mask)[0]
        # output = torch.mean(output[:, 1:-1, :],1)

        # ewiser implementation naive
        ###############################
        if 'ewiser' in self.mtype:
            # print ('Ewiser model running....')
            h0 = self.batchnorm(output)
            # print("hO", h0.shape)
            h1 = self.swish(self.wt2(h0))
            # print("h1", h1.shape)
            # Z = torch.matmul(h1, self.O.weight)
            # print("Z", Z.shape)

            Z = h1
            if self.mtype == 'ewiserc':
                # print("complete ewiser")
                neighbors = self._spmm(Z, self.A)
                if self.renormalize:
                    neighbor_sum = self._get_row_sum(self.A)
                    neighbors = neighbors / neighbor_sum.view(1, 1, -1)

                Q = neighbors + Z
                output = Q
                # print("Q:", Q.shape)
            elif self.mtype == 'ewiserc+':

                neighbors = 0
                # print('number of layers: ',len(self.A))
                # print(self.A[0])
                for i, m in enumerate(self.A):
                    ii, vv, size = m
                    vv_ = vv * self.vec[i]
                    m = (ii, vv_, size)
                    neighbors = neighbors + self._spmm(Z, m)

                    if self.renormalize:
                        neighbor_sum = self._get_row_sum(m)
                        neighbors = neighbors / neighbor_sum.view(1, 1, -1)

                Q = neighbors + Z
                output = Q

            else:
                # print("partial ewiser")
                output = Z

            # output = self.sm(Q)
            # output = Q
        ################################
        else:
            # print(self.mtype)
            # print ('Linear model running....')
            output = self.linear(output)
            # output = self.sm(self.linear(output))

        return output

    def _spmm(self, inp, params):
        ii, vv, size = params
        old_inp_size = inp.size()
        inp_flat_T = inp.view(-1, inp.size(-1)).t()  # H x D_0*D_1*...*D_n
        out_flat = torch_sparse.spmm(
            ii, vv,
            m=size[0], n=size[1],
            matrix=inp_flat_T
        ).t()
        out = out_flat.view(*old_inp_size)
        return out

    def unpack_sparse_tensor(self, sparse_tensor):
        pieces = (sparse_tensor._indices(), sparse_tensor._values(), torch.LongTensor(list(sparse_tensor.shape)))
        return pieces

    def _coalesce(self, params):
        ii, vv, size = params
        coalesced_ii, coalesced_vv = torch_sparse.coalesce(ii, vv, *size, op='max')
        ii.data = coalesced_ii
        vv.data = coalesced_vv
        return params

    def _get_row_sum(self, params):
        ii, vv, size = params
        row_sum = self._torch_scatter.scatter_add(vv, ii[0], dim_size=size[1])
        return row_sum

    def _get_col_sum(self, params):
        ii, vv, size = params
        col_sum = self._torch_scatter.scatter_add(vv, ii[1], dim_size=size[0])
        return col_sum

    def _initialize_to_1_over_n(self, params, sum='none'):
        if sum == 'none':
            return params
        ii, vv, size = params
        vv.data[:] = 1
        if sum == 'row':
            row_sum = self._get_row_sum(params)
            vv.data[:] = 1 / row_sum[ii[0]]
        elif sum == 'col':
            col_sum = self._get_col_sum(params)
            vv.data[:] = 1 / col_sum[ii[1]]
        return params


def give_same_lemma_labels_ids(lemmas, nclasses, labels):
    aa = []
    for l in lemmas:
        ids = []
        kids = []
        name = l.split('_')[4]
        for k, v in labels.items():

            if f'_{name}_' in k:
                ids.append(v)
                kids.append(k)

        # print(l, kids,':', ids, '\n')
        a = np.zeros(nclasses)
        a[ids] = 1
        aa.append(a)
    # print(aa)
    return np.asarray(aa, dtype=np.float64)


class Enet(Ewiser):
    def __init__(self, config, A=None):
        super(Enet, self).__init__(config, A)
        self.epochs = config['num_epochs']
        self.loss1 = nn.CrossEntropyLoss()
        self.loss2 = nn.NLLLoss()
        self.config = config
        self.labels = None
        self.baseline = Ewiser(self.config, A, renormalize=False, adjacency_trainable=config['adj_trainable']).to(
            self.config['device'])

    def train(self, traindata, testdata):

        if self.config['load_model']:
            return
        else:
            print('Starting training...')

        sm = nn.LogSoftmax(dim=1)

        # self.baseline.make_vocab(traindata)
        # print(self.baseline.lemma2ids)

        print(self.config, '\n')
        # logger.info(self.config, '\n')
        # best_loss = float('inf')
        # best_f1 = 0

        # encoder addition ####
        batches = make_batches2(traindata, self.encoder, batch_size=self.config['batch_size'], shuffle=True)
        #######################
        optimizer = optim.Adam(self.baseline.parameters(), self.config['learning_rate'])
        writer = SummaryWriter(f'{self.config["save_model_dir"]}/ewiser_{self.config["model_num"]}')
        early_stopping = utils.EarlyStopping(patience=5, verbose=True,
                                             save_path=f'runs/ewiser_{self.config["model_num"]}/checkpoint_{str(time.time())}.pt')

        all_loss = []
        steps = 0
        self.baseline.train()

        breakflag = False

        for epoch in range(self.epochs):
            eloss = []
            # for idx, batch in enumerate(traindata):
            for idx, batch in enumerate(batches):
                optimizer.zero_grad()
                steps += 1
                t1 = time.time()
                ## encoder addition #####
                # if len(batch[1][-1]) == 1: continue
                inputs, wsd_data = batch
                output = self.baseline(batch)  # batch['rcontext'])
                instance_ids = wsd_data[0]
                tgt = wsd_data[1]  # token indices
                batch_idx = wsd_data[-1]  # batch indices

                # target_output = output[batch_idx, tgt]
                #########################
                # before encoder
                # output = self.baseline(batch['rcontext'])
                ##########################
                # targets = torch.LongTensor([self.labels[l[0]] for l in batch["labels"]]).to(self.config['device'])
                targets = torch.LongTensor([self.labels[l[0]] for l in wsd_data[-2]]).to(self.config['device'])

                # loss = self.loss1(output, targets)
                loss = self.loss2(sm(output), targets)

                all_loss.append(loss.item())
                eloss.append(loss.item())
                t2 = time.time()

                if ((epoch * self.config['batch_size']) + idx) % 2 == 0:
                    print(
                        f"Epoch: {epoch + 1} Batch: {idx + 1}/ {len(traindata.instances) // self.config['batch_size']}, Loss {loss.item()} \t Avg time: {t2 - t1}seconds")
                    writer.add_scalar('Batch_training_loss', loss.item(), steps)
                    writer.add_scalar('Avg_training_loss', np.sum(all_loss) / len(all_loss), steps)

                loss.backward()
                optimizer.step()

                # best_loss = min(best_loss, loss.item())
                if idx % self.config['report_every'] == 0:
                    print('Making predictions ...')
                    t3 = time.time()
                    test_loss = self.predict(testdata, name=f'e{epoch}_b{idx}')
                    t4 = time.time()
                    print(f'Test loss {test_loss}, Avg prediction time: {t4 - t3} seconds')

            if self.config['early_stopping']:
                early_stopping(test_loss, self.baseline)
                if early_stopping.early_stop:
                    print("Early Stopping!")
                    break

        # change the checkpoint name as needed
        if self.config['save_model']:
            torch.save(self.baseline.state_dict(),
                       f"{self.config['save_model_dir']}/model_{self.config['model_num']}_e{epoch}_b{idx}.pth");
            print('model saved !!')
        #            if epoch == 1: break
        # writer.add_scalar('training_loss', np.mean(eloss), epoch+1)

        return np.sum(all_loss) / len(all_loss)

    def predict(self, testdata, name=None):

        all_loss = []
        acc, mask_acc = 0, 0
        predictions, masked_predictions, tki, tks = [], [], [], []
        all_labels = []
        actual_labels = []
        iids = []
        batches = make_batches2(testdata, self.encoder, batch_size=self.config['batch_size'])

        id2label = {v: k for k, v in self.labels.items()}

        # add the model name explicitly
        if self.config['load_model']:
            checkpoint = torch.load(f"{self.config['model_path']}")
            self.baseline.load_state_dict(checkpoint)
            print('Model loaded ...')

        else:
            print('Evaluating with trained model...')

        sm = nn.LogSoftmax(dim=1)
        m = nn.Softmax(dim=1)
        self.baseline.eval()
        with torch.no_grad():

            for idx, batch in tqdm(enumerate(batches)):

                inputs, wsd_data = batch
                output = self.baseline(batch)
                instance_ids = wsd_data[0]
                tgt = wsd_data[1]  # token indices
                batch_idx = wsd_data[-1]
                cl = np.asarray([self.labels[l[0]] for l in wsd_data[-2]])
                iids.extend(instance_ids)
                # print(cl)
                o = np.argmax(output.detach().cpu().numpy(), axis=1).tolist()
                predictions.extend(o)
                opt = torch.topk(m(output), 10, dim=1)
                tks.extend(opt[0].detach().cpu().numpy());
                tki.extend(opt[1].detach().cpu().numpy())
                targets = torch.LongTensor(cl).to(self.config['device'])

                all_labels.extend(cl.tolist())
                # print('Targets:',targets[:10], '\nOutputs:', o[:10])
                mask = give_same_lemma_labels_ids([l[0] for l in wsd_data[-2]], self.config['nclasses'], self.labels)
                # print(output.shape, mask.shape, cl.shape, )

                # masked softmax for non zero elements
                tsr = torch.tensor(mask * output.detach().cpu().numpy(), dtype=torch.float)
                nzmask = ((tsr.numpy() != 0) - 1) * 9999  # for -inf
                result = (tsr + nzmask).softmax(dim=-1)

                mo = np.argmax(result, axis=1).numpy()
                match = mo == cl

                masked_predictions.extend(mo)

                if isinstance(match, bool):
                    match = [match]
                mask_acc += sum(match)

                loss = self.loss2(sm(output), targets)

                all_loss.append(loss.item())

            # predictions = np.asarray(predictions)
            # masked_predictions = np.asarray(masked_predictions)
            # iids = np.asarray(iids)
            # print(len(iids), len(all_labels), len(masked_predictions), len(predictions))
            # print([(iid,ap,mp,p) for iid, ap, mp, p in zip(iids, all_labels, masked_predictions, predictions)])
            fp = open(
                f"{self.config['save_model_dir']}ewiser_{self.config['model_num']}/predictions_{self.config['model_num']}_{name}.txt",
                'w')
            print('instance_id,first_label,pred,masked_pred', file=fp)
            for iid, t, l, ml in zip(iids, all_labels, predictions, masked_predictions):
                #   print(id2label[t], id2label[l], id2label[ml])
                print(iid, id2label[t], id2label[l], id2label[ml], sep=',', file=fp)
                acc += (int(l == t))
                # print(acc)
            fp.close()
            if 'test' in name:
                fp2 = open(
                    f"{self.config['save_model_dir']}ewiser_{self.config['model_num']}/analyze_pred_{self.config['model_num']}_{name}.txt",
                    'w')
                for iid, t, i, s in zip(iids, all_labels, tki, tks):
                    print(iid, id2label[t], *[f'{id2label[ii]}/{ss}' for ii, ss in zip(i, s)], file=fp2)
                fp2.close()

        # acc /= len(all_labels)

        print(f"Correct: {acc} out of {len(all_labels)}")
        print(f"With mask Correct: {mask_acc} out of {len(all_labels)}")

        return sum(all_loss) / len(batches)


def main(args):
    utils.seed_everything()

    traindata = WSDDataReader('train').read_from_data_dirs([args.data_dir], add_context_to_instance=True, keep_mwe=True)
    devdata = WSDDataReader('dev').read_from_data_dirs([args.data_dir], add_context_to_instance=True, keep_mwe=True)
    testdata = WSDDataReader('test').read_from_data_dirs([args.data_dir], add_context_to_instance=True, keep_mwe=True)

    labels = set()  # traindata.get_labels()) | set(devdata.get_labels())

    senses = set()

    label2nid = dict()

    # Get only senses in the train/dev/test data to create the matrix
    for d in [traindata, devdata, testdata]:
        for sense in d.instances:
            nodeid = sense.instance_src  # .split('/')[-1]
            senses.add(nodeid)
            labels.add(sense.first_label)
            if sense.first_label not in label2nid:
                label2nid[sense.first_label] = {nodeid}
            else:
                label2nid[sense.first_label].add(nodeid)
            # print(nodeid, sense.first_label)

    senses = sorted(senses)

    for k, v in label2nid.items():
        if len(v) > 1:
            print(k, v)

    print('totals labels:', len(set(labels)))
    print('here is no of senses:', len(set(senses)))

    assert len(set(labels)) == len(set(senses))

    # print(len(senses))
    A = utils.load_adjacency_mat(senses, semantics=args.semantics, fragment=args.fragment, cosine=args.cosine)
    print(A)
    exit(0)
    #print(A.coalesce().indices().shape)
    #print(A.coalesce().values().shape)


    # return
    # O = utils.load_sense_mat(senses)

    # print(A, O.shape)

    model_params = {'embed_dim': int(args.embed_dim),  # 768,#1024,#768,
                    'hidden_size': int(args.hidden_dim),
                    'dropout': 0,
                    'num_epochs': int(args.num_epochs),
                    'batch_size': int(args.batch_size),
                    'learning_rate': args.learning_rate,
                    'device': args.device,
                    'mtype': args.mtype,
                    'load_model': args.load_model,
                    'save_model_dir': args.save_dir,
                    'model_path': args.model_path,
                    'model_num': args.model_num,
                    'nclasses': len(labels),
                    'lm': args.lm,
                    'save_model': False,
                    'early_stopping': True,
                    'report_every': 25,
                    'patience': args.patience,
                    'semantics': args.semantics,
                    'adj_trainable': args.trainable,
                    'report_every': args.report_every,
                    'patience': args.patience,
                    'semantics': args.semantics,
                    'fragment': args.fragment,
                    'cosine': args.cosine,
                    }

    clf = Enet(model_params, A)

    clf.labels = {l: i for i, l in enumerate(labels)}
    clf.config['nclasses'] = len(labels)

    clf.train(traindata, devdata)

    print('#' * 40, '---devTesting---', '#' * 40)
    clf.predict(devdata, name='devtest')
    print('#' * 40)
    print('*' * 40, '----Testing-----', '*' * 40)
    clf.predict(testdata, name='test')
    # print(clf.baseline)



if __name__ == '__main__':
    import argparse, pathlib

    parser = argparse.ArgumentParser(description="Running MLP...")
    parser.add_argument('--num_epochs', default=2,
                        help='Number of Epochs')
    parser.add_argument('--data_dir', required=False, type=pathlib.Path,
                        help='location to dataset files', default='/home/hchoi/GraphWSD/data/ortolang/noun/')
    parser.add_argument('--device', default=torch.device('cpu'),
                        help='gpu or cpu')
    parser.add_argument('--batch_size', default=32,
                        help='batch size')
    parser.add_argument('--mtype', default='linear',
                        help='Type of model')
    parser.add_argument('--load_model', action='store_true',
                        help='To load and run model')
    parser.add_argument('--save_dir', default='/home/hchoi/wsd-grid/wsd/scripts/runs/',
                        help='model saving dir')
    parser.add_argument('--model_num', default=1, type=str,
                        help='saved model identifier')
    parser.add_argument('--save-model', action='store_true',
                        help='to save the model')
    parser.add_argument('--model_path', help='full path of model to load')
    parser.add_argument('--early-stopping', action='store_true',
                        help='to save checkpoint early')
    parser.add_argument('--report-every', default=10, type=int,
                        help='Log report period')
    parser.add_argument('--hidden-dim', default=100, type=int,
                        help='Hidden layer size')
    parser.add_argument('--embed-dim', default=768, type=int, help='LM embedding dim')
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help="Learning rate of loss optimization")
    parser.add_argument('--patience', default=5,
                        help='patience for early_stopping')
    parser.add_argument('--lm', default='camembert-base', help='name of huggingface lm')
    parser.add_argument('--semantics', action='store_true',
                        help='To load semantics A')
    parser.add_argument('--cosine', action='store_true',
                        help='To load A with cosine similarity')
    parser.add_argument('--trainable', action='store_true', help='to train adjancency')
    parser.add_argument('--fragment', action='store_true')
    args = parser.parse_args()

    main(args)
