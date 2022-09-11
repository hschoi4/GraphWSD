# GraphWSD

Offical code repository: **Word Sense Disambiguation of French Lexicographical Examples Using Lexical Networks**

## Folder Description

`/data/` : contains instructions to organize the `data/` folder

`/scripts/` : contains individual script modules


## Steps to run the experiments:

1) Clone the repo : `git clone https://github.com/ATILF-UMR7118/GraphWSD.git`

2) Create the virtualenv

```
python3 -m venv wsdvenv
. wsdvenv/bin/activate
pip3 install --upgrade pip
cd GraphWSD/
pip3 install -r requirements.txt
```

3) Follow the instructions provided in `data/` folder

4) To run the models for NOUN/VERB wsd:

(a) STRUCT model

```
python3  ~/GraphWSD/scripts/wsd_ewiser.py \
        --data ~/GraphWSD/data/ortolang/nountmp/ \
        --save_dir ~/GraphWSD/scripts/ortolog/ \
        --num 100  --model_num onoun_ewiser_29061156 --mtype ewiser --save-model \
        --learning 0.001  --hidden 8000 --batch 64 --device cuda --embed 768 --lm camembert-base
```
(b) SEM model

```
python3  ~/GraphWSD/scripts/wsd_ewiser.py \
        --data ~/GraphWSD/data/ortolang/nountmp/ --num 100 \
        --save_dir ~/GraphWSD/scripts/ortolog/  --model_num onoun_seml_29061522 \
        --mtype ewiserc --save-model --batch 64  --device cuda --semantics\
          --hidden-dim 8000  --embed 768 --lm camembert-base
```


For any questions related to repository contact: `asinha@atilf.fr`
