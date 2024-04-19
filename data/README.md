# Instruction to organize the data to run the models

Create  ortolang/ and add dataset folder nountmp/ or verbtmp/.

Add the lf file ```15-lslf-rel_boost.csv``` in data/ folder.
Or ask for it using email ```asinha@atilf.fr```


## Preprocessing Pipeline

Needed in data
- BEL-RL-Fr/XML-POS/BEL-RL-fr_AllSources_nouns.xml and BEL-RL-fr_AllSources_verbs.xml (from version 31iii21)
- RL-fr version 2.1
- mkdir ortolang

```
python3 scripts/utils.py --name ortolang --version 1 --save_dir ~/GraphWSD/data/ortolang/ 
                         --xml ~/GraphWSD/data/BEL-RL-fr/XML-pos/ --write --keep --pos noun
```
