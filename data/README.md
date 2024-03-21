# Instruction to organize the data to run the models

Create  ortolang/ and add dataset folder nountmp/ or verbtmp/. 

Add the lf file ```15-lslf-rel_boost.csv``` in data/ folder.
Or ask for it using email ```asinha@atilf.fr```


## Preprocessing Pipeline

```
python scripts/utils.py --name ortolang --version 1 		
						--save_dir ~/Downloads/ortolang/ 
						--home ~/GraphWSD/ 
						--xml ~/GraphWSD/data/version_31iii21/BEL-RL-fr/V2-ORTOLANG/XML-POS/ --write --keep  
						--pos noun
```
