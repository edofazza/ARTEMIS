# ARTEMIS: Animal Recognition Through Enhanced Multimodal Integration System

Code implementation for “ARTEMIS: animal recognition through enhanced multimodal integration system” paper.

To run the training code for ARTEMIS simple run:

```bash
python artemis_code/main.py --model='artemis' --total_length=16 --num_workers=2 --batch_size=16 --recurrent='none' --residual --backboneresidual 
```

Specifying which residual connection to add and if to use feature alignment by passing `--contrastive='cca'` and changing the model to `artemis_contrative`.

To run the embeddings: 

```bash
python artemis_code/main_ensemble.py --num_workers=2 --batch_size=32 --type='ga' --k=0 --train
```