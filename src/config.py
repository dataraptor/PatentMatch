import os

class CFG:
    group = 'DeBERTa'    # Exp name
    name = 'BASE'      # Sub exp name
    amp = True
    
    backbone = 'microsoft/deberta-v3-small'
    n_epochs = 6
    batch_size = 16
    learning_rate = 2.0e-5
    train_last_nlayer = 0
    weight_decay = 0.01
    
    num_workers = len(os.sched_getaffinity(0))

