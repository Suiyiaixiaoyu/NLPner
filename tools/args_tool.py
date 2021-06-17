import argparse

def getparse():
    parse = argparse.ArgumentParser()

    parse.add_argument('--data_path',default='data/',type=str)
    parse.add_argument('--bert_path',default='bert',type=str)
    parse.add_argument('--batch_size',default=32,type=str)
    parse.add_argument("--GPUNUM",default = 0,type=int)
    parse.add_argument("--max_steps", default=-1, type=int,)
    parse.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parse.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parse.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parse.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")
    parse.add_argument("--learning_rate", default=5e-6, type=float,
                        help="The initial learning rate for Adam.")
    parse.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parse.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")


    return parse