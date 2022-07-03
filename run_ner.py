from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (AdamW, RobertaConfig,
                          RobertaForTokenClassification, RobertaTokenizer,
                          get_linear_schedule_with_warmup)
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from seqeval import classification_report, f1_score

class Ner(RobertaForTokenClassification):

    def __init__(self, config, num_presence_labels, num_period_labels):
        super(Ner, self).__init__(config)

        self.num_presence_labels = num_presence_labels
        self.num_period_labels = num_period_labels

        self.presence_classifier = nn.Linear(config.hidden_size, num_presence_labels)
        self.period_classifier = nn.Linear(config.hidden_size, num_period_labels)

    def forward(
        self, 
        input_ids, 
        token_type_ids=None, 
        attention_mask=None, 
        labels=None, 
        presence_labels = None, 
        period_labels = None, 
        label_weights = None, 
        presence_label_weights = None, 
        period_label_weights = None, 
        loss_weights = [1/3]*3
        ):

        sequence_output = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        presence_logits = self.presence_classifier(sequence_output)
        period_logits = self.period_classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight = torch.FloatTensor(label_weights).cuda(), ignore_index=-100) 
            loss_fct_presence = nn.CrossEntropyLoss(weight = torch.FloatTensor(presence_label_weights).cuda() ,ignore_index=-100)
            loss_fct_period = nn.CrossEntropyLoss(weight = torch.FloatTensor(period_label_weights).cuda(),ignore_index=-100)
            loss = loss_weights[0]*loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss += loss_weights[1]*loss_fct_presence(presence_logits.view(-1, self.num_presence_labels), presence_labels.view(-1))
            loss += loss_weights[2]*loss_fct_period(period_logits.view(-1, self.num_period_labels), period_labels.view(-1))
            return loss
        else:
            return (logits,presence_logits,period_logits)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, presence_label_id, period_label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.presence_label_id = presence_label_id
        self.period_label_id = period_label_id

def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label, presence_label, period_label = [], [], []
    for index,line in enumerate(f):
        if len(line)==0 or line.startswith('-DOCSTART'):
            if len(sentence) > 0:
                data.append((sentence,presence_label,period_label,label))
                sentence = []
                label, presence_label, period_label = [], [], []
            continue
        if line[0]=="\n": continue # merge sentences to form a paragraph
        splits = line.split(' ')
        sentence.append(splits[0])
        presence_label.append(splits[1])
        period_label.append(splits[2])
        label.append(' '.join(splits[3:])[:-1])

    if len(sentence) >0:
        data.append((sentence,presence_label,period_label,label))
        sentence = []
        label, presence_label, period_label = [], [], []
    # print(len(data))
    return data

class NerProcessor(object):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, train_data_dir):
        """Gets the train set."""
        return self._create_examples(
            train_data_dir, "train")

    def get_dev_examples(self, val_data_dir):
        """Gets the dev set."""
        return self._create_examples(
            val_data_dir, "dev")

    def get_test_examples(self, test_data_dir):
        """Gets the test set."""
        return self._create_examples(
            test_data_dir, "test")

    def get_labels(self):
        return ['O', 'B-Housing Insecurity', 'B-Food Insecurity', 'B-Pain', 'B-Psychiatric Symptoms', 'B-Financial Insecurity', 'B-Transitions of Care', 'B-Suicide Outcome', 'B-Substance Abuse', 'B-Patient Disability', 'B-Barriers to Care', 'B-Violence', 'B-Social Isolation', 'B-Legal Problems', 'I-Housing Insecurity', 'I-Food Insecurity', 'I-Pain', 'I-Psychiatric Symptoms', 'I-Financial Insecurity', 'I-Transitions of Care', 'I-Suicide Outcome', 'I-Substance Abuse', 'I-Patient Disability', 'I-Barriers to Care', 'I-Violence', 'I-Social Isolation', 'I-Legal Problems']
    
    def get_presence_labels(Self):
        return ['O', 'B-n_y', 'B-y', 'I-n_y', 'I-y']

    def get_period_labels(self):
        return ['O', 'B-Current', 'B-notCurrent', 'I-Current', 'I-notCurrent']

    def _create_examples(self,input_file,set_type):
        lines = readfile(input_file)
        examples = []
        for i,(sentence,presence_label,period_label,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append({
                'guid':guid,
                'text_a':text_a,
                'text_b':text_b,
                'label':label,
                'presence_label':presence_label,
                'period_label':period_label})
        return examples

def convert_examples_to_features(examples, label_list, presence_label_list, period_label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    presence_label_map = {label : i for i, label in enumerate(presence_label_list)}
    period_label_map = {label : i for i, label in enumerate(period_label_list)}

    features = []
    for (ex_index,example) in tqdm(enumerate(examples),total=len(examples)):
        textlist = example['text_a'].split(' ')
        labellist = example['label']
        presence_labellist = example['presence_label']
        period_labellist = example['period_label']
        
        ntokens, label_ids, presence_label_ids, period_label_ids = [], [], [], []
        ntokens.append("<s>")
        label_ids.append(-100)             # Mapped id for [CLS]
        presence_label_ids.append(-100)    # Mapped id for [CLS]
        period_label_ids.append(-100)      # Mapped id for [CLS]
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            ntokens.extend(token)
            for m in range(len(token)):
                if m == 0:
                    label_ids.append(label_map[labellist[i]])
                    presence_label_ids.append(presence_label_map[presence_labellist[i]])
                    period_label_ids.append(period_label_map[period_labellist[i]])
                else:
                    label_ids.append(-100)
                    presence_label_ids.append(-100)
                    period_label_ids.append(-100)

        if len(ntokens) > max_seq_length - 1:
            ntokens = ntokens[0:(max_seq_length - 1)]
            label_ids = label_ids[0:(max_seq_length - 1)]
            presence_label_ids = presence_label_ids[0:(max_seq_length - 1)]
            period_label_ids = period_label_ids[0:(max_seq_length - 1)]

        ntokens.append("</s>")
        label_ids.append(-100)             # Mapped id for [SEP]
        presence_label_ids.append(-100)    # Mapped id for [SEP]
        period_label_ids.append(-100)      # Mapped id for [SEP]
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)  # attention mask
        segment_ids = [0] * len(input_ids) # token_type_ids
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(-100)
            presence_label_ids.append(-100)
            period_label_ids.append(-100)

        assert len(input_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              presence_label_id=presence_label_ids,
                              period_label_id=period_label_ids))
    return features

def eval(
    args, 
    eval_features, 
    model, 
    label_list, 
    presence_label_list, 
    period_label_list, 
    tokenizer, 
    device, 
    logger, 
    loss_weights = None, 
    eval_on = 'test'
    ):

        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            logger.info("***** Running evaluation on {:}*****".format(eval_on))
            logger.info("  Num examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            all_presence_label_ids = torch.tensor([f.presence_label_id for f in eval_features], dtype=torch.long)
            all_period_label_ids = torch.tensor([f.period_label_id for f in eval_features], dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_presence_label_ids,all_period_label_ids)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            y_true, y_pred = [], []
            y_presence_true, y_presence_pred = [], []
            y_period_true, y_period_pred = [], []
            label_map = {i : label for i, label in enumerate(label_list)}
            presence_label_map = {i : label for i, label in enumerate(presence_label_list)}
            period_label_map = {i : label for i, label in enumerate(period_label_list)}
            for input_ids, input_mask, segment_ids, label_ids, presence_label_ids, period_label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                # label_ids = label_ids.to(device)
                # presence_label_ids = presence_label_ids.to(device)
                # period_label_ids = period_label_ids.to(device)

                with torch.no_grad():
                    logits, presence_logits, period_logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

                logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
                logits = logits.detach().cpu().numpy()
                presence_logits = torch.argmax(F.log_softmax(presence_logits,dim=2),dim=2)
                presence_logits = presence_logits.detach().cpu().numpy()
                period_logits = torch.argmax(F.log_softmax(period_logits,dim=2),dim=2)
                period_logits = period_logits.detach().cpu().numpy()
                label_ids = label_ids.numpy()
                presence_label_ids = presence_label_ids.numpy()
                period_label_ids = period_label_ids.numpy()
                input_mask = input_mask.to('cpu').numpy()

                for i, (label, presence_label, period_label) in enumerate(zip(label_ids,presence_label_ids,period_label_ids)):
                    temp_1, temp_2 = [], []
                    for j,m in enumerate(label):
                        if label_ids[i][j] != -100: 
                            temp_1.append(label_map[label_ids[i][j]])
                            temp_2.append(label_map[logits[i][j]])
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    
                    temp_1, temp_2 = [], []
                    for j,m in enumerate(presence_label):
                        if presence_label_ids[i][j] != -100: 
                            temp_1.append(presence_label_map[presence_label_ids[i][j]])
                            temp_2.append(presence_label_map[presence_logits[i][j]])
                    y_presence_true.append(temp_1)
                    y_presence_pred.append(temp_2)
                    
                    temp_1, temp_2 = [], []
                    for j,m in enumerate(period_label):
                        if period_label_ids[i][j] != -100: 
                            temp_1.append(period_label_map[period_label_ids[i][j]])
                            temp_2.append(period_label_map[period_logits[i][j]])
                    y_period_true.append(temp_1)
                    y_period_pred.append(temp_2)
            
            file_save_dict = {'dev':None, 'test':'bio_sdoh_feb8'}
            report, macro_f1_score = classification_report(y_true, y_pred, digits=4, criteria = args.eval_criteria) 
            logger.info("\n%s", report)
            y_presence_pred = post_process_attribute_predictions(y_pred, y_presence_pred)
            report_presence, macro_f1_score_presence = classification_report(y_presence_true, y_presence_pred, digits=4, criteria = args.eval_criteria) 
            logger.info("\n%s", report_presence)
            y_period_pred = post_process_attribute_predictions(y_pred, y_period_pred)
            report_period, macro_f1_score_period = classification_report(y_period_true, y_period_pred, digits=4, criteria = args.eval_criteria) 
            logger.info("\n%s", report_period)

            writefile(y_true,'ytrue.json')
            writefile(y_pred,'ypred.json')
            writefile(y_presence_true,'y_presence_true.json')
            writefile(y_presence_pred,'y_presence_pred.json')
            writefile(y_period_true,'y_period_true.json')
            writefile(y_period_pred,'y_period_pred.json')


            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                writer.write(report)

            if loss_weights is not None:
                combined_f1_score = float(loss_weights[0])*macro_f1_score+float(loss_weights[1])*macro_f1_score_presence+float(loss_weights[2])*macro_f1_score_period
            else:
                combined_f1_score = None
            
            return combined_f1_score

def writefile(y, filename):
    with open(filename,'w') as f:
        json.dump(y,f)

def post_process_attribute_predictions(pred, attrib_pred):
    '''
    Replace attribute prediction with 'O' if ner prediction is 'O'
    '''
    post_processed_preds = []
    for i,j in zip(pred, attrib_pred):
        temp = []
        for token1, token2 in zip(i,j):
            if token1 == 'O': temp.append('O')
            else: temp.append(token2)
        post_processed_preds.append(temp)
    return post_processed_preds

def compute_weights(dist, alpha = 1):
    keys = sorted(dist.keys()) # maintain the order
    total = np.log10(alpha*sum(dist[k] for k in keys if k!=-100 and k!=0 ))
    weights = np.array([np.log10(dist[k]) for k in keys if k!=-100 and k!=0])
    return np.append(np.array([1]),np.maximum(1,total-weights)) # weight 0 for [CLS]/[SEP]/[PAD],1 for 'O' class

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_dir",
                        default=None,
                        type=str,
                        help="The training data dir.")
    parser.add_argument("--dev_data_dir",
                        default=None,
                        type=str,
                        help="The dev data dir.")
    parser.add_argument("--test_data_dir",
                        default=None,
                        type=str,
                        help="The test data dir.")
    parser.add_argument("--bert_model", default="bert-base-cased", type=str, required=True,
                        help="Any bert-based pre-trained model.")
    parser.add_argument("--task_name",
                        default="ner",
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--logfile",default=None,type=str,required=True,
                        help="The output log file.")


    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--eval_criteria",
                        default=None,
                        type=str,
                        required=True,
                        help="Evaluation criteria, either 'exact' or 'relaxed'.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--patience', type=int, default=5, 
                        help='patience for early stopping, use any negative value for early stopping')
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    
    logging.basicConfig(filename=args.logfile, filemode='a',
                    format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if args.eval_criteria not in ['exact','relaxed']:
        raise ValueError("Must be 'exact' or 'relaxed'.")
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    processor = NerProcessor()
    label_list = processor.get_labels()
    presence_label_list = processor.get_presence_labels()
    period_label_list = processor.get_period_labels()
    num_labels = len(label_list)
    num_presence_labels = len(presence_label_list)
    num_period_labels = len(period_label_list)

    tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = processor.get_train_examples(args.train_data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Prepare model
    config = RobertaConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
    model = Ner.from_pretrained(args.bert_model,
              from_tf = False,
              config = config,
              num_presence_labels = num_presence_labels, 
              num_period_labels = num_period_labels)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer = optimizer , 
		                                        num_warmup_steps = warmup_steps, 
		                                        num_training_steps = num_train_optimization_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    label_map = {i : label for i, label in enumerate(label_list)}
    if args.do_train:
        dev_examples = processor.get_dev_examples(args.dev_data_dir)
        test_examples = processor.get_test_examples(args.test_data_dir)

        train_features = convert_examples_to_features(
            train_examples, label_list, presence_label_list, period_label_list, args.max_seq_length, tokenizer)
        dev_features = convert_examples_to_features(
            dev_examples, label_list,presence_label_list, period_label_list,args.max_seq_length, tokenizer)
        test_features = convert_examples_to_features(
            test_examples, label_list,presence_label_list, period_label_list,args.max_seq_length, tokenizer)


        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_presence_label_ids = torch.tensor([f.presence_label_id for f in train_features], dtype=torch.long)
        all_period_label_ids = torch.tensor([f.period_label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_presence_label_ids,all_period_label_ids)
        
        flattened_label_ids, flattened_presence_label_ids, flattened_period_label_ids = [], [], []
        for f in train_features:
            flattened_label_ids += f.label_id
            flattened_presence_label_ids += f.presence_label_id
            flattened_period_label_ids += f.period_label_id

        label_dist = dict(Counter(flattened_label_ids))
        presence_label_dist = dict(Counter(flattened_presence_label_ids))
        period_label_dist = dict(Counter(flattened_period_label_ids))

        label_weights = compute_weights(label_dist, alpha = 5)
        presence_label_weights = compute_weights(presence_label_dist, alpha = 15)
        period_label_weights = compute_weights(period_label_dist, alpha = 15)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        best_f1 = 0.
        no_improvement = 0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            if epoch<3: loss_weights = [0.5, 0.25, 0.25]
            else: loss_weights = [1/3]*3
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, presence_label_ids, period_label_ids = batch  
                loss = model(
                    input_ids, 
                    token_type_ids = segment_ids, 
                    attention_mask = input_mask, 
                    labels = label_ids, 
                    presence_labels = presence_label_ids, 
                    period_labels = period_label_ids,
                    label_weights = label_weights, 
                    presence_label_weights = presence_label_weights, 
                    period_label_weights = period_label_weights, 
                    loss_weights = loss_weights
                    )
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                # if step==10:break
            
            logger.info("Epoch {:}, Loss: {:}".format(epoch, tr_loss))
            macro_f1_score = eval(args, dev_features, model, label_list, presence_label_list, period_label_list, tokenizer, device, logger, loss_weights, 'dev')
            if args.patience > 0: # Save model only when val performance improves and breaks at args.patience level
                if best_f1 < macro_f1_score:
                    best_f1 = macro_f1_score
                    no_improvement = 0
                    eval(args, test_features, model, label_list, presence_label_list, period_label_list, tokenizer, device, logger, loss_weights)
                    # Save a trained model and the associated configuration
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    model_to_save.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    label_map = {i : label for i, label in enumerate(label_list)}
                    model_config ={"bert_model":args.bert_model,"do_lower":args.do_lower_case,"max_seq_length":args.max_seq_length,
                                   "num_labels":len(label_list)+1,"label_map":label_map}
                    json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"))
                else:
                    no_improvement += 1
                    if no_improvement == args.patience:
                        break
            elif epoch==args.num_train_epochs: # Save model at the last epoch
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                label_map = {i : label for i, label in enumerate(label_list)}
                model_config ={"bert_model":args.bert_model,"do_lower":args.do_lower_case,"max_seq_length":args.max_seq_length,
                               "num_labels":len(label_list)+1,"label_map":label_map}
                json.dump(model_config,open(os.path.join(args.output_dir,"model_config.json"),"w"))
            else: # do nothing
                pass

    else:
        # Load a trained model and vocabulary that you have fine-tuned
        model = Ner.from_pretrained(args.output_dir,num_presence_labels = num_presence_labels,num_period_labels = num_period_labels)
        tokenizer = RobertaTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(device)

        test_examples = processor.get_test_examples(args.test_data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list,presence_label_list, period_label_list,args.max_seq_length, tokenizer)

        macro_f1_score = eval(args, test_features, model, label_list, presence_label_list, period_label_list, tokenizer, device, logger)


if __name__ == "__main__":
    main()
