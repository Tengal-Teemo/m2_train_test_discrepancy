

from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import os
import argparse

from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import pdb
import torch
from tabulate import tabulate

from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from typing import Optional, cast
from datetime import datetime

import torch
import torch.nn.functional as F

from src.embeddings.create_LoCo import load_tau_scrolls_for_summ_screen_fd_gov_report_qmsum, load_qasper, load_long_bench
from src.embeddings.dres import DenseRetrievalExactSearch as DRES

from embeddings_inference import OpenAI_Encoder, Voyager_Encoder, Cohere_Encoder, M2_BERT_Encoder, Together_Encoder, MiniLM6Encoder, M2_BERT_Raw_encoder

import time
from sentence_transformers import SentenceTransformer

######################################################################

from beir import util, LoggingHandler
import pathlib, os
import pandas as pd
from beir.datasets.data_loader import GenericDataLoader
import logging

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

######################################################################

import argparse

parser = argparse.ArgumentParser(description='Your program description here.')

model_options = [
    'm2',
    'm2-non-ft',
    'bm25',
    'minilmv6',
    'sentence-bert',
    'openai',
    'voyager',
    'cohere'
]

'''
suggested model names:
sentence_bert_model: "BAAI/bge-large-en-v1.5"
openai_embedding_model: "text-embedding-ada-002"
voyager_embedding_model: "voyage-01"
cohere_embedding_model: "embed-english-v3.0"
'''

# Boolean flags
parser.add_argument('--model', type=str, default='m2', choices=model_options, help='Model type')
parser.add_argument('--model-name', type=str, default="togethercomputer/m2-bert-80M-32k-retrieval", help='Model name')

parser.add_argument('--iterations', type=int, default=100, help='how many subsets to take from train')

parser.add_argument('--together-api', action='store_true', help='Use Together API')

# File paths
parser.add_argument('--yaml-file', type=str, default="yamls/embeddings/m2-bert-80M-32k-retrieval.yaml", help='Path to YAML file')
parser.add_argument('--checkpoint', type=str, help='M2 pretrained checkpoint')

# Integer argument
parser.add_argument('--batch-size', type=int, default=1, help='Batch size for encoding')

parser.add_argument('--revision', type=str, default=None, help='HF Commit to use')

# Baselines
parser.add_argument('--perform-BM25-and-reranking-with-BGE', action='store_true', help='Perform BM25 and reranking with BGE')

args = parser.parse_args()

# Model Selection
use_M2_BERT = args.model == 'm2'
use_M2_BERT_NO_FT = args.model == 'm2-non-ft'
use_BM25 = args.model == 'bm25'
use_MINILMV6 = args.model == 'minilmv6'
use_sentence_BERT_model = args.model == 'sentence-bert'
use_OpenAI = args.model == 'openai'
use_Voyager = args.model == 'voyager'
use_Cohere = args.model == 'cohere'

use_Together_API = args.together_api
if use_Together_API:
    try:
        TOGETHER_API_KEY = os.environ['TOGETHER_API_KEY']
    except:
        'Please set your Together API key as an environment variable called TOGETHER_API_KEY'

yaml_file = args.yaml_file
checkpoint = args.checkpoint

if use_M2_BERT and not use_Together_API and checkpoint is None:
    checkpoint = hf_hub_download(
        repo_id=args.model_name,
        filename="model.bin" if args.revision is not None else "pytorch_model.bin",
        revision=args.revision
    )

batch_size_for_encoding = args.batch_size
perform_BM25_and_reranking_with_BGE = args.perform_BM25_and_reranking_with_BGE

# dataset_name: str, split: str, document_column: str, query_column: str, subset=None

tau_scrolls_summ_screen_fd_config = ("tau/scrolls", "", "input", "output", "summ_screen_fd", "scrolls/summ_screen_fd")
tau_scrolls_gov_report_config = ("tau/scrolls", "", "input", "output", "gov_report", "scrolls/gov_report")
tau_scrolls_qmsum_config = ("tau/scrolls", "", "input", "output", "qmsum", "scrolls/qmsum")
qasper_title_config = ("qasper", "", "full_text", "title", None, "quasper/title")
qasper_abstract_config = ("qasper", "", "full_text", "abstract", None, "quasper/abstract")

total_datasets = [
    tau_scrolls_summ_screen_fd_config, tau_scrolls_qmsum_config, qasper_title_config
]

column_names = ["Dataset", "NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10", "NDCG@100", "NDCG@1000"]
rows = []

with open(yaml_file) as f:
    yaml_cfg = om.load(f)
cfg = yaml_cfg

cfg = cfg.model

if use_M2_BERT and not use_Together_API:
    print("Model YAML Used")
    print(yaml_file)

######################################################################

document_statistics_columns = ['Dataset', "Query Average Length", "Document Average Length",
                               "Query Median Length", "Document Median Length",
                               "Query Min. Length", "Query Max. Length", 
                               "Document Min. Length", "Document Max. Length"]
document_statistics_rows = [document_statistics_columns]


# define
def calculate_ndcg_10(corpus, qrels, queries, model):
    # Absolutely disgusting scuffed mess
    indexes = random.sample(range(len(corpus)), len(validation_corpus))
    qkeys = ['Query_' + str(index) for index in indexes]
    ckeys = ['Passage_' + str(index) for index in indexes]

    # sub_corpus = dict(zip(ckeys, [list(corpus.values())[i] for i in indexes]))
    # sub_qrels = dict(zip(qkeys, [list(qrels.values())[i] for i in indexes]))
    # sub_queries = dict(zip(qkeys, [list(queries.values())[i] for i in indexes]))

    sub_corpus = dict(zip(ckeys, [corpus[key] for key in ckeys]))
    sub_qrels = dict(zip(qkeys, [qrels[key] for key in qkeys]))
    sub_queries = dict(zip(qkeys, [queries[key] for key in qkeys]))


    if use_BM25:
        from beir.retrieval.search.lexical import BM25Search as BM25

        #### Provide parameters for elastic-search
        hostname = "http://localhost:9200"
        index_name = "loco-" + dataset[0].replace("/", "-") + "-" + (
            dataset[-2] if dataset[-2] is not None else dataset[-3])
        initialize = True  # True, will delete existing index with same name and reindex all documents

        model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)

    retriever = EvaluateRetrieval(model, score_function="cos_sim")

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(sub_corpus, sub_queries)

    logging.info("Dataset Evaluated: " + str(dataset[0]) + "_" + str(dataset[4]) + "_" + str(dataset[3]))

    ndcg, _map, recall, precision = retriever.evaluate(sub_qrels, results, retriever.k_values)
    print("NDCG")
    print(ndcg)
    logging.info("--------------------------------------------------------------")

    ndcg10 = ndcg['NDCG@10']
    return ndcg10


# train and validation dataset generation
for dataset in total_datasets:
        
    print(f"Starting on {dataset[0]}_{dataset[4]}_{dataset[3]}!")
    dataset_name = f'{dataset[0]}_{dataset[4]}_{dataset[3]}'

    if dataset[0] == "tau/scrolls" and dataset[4] == "summ_screen_fd":
        train_corpus, train_queries, train_qrels = load_tau_scrolls_for_summ_screen_fd_gov_report_qmsum(dataset[0],
                                                                                                        'train',
                                                                                                        dataset[2],
                                                                                                        dataset[3],
                                                                                                        dataset[4])
        validation_corpus, validation_queries, validation_qrels = load_tau_scrolls_for_summ_screen_fd_gov_report_qmsum(
            dataset[0], 'validation', dataset[2], dataset[3], dataset[4])
    elif dataset[0] == "tau/scrolls" and dataset[4] == "gov_report":
        train_corpus, train_queries, train_qrels = load_tau_scrolls_for_summ_screen_fd_gov_report_qmsum(dataset[0],
                                                                                                        'train',
                                                                                                        dataset[2],
                                                                                                        dataset[3],
                                                                                                        dataset[4])
        validation_corpus, validation_queries, validation_qrels = load_tau_scrolls_for_summ_screen_fd_gov_report_qmsum(
            dataset[0], 'validation', dataset[2], dataset[3], dataset[4])
    elif dataset[0] == "tau/scrolls" and dataset[4] == "qmsum":
        train_corpus, train_queries, train_qrels = load_tau_scrolls_for_summ_screen_fd_gov_report_qmsum(dataset[0],
                                                                                                        'train',
                                                                                                        dataset[2],
                                                                                                        dataset[3],
                                                                                                        dataset[4])
        validation_corpus, validation_queries, validation_qrels = load_tau_scrolls_for_summ_screen_fd_gov_report_qmsum(
            dataset[0], 'validation', dataset[2], dataset[3], dataset[4])
    elif dataset[0] == "qasper":
        train_corpus, train_queries, train_qrels = load_qasper(dataset[0],
                                                               'train',
                                                               dataset[2],
                                                               dataset[3],
                                                               dataset[4])
        validation_corpus, validation_queries, validation_qrels = load_qasper(
            dataset[0], 'test', dataset[2], dataset[3], dataset[4])
    else:
        print("LoCo Dataset not found!")
        assert False

    if use_OpenAI:
        print("Initializing OpenAI Encoder!")
        openai_encoder = OpenAI_Encoder(embedding_model=args.model_name)
        model = DRES(openai_encoder, batch_size=batch_size_for_encoding)
    elif use_Voyager:
        print("Initializing Voyager Encoder!")
        openai_encoder = Voyager_Encoder(embedding_model=args.model_name)
        model = DRES(openai_encoder, batch_size=batch_size_for_encoding)
    elif use_Cohere:
        print("Initializing Cohere Encoder!")
        openai_encoder = Cohere_Encoder(truncation="END", embedding_model=args.model_name)
        model = DRES(openai_encoder, batch_size=batch_size_for_encoding)
    elif not use_M2_BERT and use_sentence_BERT_model:
        model = DRES(models.SentenceBERT(args.model_name), batch_size=batch_size_for_encoding)
    elif use_MINILMV6:
        minilm_encoder =MiniLM6Encoder(cfg=cfg)
        model = DRES(minilm_encoder, batch_size=batch_size_for_encoding)
    elif use_BM25:
        model = None
        pass
    elif use_M2_BERT_NO_FT:
        m2_encoder_no_ft = M2_BERT_Raw_encoder(cfg=cfg)
        model = DRES(m2_encoder_no_ft, batch_size=batch_size_for_encoding)
    else:
        if use_Together_API:
            m2_encoder = Together_Encoder(cfg=cfg, api_key=TOGETHER_API_KEY, together_model_name=args.model_name)
            model = DRES(m2_encoder, batch_size=batch_size_for_encoding)
        else:
            m2_encoder = M2_BERT_Encoder(checkpoint=checkpoint, cfg=cfg)
            model = DRES(m2_encoder, batch_size=batch_size_for_encoding)

    import random
    import tqdm

    current_row = [dataset[0]]

    # Calculate Rows
    for test in tqdm.tqdm(range(0, args.iterations)):
        try:
            current_row.append(calculate_ndcg_10(train_corpus, train_qrels, train_queries, model))
        except:
            # Handles rare case of Elasticsearch breaking
            print('broke: ')
            current_row.append(-1.0)
            continue
    current_row.append(calculate_ndcg_10(validation_corpus, validation_qrels, validation_queries, model))

    rows.append(current_row)

args.revision = '-'+str(args.revision)

row_array = np.array(rows).T
np.save(f'results/{args.iterations}-{args.model}{args.revision}.npy', row_array)

# df = pd.DataFrame(rows)
#
# print("------------------------------------------------")
# print(tabulate(df, tablefmt="grid"))
# print("------------------------------------------------")
