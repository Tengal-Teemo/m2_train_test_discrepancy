#!/bin/bash

iterations=2

python3 loco_eval.py --model m2 --model-name togethercomputer/m2-bert-80M-2k-retrieval --together-api --batch-size 12 --iterations $iterations

#python3 loco_eval.py --model minilmv6 --batch-size 12 --iterations $iterations
#
#python3 loco_eval.py --model m2 --revision 35984ead7f82ebdd33ee9b1f23a1394934b9519e --model-name togethercomputer/m2-bert-80M-2k-retrieval --batch-size 12 --iterations $iterations --yaml-file yamls/embeddings/m2-bert-80M-2k-retrieval.yaml
#
#python3 loco_eval.py --model m2 --revision 6eca2df31193ee3101e7ff10e11e4a5a8075619b --model-name togethercomputer/m2-bert-80M-2k-retrieval --batch-size 12 --iterations $iterations --yaml-file yamls/embeddings/m2-bert-80M-2k-retrieval.yaml
#
#python3 loco_eval.py --model m2 --revision 7ebf4f26a79377d64f7f7c0bed02e84711de33c2 --model-name togethercomputer/m2-bert-80M-2k-retrieval --batch-size 12 --iterations $iterations --yaml-file yamls/embeddings/m2-bert-80M-2k-retrieval.yaml
#
#python3 loco_eval.py --model bm25 --batch-size 12 --iterations $iterations
