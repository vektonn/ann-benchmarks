#!/bin/bash

python3 run.py --dataset glove-50-angular --algorithm vkt-faiss-bf
python3 run.py --dataset glove-50-angular --algorithm vkt-faiss-bf --batch
python3 run.py --dataset glove-50-angular --algorithm vektonn-bf
python3 run.py --dataset glove-50-angular --algorithm vektonn-bf --batch

python3 run.py --dataset glove-100-angular --algorithm vkt-faiss-bf
python3 run.py --dataset glove-100-angular --algorithm vkt-faiss-bf --batch
python3 run.py --dataset glove-100-angular --algorithm vektonn-bf
python3 run.py --dataset glove-100-angular --algorithm vektonn-bf --batch

python3 run.py --dataset glove-50-angular --algorithm vkt-faiss-hnsw
python3 run.py --dataset glove-50-angular --algorithm vkt-faiss-hnsw --batch
python3 run.py --dataset glove-50-angular --algorithm vektonn-hnsw --batch
python3 run.py --dataset glove-50-angular --algorithm vektonn-hnsw

python3 run.py --dataset glove-100-angular --algorithm vkt-faiss-hnsw
python3 run.py --dataset glove-100-angular --algorithm vkt-faiss-hnsw --batch
python3 run.py --dataset glove-100-angular --algorithm vektonn-hnsw --batch
python3 run.py --dataset glove-100-angular --algorithm vektonn-hnsw

python3 run.py --dataset sift-128-euclidean --algorithm vkt-faiss-bf
python3 run.py --dataset sift-128-euclidean --algorithm vkt-faiss-bf --batch
python3 run.py --dataset sift-128-euclidean --algorithm vektonn-bf
python3 run.py --dataset sift-128-euclidean --algorithm vektonn-bf --batch

python3 run.py --dataset sift-128-euclidean --algorithm vkt-faiss-hnsw
python3 run.py --dataset sift-128-euclidean --algorithm vkt-faiss-hnsw --batch
python3 run.py --dataset sift-128-euclidean --algorithm vektonn-hnsw --batch
python3 run.py --dataset sift-128-euclidean --algorithm vektonn-hnsw

python3 run.py --dataset glove-50-angular --algorithm opensearchknn --timeout 36000
python3 run.py --dataset glove-100-angular --algorithm opensearchknn --timeout 36000
python3 run.py --dataset sift-128-euclidean --algorithm opensearchknn --timeout 36000
