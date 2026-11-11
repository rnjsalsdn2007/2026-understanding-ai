# Project 2. Poker Hand Prediction by Classification - train.py
#
# 3장/4장의 카드만 보고 "최종 5장 패의 점수(0~9)"를 예측하는 분류기를 학습하고
# poker-model3.sav, poker-model4.sav 로 저장한다.
#
# 아이디어
#  1) 전처리(feature engineering): (문양, 숫자) 원본 값은 순서/크기에 의미가 없어서
#     분류기가 패턴을 배우기 어렵다. 대신 포커 점수를 결정하는 성질을 특징으로 만든다.
#       - 같은 숫자가 최대 몇 장인지 (max_rank_count: 1~4)
#       - 페어의 개수 (n_pairs: 0~2)
#       - 공개된 카드가 모두 같은 문양인지 (flush_live: 플러시 가능성)
#       - 스트레이트가 가능한 5칸 구간의 개수 (n_str_windows: 0~3, 에이스는 1과 14 모두 고려)
#       - 로열 플러시가 가능한지 (royal_live)
#     이 조합을 one-hot 벡터(192차원)로 만들어 분류기에 넣는다.
#  2) 데이터 증강: 카드가 공개되는 순서는 무작위이므로, 학습 데이터의 각 5장 패에서
#     가능한 모든 3장 조합(10개)과 4장 조합(5개)을 학습 샘플로 사용한다.
#  3) 분류기: 다항 로지스틱 회귀(LogisticRegression)로 P(최종 점수 = k | 공개된 카드)를 추정한다.
#     (원본 특징을 그대로 쓴 로지스틱 회귀/의사결정나무와 비교)
#  4) 베팅 결정(poker.py): 예측 확률로 이길 확률/질 확률을 구하고 기대 수익이 큰 쪽을 선택한다.

import itertools
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss

N_STATES = 4 * 3 * 2 * 4 * 2   # 192


def hand_features(suits, ranks):
    """suits, ranks: (N, n) 배열 (n = 3 또는 4). 반환: (N, 192) one-hot 특징 행렬"""
    suits = np.asarray(suits, dtype=int)
    ranks = np.asarray(ranks, dtype=int)
    if suits.ndim == 1:
        suits = suits.reshape(1, -1)
        ranks = ranks.reshape(1, -1)
    N, n = ranks.shape
    rows = np.arange(N)

    # 숫자별 개수 (인덱스 1~13, 14는 에이스를 하이로 볼 때)
    rc = np.zeros((N, 15), dtype=int)
    for j in range(n):
        np.add.at(rc, (rows, ranks[:, j]), 1)
    rc[:, 14] = rc[:, 1]
    # 문양별 개수
    sc = np.zeros((N, 5), dtype=int)
    for j in range(n):
        np.add.at(sc, (rows, suits[:, j]), 1)

    counts = rc[:, 1:14]
    max_rank = counts.max(axis=1)                       # 1~4
    n_pairs = (counts == 2).sum(axis=1)                 # 0~2
    distinct = (counts > 0).sum(axis=1)
    flush_live = (sc.max(axis=1) == n).astype(int)      # 0/1

    present = rc > 0
    n_str = np.zeros(N, dtype=int)
    royal_window = np.zeros(N, dtype=bool)
    for s in range(1, 11):                               # 구간 s ~ s+4 (A-2-3-4-5 ~ 10-J-Q-K-A)
        fits = (present[:, s:s + 5].sum(axis=1) == distinct) & (distinct == n)
        n_str += fits
        if s == 10:
            royal_window = fits
    n_str = np.minimum(n_str, 3)                         # 0~3
    royal_live = (royal_window & (flush_live == 1)).astype(int)

    state = ((((max_rank - 1) * 3 + n_pairs) * 2 + flush_live) * 4 + n_str) * 2 + royal_live
    X = np.zeros((N, N_STATES))
    X[rows, state] = 1.0
    return X


def raw_to_arrays(df, idx):
    """df의 카드 컬럼 중 idx(카드 번호 리스트)만 골라 suits, ranks 배열 반환"""
    v = df.iloc[:, :10].to_numpy()
    suits = np.stack([v[:, 2 * i] for i in idx], axis=1)
    ranks = np.stack([v[:, 2 * i + 1] for i in idx], axis=1)
    return suits, ranks


def build_dataset(df, n_cards, all_subsets=True):
    """각 5장 패에서 n_cards장 조합으로 (특징, 최종 점수) 샘플을 만든 뒤, 같은 상태끼리 묶어 가중치로 집계"""
    y = df.iloc[:, -1].to_numpy()
    subsets = list(itertools.combinations(range(5), n_cards)) if all_subsets else [tuple(range(n_cards))]
    states, labels = [], []
    for idx in subsets:
        s, r = raw_to_arrays(df, idx)
        states.append(hand_features(s, r).argmax(axis=1))
        labels.append(y)
    states = np.concatenate(states)
    labels = np.concatenate(labels)
    # (상태, 점수) 별 빈도를 sample_weight 로 사용 -> 수백만 샘플을 빠르게 학습
    agg = pd.DataFrame({'state': states, 'y': labels}).value_counts().reset_index()
    agg.columns = ['state', 'y', 'w']
    X = np.zeros((len(agg), N_STATES))
    X[np.arange(len(agg)), agg['state'].to_numpy()] = 1.0
    return X, agg['y'].to_numpy(), agg['w'].to_numpy().astype(float)


def train_model(df, n_cards):
    X, y, w = build_dataset(df, n_cards)
    # 기본 tol(1e-4)로는 최적화가 일찍 멈춰서 표본이 적은 상태(예: 4장이 모두 같은 숫자)의
    # 확률을 제대로 학습하지 못했다. tol 을 1e-8 로 낮춰 끝까지 수렴시킨다.
    model = LogisticRegression(C=100.0, max_iter=20000, tol=1e-8)
    model.fit(X, y, sample_weight=w)
    return model


def evaluate_probs(model, df, n_cards, featurize=True):
    s, r = raw_to_arrays(df, range(n_cards))
    y = df.iloc[:, -1].to_numpy()
    if featurize:
        X = hand_features(s, r)
    else:
        X = df.iloc[:, :2 * n_cards].to_numpy()
    p = model.predict_proba(X)
    pred = model.classes_[p.argmax(axis=1)]
    # 테스트에만 있는 클래스가 있을 수 있어 labels 지정
    full = np.zeros((len(y), 10))
    full[:, model.classes_] = p
    full = np.clip(full, 1e-12, 1)
    full /= full.sum(axis=1, keepdims=True)
    return accuracy_score(y, pred), log_loss(y, full, labels=list(range(10)))


if __name__ == '__main__':
    train = pd.read_csv('poker-hand-training.csv')
    test = pd.read_csv('poker-hand-testing.csv')
    train = train.apply(pd.to_numeric, errors='coerce').dropna().astype(int)
    test = test.apply(pd.to_numeric, errors='coerce').dropna().astype(int)
    print('train:', train.shape, ' test:', test.shape)

    sub_tr = train.sample(50000, random_state=0)
    for n in (3, 4):
        print('\n===== %d장 공개 모델 =====' % n)
        # (비교 1) 원본 특징 + 로지스틱 회귀 (샘플 코드 방식)
        base = LogisticRegression(max_iter=500)
        base.fit(sub_tr.iloc[:, :2 * n], sub_tr.iloc[:, -1])
        acc, ll = evaluate_probs(base, test, n, featurize=False)
        print('[원본 특징 + LogisticRegression]  accuracy=%.4f  log_loss=%.4f' % (acc, ll))
        # (비교 2) 원본 특징 + 의사결정나무
        tree = DecisionTreeClassifier(max_depth=10, random_state=0)
        tree.fit(sub_tr.iloc[:, :2 * n], sub_tr.iloc[:, -1])
        acc, ll = evaluate_probs(tree, test, n, featurize=False)
        print('[원본 특징 + DecisionTree]        accuracy=%.4f  log_loss=%.4f' % (acc, ll))
        # (최종) 전처리 특징 + 데이터 증강 + 로지스틱 회귀
        model = train_model(train, n)
        acc, ll = evaluate_probs(model, test, n)
        print('[전처리 특징 + LogisticRegression] accuracy=%.4f  log_loss=%.4f' % (acc, ll))
        filename = 'poker-model%d.sav' % n
        pickle.dump(model, open(filename, 'wb'))
        print('saved ->', filename)
