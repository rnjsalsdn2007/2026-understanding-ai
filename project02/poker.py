from pydoc import cli
import pygame, sys
pygame.init()

# NOTE: The following implementation is a sample and will not run without 
# the necessary model files (e.g., SAV files).
## ---------------- FILL HERE ---------------------------------
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------------------
# 전처리: 공개된 카드(3장 또는 4장)를 포커 점수와 관련된 특징으로 변환
#   - max_rank_count : 같은 숫자 카드의 최대 개수 (1~4)
#   - n_pairs        : 페어 개수 (0~2)
#   - flush_live     : 공개된 카드가 모두 같은 문양인지 (플러시 가능성)
#   - n_str_windows  : 스트레이트가 가능한 5칸 구간 수 (0~3, 에이스는 1/14 모두 고려)
#   - royal_live     : 로열 플러시 가능 여부
# 위 조합을 192차원 one-hot 벡터로 만든다. (train.py 와 동일한 함수)
# ------------------------------------------------------------------
N_STATES = 4 * 3 * 2 * 4 * 2


def hand_features(suits, ranks):
    suits = np.asarray(suits, dtype=int)
    ranks = np.asarray(ranks, dtype=int)
    if suits.ndim == 1:
        suits = suits.reshape(1, -1)
        ranks = ranks.reshape(1, -1)
    N, n = ranks.shape
    rows = np.arange(N)
    rc = np.zeros((N, 15), dtype=int)
    for i in range(N):          # 손패마다
        for j in range(n):      # 카드마다
            rc[i, ranks[i, j]] += 1
    rc[:, 14] = rc[:, 1]
    sc = np.zeros((N, 5), dtype=int)
    for i in range(N):          # 손패마다
        for j in range(n):      # 카드마다
            sc[i, suits[i, j]] += 1
    counts = rc[:, 1:14]
    max_rank = counts.max(axis=1)
    n_pairs = (counts == 2).sum(axis=1)
    distinct = (counts > 0).sum(axis=1)
    flush_live = (sc.max(axis=1) == n).astype(int)
    present = rc > 0
    n_str = np.zeros(N, dtype=int)
    royal_window = np.zeros(N, dtype=bool)
    for s in range(1, 11):
        fits = (present[:, s:s + 5].sum(axis=1) == distinct) & (distinct == n)
        n_str += fits
        if s == 10:
            royal_window = fits
    n_str = np.minimum(n_str, 3)
    royal_live = (royal_window & (flush_live == 1)).astype(int)
    state = ((((max_rank - 1) * 3 + n_pairs) * 2 + flush_live) * 4 + n_str) * 2 + royal_live
    X = np.zeros((N, N_STATES))
    X[rows, state] = 1.0
    return X


# load the model from disk
#  (sklearn 버전 차이 등으로 sav 로딩이 실패하면 train.py 로 즉석에서 다시 학습)
def _load_models():
    try:
        m3 = pickle.load(open('poker-model3.sav', 'rb'))
        m4 = pickle.load(open('poker-model4.sav', 'rb'))
        return m3, m4
    except Exception as e:
        print('model load failed (%s); retraining from poker-hand-training.csv' % e)
        from train import train_model
        df = pd.read_csv('poker-hand-training.csv')
        df = df.apply(pd.to_numeric, errors='coerce').dropna().astype(int)
        return train_model(df, 3), train_model(df, 4)


model3, model4 = _load_models()

# 52장 전체 덱 (문양 1~4, 숫자 1~13)
FULL_DECK = [(s, r) for s in range(1, 5) for r in range(1, 14)]


def _win_lose_prob(model, proba, opposite_score):
    classes = model.classes_
    p_win = proba[:, classes > opposite_score].sum(axis=1)
    p_lose = proba[:, classes < opposite_score].sum(axis=1)
    return p_win, p_lose


# given parameters:
#   hand: list of tuples (suit, rank); length is 3 or 4
#   opposite_score: the score of opposite player
#
# return:
#   True if you want to call,
#   False if you want to fold
#
# 기대 수익 기준 결정 (판돈 $1씩, 승리 시 +$3 / 패배 시 -$3 / 무승부 $0)
#   - 4장: 콜하면 3*(P승 - P패), 폴드하면 -2  ->  3*(P승 - P패) > -2 이면 콜
#   - 3장: 폴드하면 -1. 콜하면 다음 카드(남은 49장 중 하나)를 보고 다시 최선의 선택을 하므로
#          기대값 = 평균_{다음 카드} max(-2, 3*(P승 - P패))  ->  이 값이 -1 보다 크면 콜
def predict_call(hand, opposite_score):
    suits = [int(s) for s, r in hand]
    ranks = [int(r) for s, r in hand]

    if len(hand) == 4:
        proba = model4.predict_proba(hand_features(suits, ranks))
        p_win, p_lose = _win_lose_prob(model4, proba, opposite_score)
        return bool(3 * (p_win[0] - p_lose[0]) > -2)

    # 3장: 가능한 4번째 카드를 모두 고려
    known = set(zip(suits, ranks))
    nexts = [c for c in FULL_DECK if c not in known]
    S = np.array([suits + [c[0]] for c in nexts])
    R = np.array([ranks + [c[1]] for c in nexts])
    proba = model4.predict_proba(hand_features(S, R))
    p_win, p_lose = _win_lose_prob(model4, proba, opposite_score)
    value_if_call = np.maximum(-2, 3 * (p_win - p_lose)).mean()
    return bool(value_if_call > -1)
## -------------------------------------------------

import pandas as pd
cand_hands = pd.read_csv('poker-hand-testing.csv')

def pick_opposite_score():
    return cand_hands['Poker Hand'].sample().values[0]

def pick_five_cards():
    sam = cand_hands.sample().values[0]
    score = sam[-1]
    lis = sam[:-1]
    return (
        [ (a, b) for a, b in zip(lis[::2], lis[1::2]) ],
        score)


# game money: initially, $1000
money = 1000

# game count
#game_count = 1000

# initially, pick five cards
full_hand, hand_score = pick_five_cards()
print(full_hand, hand_score)

# currently, three flipped cards
flipped = 3
current_cards = full_hand[:flipped]

# current game's opposite hand score
opposite_hand_score = pick_opposite_score()

#game_count = game_count - 1
money = money - 1

font = pygame.font.SysFont("comicsans",20)
SCREEN_WIDTH, SCREEN_HEIGHT = 512, 512

# create window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("POKER")

CARD_WIDTH, CARD_HEIGHT = 60, 100
SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT = 32, 64

# create a surface object, image is drawn on it.
imp = pygame.image.load("cards.jpg")
imp = pygame.transform.scale(imp, (SUIT_ICON_WIDTH*4, SUIT_ICON_HEIGHT))

# list of suit icons (0: hear, 1: spade, 2: diamond, 3: club)
suits = list()

# 0: heart
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (SUIT_ICON_WIDTH*3, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# 1: spade
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (0, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# 2: diamond
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (SUIT_ICON_WIDTH, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# 3: club
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (SUIT_ICON_WIDTH*2, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# colors
white = (255, 255, 255)
black = (0, 0, 0)
gray = (128, 128, 128)

def draw_card(suit, rank, pos):
    # white card
    rect = pygame.Rect( pos, (CARD_WIDTH, CARD_HEIGHT) )
    pygame.draw.rect(screen, white, rect)

    # suit icon
    screen.blit(
        suits[suit], 
        (pos[0] + CARD_WIDTH/2 - SUIT_ICON_WIDTH/2, 
            pos[1] + CARD_HEIGHT*2/3 - SUIT_ICON_HEIGHT/2))

    # rank
    rank_text = font.render("{}".format(rank), True, black)
    rank_rect = rank_text.get_rect()
    rank_rect.center = (pos[0] + CARD_WIDTH/2, pos[1] + CARD_HEIGHT/4)
    screen.blit(rank_text, rank_rect)

def gamemoney_text():
    text = font.render("Your money: ${}".format(money), True, white)
    rank_rect = text.get_rect()
    rank_rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/6)
    screen.blit(text, rank_rect)

def opposite_hand_text():
    text = font.render(
        "The opposite's hand score: {}".format(opposite_hand_score), 
        True, white)
    rank_rect = text.get_rect()
    rank_rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/6 + rank_rect.height)
    screen.blit(text, rank_rect)

BUTTON_WIDTH, BUTTON_HEIGHT = 120, 50

def draw_button(center_pos, button_text, button_color):
    # rect button
    #rect = pygame.Rect(
    #    (center_pos[0] - BUTTON_WIDTH/2, center_pos[1] - BUTTON_HEIGHT/2),
    #    (BUTTON_WIDTH, BUTTON_HEIGHT) )
    rect = pygame.Rect((0, 0, BUTTON_WIDTH, BUTTON_HEIGHT))
    rect.center = center_pos
    pygame.draw.rect(screen, button_color, rect) 

    # button text
    text = font.render(button_text, True, black)
    text_rect = text.get_rect()
    text_rect.center = rect.center
    screen.blit(text, text_rect)

msg = None
MESSAGE_HEIGHT = 40
def display_message(_msg):
    if _msg is not None:
        # white message board at the top of screen
        rect = pygame.Rect( (0, 0), (SCREEN_WIDTH, MESSAGE_HEIGHT) )
        pygame.draw.rect(screen, white, rect)
        
        # display message text
        text = font.render(_msg, True, black)
        rect = text.get_rect()
        rect.center = (SCREEN_WIDTH/2, MESSAGE_HEIGHT/2)
        screen.blit(text, rect)

# bet and die buttons at the bottom of screen
BET_BUTTON_CENTER = (SCREEN_WIDTH/4, SCREEN_HEIGHT*4/5)
bet_button_color = white

def draw_bet_button():
    draw_button(BET_BUTTON_CENTER, "CALL", bet_button_color)

# answer if pos is on the button
def on_bet_button(pos):
    return (abs(pos[0] - BET_BUTTON_CENTER[0]) <= BUTTON_WIDTH/2) \
        and (abs(pos[1] - BET_BUTTON_CENTER[1]) <= BUTTON_HEIGHT/2)

DIE_BUTTON_CENTER = (SCREEN_WIDTH/2, SCREEN_HEIGHT*4/5)
die_button_color = white

def draw_die_button():
    draw_button(DIE_BUTTON_CENTER, "FOLD", die_button_color)

def on_die_button(pos):
    return (abs(pos[0] - DIE_BUTTON_CENTER[0]) <= BUTTON_WIDTH/2) \
        and (abs(pos[1] - DIE_BUTTON_CENTER[1]) <= BUTTON_HEIGHT/2)

# auto play with classifier
CLS_BUTTON_CENTER = (SCREEN_WIDTH*3/4, SCREEN_HEIGHT*4/5)
cls_button_color = white

def draw_cls_button():
    draw_button(CLS_BUTTON_CENTER, "AUTO", cls_button_color)

def on_cls_button(pos):
    return (abs(pos[0] - CLS_BUTTON_CENTER[0]) <= BUTTON_WIDTH/2) \
        and (abs(pos[1] - CLS_BUTTON_CENTER[1]) <= BUTTON_HEIGHT/2)

# play sound when cards are revealed
flip_sound = pygame.mixer.Sound("flipcard.wav")
def play_cardflip_sound(rep):
    pygame.mixer.Sound.play(flip_sound, loops=rep-1)

def draw_current_cards():
    for i, (suit, rank) in enumerate(current_cards):
        draw_card(
            suit-1, rank, # suit and rank 
            ((i+1)*SCREEN_WIDTH/7, SCREEN_HEIGHT*2/5))

# game state for updating screen
# 0: beginning of game - three cards are flipped
# 1: bet - one more card is flipped
# 2: bet & game over - the last card is flipped and
#    win/tie/lose messages are shown
#    then, go to the next game after 5 secs
# 3: die & game over - message shown then,
#    go to the next game after 5 secs
# 9: wait for the next action
update_state = 0

def draw_items_on_screen():
    screen.fill(black)
    opposite_hand_text()
    gamemoney_text()
    draw_bet_button()
    draw_die_button()
    draw_cls_button()
    draw_current_cards()

    if msg is not None:
        display_message(msg)

    pygame.display.flip()
    pygame.time.delay(200)

# drawing cards and playing sound
def update_cards_and_sounds():
    global flipped, update_state, full_hand, hand_score,\
        money, msg, current_cards, opposite_hand_score

    if update_state == 0:
        # at the beginning of game, three cards are flipped
        play_cardflip_sound(3)
        draw_items_on_screen()

        # then, the screen of waiting the next action
        update_state = 9

    elif update_state == 1:
        # check button clicked and now four cards revealed
        play_cardflip_sound(1)
        draw_items_on_screen()

        # then, the screen of waiting the next action
        update_state = 9

    elif update_state == 2 or update_state == 3:
        # the last cards are revealed
        play_cardflip_sound(1)
        draw_items_on_screen()

        # delay another 1 sec for showing message
        pygame.time.delay(1000)
        msg = None
        draw_items_on_screen()

        # a new poker hand is drawn
        full_hand, hand_score = pick_five_cards()
        print(full_hand, hand_score)

        # three flipped cards
        flipped = 3
        current_cards = full_hand[:flipped]

        # opposite's hand score
        opposite_hand_score = pick_opposite_score()
        money = money - 1
        update_state = 0


game_stop = False
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and not game_stop:
            click_pos = pygame.mouse.get_pos()

            auto_bet = False
            auto_die = False
            if on_cls_button(click_pos):
                if predict_call(current_cards, opposite_hand_score):
                    print("auto call!")
                    auto_bet = True
                    display_message("AI calls")
                    pygame.display.update()
                    pygame.time.delay(800)
                else:
                    print("auto fold!")
                    auto_die = True
                    display_message("AI folds")
                    pygame.display.update()
                    pygame.time.delay(800)

            if on_bet_button(click_pos) or auto_bet:
                flipped = flipped + 1
                current_cards = full_hand[:flipped]
                money = money - 1

                # if game over,
                if flipped == 5:
                    # case 1: win
                    if hand_score > opposite_hand_score:
                        money = money + 6
                        msg = "Win"
                        update_state = 2
                    # case 2: tie
                    elif hand_score == opposite_hand_score:
                        money = money + 3
                        msg = "Tie"
                        update_state = 2
                    # case 3: lose
                    else:
                        msg = "Lose"
                        update_state = 2
                # if cards can be flipped more,
                else:
                    update_state = 1
                        
            elif on_die_button(click_pos) or auto_die:
                msg = "Fold, go on to the next"
                update_state = 3

    # show cards and info message including sound
    update_cards_and_sounds()

    pygame.time.delay(200)