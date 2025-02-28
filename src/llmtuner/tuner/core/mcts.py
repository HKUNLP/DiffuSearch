import logging
import math
import chess.engine
import numpy as np
import torch

from llmtuner.extras.logging import get_logger
import torch.nn.functional as F
import chess
import chess.pgn

import re

def replace_dots(s):
    pattern = r'\.{1,}' 
    matches = re.findall(pattern, s) 
    for match in matches:
        count = str(len(match)) 
        s = s.replace(match, count, 1) 
    return s


def new_to_ori_fen(fen):
    moves = '/'.join([replace_dots(fen[i:i+8]) for i in range(0, 64, 8)])
    remain = fen[64:]
    player = remain[0]
    castling = remain[1:5].replace('.', '')
    en_passant = remain[5:7].replace('.', '')
    halfmove = remain[7:9].replace('.', '')
    fullmove = remain[9:].replace('.', '')
    fen = ' '.join([moves, player, castling, en_passant, halfmove, fullmove])
    return fen

def process_moves(fen_moves):
    return fen_moves.replace('1', '.').replace('2', '.'*2).replace('3', '.'*3).\
                     replace('4', '.'*4).replace('5', '.'*5).replace('6', '.'*6).\
                     replace('7', '.'*7).replace('8', '.'*8).replace('/', '')

def ori_to_new_fen(fen):
    splits = fen.split(' ')
    splits[0] = process_moves(splits[0])
    splits[2] = splits[2]+'.'*(4-len(splits[2]))
    splits[3] = splits[3]+'.'*(2-len(splits[3]))
    splits[4] = splits[4]+'.'*(2-len(splits[4]))
    splits[5] = splits[5]+'.'*(3-len(splits[5]))
    return ''.join(splits)
    
def extract_number_from_string(input_string):
    match = re.search(r"WIN\[(\d+)\]", input_string)
    if match:
        return int(match.group(1))
    else:
        return None

EPS = 1e-8
logger = get_logger(__name__)

class State:
    def __init__(self, game_state : chess.Board, player=1):
        self.game_state = game_state
        self.player = player  # 1 or -1, indicating which player's turn

class Node:
    def __init__(self, state : State, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        # self.children = []
        self.untried_moves = list(state.game_state.legal_moves)  # initialize with all legal moves
        self.player = state.player  # track player at this node
        self.fen = state.game_state.fen()

    # def add_child(self, move, state):
    #     child_node = Node(state=state, parent=self, move=move)
    #     # self.children.append(child_node)
    #     return child_node

    def is_terminal(self):
        return self.state.game_state.is_game_over()


class MCTS:
    def __init__(self, value_model, action_model, value_model_tokenizer, action_model_tokenizer, cpuct=1.41):
        self.value_model = value_model
        self.value_model = self.value_model.eval()
        self.action_model = action_model
        self.action_model = self.action_model.eval()
        self.value_model_tokenizer = value_model_tokenizer
        self.action_model_tokenizer = action_model_tokenizer

        self.cpuct = cpuct
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        
        self.max_depth = 0  # to track the maximum depth
        
    def predict_action_probs(self, node, s):
        encoded_input = self.action_model_tokenizer(s, return_tensors='pt', padding=True)
        encoded_input.to(self.action_model.device)
        attention_mask = encoded_input['attention_mask']
        input_ids = encoded_input['input_ids']
        
        sep = torch.full((input_ids.shape[0], 1), self.action_model_tokenizer.sep_token_id).to('cuda')
        input_ids = torch.cat([input_ids, sep], dim=-1).to('cuda')
        sep_mask = torch.full((attention_mask.shape[0], 1), 1).to('cuda')
        attention_mask = torch.cat([attention_mask, sep_mask], dim=-1).to('cuda')

        with torch.no_grad():
            outputs = self.action_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        next_token_logits = logits[:, -1, :]
        legal_moves = [move.uci() for move in node.state.game_state.legal_moves]
        legal_move_ids = [self.action_model_tokenizer.convert_tokens_to_ids(move) for move in legal_moves]

        if len(legal_move_ids) == 0:
            raise ValueError("No legal moves found after filtering.")
        # # Gather the logits for the legal moves
        # move_probs = next_token_logits[0, legal_move_ids]
        # # Apply softmax to the legal move logits to get probabilities
        # action_probs = F.softmax(move_probs, dim=-1)
        # sum_probs = action_probs.sum()
        
        all_action_probs = F.softmax(next_token_logits, dim=-1)
        action_probs  = all_action_probs[0, legal_move_ids]
        sum_probs = action_probs.sum()
        # Avoid division by zero
        if sum_probs > 0:
            action_probs = action_probs / sum_probs
        else:
            action_probs = torch.zeros_like(action_probs)
        return {a: r for a, r in zip(legal_moves,action_probs.tolist())}
        # return action_probs.tolist()
    
    def logits_to_values(self, logits):
        # return win_rate
        full_linspace = torch.linspace(0.0, 1.0, 128 + 1, device=logits.device)
        uniform_values = (full_linspace[:-1] + full_linspace[1:]) / 2
        value_probs = logits[...,-128:].softmax(-1)
        value_probs *= uniform_values
        values = value_probs.sum(-1) # (b,)
        # logger.info(values)
        return values
            
    def predict(self, node: Node):
        v = 0
        s = node.state.game_state.fen()
        # board = chess.Board(s)
        s = ori_to_new_fen(s) 
        actions = s
        encoded_input = self.value_model_tokenizer(actions, return_tensors='pt', padding=True)
        encoded_input.to(self.value_model.device)
        attention_mask = encoded_input['attention_mask']
        encoded_input = encoded_input['input_ids']
       
        sep = torch.full((encoded_input.shape[0], 1), self.value_model_tokenizer.sep_token_id).to('cuda')
        encoded_input = torch.cat([encoded_input, sep], dim=-1).to('cuda')
        sep_mask = torch.full((attention_mask.shape[0], 1), 1).to('cuda')
        attention_mask = torch.cat([attention_mask, sep_mask], dim=-1).to('cuda')
        with torch.no_grad():
            output = self.value_model(input_ids=encoded_input, attention_mask=attention_mask)
        logits = output.logits
        logits = logits[:, -1, :]
        
        v = self.logits_to_values(logits)
        v = v.tolist()[0]
        if v > 1 or v < 0:
            print(v)
        return self.predict_action_probs(node, s), v
        

    def search(self, node : Node, depth=0):
        s = node.fen
        self.max_depth = max(self.max_depth, depth)

        if s not in self.Es:
            if node.state.game_state.is_game_over():
                if node.state.game_state.is_checkmate():
                    # Determine if the current player is in checkmate
                    if (node.state.game_state.turn == chess.WHITE and node.player == -1) or (node.state.game_state.turn == chess.BLACK and node.player == 1):
                        self.Es[s] = 1 # Current player won
                    else:
                        self.Es[s] = -1
                elif node.state.game_state.is_stalemate() or node.state.game_state.is_insufficient_material() or node.state.game_state.is_seventyfive_moves() or node.state.game_state.is_fivefold_repetition():
                    self.Es[s] = EPS # Game is a draw due to a draw condition
            else:
                self.Es[s] = 0

        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps.keys():
            # Leaf node
            action_probs, v = self.predict(node)  # Assume model returns a scalar value v
            
            # logger.info(v)
            # sum_win_probs = np.sum(win_probs)
            # if (sum_win_probs != 0):
            #     win_probs /= sum_win_probs 
            self.Ps[s] = {move: prob for move, prob in action_probs.items()}
            # print(self.Ps[s])
            self.Ns[s] = 0
            self.Vs[s] = [move.uci() for move in node.state.game_state.legal_moves]
            return -v
        # elif s in self.Ps and depth >= 4:
        #     self.Ns[s] += 1
        #     return -self.Vs[s]
            
        # Normal MCTS logic to select the best action based on UCB
        best_action = None
        max_ucb = float('-inf')

        for move in self.Vs[s]:
            a = move
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Avoid division by zero

            if u > max_ucb:
                max_ucb = u
                best_action = a

        # Expand the node with the best action found
        next_s = node.state.game_state.copy()
        next_s.push(chess.Move.from_uci(best_action))
        next_node = Node(State(next_s, -node.state.player), node, best_action)
        
        # node.add_child(best_action, State(next_s, -node.state.player))

        v = self.search(next_node, depth + 1)
        

        if (s, best_action) in self.Qsa:
            self.Qsa[(s, best_action)] = (self.Nsa[(s, best_action)] * self.Qsa[(s, best_action)] + v) / (self.Nsa[(s, best_action)] + 1)
            self.Nsa[(s, best_action)] += 1
        else:
            self.Qsa[(s, best_action)] = v
            self.Nsa[(s, best_action)] = 1

        self.Ns[s] += 1
        # print("search ", v)
        return -v

    # def get_best_move(self, node : Node, simulations_number=25):
    #     """ Performs 'simulations_number' simulations starting from 'node' to determine the best move. """
    #     self.max_depth = 0
    #     # print(self.Ns)
    #     for _ in range(simulations_number):
    #         self.search(node)

    #     best_move = None
    #     max_visits = -float('inf')

    #     # Select the move with the highest visit count from this node
    #     for move in node.state.game_state.legal_moves:
    #         if (node.state.game_state.fen(), move.uci()) in self.Nsa and self.Nsa[(node.state.game_state.fen(), move.uci())] > max_visits:
    #             best_move = move.uci()
    #             max_visits = self.Nsa[(node.state.game_state.fen(), move.uci())]

    #     return best_move, self.max_depth
    def get_best_move(self, node: Node, simulations_number=25):
        """Performs 'simulations_number' simulations starting from 'node' to determine the best move."""
        self.max_depth = 0

        for _ in range(simulations_number):
            self.search(node)

        best_move = None
        max_visits = -float('inf')
        best_qsa = -float('inf')

        # get the action that visit the most times
        for move in node.state.game_state.legal_moves:
            move_key = (node.state.game_state.fen(), move.uci())
            
            visits = self.Nsa.get(move_key, 0)
            qsa = self.Qsa.get(move_key, -float('inf'))

            # compare the Q(s, a) if has the same visit time
            if visits > max_visits or (visits == max_visits and qsa > best_qsa):
                best_move = move.uci()
                max_visits = visits
                best_qsa = qsa

        return best_move, self.max_depth
