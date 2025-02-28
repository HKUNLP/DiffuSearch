import torch
import math
import chess
import logging
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from llmtuner.tuner.core.mcts import MCTS, Node, State, ori_to_new_fen, new_to_ori_fen
from llmtuner.tuner.core.custom_tokenizer import CustomTokenizer
import argparse

def main(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    action_model_tokenizer = CustomTokenizer.from_pretrained(args.action_model_to_load)
    value_model_tokenizer = CustomTokenizer.from_pretrained(args.value_model_to_load)
    value_model = AutoModelForCausalLM.from_pretrained(args.value_model_to_load).to('cuda')
    action_model = AutoModelForCausalLM.from_pretrained(args.action_model_to_load).to('cuda')

    value_model_tokenizer.padding_side = "left"
    action_model_tokenizer.padding_side = "left"

    log_file = f'{args.log_name}.txt'
    max_depth_distribution_file = f'{args.distribution_file_name}.json'

    right_answer = 0
    idx = 0
    max_depths = [0 for _ in range(1000)]

    with open(args.file_path, 'r') as file, open(log_file, 'a') as log:
        lines = file.readlines()[args.start_idx:args.start_idx + args.test_len]
        for line in tqdm(lines, desc="Reading data", total=len(lines)):
            mcts = MCTS(value_model, action_model, value_model_tokenizer, action_model_tokenizer, cpuct=args.cpuct)
            data = json.loads(line)
            state = data['state']
            action = data['action']
            idx += 1

            initial_board = chess.Board(new_to_ori_fen(state))
            player = 1 if initial_board.turn == chess.BLACK else -1
            initial_state = State(game_state=initial_board, player=player)
            
            root_node = Node(initial_state)
            best_move, max_depth = mcts.get_best_move(root_node, simulations_number=args.simulations_number)
            max_depths[max_depth] += 1
            if best_move != None and best_move.strip() == action:
                right_answer += 1
            else:
                logger.info(f"{best_move}, while gold is: {action}")
                log.write(f"{best_move}, while gold is: {action}\n")
                log.write(f"{mcts.Ps[new_to_ori_fen(state)]}\n")
                move_visit_time_dict = {}
                for move in initial_board.legal_moves:
                    move_visit_time_dict[move.uci()] = 0
                    if (new_to_ori_fen(state), move.uci()) in mcts.Nsa.keys():
                        move_visit_time_dict[move.uci()] = mcts.Nsa[(new_to_ori_fen(state), move.uci())]
                log.write(f"{move_visit_time_dict}\n")
                log.flush()
            now_acc = right_answer / idx
            logger.info(f"now_acc: {now_acc}, max_depth: {max_depth}\n")
            log.write(f"now_acc: {now_acc}, max_depth: {max_depth}\n")
            log.flush()

    acc = right_answer / len(lines)
    logger.info(f"acc : {acc}")
    with open(log_file, 'a') as log:
        log.write(f"Final acc: {acc}\n")
        log.write(f"Max depth distribution: {max_depths}\n")
        log.flush()
    
    with open(max_depth_distribution_file, 'w') as f:
        json.dump(max_depths, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTS Chess Simulation")

    parser.add_argument("--file_path", type=str, default="data/chess_test.jsonl", required=False, help="Path to the chess data file.")
    # parser.add_argument("--value_model_to_load", type=str, default="output/chess100k_s_r/gpt2-model-bs1024-lr3e-4-ep40-20240826-184019", required=False, help="Path to the value model.")
    # parser.add_argument("--action_model_to_load", type=str, default="output/chess100k_gold_s_a/gpt2-model-bs1024-lr3e-4-ep100-20240807-113851", required=False, help="Path to the action model.")
    parser.add_argument("--value_model_to_load", type=str, default="output/chess10k_s_r/gpt2-model-bs1024-lr3e-4-ep40-20240826-173757", required=False, help="Path to the value model.")
    parser.add_argument("--action_model_to_load", type=str, default="output/chess10k_gold/gpt2-model-bs1024-lr3e-4-ep40-20240820-104029", required=False, help="Path to the action model.")
    parser.add_argument("--log_name", type=str, default="mcts_100k_sim_100", required=False, help="Log file name.")
    parser.add_argument("--distribution_file_name", type=str, default="mcts_100k_sim_100", required=False, help="Distribution file name.")
    parser.add_argument("--simulations_number", type=int, default=100, required=False, help="Number of simulations for MCTS.")
    parser.add_argument("--tot_answer", type=int, default=62561, required=False, help="Total number of test cases.")
    parser.add_argument("--start_idx", type=int, default=0, required=False, help="Starting index for the test data.")
    parser.add_argument("--test_len", type=int, default=62561, required=False, help="Number of test cases to run.")
    parser.add_argument("--cpuct", type=float, default=0.1, required=False, help="PUCT.")

    args = parser.parse_args()
    main(args)
