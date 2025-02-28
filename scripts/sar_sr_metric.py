import json
import numpy as np
import random
import sys

random.seed(1)

exp = sys.argv[1]
gold = sys.argv[2]

pred = f'{exp}/generated_predictions.jsonl'

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def cal_metric(pred, gold):
    ''' pred: label, predict
        gold: idx, cur_a, gold_a, input, etc.
    '''
    tmp = {}
    for p, g in zip(pred, gold):
        splits = g['input'].split(' ')
        if len(splits) == 1: ### s_r data
            is_sr = True
        else: ### sa_r data
            is_sr = False
            
        tmp.setdefault(g['idx'], {'pred': {}, 'label': g['gold_a']})
        try:
            # tmp[g['idx']]['pred'][g['cur_a']] = int(p['predict'].split(']')[0].strip('WIN['))
            tmp[g['idx']]['pred'][g['cur_a']] = float(p['predict'])
        except:
            continue
    ## get argmax-scored action for each state
    corr = 0
    total = 0
    for idx, infos in tmp.items():
        pred_actions = list(infos['pred'].keys())
        pred_values = np.array([infos['pred'][i] for i in pred_actions])
        # pred_a = random.choice(pred_actions)
        if is_sr:
            pred_a = pred_actions[pred_values.argmin()]
        else:
            pred_a = pred_actions[pred_values.argmax()]
        if infos['label']==pred_a:
            corr += 1
        total += 1
        
    print(f'total: {total}; acc: {corr/total}')

pred = read_jsonl_file(pred)
gold = read_jsonl_file(gold)
s2label = cal_metric(pred, gold)

