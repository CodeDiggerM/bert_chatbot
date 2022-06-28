import torch

from config import Config
from nn import build_model
from tokenizer import Tokenizer
from utils import evaluate

if __name__ == '__main__':

    # device = torch.device(Config.device)
    device = torch.device('cpu')

    state_dict = torch.load(f'{Config.data_dir}/{Config.fn}.pth', map_location=device)

    tokenizer = Tokenizer.from_pretrained(Config.model_name)

    model = build_model(Config).to(device)
    model.load_state_dict(state_dict['model'])
    model.eval()
    model.freeze()
    buff = ""
    answer_dict = {"お名前は何ですか":"bert botです",
                   "天気はどうですか": "晴れです",
                   "何にをしていますか":"暇です",
                   "日本語以外の言語喋れるか":"pythonができます",
                   "何歳ですか":"10歳だよ"}
    while True:

        s = input('You>')
        if  "#" in s:
            buff = buff.strip()[:-2]
            if buff.strip() in answer_dict:
                print('BOT>' + answer_dict[buff], end='')
            else:
                print('BOT>', end='')
                text = evaluate(Config, buff, tokenizer, model, device, True)
                buff = ""
        else:
            buff += s
        if s == 'q':
            break