import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, M2M100Tokenizer

class m2m100_bt():
    def __init__(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else: 
            self.device = 'cpu'
        self.model1 = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M")
        self.model1.load_state_dict(torch.load('./EN_ZH_epoch_5_valid_bleu_55.37_model_weights.bin'))
        self.model2 = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M")
        self.model2.load_state_dict(torch.load('./epoch_5_valid_bleu_62.34_model_weights.bin'))
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        self.tokenizer.src_lang = 'en'
        self.model1.to(self.device)
        self.model2.to(self.device)
    
    def back_translate(self, target_lang, text):
        max_input_length = 200
        max_target_length = 160
        min_target_length = 10
        phase_1 = {}
        for idx, sentence in enumerate(tqdm(text)):
            sentence = self.tokenizer(sentence, max_length=max_input_length, return_tensors='pt')
            sentence = sentence.to(self.device)
            gen_tokens = self.model1.generate(**sentence, forced_bos_token_id=self.tokenizer.get_lang_id(target_lang), 
                                                max_length=max_target_length, min_length=min_target_length)
            out = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            phase_1[str(text.index[idx])] = out[0]
        
        self.tokenizer.src_lang = target_lang
        phase_2 = {}
        for k in tqdm(phase_1.keys()):
            sentence = phase_1[k]
            sentence = self.tokenizer(sentence, max_length=max_input_length, return_tensors="pt")
            sentence = sentence.to(self.device)
            gen_tokens = self.model2.generate(**sentence, forced_bos_token_id=self.tokenizer.get_lang_id('en'), 
                                                max_length = max_target_length, min_length=min_target_length)
            out = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            phase_2[k] = out[0]
        
        return phase_1, phase_2
        
    