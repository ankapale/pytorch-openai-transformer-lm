#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tkinter

from tkinter import *
import numpy
import random
import torch
import numpy as np
from text_utils import TextEncoder
from model_pytorch import *

class CustomLMModel(torch.nn.Module):
    """ Transformer with language model head only """
    def __init__(self, cfg, vocab=40990, n_ctx=512, return_probs=True,
                 encoder_path='./model/encoder_bpe_40000.json', bpe_path='./model/vocab_40000.bpe'):
        super(CustomLMModel, self).__init__()
        self.transformer = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        self.lm_head = LMHead(self.transformer, cfg, trunc_and_reshape=False)
        self.return_probs = return_probs
        self.text_encoder = TextEncoder(encoder_path,bpe_path)
        
        if self.return_probs:
            pos_emb_mask = torch.zeros(1, 1, vocab)
            pos_emb_mask[:, :, -n_ctx:] = -1e12
            self.register_buffer('pos_emb_mask', pos_emb_mask)


    def forward(self, x):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        if self.return_probs:
            lm_logits = F.softmax(lm_logits + self.pos_emb_mask, dim=-1)
        return lm_logits

lm_model = torch.load('./trained_lm_model')
device = "cpu"

def make_batch(X, n_vocab):
    X = np.array(X)
    assert X.ndim in [1, 2]
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    pos_enc = np.arange(n_vocab, n_vocab + X.shape[-1])
    pos_enc = np.expand_dims(pos_enc, axis=0)
    batch = np.stack([X, pos_enc], axis=-1)
    batch = torch.tensor(batch, dtype=torch.long).to(device)
    return batch

def append_batch(X, next_idx):
    next_pos = X[:, -1:, 1] + 1
    next_x = torch.cat((next_idx, next_pos), -1).unsqueeze(1)
    return torch.cat((X, next_x), 1)
    
def predict_next_word(text, gen_len=20, topk=10):
    n_vocab = len(lm_model.text_encoder.encoder)
    encoded_text = lm_model.text_encoder.encode([text,])
    encoded_text = make_batch(encoded_text, n_vocab)
    
    for _ in range(gen_len):
        lm_probs = lm_model(encoded_text)
        values, indices = lm_probs[:, -1, :].topk(topk)
        next_idx = indices.gather(-1, torch.multinomial(values, 1))
        next_token = lm_model.text_encoder.decoder[next_idx.item()].replace('</w>', '')
        print(next_token, end=' ')
        encoded_text = append_batch(encoded_text, next_idx)
        

# Create TK instance
top = Tk()
top.title = 'Predict the next word'
top.geometry('600x280')

# Variable which stored the entered text prompt
text_var = tkinter.StringVar()
# Text entry widget to enter text
text_entry = Entry(top, textvariable=text_var)
text_entry.grid(row=0, column=0)

# Function reads the text and predicts next word
def predict():
	# Read the text from the entry widget
	entered_text = text_var.get()
	# Randomly select a word (This will be replaced by a neural network)
	next_word = random.choice(word_list)
	textvar = "The next word is : {}".format(next_word)
    # Update the entry
	t1.delete(0.0, tkinter.END)
	t1.insert('insert', textvar+'\n')
	t1.update()

# Function reads the text and predicts next word
def generate_text():
	# Read the text from the entry widget
	entered_text = text_var.get()
	textvar = entered_text
	# Generate text
  textvar = predict_next_word(textvar, gen_len=10, topk=10)
    # Update the entry
	t1.delete(0.0, tkinter.END)
	t1.insert('insert', textvar+'\n')
	t1.update()

# "Next word" button: triggers the predict function
next_word_button = Button(top, text ='Next Word', command = predict)
next_word_button.grid(row=1, column=0)

# "Gennerate Text" button: triggers the predict function
generate_text_button = Button(top, text ='Generate Text', command = generate_text)
generate_text_button.grid(row=2, column=0)
    

l1=Label(top,text='Please type the text prompt, then press <Next Word>')
l1.grid(row=3)


t1=Text(top,bd=0, width=20,height=10,font='Fixdsys -14')
t1.grid(row=0, column=1)

top.mainloop()
