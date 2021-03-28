import os

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["USE_CPU"]="1"

import sys
import argparse
import numpy as np
import pandas as pd
from train import a
from train import g
import torch

vowels = sorted(['y', 'é', 'ö', 'a', 'i', 'å', 'u', 'ä', 'e', 'o'])

def createInstances(input_data, vocab):
    gt = []
    gr = []
    for v in range(len(input_data) - 4):

        context_in_vocab = True
        for char in [input_data[v], input_data[v+1], input_data[v+3], input_data[v+4]]:
            if char not in vocab:
                context_in_vocab = False

        if (input_data[v+2] not in vowels) or not context_in_vocab:
            continue
        
        h2 = vowels.index(input_data[v+2])
        gt.append(h2)
        r = np.concatenate([g(x, vocab) for x in [input_data[v], input_data[v+1], input_data[v+3], input_data[v+4]]])
        gr.append(r)

    return np.array(gr), np.array(gt)

def predictVowels(model, context_chars):
    predictions = model(torch.Tensor(context_chars)).detach().numpy()
    predicted_vowels = np.argmin(np.abs(predictions), axis=1)

    return predicted_vowels

def predictText(input_data, predicted_vowels, vocab, outputfile):

    i = 0
    output_data = input_data
    for v in range(len(input_data) - 4):

        context_in_vocab = True
        for char in [input_data[v], input_data[v+1], input_data[v+3], input_data[v+4]]:
            if char not in vocab:
                context_in_vocab = False

        if (input_data[v+2] in vowels) and context_in_vocab:
            output_data[v+2] = vowels[predicted_vowels[i]]
            i += 1

    outF = open(outputfile, "w")
    outF.write(''.join(output_data[2:-2]))
    outF.close()
    return

def calculateAccuracy(predicted_vowels, vowel_list):
    # Accuracy = TP / all instances
    tp = 0
    for i in range(len(vowel_list)):
        if vowel_list[i] == predicted_vowels[i]:
            tp += 1
    accuracy = (tp/len(vowel_list))

    return accuracy

def calculatePerplexity(loss):
    perplexity = torch.exp(loss)
    return perplexity.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("testdata", type=str)
    parser.add_argument("outputfile", type=str)
    parser.add_argument('--perplexity', action='store_true')
    args = parser.parse_args()

    model = torch.load(args.model)
    input_data, char_list_testing = a(args.testdata)
    context_chars, vowel_list  = createInstances(input_data, model.vocab)

    predicted_vowels = predictVowels(model, context_chars)
    output_data = predictText(input_data, predicted_vowels, model.vocab, args.outputfile)

    print("The text with the predicted vowels can be found in 'output.txt'.")
    print("Accuracy of the model: ", calculateAccuracy(predicted_vowels, vowel_list))
    if args.perplexity:
        print("Perplexity of the model: ", calculatePerplexity(model.loss))