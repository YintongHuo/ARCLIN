import sys
sys.path.append('APIRecognition')
from recognizer import main_recog
sys.path.append('APILinking')
from linker import main_link
from context_pred import ContextPredictor, BERTContext
from twokenize import tokenize


sentence = "However, torch.tensors are designed to be used in the context of gradient descent optimization"
tokens = tokenize(sentence)
tokens, link_preds = main_recog(tokens)
# print(tokens, link_preds)
output = main_link(tokens, link_preds)
print(tokens)
print(output)

