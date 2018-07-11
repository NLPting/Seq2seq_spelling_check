import onmt
import onmt.io
import onmt.translate
import onmt.ModelConstructor
import io
from collections import namedtuple
from itertools import count
import re
import aspell
s = aspell.Speller('lang', 'en')



import difflib
def difflib_leven(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
       #print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))
        if tag == 'replace':
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':
            leven_cost += (j2-j1)
        elif tag == 'delete':
            leven_cost += (i2-i1)
    return leven_cost



def Load_model(model):
    Opt = namedtuple('Opt', ['model', 'data_type', 'reuse_copy_attn', "gpu"])
    opt = Opt(model, "text",False, 0)
    fields, model, model_opt =  onmt.ModelConstructor.load_test_model(opt,{"reuse_copy_attn":False})
    return (fields, model, model_opt)

    

def ch_OpenNMT_candidate(detect_sentence_arr,mes):
    
    fields, model, model_opt = mes[0],mes[1],mes[2]
    ch_candidate = {}
    mispel = detect_sentence_arr[0]
    #print(mispel)
    text = '\n'.join(' '.join(word) for word in detect_sentence_arr)
    input_text = io.StringIO(text)
    
    
    data = onmt.io.build_dataset(fields, "text", input_text, None, use_filter_pred=False)
    data_iter = onmt.io.OrderedIterator(
        dataset=data, device=0,
        batch_size=1, train=False, sort=False,
        sort_within_batch=True, shuffle=False)
    
    # Translator
    scorer = onmt.translate.GNMTGlobalScorer(0,
                                             0,
                                             None,
                                             None)
   
    
    
    translator = onmt.translate.Translator(model, fields,
                                               beam_size=30,
                                               n_best=30,
                                               global_scorer=scorer,
                                               cuda=True)
    builder = onmt.translate.TranslationBuilder(
            data, translator.fields,
            30, False, None)
    
    for batch in data_iter:
        batch_data = translator.translate_batch(batch, data)
        translations = builder.from_batch(batch_data)
        for trans in translations:
            n_best_preds = [" ".join(pred) for pred in trans.pred_sents[:30]]
        
        ch_candidate[' '.join(translations[0].src_raw).replace(' ','')] = n_best_preds
        
        
    
        
        
    
    can = [c.replace(' ','').replace(',','').replace('.','') for ch in ch_candidate.values() for c in ch ]
    
    cann = [ (c , difflib_leven(c , mispel)) for c in can if s.check(c) and c!='']
    #print(cann)
    can = sorted(cann , key = lambda x : x[1])
    can = [c[0] for c in can]
    #can = [c for c in can if s.check(c) and c!='' and difflib_leven(c, mispel)<=3]

        
    
    
    return can

