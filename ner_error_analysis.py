import json

def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART'):
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        if line[0]=="\n": continue # merge sentences to form a paragraph
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(' '.join(splits[1:])[:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []
    # print(len(data))
    return data

def readfile_prop(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label, presence_label, period_label = [], [], []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART'):
            if len(sentence) > 0:
                data.append((sentence,presence_label,period_label,label))
                sentence = []
                label, presence_label, period_label = [], [], []
            continue
        if line[0]=="\n": continue # merge sentences to form a paragraph
        splits = line.split(' ')
        sentence.append(splits[0])
        presence_label.append(splits[1])
        period_label.append(splits[2])
        label.append(' '.join(splits[3:])[:-1])

    if len(sentence) >0:
        data.append((sentence,presence_label,period_label,label))
        sentence = []
        label, presence_label, period_label = [], [], []
    # print(len(data))
    return data


with open('ypred.json') as f:
	preds = json.load(f)
with open('ytrue.json') as f:
	golds = json.load(f)
with open('y_presence_pred.json') as f:
    preds_presence = json.load(f)
with open('y_presence_true.json') as f:
    golds_presence = json.load(f)
with open('y_period_pred.json') as f:
    preds_period = json.load(f)
with open('y_period_true.json') as f:
    golds_period = json.load(f)

with open('../ner_data/test_file_names.json') as f:
	filenames = json.load(f)

# print(len(preds), len(golds))
data = readfile_prop('../ner_data/ehr_BIO_sdoh_test_lor_updated.txt')
print(len(filenames))
with open('error_analysis_dec2_onlyWrongPreds.csv','w') as ea:
    ea.write('token, gold_sdoh, pred_sdoh, gold_presence, pred_presence, gold_period, pred_period \n ')
    for i,(s,pres_l,perd_l,l) in enumerate(data):
        ea.write('--------------,--------------,--------------\n')
        ea.write(filenames[i]+'\n')
        ea.write('--------------,--------------,--------------\n')
        for token, gold, pred, gold_presence, pred_presence, gold_period, pred_period in zip(s,l,preds[i],pres_l,preds_presence[i],perd_l,preds_period[i]):
            f_token = '"'+token+'"' if ',' in token else token
            if gold==pred and gold_presence==pred_presence and gold_period == pred_period:
                pass
                # ea.write(','.join([f_token, gold, pred, gold_presence, pred_presence, gold_period, pred_period])+'\n')
            else:
                if gold != pred: pred = '##'+pred
                if gold_presence!=pred_presence: pred_presence = '##'+pred_presence
                if gold_period!=pred_period: pred_period = '##'+pred_period
                ea.write(','.join([f_token, gold, pred, gold_presence, pred_presence, gold_period, pred_period])+'\n')

        # assert l == golds[i], "{:} || {:}".format(l,golds[i])
        # assert pres_l == golds_presence[i], "{:} || {:}".format(pres_l,golds_presence[i])
        # assert perd_l == golds_period[i], "{:} || {:}".format(perd_l,golds_period[i])
        print('{:.3f}% files processed.'.format((i+1)*100/len(data)),end='\r')
