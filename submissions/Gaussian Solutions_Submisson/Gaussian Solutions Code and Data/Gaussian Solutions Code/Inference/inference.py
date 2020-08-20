# given a doc or pdf file, parse, infer, and return clause, label, probablility
import pickle
import os
import sys
import nlp_utils as nu

import sys
sys.path.insert(1, '../common')


def get_confidence(row):
    if row['label'] == 1:
        confidence = row['probability']
    else:
        confidence = 1 - row['probability']

    confidence = round(confidence*100, 2)
    return confidence

def infer(doc, parser):
    """
    Given a doc, or pdf, parses, and returns list of [clause, label, probability]
    """
    model_dict = get_model()
    infer_info = {}
    docdf = parser(doc)
    # if docdf is None, that means the document couldn;t be parsed, return
    if docdf is None:
        infer_info['error'] = " parsing document failed"
        return infer_info

    
    vectorizer = model_dict['vectorizer']
    model = model_dict['model']
    preprocessor = model_dict['preprocessor']
    # for each clause, transform text, and predict
    print("Preprocessing data")
    X_prep = [str(preprocessor(x)) for x in docdf['Clause'] ]

    # vectorize the input 
    print("Vecotirizng data")
    X_test = vectorizer.transform(X_prep)

    print("share of X_test is ", X_test.shape)
    # make the predictions
    y_pred = model.predict(X_test)
    # get the probabilities of prediction
    try:
        y_prob = model.predict_proba(X_test)[:,1]
    except AttributeError:
        y_prob = model.decision_function(X_test)    

    docdf['label'] = y_pred
    docdf['probability'] = y_prob
    docdf['confidence']  = docdf.apply(lambda row: get_confidence(row), axis=1)
    #docdf['confidence'] = docdf['probability']

    infer_info['inference'] = docdf

    print("Max:{}, Min:{} of probabitilies".format(max(y_prob), min(y_prob)))

    return infer_info

def get_model():
    model_dict = {}

    print("Current directory  is: ", os.getcwd())   
    sys.stdout.flush() 
    preprocessor = nu.process_text
    vectorizer = pickle.load(open('../Inference/TFIDF-2vectorizer.pkl', 'rb'))
    model = pickle.load(open('../Inference//SVM_TFIDF-2.pkl', 'rb'))
    model_dict['preprocessor'] = preprocessor
    model_dict['vectorizer'] = vectorizer
    model_dict['model'] = model
    return model_dict
