from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
import json
import sys
import os
import codecs
reload(sys)

sys.setdefaultencoding('utf8')


def _get_json_format(pred_sents, name):
    '''
    Format to be written in json file. Excepted by coco lib
    '''
    data_pred = []
    for id, s in enumerate(pred_sents):
        line = {}
        line['image_id'] = id
        line['caption'] = s
        data_pred.append(line)

    return [data_pred, name]


def loadJsonToMap(json_data):
    data = json_data
    imgToAnns = {}
    for entry in data:
        # print entry['image_id'],entry['caption']
        if entry['image_id'] not in imgToAnns.keys():
            imgToAnns[entry['image_id']] = []
        summary = {}
        summary['caption'] = entry['caption']
        summary['image_id'] = entry['caption']
        imgToAnns[entry['image_id']].append(summary)
    return imgToAnns


class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.keys()}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco[imgId]  # .imgToAnns[imgId]
            res[imgId] = self.cocoRes[imgId]  # .imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print 'tokenization...'
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print 'setting up scorers...'
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        ]

        # =================================================
        # Compute scores
        # =================================================
        final_score = 0
        for scorer, method in scorers:
            print 'computing %s score...' % (scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
                    print "%s: %0.3f" % (m, sc)
                    final_score = sc
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
                print "%s: %0.3f" % (method, score)
        self.setEvalImgs()
        return final_score

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


def evaluate(annFile, resFile, phase_codename, **kwargs):

    print "Starting evaluation...."
    print "Reading annotation file..."
    file_1 = open(annFile)
    print "Annotations read successfully."
    print "Reading submitted file..."
    file_2 = open(resFile)
    print "Successfully read submission file."

    ref_sents = file_1.readlines()
    pred_sents = file_2.readlines()

    file_1 = _get_json_format(ref_sents, 'reference')[0]
    file_2 = _get_json_format(pred_sents, 'predicted')[0]

    print "Evaluating now..."
    coco = loadJsonToMap(file_1)
    cocoRes = loadJsonToMap(file_2)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.keys()

    bleu_score = cocoEval.evaluate()
    print "Bleu Score: ", bleu_score
    print "Successfully completed the process of evaluation."
    output = {}

    # The data stored in 'host_metadata' dict will 
    # only be visible to the challenge host
    host_metadata = {"submission_details": kwargs['submission_metadata']}
    host_metadata['predicted'] = file_2

    output['result'] = [
        {
            'val': {
                'bleu': bleu_score,
            }
        }
    ]
    host_metadata['result'] = output['result']

    output['submission_metadata'] = host_metadata
    return output
