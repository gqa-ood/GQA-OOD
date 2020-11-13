import argparse

import json
import os
import numpy as np
import random
import copy

from utils import load_json, save_json, entropy, std, n_entropy
from paths import *
from tqdm import tqdm

class GQAdataset():
    
    def __init__(self, fast=False, data=None, ver='val'):

        self.ver = ver
        data_path = {'val': val_bal_questions,
                     'testdev': testdev_bal_questions}

        self.stats = {  'groups_global': {},
                        'groups_local': {},
                        'types_semantic': {},
                        'types_structural': {},
                        'types_detailed': {},
                        }
        self.answers_distribution = {   'overall': {},
                                        'per_groups_global': {}}
        self.answer_local_groups = {}
        self.qid2answergroup = {}
        self.count_answer_local_groups = 0
        self.count_local_groups = 0
        self.qid2imgid = {}
        self.imgid2qid = {}
        self.qids = []
        self.qids2idx = {}

        # Load Questions and annotations
        if data is None:
            self.data = load_json(data_path[ver])
            print('Load %d questions!' % len(self.data))
            if fast:  # debug
                fast_keys = list(self.data.keys())[:512]
                self.data = {k:self.data[k] for k in fast_keys}
        else:
            self.data = data

    def save(self, name, path):
        print("Saving %s in %s..."%(name, path))
        filename = os.path.join(path, '%s.json' % name)
        with open(filename, 'w') as fp:
            json.dump(self.data, fp, sort_keys=True, indent=4)
        print('Done!')

    def length(self):
        return len(self.qid2imgid)

    def statistics(self, all_stats=False):

        for qid, data in tqdm(self.data.items(), desc='Computing stats...'):
            # question distribution
            if all_stats:
                self.stat_counter('types_semantic', data['types']['semantic'])
                self.stat_counter('types_structural', data['types']['structural'])
                self.stat_counter('types_detailed', data['types']['detailed'])

                # answer distribution
                self.ans_counter('overall', data['answer'])
                self.ans_counter('group_global_'+self.filter_none(data['groups']['global']), data['answer'])
                self.ans_counter('types_semantic_'+self.filter_none(data['types']['semantic']), data['answer'])
                self.ans_counter('types_structural_'+self.filter_none(data['types']['structural']), data['answer'])
                self.ans_counter('group_local_'+self.filter_none(data['groups']['local']), data['answer'])

            # others
            self.qid2imgid[qid] = data['imageId']
            if data['imageId'] not in self.imgid2qid:
                self.imgid2qid[data['imageId']] = [qid]
            else:
                self.imgid2qid[data['imageId']] += [qid]
            self.stat_counter('groups_local', data['groups']['local'])
            self.stat_counter('groups_global', data['groups']['global'])
        
        self.count_local_groups = len(self.stats['groups_local'])

    def filter_none(self, x):
        if x is None:
            return 'none'
        else:
            return x

    def stat_counter(self, key, cls):
        cls = self.filter_none(cls)
        if cls not in self.stats[key]:
            self.stats[key][cls] = 1
        else:
            self.stats[key][cls] += 1

    def ans_counter(self, key, cls):
        cls = self.filter_none(cls)
        if key not in self.answers_distribution:
            self.answers_distribution[key] = {}
        if cls not in self.answers_distribution[key]:
            self.answers_distribution[key][cls] = 1
        else:
            self.answers_distribution[key][cls] += 1    
    
    def normalize(self, dict):
        total = sum(dict.values())
        dict_norm = {k: v / float(total) for k,v in dict.items()}
        return dict_norm

    def add_qid_to_answer_local_group(self, local_group, answer, qid):
        if answer in self.answer_local_groups[local_group]:
            self.answer_local_groups[local_group][answer] += [qid]
        else:  # create new answer local group
            self.answer_local_groups[local_group][answer] = [qid]
            self.count_answer_local_groups += 1

    def build_answer_local_groups(self):
        # init
        for local_group, count in self.stats['groups_local'].items():
            self.answer_local_groups[self.filter_none(local_group)] = {}
        
        # iterate on dataset
        index = 0
        for qid, data in tqdm(self.data.items(), desc='Constructing answer local groups...'): 
            answer      = data['answer']
            local_group = self.filter_none(data['groups']['local'])
            self.add_qid_to_answer_local_group(local_group, answer, qid)
            self.qids += [qid]
            self.qids2idx[qid] = index
            index += 1
            self.qid2answergroup[qid] = "%s/%s" % (local_group, answer)

        print('%d answer local groups have been created!'%self.count_answer_local_groups)


class VQA_SET():
    """ Used to store a set of (question, answer, image) """

    def __init__(self, name):

        self.name = name
        self.qids   = []
        self.imgids = []
        self.qid2imgid = {}

    def save(self, gqa_dataset, path):
        self.to_gqaDataset(gqa_dataset)
        self.dataset.save(self.name, path)

    def to_gqaDataset(self, gqadataset):
        """ Transform to the GQA dataset format (with all standard annotations) """

        # gather data information from original dataset
        data = {}
        for qid in self.qids:
            data[qid] = gqadataset.data[qid]
        # create gqadataset from the current set
        self.dataset = GQAdataset(fast=False, data=data)

    def add_question_batch(self, qid_batch, gqa_dataset,):
        for qid in qid_batch:
            imgid = gqa_dataset.qid2imgid[qid]
            if not imgid in self.imgids:  # add imgid
                self.imgids += [imgid]
            self.add_question(qid, imgid)

    def add_question(self, qid, imgid):
        self.qids += [qid]
        self.qid2imgid[qid] = imgid

    def display_stats(self):
        print("Set [%s] is composed of %d questions and %d images."%(self.name, len(self.qids), len(self.imgids)))

class GQA_OOD():
    
    def __init__(self, GQAdataset, alpha):

        self.ver = GQAdataset.ver
        self.all = VQA_SET(name="%s_all"%self.ver)
        self.head = VQA_SET(name="%s_head_%.2f"%(self.ver, alpha))
        self.tail = VQA_SET(name="%s_tail_%.2f"%(self.ver, alpha))

        self.gqa_dataset = GQAdataset
        self.answer_local_groups = {}
        self.group_entropy ={}

        self.nb_questions = self.gqa_dataset.length()
        self.nb_images = len(self.gqa_dataset.imgid2qid)

        self.alpha = alpha

    def construct_sets(self):
        
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # 1) Compute gqa's answer distribution  per question group
        #    and  compute their imbalance using shanon's entropy
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        # Get answer histogram using GQA annotations
        # We group questions by answers in local groups: [local_group] -> [answer_group] -> [questions]
        per_local_group_gqa_dist = {}
        for group_name, local_group in self.gqa_dataset.answer_local_groups.items():
            per_local_group_gqa_dist[group_name] = {}
            for answer, answer_group in local_group.items():
                per_local_group_gqa_dist[group_name][answer] = len(answer_group)

        nb_groups = sum([len(group) for group in per_local_group_gqa_dist.values()])
        nb_questions = sum([sum(group.values()) for group in per_local_group_gqa_dist.values()])

        # Compute the answer distribution for each local group
        self.gqa_dist = {}
        self.dist_label = {}
        for group_name, local_group in per_local_group_gqa_dist.items():
            dist_vector = np.array(list(local_group.values()))
            dist_vector = dist_vector / dist_vector.sum()  # normalize
            self.gqa_dist[group_name] = dist_vector
            self.dist_label[group_name] = list(local_group.keys())
            self.group_entropy[group_name] = n_entropy(dist_vector) # get group's imabalance
        print('\n1) GQA question group distribution has been computed!')

        
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # 2) Discard the less imbalanced groups
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 
        mean_entropy = np.array(list(self.group_entropy.values())).mean()
        std_entropy = np.array(list(self.group_entropy.values())).std()
        imbalanced_treshold = mean_entropy - std_entropy  # discard all groups weaker than this treshold
        self.group_imbalanced = {k:(v<=imbalanced_treshold) for k,v in self.group_entropy.items()}
        print("\n2) Less imbalanced question groups have been discarde!")

        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # 3) From the remaining groups, extract head and tail questions
        # * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        for group_name, local_group_dist in tqdm(self.gqa_dist.items(), desc="Constructing tail, head and all..."):  # iterate on question groups

            # skip if not imbalanced
            if not self.group_imbalanced[group_name]:
                continue

            avg_ans_count = local_group_dist.mean()
            tail_treshold = avg_ans_count * (self.alpha + 1)

            local_group = self.gqa_dataset.answer_local_groups[group_name]  # [local_group] -> [answer_group] -> [questions]
                
            for idx, (answer, questions) in enumerate(local_group.items()):  # iterate on answers, each answer has N questions
                
                ans_count = self.gqa_dist[group_name][idx]

                if ans_count < tail_treshold:
                    # Tail
                    self.tail.add_question_batch(questions, self.gqa_dataset)
                else:
                    # Head
                    self.head.add_question_batch(questions, self.gqa_dataset)

                # In both cases, add to All:
                self.all.add_question_batch(questions, self.gqa_dataset)

        print("\n3) Sets have been constructed:\n")
        self.all.display_stats()
        self.tail.display_stats()
        self.head.display_stats()

    def save_sets(self, path):
        self.all.save(self.gqa_dataset, path)
        self.head.save(self.gqa_dataset, path)
        self.tail.save(self.gqa_dataset, path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GQA-OOD options.')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'testdev'], help='Which split to construct')
    parser.add_argument('--alpha', type=float, default=0.2, help='Size of the tail.')
    args = parser.parse_args()

    # Load original GQA dataset
    gqa = GQAdataset(fast=False, ver=args.split)
    gqa.statistics()
    gqa.build_answer_local_groups()

    # Construct OOD version
    gqa_ood = GQA_OOD(gqa, args.alpha)
    gqa_ood.construct_sets()
    gqa_ood.save_sets('../data/')

