import torch
import torch.nn as nn
import numpy as np
from tac_grammar import CFG
from .tactic_decoder import TacticDecoder
from .term_encoder import TermEncoder
import pdb
import os
from itertools import chain
import sys
sys.path.append(os.path.abspath('.'))
from time import time


class Prover(nn.Module):

    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.tactic_decoder = TacticDecoder(CFG(opts.tac_grammar, 'tactic_expr'), opts)
        self.term_encoder = TermEncoder(opts)
        

    def freeze_encoding(self):
        """
        Freeze the encoding parameters so we can measure novelty of states during exploration.
        """
        for param in self.term_encoder.parameters():
            param.requires_grad = False

    def embed_terms(self, environment, local_context, goal):
        all_asts = list(chain([env['ast'] for env in chain(*environment)], [context['ast'] for context in chain(*local_context)], goal))
        all_embeddings = self.term_encoder(all_asts)

        batchsize = len(environment)
        environment_embeddings = []
        j = 0
        for n in range(batchsize):
            size = len(environment[n])
            environment_embeddings.append(torch.cat([torch.zeros(size, 3, device=self.opts.device), 
                                                     all_embeddings[j : j + size]], dim=1))
            environment_embeddings[-1][:, 0] = 1.0
            j += size

        context_embeddings = []
        for n in range(batchsize):
            size = len(local_context[n])
            context_embeddings.append(torch.cat([torch.zeros(size, 3, device=self.opts.device), 
                                                 all_embeddings[j : j + size]], dim=1))
            context_embeddings[-1][:, 1] = 1.0
            j += size

        goal_embeddings = []
        for n in range(batchsize):
            goal_embeddings.append(torch.cat([torch.zeros(3, device=self.opts.device), all_embeddings[j]], dim=0))
            goal_embeddings[-1][2] = 1.0
            j += 1
        goal_embeddings = torch.stack(goal_embeddings)

        return environment_embeddings, context_embeddings, goal_embeddings

    @staticmethod
    def _dist_embeddings(embedding1, embedding2):
        """
        Computes cosine distance between embeddings.

        Returns 1/3 * (sum {cos_dist(e1, e2)} over env, context and goal embeddings)
        """
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        dist = 0
        for (e1, e2) in zip(embedding1, embedding2):
            cos_dists = []
            for (emb1, emb2) in zip(e1, e2):
                # Context embedding sometimes has 0 rows
                if len(emb1) == 0 or len(emb2) == 0:
                    continue
                if len(emb1.size()) < 2 or len(emb2.size()) < 2:
                    emb1 = emb1.unsqueeze(0)
                    emb2 = emb2.unsqueeze(0)

                # yes this is practically CosineEmbeddingLoss, but creating
                # the necessary y-tensor of ones seemed tedious
                dist = (1 - cos(emb1, emb2)).detach()
                cos_dists.append(dist)
            if len(cos_dists) > 0:
                dist += torch.mean(torch.cat(cos_dists)) / 3
        return dist

    def forward(self, environment, local_context, goal, actions, teacher_forcing):
        environment_embeddings, context_embeddings, goal_embeddings = \
          self.embed_terms(environment, local_context, goal)
        environment = [{'idents': [v['qualid'] for v in env], 
                        'embeddings': environment_embeddings[i], 
                        'quantified_idents': [v['ast'].quantified_idents for v in env]}
                          for i, env in enumerate(environment)]
        local_context = [{'idents': [v['ident'] for v in context], 
                          'embeddings': context_embeddings[i],
                          'quantified_idents': [v['ast'].quantified_idents for v in context]}
                            for i, context in enumerate(local_context)]
        goal = {'embeddings': goal_embeddings, 'quantified_idents': [g.quantified_idents for g in goal]}
        asts, loss = self.tactic_decoder(environment, local_context, goal, actions, teacher_forcing)
        return asts, loss


    def beam_search(self, environment, local_context, goal, sampling=None):
        environment_embeddings, context_embeddings, goal_embeddings = \
          self.embed_terms([environment], [local_context], [goal])
        environment = {'idents': [v['qualid'] for v in environment],
                       'embeddings': environment_embeddings[0],
                       'quantified_idents': [v['ast'].quantified_idents for v in environment]}
        local_context = {'idents': [v['ident'] for v in local_context],
                         'embeddings': context_embeddings[0],
                         'quantified_idents': [v['ast'].quantified_idents for v in local_context]}

        if len(goal_embeddings) != 1:
            print("Goal embeddings length:", len(goal_embeddings))
        goal = {'embeddings': goal_embeddings, 'quantified_idents': goal.quantified_idents}
        if sampling == "PG":
            asts = self.tactic_decoder.beam_search_train(environment, local_context, goal)
            # asts = self.tactic_decoder.simple_search(environment, local_context, goal)
        elif sampling == "DFS":
            asts = self.tactic_decoder.beam_search_train(environment, local_context, goal)
        else:
            asts = self.tactic_decoder.beam_search(environment, local_context, goal)
        return asts
