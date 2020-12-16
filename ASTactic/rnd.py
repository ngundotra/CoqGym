import torch
import torch.nn as nn
from models.term_encoder import TermEncoder
from itertools import chain

class RandomDistillation(nn.Module):
    """
    The fixed method is the xavier uniform initialized method.
    """

    def __init__(self, opts, fixed=True):
        super().__init__()
        self.opts = opts
        self.term_encoder = TermEncoder(opts).to(opts.device)
        self.head = nn.Linear(in_features=opts.term_embedding_dim, out_features=1).to(opts.device)
        if fixed:
            torch.nn.init.xavier_uniform_(self.head.weight)
        else:
            torch.nn.init.normal_(self.head.weight)
    
    def forward(self, environment, local_context, goal, actions):
        embeddings = self.embed_terms(environment, local_context, goal)
        statics = []
        for emb in embeddings:
            print(emb.size())
            statics.append(self.head(emb.to(self.head.weight.device)))
        statics = torch.cat(statics)
        print("statics size:", statics.size())
        return statics

    def embed_terms(self, environment, local_context, goal):
        all_asts = list(chain([env['ast'] for env in chain(*environment)], [context['ast'] for context in chain(*local_context)], goal))
        return self.term_encoder(all_asts)

    @staticmethod
    def compare(s1, s2):
        dist = torch.norm(s1 - s2).mean()
        return dist