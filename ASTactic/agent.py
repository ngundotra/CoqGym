import numpy as np
import torch
import torch.nn.functional as F
import os
from gallina import GallinaTermParser
from utils import SexpCache, log
from eval_env import FileEnv
import re
import pickle
from progressbar import ProgressBar
from glob import glob
import json
from random import random
import pdb
from hashlib import sha1
import gc
from copy import deepcopy
from time import time

# Custom Sampling Techniques (highly advanced, >9000 IQ)
from sampler import ParallelSampler

def action_seq_loss(logits_batch, actions_batch, opts):
    assert len(logits_batch) == len(actions_batch)
    loss = 0
    for logits, actions in zip(logits_batch, actions_batch):
        length = min(logits.shape[0], actions.shape[0])
        loss += F.cross_entropy(logits[:length], actions[:length].to(opts.device))
    loss /= len(logits_batch)
    return loss


# merge this with extract_proof_steps.py
term_parser = GallinaTermParser(caching=True)
sexp_cache = SexpCache('../sexp_cache', readonly=True)


def filter_env(env):
    filtered_env = []
    for const in [const for const in env['constants'] if const['qualid'].startswith('SerTop')][-10:]:
        ast = sexp_cache[const['sexp']]
        filtered_env.append({'qualid': const['qualid'], 'ast': term_parser.parse(ast)})
    return filtered_env


def parse_goal(g):
    goal = {'id': g['id'], 'text': g['type'], 'ast': term_parser.parse(g['sexp'])}
    local_context = []
    for i, h in enumerate(g['hypotheses']):
        for ident in h['idents']:
            local_context.append({'ident': ident, 'text': h['type'], 'ast': term_parser.parse(h['sexp'])})
    return local_context, goal['ast']


def print_single_goal(g):
    for h in g['hypotheses']:
        for ident in h['idents']:
            print('\t%s: %s' % (ident, h['type']))
    print('---------------')
    print('\t%s' % g['type'])
    print('##########')


def print_goals(obs):
    if 'fg_goals' not in obs:
        print('##########')
        return
    print('########## fg_goals ##########')
    for g in obs['fg_goals']:
        print_single_goal(g)
    print('########## bg_goals ##########')
    for g in obs['bg_goals']:
        print_single_goal(g)
    print('########## shelved_goals ##########')
    for g in obs['shelved_goals']:
        print_single_goal(g)
    print('########## given_up_goals ##########')
    for g in obs['given_up_goals']:
        print_single_goal(g)


def get_goal_signature(goal):
    sexp = goal['sexp'] + ''.join([h['sexp'] for h in goal['hypotheses']])
    return sha1(sexp.encode('utf-8')).hexdigest()


class Agent:

    def __init__(self, model, optimizer, dataloader, opts):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.opts = opts
        self.projs_split = json.load(open(opts.projs_split))

    def train(self, n_epoch):
        self.model.train()
        log('training with teacher forcing %f..' % self.opts.teacher_forcing)

        bar = ProgressBar(max_value=len(self.dataloader['train']))
        for i, data_batch in enumerate(self.dataloader['train']):
            use_teacher_forcing = random() < self.opts.teacher_forcing
            asts, loss = self.model(data_batch['env'], data_batch['local_context'],
                                    data_batch['goal'], data_batch['tactic_actions'], use_teacher_forcing)
            log('\nteacher forcing = %s, loss = %f' % (str(use_teacher_forcing), loss.item()))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            gc.collect()
            bar.update(i)
            if self.opts.smoke and i == 11:
                break

        log('\ntraining losses: %f' % loss)

    def train_RL(self, n_epoch, filename, logger, proof_name=None, sample='DFS'):
        """
        TODO:
            - reset the env
            - perform batched update at the end of an epoch (?)
            - 150 trajectories per update (?)
            - produce residual reward by # of open & closed goals from each environment (?)
        """
        
        self.model.train()
        log('training with {}'.format(sample))

        if 'hammer' in self.opts.method:
            for atp in ['Vampire', 'Z3', 'CVC4', 'Eprover']:
                if ('hammer_' + atp) in self.opts.method:
                    with_hammer = atp
                    self.opts.method = self.opts.method.replace('hammer_' + atp, 'hammer')
                    break
            else:
                with_hammer = 'All'
        else:
            with_hammer = None
        assert 'hammer_' not in self.opts.method
        hammer_timeout = self.opts.hammer_timeout if 'ours' in self.opts.method else self.opts.timeout

        # TODO: train with training `data_batch` instead. Create `proof_env` for each data.
        # {proof_name: [lowest loss, success]}
        if sample == "vanilla":
            results, total_collected = self.train_RL_PG(n_epoch, 5, filename, logger, with_hammer, hammer_timeout)
            return results + [total_collected]
        elif sample == "DFS":
            return self.train_RL_DFS(n_epochs, with_hammer, hammer_timeout)
        else:
            raise ValueError('Sampling method not found.')

    def train_RL_PG(self, n_epoch, epochs_per_update, filename, logger, with_hammer, hammer_timeout, use_dfs=False):
        """
        Collects hella samples for Policy Gradients.
        Uses parallel workers if `opts.parallel`
        """
        tac_template = self.get_tac_template()
        file_env_args = (filename, self.opts.max_num_tactics, self.opts.timeout, with_hammer, hammer_timeout)

        loss = None
        total_collected = 0 
        all_results = []
        loss_graph = []
        for ep in range(n_epoch):
            print("\n>>>>>>>>>>>>>>>>>>>>EPOCH: {}<<<<<<<<<<<<<<<<<<<<<<\n".format(ep))
            results, grads, collected, losses = self.sample_parallel(epochs_per_update, tac_template=tac_template, file_env_args=file_env_args, train=True)
            all_results += results
            total_collected += collected
            
            avg_loss = sum(losses)/len(losses)

            for idx, (layer, grad) in enumerate(zip(self.model.parameters(), grads)):
                if grad is not None:
                    for p, g in zip(layer, grad):
                        p.grad = g / collected
            
            # losses_env = [((-logprob)
            #                 * (reward).to(logprob.device)).unsqueeze(0)
            #                 for logprob, reward in samples]
            # Do loss
            # if loss is None:
            #     loss = torch.cat(losses_env).mean()
            # else:
            #     loss += torch.cat(losses_env).mean()
            # if torch.isnan(loss):
            #     print("=======NAN=======")
            #     pdb.set_trace()
            # Update
            # print("\tLoss: {}".format(loss.item()))
            # self.optimizer.zero_grad()
            # loss.backward()
            print("\tEpoch loss{}: {}".format(ep, avg_loss))
            logger.log_value('loss', avg_loss, ep)
            loss_graph.append(avg_loss)
            self.optimizer.step()
        self.save(n_epoch, "train-PG-ckpt/")
        
        return results, total_collected

    def train_RL_DFS(self, n_epoch, with_hammer, hammer_timeout):
        """
        TODO: put in `RL_Trainer` sort of file ...?
        Collects samples & updates the model in accordance with DFS sampling
        """
        for curr_epoch in range(n_epoch):
            print("\n---EPOCH---")
            print("-----{}-----\n".format(curr_epoch))
            with FileEnv(filename, self.opts.max_num_tactics, self.opts.timeout, with_hammer=with_hammer,
                         hammer_timeout=hammer_timeout) as file_env:
                results = []
                loss = None
                for proof_env in file_env:  # start a proof
                    curr_name = proof_env.proof['name']
                    if proof_name is not None and curr_name != proof_name:
                        continue
                    print('proof: ', proof_env.proof['name'])
                    # success, proof_pred, time, num_tactics, trajectory = self.prove(proof_env, train=True)
                    samples, Nsa, Ns = self.prove(proof_env, train=True, sample=sample) # TODO: control number of samples better

                    losses_env = [((Ns[state]/Nsa[state][action]).to(logprob.device)
                                    * torch.exp(logprob)
                                    * (-logprob)
                                    * (reward).to(logprob.device)).unsqueeze(0)
                                    for state, action, logprob, reward in samples]

                    if loss is None:
                        loss = torch.cat(losses_env).mean()
                    else:
                        loss += torch.cat(losses_env).mean()
                    if torch.isnan(loss):
                        print("=======NAN=======")
                        pdb.set_trace()

                    if proof_name is not None:
                        break

            print("\tLoss: {}".format(loss.item()))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.save(n_epoch, "train-ckpt/")
        return results

    def valid(self, n_epoch):
        self.model.eval()
        log('validating..')
        loss_avg = 0
        predictions = []
        num_correct = 0
        bar = ProgressBar(max_value=len(self.dataloader['valid']))

        for i, data_batch in enumerate(self.dataloader['valid']):
            asts, loss = self.model(data_batch['env'], data_batch['local_context'],
                                    data_batch['goal'], data_batch['tactic_actions'], False)
            loss_avg += loss.item()

            for n in range(len(data_batch['file'])):
                tac_gt = data_batch['tactic_str'][n]
                tac_pred = asts[n].to_tokens()
                if tac_gt.replace(' ', '') == tac_pred.replace(' ', ''):
                    num_correct += 1
                predictions.append({'file_name': data_batch['file'][n],
                                    'proof_name': data_batch['proof_name'][n],
                                    'n_step': data_batch['n_step'][n],
                                    'tac_gt': tac_gt,
                                    'tac_pred': tac_pred})
            gc.collect()
            bar.update(i)
            if self.opts.smoke and i == 11:
                break

        pickle.dump(predictions, open(os.path.join(self.opts.log_dir, 'predictions/pred_%03d.pickle' % n_epoch), 'wb'))

        loss_avg /= len(self.dataloader['valid'])
        log('\nvalidation losses: %f' % loss_avg)
        acc = num_correct / len(predictions)
        log('validation accuracy: %f' % acc)
        return loss_avg

    def gloop_evaluate(self, filename, proof_name=None):
        if self.model is not None:
            self.model.eval()

        if 'hammer' in self.opts.method:
            for atp in ['Vampire', 'Z3', 'CVC4', 'Eprover']:
                if ('hammer_' + atp) in self.opts.method:
                    with_hammer = atp
                    self.opts.method = self.opts.method.replace('hammer_' + atp, 'hammer')
                    break
            else:
                with_hammer = 'All'
        else:
            with_hammer = None
        assert 'hammer_' not in self.opts.method
        hammer_timeout = self.opts.hammer_timeout if 'ours' in self.opts.method else self.opts.timeout

        with FileEnv(filename, self.opts.max_num_tactics, self.opts.timeout, with_hammer=with_hammer,
                     hammer_timeout=hammer_timeout) as file_env:
            results = []
            # Combine constants, inductives, and foreground goals
            proof_env = file_env.coagulated_env()

            if proof_name is not None and proof_env.proof['name'] != proof_name:
                return results
            print('proof: ', proof_env.proof['name'])
            # print('cuda memory allocated before proof: ', torch.cuda.memory_allocated(self.opts.device), file=sys.stderr)
            success, proof_pred, time, num_tactics = self.prove(proof_env)

            # Append separate proof per goal n coagulated environment
            results.append({
                'filename': filename, 'proof_name': proof_env.proof['name'], 'success': success,
                'proof_gt': [step['command'][0] for step in proof_env.proof['steps'] if
                             step['command'][1] != 'VernacEndProof'],
                'proof_pred': proof_pred,
                'time': time,
                'num_tactics': num_tactics, })
            if proof_name is not None:
                return results
        return results

    def evaluate(self, filename, proof_name=None):
        if self.model is not None:
            self.model.eval()

        if 'hammer' in self.opts.method:
            for atp in ['Vampire', 'Z3', 'CVC4', 'Eprover']:
                if ('hammer_' + atp) in self.opts.method:
                    with_hammer = atp
                    self.opts.method = self.opts.method.replace('hammer_' + atp, 'hammer')
                    break
            else:
                with_hammer = 'All'
        else:
            with_hammer = None
        assert 'hammer_' not in self.opts.method
        hammer_timeout = self.opts.hammer_timeout if 'ours' in self.opts.method else self.opts.timeout

        with FileEnv(filename, self.opts.max_num_tactics, self.opts.timeout, with_hammer=with_hammer,
                     hammer_timeout=hammer_timeout) as file_env:
            results = []
            for proof_env in file_env:  # start a proof
                if proof_name is not None and proof_env.proof['name'] != proof_name:
                    continue
                print('proof: ', proof_env.proof['name'])
                # print('cuda memory allocated before proof: ', torch.cuda.memory_allocated(self.opts.device), file=sys.stderr)
                success, proof_pred, time, num_tactics = self.prove(proof_env)
                results.append({
                    'filename': filename, 'proof_name': proof_env.proof['name'], 'success': success,
                    'proof_gt': [step['command'][0] for step in proof_env.proof['steps'] if
                                 step['command'][1] != 'VernacEndProof'],
                    'proof_pred': proof_pred,
                    'time': time,
                    'num_tactics': num_tactics, })
                if proof_name is not None:
                    break
        return results

    def prove_one_tactic(self, proof_env, tac):
        obs = proof_env.init()
        print_goals(obs)
        obs = proof_env.step(tac + '.')
        print(obs['result'])
        print_goals(obs)
        time = self.opts.timeout - obs['time_left']
        if obs['result'] == 'SUCCESS':
            return True, [tac], time, 1
        else:
            return False, [tac], time, 1

    def prove(self, proof_env, train=False, sample='DFS'):
        'prove a theorem interactively'
        if 'ours' not in self.opts.method:  # auto, hammer, etc.
            return self.prove_one_tactic(proof_env, self.opts.method)

        tac_template = self.get_tac_template()
        # pdb.set_trace()
        if train:
            if sample == 'DFS':
                return self.sample_DFS(proof_env, tac_template, train=True)
            else:
                raise ValueError('Sampling method not found.')
        return self.prove_DFS(proof_env, tac_template)

    def sample_parallel(self, epochs, file_env_args, tac_template, train=False):
        parallel_sampler = ParallelSampler(file_env_args, tac_template, self, train)        
        return parallel_sampler.sample_trajectories(epochs)
    
    def sample(self, epochs, tac_template, train=False, file_env_args=None, proof_env=None):
        if self.opts.parallel_sample:
            assert file_env_args is not None
            return self.sample_parallel(epochs, file_env_args, tac_template, train)
        if proof_env:
            assert file_env_args is None and epochs == 1
            return self.sample_once(proof_env, tac_template, train)
        raise NotImplementedError

    def sample_once(self, proof_env, tac_template, train=False):
        obs = proof_env.init()
        env = filter_env(obs['env'])

        if 'fg_goals' not in obs:
            print(proof_env.proof['name'])
            pdb.set_trace()
        first_goal_signatures = {get_goal_signature(obs['fg_goals'][0])}

        # store logprobs (along the trajectory) to be rewarded later
        prob_list = []

        # initialize
        local_context, goal = parse_goal(obs['fg_goals'][0])
        tactics = self.model.beam_search(env, local_context, goal, train)
        tacs = [tac_template % tac.to_tokens() for tac, _ in tactics]
        probs = torch.cat([prob.unsqueeze(0) for _, prob in tactics])
        script = []

        steps = 0
        while True and steps < 1e3:
            steps += 1

            m = torch.distributions.Categorical(probs)
            idx = m.sample()
            tac, prob = tacs[idx], probs[idx]

            prob_list.append(prob)
            obs = proof_env.step(tac)
            fg_goals, bg_goals, shelved_goals, _ = proof_env.serapi.query_goals()

            if obs['result'] == 'SUCCESS':
                script.append(tac)
                time = self.opts.timeout - obs['time_left']
                num_tactics = self.opts.max_num_tactics - obs['num_tactics_left']
                samples = [(logprob, 1.0) for logprob in prob_list]
                return {'samples': samples, 'results': (True, script, time, num_tactics)}
            elif obs['result'] in ['MAX_NUM_TACTICS_REACHED', 'MAX_TIME_REACHED']:
                time = self.opts.timeout - obs['time_left']
                num_tactics = self.opts.max_num_tactics - obs['num_tactics_left']
                samples = [(logprob, -0.1) for logprob in prob_list] #TODO: set reward to 0 or -0.1?
                return {'samples': samples, 'results': (False, script, time, num_tactics)}
            elif obs['result'] in ['ERROR']:  # Tactic is misapplied, nothing happened
                # samples = [(logprob, -0.1) for logprob in prob_list]
                # return samples
                continue
            else:
                assert obs['result'] == 'PROVING'
                script.append(tac)
                sig = get_goal_signature(obs['fg_goals'][0]) #TODO: should we care about this in sampling?
                if sig in first_goal_signatures:
                    proof_env.step('Undo.')
                    script.pop()
                    continue
                first_goal_signatures.add(sig)

                if len(script) >= self.opts.depth_limit:
                    samples = [(logprob, -0.1) for logprob in prob_list]  #TODO: set reward to 0 or -0.1?
                    return samples

                local_context, goal = parse_goal(obs['fg_goals'][0])
                tactics = self.model.beam_search(env, local_context, goal, train)
                tacs = [tac_template % tac.to_tokens() for tac, _ in tactics]
                probs = torch.cat([prob.unsqueeze(0) for _, prob in tactics])

        raise RuntimeError('huh?')

    def sample_DFS(self, proof_env, tac_template, train=True):
        """
        Single attempt to prove something
        ENDS
            - when error happens
            - when success is reached
            - when timelimit hit
        """
        # number of time state, action tuple is visited
        Nsa = {}
        # number of time state is visited
        Ns = {}

        obs = proof_env.init()
        env = filter_env(obs['env'])
        # pdb.set_trace()
        if 'fg_goals' not in obs:
            print(proof_env.proof['name'])
            pdb.set_trace()
        first_goal_signatures = {get_goal_signature(obs['fg_goals'][0])}

        # store samples to be returned
        sample_list = []

        # store logprobs (along the trajectory) to be rewarded later
        prob_list = []

        # initialize
        local_context, goal = parse_goal(obs['fg_goals'][0])
        tactics = self.model.beam_search(env, local_context, goal, train)
        stack = [[(tac_template % tac.to_tokens(), prob) for tac, prob in tactics[::-1]]]
        script = []

        # depth-first search starting from the trace
        while stack != [[]]:
            # print('stack: ', stack)
            # pick a tactic
            if stack[-1] == []:  # all candidate have been tried, backtrack
                stack.pop()
                script.pop()
                proof_env.step('Undo.')
                continue
            else:
                tac, logprob = stack[-1].pop()

            # -----Exploration-----
            # obs_string = None #TODO: a string identifier for the current obs
            # tac_string = None #TODO: a string identifier for the current tac
            # f - f_hat
            # f is input -> embedding
            # maximize(f-f_hat)

            # Nsa
            if obs_string not in Nsa.keys():
                Nsa[obs_string] = {}
            if tac_string not in Nsa[obs_string].keys():
                Nsa[obs_string][tac_string] = 1
            else:
                Nsa[obs_string][tac_string] += 1

            # Ns
            if obs_string not in Ns.keys():
                Ns[obs_string] = 1
            else:
                Ns[obs_string] += 1

            prob_list.append((obs_string, tac_string, logprob))
            obs = proof_env.step(tac)
            fg_goals, bg_goals, shelved_goals, _ = proof_env.serapi.query_goals()

            if obs['result'] == 'SUCCESS':
                script.append(tac)
                samples = [(obs_string, tac_string, logprob, 1) for obs_string, tac_string, logprob in prob_list]
                sample_list.extend(samples)
                prob_list.pop(-1)
                proof_env.serapi.pop() #TODO: find out if this works
                continue
            elif obs['result'] in ['MAX_NUM_TACTICS_REACHED', 'MAX_TIME_REACHED']:
                # TODO: reassure that its the total number of tactics over the whole beam search but not along the trajectory.
                #       Change the reward to -0.1 if its the latter.
                samples = [(obs_string, tac_string, logprob, 0) for obs_string, tac_string, logprob in prob_list]
                sample_list.extend(samples)
                return sample_list, Nsa, Ns
            elif obs['result'] in ['ERROR']:  # Tactic is misapplied, nothing happened
                samples = [(obs_string, tac_string, logprob, -0.1) for obs_string, tac_string, logprob in prob_list]  # TODO: scale the reward for failing
                sample_list.extend(samples)
                prob_list.pop(-1)
                continue
            else:
                script.append(tac)
                sig = get_goal_signature(obs['fg_goals'][0])
                if sig in first_goal_signatures or len(script) >= self.opts.depth_limit:
                    proof_env.step('Undo.')
                    script.pop()
                    continue
                first_goal_signatures.add(sig)
                local_context, goal = parse_goal(obs['fg_goals'][0])
                tactics = self.model.beam_search(env, local_context, goal, train)
                stack.append([(tac_template % tac.to_tokens(), prob) for tac, prob in tactics[::-1]])

        obs = proof_env.step('Admitted.')
        # print(obs['result'])
        return sample_list, Nsa, Ns

    def prove_DFS(self, proof_env, tac_template):
        obs = proof_env.init()
        env = filter_env(obs['env'])
        first_goal_signatures = {get_goal_signature(obs['fg_goals'][0])}

        # initialize the stack
        local_context, goal = parse_goal(obs['fg_goals'][0])
        tactics = self.model.beam_search(env, local_context, goal)
        stack = [[tac_template % tac.to_tokens() for tac in tactics[::-1]]]
        script = []
        # pdb.set_trace()

        # depth-first search starting from the trace
        while stack != [[]]:
            # print('stack: ', stack)
            # pick a tactic
            if stack[-1] == []:  # all candidate have been tried, backtrack
                stack.pop()
                script.pop()
                proof_env.step('Undo.')
                continue
            else:
                tac = stack[-1].pop()

            obs = proof_env.step(tac)
            # print(obs['result'])
            # print_goals(obs)

            if obs['result'] == 'SUCCESS':
                script.append(tac)
                time = self.opts.timeout - obs['time_left']
                num_tactics = self.opts.max_num_tactics - obs['num_tactics_left']
                return True, script, time, num_tactics
            elif obs['result'] in ['MAX_NUM_TACTICS_REACHED', 'MAX_TIME_REACHED']:
                time = self.opts.timeout - obs['time_left']
                num_tactics = self.opts.max_num_tactics - obs['num_tactics_left']
                return False, script, time, num_tactics
            elif obs['result'] in ['ERROR']:
                continue
            else:
                assert obs['result'] == 'PROVING'
                script.append(tac)
                sig = get_goal_signature(obs['fg_goals'][0])
                if sig in first_goal_signatures or len(script) >= self.opts.depth_limit:
                    proof_env.step('Undo.')
                    script.pop()
                    continue
                first_goal_signatures.add(sig)
                local_context, goal = parse_goal(obs['fg_goals'][0])
                tactics = self.model.beam_search(env, local_context, goal)
                stack.append([tac_template % tac.to_tokens() for tac in tactics[::-1]])

        obs = proof_env.step('Admitted.')
        print(obs['result'])
        time = self.opts.timeout - obs['time_left']
        num_tactics = self.opts.max_num_tactics - obs['num_tactics_left']
        return False, script, time, num_tactics

    def prove_IDDFS(self, proof_env, tac_template):
        obs = proof_env.init()
        env = filter_env(obs['env'])
        first_goal_signatures = {get_goal_signature(obs['fg_goals'][0])}
        depth_limit = self.opts.depth_limit
        traces = [[]]

        # iterative deepening depth-first search
        while traces != []:
            # depth-first search with depth_limit
            new_traces = []  # the newly-discovered truncated proofs
            for script in traces:
                # execute the tactics in the trace
                for tac in script:
                    obs = proof_env.step(tac)
                print(obs['result'])
                print_goals(obs)
                if obs['result'] != 'PROVING':
                    assert obs['result'] in ['MAX_NUM_TACTICS_REACHED', 'MAX_TIME_REACHED']
                    time = self.opts.timeout - obs['time_left']
                    num_tactics = self.opts.max_num_tactics - obs['num_tactics_left']
                    return False, script, time, num_tactics
                # initialize the stack
                local_context, goal = parse_goal(obs['fg_goals'][0])
                tactics = self.model.beam_search(env, local_context, goal)
                stack = [[tac_template % tac.to_tokens() for tac, _ in tactics[::-1]]]

                # depth-first search starting from the trace
                while stack != [[]]:
                    print('stack: ', stack)
                    # pick a tactic
                    if stack[-1] == []:  # all candidate have been tried, backtrack
                        stack.pop()
                        script.pop()
                        proof_env.step('Undo.')
                        continue
                    else:
                        tac = stack[-1].pop()

                    obs = proof_env.step(tac)
                    print(obs['result'])
                    print_goals(obs)

                    if obs['result'] == 'SUCCESS':
                        script.append(tac)
                        time = self.opts.timeout - obs['time_left']
                        num_tactics = self.opts.max_num_tactics - obs['num_tactics_left']
                        return True, script, time, num_tactics
                    elif obs['result'] in ['MAX_NUM_TACTICS_REACHED', 'MAX_TIME_REACHED']:
                        time = self.opts.timeout - obs['time_left']
                        num_tactics = self.opts.max_num_tactics - obs['num_tactics_left']
                        return False, script, time, num_tactics
                    elif obs['result'] in ['ERROR']:
                        continue
                    else:
                        assert obs['result'] == 'PROVING'
                        script.append(tac)
                        sig = get_goal_signature(obs['fg_goals'][0])
                        if sig in first_goal_signatures or len(script) >= depth_limit:
                            if len(script) >= depth_limit and sig not in first_goal_signatures:
                                new_traces.append(deepcopy(script))
                            proof_env.step('Undo.')
                            script.pop()
                            continue
                        first_goal_signatures.add(sig)
                        local_context, goal = parse_goal(obs['fg_goals'][0])
                        tactics = self.model.beam_search(env, local_context, goal)
                        stack.append([tac_template % tac.to_tokens() for tac, _ in tactics[::-1]])

                proof_env.step('Restart.')
                gc.collect()

            depth_limit *= 2
            traces = new_traces

        obs = proof_env.step('Admitted.')
        print(obs['result'])
        time = self.opts.timeout - obs['time_left']
        num_tactics = self.opts.max_num_tactics - obs['num_tactics_left']
        return False, script, time, num_tactics

    def save(self, n_epoch, dirname):
        torch.save({'state_dict': self.model.state_dict(), 'n_epoch': n_epoch,
                    'optimizer': self.optimizer.state_dict()}, os.path.join(dirname, 'model_%03d.pth' % n_epoch))

    def get_tac_template(self):
        """
        Smiles at you kindly :D
        come warm up by the fire, weary sojourner, and enjoy the fruits of modularity
        """
        m = re.fullmatch(r'ours\+(?P<auto_tac>\w+)', self.opts.method)  # ours+auto/hammer/etc.
        if m is not None:
            tac_template = m['auto_tac'] + '; %s.'
        else:
            tac_template = '%s.'
            
        return tac_template