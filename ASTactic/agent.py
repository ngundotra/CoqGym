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
import time
from models.prover import Prover
from rnd import RandomDistillation

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
        if opts.freeze: # Freeze prover encoder module
            self.model.term_encoder.requires_grad_ = False
            for p in self.model.term_encoder.parameters():
                p.requires_grad = False

        if opts.RND:
            self.RND_fixed = RandomDistillation(opts, fixed=True)
            self.RND_fixed.requires_grad = False
            self.RND_train = RandomDistillation(opts, fixed=False)
            self.RND_optimizer = torch.optim.RMSprop(self.RND_train.parameters(), lr=3e-5,
                                        momentum=0.9,
                                        weight_decay=1e-6)
        # Just set this to None so we can check if it exists for exploration bonuses
        self.exp_model = None

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

    def train_RL(self, n_epoch, file_list, logger, proof_name=None, sample='DFS'):
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
        if sample == "vanilla":
            results, total_collected = self.train_RL_PG(n_epoch, self.opts.workers, file_list, logger, with_hammer, hammer_timeout)
            return results + [total_collected]
        elif sample == "DFS":
            return self.train_RL_DFS(logger, n_epoch, file_list, with_hammer, hammer_timeout)
        else:
            raise ValueError('Sampling method not found.')


    def train_RL_PG(self, n_epoch, epochs_per_update, file_list, logger, with_hammer, hammer_timeout, use_dfs=False):
        """
        Collects hella samples for Policy Gradients.
        Uses parallel workers if `opts.parallel`
        """
        print("Making sure save_folder exists...")
        save_folder= "train-PG-ckpt/{}/".format(self.descriptor)
        print("+ Good to go +")
        os.makedirs(save_folder, exist_ok=True)
        tac_template = self.get_tac_template()
        file_env_args = [(filename, self.opts.max_num_tactics, self.opts.timeout, with_hammer, hammer_timeout)
                for filename in file_list]

        loss = None
        total_collected = 0 
        all_results = []
        last_ep = 0
        try:
            for ep in range(n_epoch):
                start = time.time()
                last_ep = ep
                print("\n>>>>>>>>>>>>>>>>>>>>EPOCH: {}<<<<<<<<<<<<<<<<<<<<<<\n".format(ep))
                self.optimizer.zero_grad()
                results, grads, collected, losses, len_fg_bg, expl_bonus = \
                    self.sample_parallel(epochs_per_update, tac_template=tac_template, file_env_args=file_env_args, train=True)
                all_results += results
                total_collected += collected

                params = self.model.parameters()
                for idx, (layer, grad) in enumerate(zip(params, grads['model'])):
                    if grad is not None:
                        layer.grad = grad / collected
                self.optimizer.step()
                self._log_epoch(logger, ep, start, results, collected, losses, len_fg_bg, expl_bonus)

                if self.opts.RND:
                    self.RND_optimizer.zero_grad()
                    for idx, (layer, grad) in enumerate(zip(self.RND_train.parameters(), grads['RND'])):
                        if grad is not None:
                            layer.grad = grad / collected
                    self.RND_optimizer.step()

        except KeyboardInterrupt as kb:
            print("Ended on epoch: {}".format(last_ep))

        print("***Saving model***")
        self.save(last_ep+1, save_folder)
        print("Saved model on epoch {} to {}".format(last_ep+1, save_folder))
        return results, total_collected

    
    def _log_epoch(self, logger, ep, start, results, collected, losses, len_fg_bg, expl_bonuses):
        """
        Logs values to tensorboard:
        Loss, num timesteps collected, num opened fg goals, num opened bg goals,
        num successes, num failures, and time for epoch
        """
        num_success = sum([int(result[0]) for result in results])
        num_fail = sum([int(not result[0]) for result in results])
        avg_loss = sum(losses)/len(losses)
        print("\tEpoch loss{}: {}".format(ep, avg_loss))
        logger.log_value('loss', avg_loss, ep)
        logger.log_value("num_collected", collected, ep)
        logger.log_value("num_fg", len_fg_bg[0], ep)
        logger.log_value("num_bg", len_fg_bg[1], ep)
        logger.log_value("num_success", num_success, ep)
        logger.log_value("num_fail", num_fail, ep)
        logger.log_value('time', time.time() - start, ep)
        for proof_name, bonuses in expl_bonuses.items():
            if bonuses['added'] == 0:
                continue
            for key, val in bonuses.items():
                logger.log_value(proof_name + '/' + key, val, ep)
        # todo: how to get numsteps?


    def train_RL_DFS(self, logger, n_epoch, file_list, with_hammer, hammer_timeout):
        """
        TODO: put in `RL_Trainer` sort of file ...?
        Collects samples & updates the model in accordance with DFS sampling
        """
        save_folder = "train-DFS-ckpt/{}/".format(self.descriptor)
        print("Making sure save folder exists...")
        os.makedirs(save_folder, exist_ok=True)
        print("+ Good to go +")

        tac_template = self.get_tac_template()
        last_ep = 0
        try:
            for curr_epoch in range(n_epoch):
                start = time.time()
                last_ep = curr_epoch
                print("\n-------------EPOCH-------------")
                print("---------------{}---------------\n".format(curr_epoch))
                losses = []
                expl_bonuses = {}
                results = []
                for filename in file_list:
                    with FileEnv(filename, self.opts.max_num_tactics, self.opts.timeout, with_hammer=with_hammer) as file_env:
                        for proof_env in file_env:  # start a proof
                            curr_name = proof_env.proof['name']
                            print('proof: ', proof_env.proof['name'])
                            # success, proof_pred, time, num_tactics, trajectory = self.prove(proof_env, train=True)
                            samples, result, expl_bonus = \
                                self.sample_DFS(proof_env, tac_template) # TODO: control number of samples better

                            losses_env = torch.cat([ ((-logprob)
                                            * (reward)).unsqueeze(0)
                                            for logprob, reward in samples])
                            losses.append(torch.mean(losses_env))
                            expl_bonus['added'] = 0
                            if self.opts.RND:
                                expl_bonus['added'] = 1
                            del expl_bonus['grads']
                            expl_bonuses[curr_name] = expl_bonus
                            results.append(result)

                loss = sum(losses) / len(losses)
                print("\tLoss: {}".format(loss.item()))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._log_epoch(logger, curr_epoch, start,
                    results, 1, losses, [1, 1], expl_bonuses)

                if self.opts.RND and expl_bonus['exp_avg'] is not None:
                    self.RND_optimizer.zero_grad()
                    bonus_loss = sum([b['exp_avg'] for k, b in expl_bonuses.items()]) / len(expl_bonuses)
                    bonus_loss.backward()
                    self.RND_optimizer.step()
        except KeyboardInterrupt as kb:
            print("Excepted kb interrupt")
        print("Saving model*")
        self.save(last_ep, save_folder)
        print("-Saved model-")
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
        """
        Train is not used for anything
        """
        obs = proof_env.init()
        env = filter_env(obs['env'])

        first_goal_signatures = {get_goal_signature(obs['fg_goals'][0])}

        return self._rollout_asts(proof_env, obs, env, first_goal_signatures)


    def _get_tactics(self, obs, env, sampling="PG"):
        """
        Helper method of sample_once that returns a tactic and prob_list form 
        """
        tac_template = self.get_tac_template()
        local_context, goal = parse_goal(obs['fg_goals'][0])
        tactics = self.model.beam_search(env, local_context, goal, sampling)
        tacs = [tac_template % tac.to_tokens() for tac, _ in tactics]
        probs = torch.cat([prob.unsqueeze(0) for _, prob in tactics])
        return tactics, tacs, probs


    def make_exp_results(self, bonuses, needs_grad=True):
        exp_avg = None
        exp_std = None
        exp_ct = None
        grads = None
        if len(bonuses) > 0:
            # we need average for backprop
            exp_avg = sum(bonuses) / len(bonuses)
            if needs_grad:
                exp_avg.backward()
                grads = [p.grad if p.grad is not None else None for p in self.RND_train.parameters()]
            bonuses = np.array([b.item() for b in bonuses])
            exp_std = np.std(bonuses)
            exp_ct = len(bonuses)

        exp_results = {'exp_avg': exp_avg,
                        'exp_std': exp_std,
                        'exp_ct': exp_ct,
                        'grads': grads}
        return exp_results


    def _rollout_asts(self, proof_env, init_obs, env, first_goal_signatures):
        """
        Helper method of sample_once that abstracts away process of applying a tactic
        and interacting with the proof environment.

        Rolls out multiple times until end conditions are reached.

        Returns {samples: samples, results: results}
        """
        # store logprobs (along the trajectory) to be rewarded later
        prob_list = []

        # initialize
        tactics, tacs, probs = self._get_tactics(init_obs, env)
        script = []
        samples = []
        bonuses = []
        steps = 0
        while True and steps < self.opts.max_num_tactics + 1:
            steps += 1

            m = torch.distributions.Categorical(probs)
            idx = m.sample()
            tac, prob = tacs[idx], probs[idx]

            prob_list.append(prob)
            obs = proof_env.step(tac)

            fg_goals, bg_goals, shelved_goals, _ = proof_env.serapi.query_goals()
            # Keep track of these things in case we exit in if-else block
            time = self.opts.timeout - obs['time_left']
            num_tactics = self.opts.max_num_tactics - obs['num_tactics_left']

            reward = -0.1
            if obs['result'] == 'SUCCESS':
                script.append(tac)
                samples.append((prob, 5.0))
                exp_results = self.make_exp_results(bonuses)
                return {'samples': samples, 'results': (True, script, time, num_tactics), 'exp': exp_results}
            elif obs['result'] in ['MAX_NUM_TACTICS_REACHED', 'MAX_TIME_REACHED']:
                script.append(tac)
                samples.append((prob, -0.1))
                exp_results = self.make_exp_results(bonuses)
                return {'samples': samples, 'results': (False, script, time, num_tactics), 'exp': exp_results}
            elif obs['result'] in ['ERROR']:  # Tactic is misapplied, nothing happened
                reward = -3.0
            else:
                assert obs['result'] == 'PROVING'

            script.append(tac)
            sig = get_goal_signature(fg_goals[0]) #TODO: should we care about this in sampling?
            if sig in first_goal_signatures:
                proof_env.step('Undo.')
                script.pop()
                samples.append((prob, reward))
                continue
            first_goal_signatures.add(sig)

            # Can only apply exploration reward while still PROVING
            if self.opts.RND:
                exp_reward = self.get_exp_reward(obs, env)
                bonuses.append(exp_reward)
                reward += exp_reward.detach().item()
            samples.append((prob, reward))

            if len(script) >= self.opts.depth_limit:
                exp_results = make_exp_results(bonuses)
                return {'samples': samples, 'results': (False, script, time, num_tactics), 'exp': self.exp_results}

            # Sample again if we ran out of tactics
            tactics, tacs, probs = self._get_tactics(obs, env)

        raise ValueError("Should have exited before hitting this block")
    
    def get_exp_reward(self, obs, env):
        """
        Returns exp bonus
        """
        local_context, goal = parse_goal(obs['fg_goals'][0])
        fixed_emb = self.RND_fixed.embed_terms([env], [local_context], [goal]).detach()
        train_emb = self.RND_train.embed_terms([env], [local_context], [goal])
        exp_reward = RandomDistillation.compare(fixed_emb, train_emb)
        return exp_reward

    def sample_DFS(self, proof_env, tac_template, train=True):
        """
        Single attempt to prove something
        ENDS
            - when error happens
            - when success is reached
            - when timelimit hit
        """
        obs = proof_env.init()
        env = filter_env(obs['env'])
        first_goal_signatures = {get_goal_signature(obs['fg_goals'][0])}

        # initialize
        local_context, goal = parse_goal(obs['fg_goals'][0])
        tactics = self.model.beam_search(env, local_context, goal, "DFS")
        stack = [[(tac_template % tac.to_tokens(), prob) for tac, prob in tactics[::-1]]]
        script = []
        samples = []
        bonuses = []

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

            obs = proof_env.step(tac)
            time = self.opts.timeout - obs['time_left']
            num_tactics = self.opts.max_num_tactics - obs['num_tactics_left']
            if obs['result'] == 'SUCCESS':
                script.append(tac)
                samples.append((logprob, 5.0))
                return samples, (True, script, time, num_tactics), self.make_exp_results(bonuses, False)
            elif obs['result'] in ['MAX_NUM_TACTICS_REACHED', 'MAX_TIME_REACHED']:
                # TODO: reassure that its the total number of tactics over the whole beam search but not along the trajectory.
                #       Change the reward to -0.1 if its the latter.
                samples.append((logprob, -0.1))
                return samples, (True, script, time, num_tactics), self.make_exp_results(bonuses, False)
            elif obs['result'] in ['ERROR']:  # Tactic is misapplied, nothing happened
                samples.append((logprob, -3.0))
                continue
            else:
                script.append(tac)
                sig = get_goal_signature(obs['fg_goals'][0])
                if sig in first_goal_signatures or len(script) >= self.opts.depth_limit:
                    proof_env.step('Undo.')
                    script.pop()
                    continue
                first_goal_signatures.add(sig)

                reward = -0.1
                if self.opts.RND:
                    exp_reward = self.get_exp_reward(obs, env)
                    bonuses.append(exp_reward)
                    reward += exp_reward.detach().item()
                samples.append((logprob, reward))

                local_context, goal = parse_goal(obs['fg_goals'][0])
                tactics = self.model.beam_search(env, local_context, goal, "DFS")
                stack.append([(tac_template % tac.to_tokens(), prob) for tac, prob in tactics[::-1]])

        obs = proof_env.step('Admitted.')
        # print(obs['result'])
                            # samples, result, expl_bonus = \
        time = self.opts.timeout - obs['time_left']
        num_tactics = self.opts.max_num_tactics - obs['num_tactics_left']
        return samples, (obs['result'] == 'SUCCESS', script, time, num_tactics), self.make_exp_results(bonuses, False)

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