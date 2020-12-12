import torch.multiprocessing as mp
from torch.multiprocessing import set_sharing_strategy, set_start_method
try:
    set_start_method('spawn')
    set_sharing_strategy("file_descriptor")
except RuntimeError:
    pass
import torch
import os, sys, pdb
from eval_env import FileEnv
from utils import log

done = mp.Event()
class ParallelSampler:
    """
    Manages multithreaded sampling from a FileEnv
    """    
    NUM_WORKERS = 1

    def __init__(self, file_env_args, tac_template, agent, train=False):
        "sampler"
        self.tac_template = tac_template
        self.file_env_args = file_env_args
        self.agent = agent
        self.train = train
        agent.model.share_memory()
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def sample_trajectories(self, n_epochs=1, **kwargs):
        """
        Uses 3-5 processes to collect trajectories.
        """
        done.clear()
        queue = mp.Queue()
        log("------------------STARTING ASYNC-------------------")
        producers = []
        for i in range(n_epochs):
            proc = mp.Process(target=self.async_trajectories, args=(i,queue,done))
            proc.start()
            producers.append(proc)
            
        ended_workers = 0
        grads = None
        collected = 0
        results = []
        losses = []
        while ended_workers != n_epochs:
            res = queue.get()
            if res is None:
                ended_workers += 1
                log("------------COLLECTED {} SAMPLES-------------".format(collected))
            else:
                async_grads = res['grads']
                async_results = res['results']
                async_collected = res['collected']
                async_loss = res['loss']

                # Process gradients (sum them)
                grads = ParallelSampler._join_grads(grads, async_grads, clone=True)
                del async_grads
                collected += async_collected
                results += async_results
                losses.append(async_loss)

        done.set()
        for proc in producers:
            proc.join()
        del queue
        print("->\tDone!")
        log("------------------FINISHED ASYNC-------------------")
        # assert collected == len(rewards), "{} collected != {} rewards".format(collected, len(rewards))
        return results, grads, collected, losses

    def sample_dfs_trajectories(self, n_epochs=1, **kwargs):
        """
        Uses a pool to collect trajectories

        Problems:
        - Need many rollouts of same proof environment
        - Would like to document number of times each node is visited
        
        file-env: [proof1, proof2, proof3]
        Process 1 [1st 33% environments]
        for i, p_env in enumerate(fenv):
            if i % 3 == 0:
                s = sample_dfs()
        Process 2 [2nd 33%]
        for i, p_env in enumerate(fenv):
            if i % 3 == 1:
                s = sample_dfs()
        Process 3 [3rd 33%]
        for i, p_env in enumerate(fenv):
            if i % 3 == 2:
                s = sample_dfs()

        return self._join_counts(Nsa1, Nsa2, Nsa3)
        """
        done.clear()
        queue = mp.Queue()
        log("------------------STARTING ASYNC-------------------")
        producers = []
        for i in range(n_epochs):
            proc = mp.Process(target=self.async_trajectories, args=(i,queue,done))
            proc.start()
            producers.append(proc)
            
        ended_workers = 0
        grads = None
        collected = 0
        results = []
        losses = []
        while ended_workers != n_epochs:
            res = queue.get()
            if res is None:
                ended_workers += 1
                log("------------COLLECTED {} SAMPLES-------------".format(collected))
            else:
                async_grads = res['grads']
                async_results = res['results']
                async_collected = res['collected']
                async_loss = res['loss']

                # Process gradients (sum them)
                grads = ParallelSampler._join_grads(grads, async_grads, clone=True)
                del async_grads
                collected += async_collected
                results += async_results
                losses.append(async_loss)

        done.set()
        for proc in producers:
            proc.join()
        del queue
        print("->\tDone!")
        log("------------------FINISHED ASYNC-------------------")
        # assert collected == len(rewards), "{} collected != {} rewards".format(collected, len(rewards))
        return results, grads, collected

    def async_trajectories(self, *args):
        """
        Collects an epoch worth of data
        Epoch = [1 rollout per proof_env]

        Waits until the `done` event passed in as an arg is set().
        Otherwise data cannot be retrieved from the IPC file descriptor, possibly for refcount deletion
        """
        pid, queue, done = args[0], args[1], args[2]
        print("{}: started collection".format(pid))
        for fenvargs in self.file_env_args:
        # try:
            with FileEnv(*fenvargs) as fenv:
                prob_grads = None
                for proof_env in fenv:
                    self.agent.optimizer.zero_grad()
                    # Collect data we can backprop
                    data = self.agent.sample_once(proof_env, self.tac_template, train=True)
                    trajectory, results = data['samples'], data['results']
                    collected = len(trajectory)

                    # Backpropagate loss
                    losses = torch.cat([(prob * -r).unsqueeze(0) for prob, r in trajectory]).to(trajectory[0][0].device)
                    loss = torch.mean(losses)
                    loss.backward() # loss.backward(retain_graph=True) is VERY expensive
                    grads = [p.grad if p.grad is not None else None for p in self.agent.model.parameters()]

                    print("{}: collected {}".format(pid, collected))
                    print("{}: results {}".format(pid, results))
                    queue.put({'grads': grads, 
                    'collected': collected, 
                    'results': results,
                    'loss': loss.detach().item()})
        # except Exception as e:
        #     print("{}: ERROR-{}".format(pid,e))
        queue.put(None)
        print("{}: finished & waiting".format(pid))
        done.wait()
        del self.agent.model
        torch.cuda.empty_cache()
        sys.exit(0)

    @staticmethod
    def _join_grads(old, new, clone=False):
        """
        Joins two gradients by adding
        """
        if old is None:
            old = new
        else:
            assert len(old) == len(new), "got-{} | expected-{}".format(len(old), len(new))
            for i, grad in enumerate(new):
                if clone and grad is not None:
                    grad = grad.clone()

                if old[i] is not None and grad is not None:
                    old[i] += grad
                elif grad is not None:
                    old[i] = grad
        return old        
