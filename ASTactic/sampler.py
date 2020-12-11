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
        Uses a pool to collect trajectories
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
        while ended_workers != n_epochs:
            res = queue.get()
            if res is None:
                ended_workers += 1
                log("------------COLLECTED {} SAMPLES-------------".format(collected))
            else:
                async_grads = res['grads']
                # async_rewards = res['rewards']
                async_collected = res['collected']

                # Process gradients (sum them)
                grads = ParallelSampler._join_grads(grads, async_grads, clone=True)
                del async_grads
                collected += async_collected

        done.set()
        for proc in producers:
            proc.join()
        del queue
        print("->\tDone!")
        log("------------------FINISHED ASYNC-------------------")
        # assert collected == len(rewards), "{} collected != {} rewards".format(collected, len(rewards))
        return grads, collected

    def async_trajectories(self, *args):
        """
        Collects an epoch worth of data
        Epoch = [1 rollout per proof_env]

        Waits until the `done` event passed in as an arg is set().
        Otherwise data cannot be retrieved from the IPC file descriptor, possibly for refcount deletion
        """
        pid, queue, done = args[0], args[1], args[2]
        print("{}: started collection".format(pid))
        try:
            with FileEnv(*self.file_env_args) as fenv:
                prob_grads = None
                rewards = []
                for proof_env in fenv:
                    self.agent.optimizer.zero_grad()
                    trajectory = self.agent.sample_once(proof_env, self.tac_template, train=True)
                    collected = len(trajectory)

                    losses = torch.cat([(prob * r).unsqueeze(0) for prob, r in trajectory]).to(trajectory[0][0].device)
                    loss = torch.mean(losses)
                    loss.backward()
                    grads = [p.grad if p.grad is not None else None for p in self.agent.model.parameters()]
                    # prob_grads = ParallelSampler._join_grads(prob_grads, grads)
                    print("{}: collected {}".format(pid, collected))
                    queue.put({'grads': grads, 'collected': collected})
        except Exception as e:
            print("{}: ERROR-{}".format(pid,e))
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


# class CoqAgent(torch.multiprocessing.Process):
#     def __init__(self, pid, ...):
#         super(CoqAgent, self).__init__()
#         self.pid = pid

#     def run(self):
#         pass

#     def 