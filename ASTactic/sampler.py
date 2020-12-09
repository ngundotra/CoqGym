import torch
import torch.multiprocessing as mp
from torch.multiprocessing import set_sharing_strategy, set_start_method
try:
    set_start_method('spawn')
    set_sharing_strategy("file_descriptor")
except RuntimeError:
    pass
import os
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
        # self.pool.submit
        log("------------------STARTING ASYNC-------------------")
        producers = []
        for i in range(ParallelSampler.NUM_WORKERS):
            proc = mp.Process(target=self.async_trajectories, args=(i,queue,done))
            proc.start()
            producers.append(proc)
            
        ended_workers = 0
        grads = None
        rewards = []
        collected = 0
        while ended_workers != ParallelSampler.NUM_WORKERS:
            res = queue.get()
            if res is None:
                ended_workers += 1
                log("-----------COLLECTED {} TRAJECTORIES-------------".format(ended_workers))
            else:
                async_grads = res['grads']
                async_rewards = res['rewards']
                async_collected = res['collected']
                # print("-----COLLECTED: {}".format(async_grads))
                # print("-----COLLECTED: {}".format(async_rewards))

                # Process rewards
                rewards.append(async_rewards.clone())
                del async_rewards

                # Process gradients (sum them)
                if grads is None:
                    # grads = [[x.clone() if x is not None else None for x in layer] for layer in async_grads]
                    grads = [[x.clone() for x in layer] for layer in async_grads]
                    grads = async_grads
                else:
                    grads = [
                                # [g1+g2.clone() if g1 is not None else None for g1, g2 in zip(layer, async_layer)] 
                                [g1+g2.clone() for g1, g2 in zip(layer, async_layer)] 
                                for layer, async_layer in zip(grads, async_grads)
                    ]
                del async_grads
                collected += 1

        log("------------------FINISHED ASYNC-------------------")
        done.set()
        print("->\tDone!")
        assert collected == len(rewards), "# collected != # rewards"
        return grads, rewards, collected

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
                for proof_env in fenv:
                    try:
                        trajectories = self.agent.sample_once(proof_env, self.tac_template, train=True)
                        prob_grads = []
                        rewards = []
                        collected = 0
                        for prob, r in trajectories:
                            self.agent.optimizer.zero_grad()
                            prob.backward(retain_graph=True)
                            # grads = [p.grad * r if p.grad is not None else None for p in self.agent.model.parameters()]
                            grads = [p.grad * r for p in self.agent.model.parameters()]
                            prob_grads.append(grads)
                            rewards.append(r)
                            collected += 1
                        rewards = torch.Tensor(rewards)
                        print("{}: rewards-{}".format(pid, len(rewards)))
                        queue.put({'grads': prob_grads, 'rewards': rewards, 'collected': collected})
                    except Exception:
                        continue
        except Exception:
            pass
        queue.put(None)
        print("{}: finished & waiting".format(pid))
        done.wait()