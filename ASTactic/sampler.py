from multiprocessing.pool import Pool
import pdb

class ParallelSampler:
    """
    Manages multithreaded sampling from a FileEnv
    """    
    NUM_WORKERS = 8

    #todo: can't do for() because proof_env is used up
    def __init__(self, file_env_args, tac_template, agent, train=False):
        "sampler"
        self.tac_template = tac_template
        self.file_env_args = file_env_args
        self.agent = agent
        self.train = train
        self.pool = Pool(processes=ParallelSampler.NUM_WORKERS)

    def sample_trajectories(self, n_epochs=1, **kwargs):
        """
        Uses a pool to collect trajectories
        """
        # self.pool.submit
        pdb.set_trace()
        pool_results = self.pool.map(self.sample_once, [(self, kwargs) for _ in range(n_epochs)])

        # threading.Thread(target=thread_function, args=(index,))
        # for i in range(num_threads):
        # pool_output = p.map(hello, range(3))
        #     x = threading.Thread(target=thread_function, args=(index,))
        #     self.thread_pool.append(x)
        #     x.start()

        return pool_results

    def sample_once(self, **kwargs):
        """
        Collects an epoch
        Epoch = [1 rollout per proof_env]
        """
        results = []
        with FileEnv(*self.file_env_args) as fenv:
            for proof_env in fenv:
                results += self.agent.sample_once(proof_env, self.tac_template, train=True)

        return results
        
    
