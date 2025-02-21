from torch.utils.data import Dataset, DataLoader
import numpy as np
class DummyDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]


class ReplayBuffer:
    def __init__(self, batch_size=2, capacity=10000):
        self.max_size = capacity
        self.size = 0
        self.observations = None
        self.observations_ref = None
        self.rewards = None
        self.next_observations = None
        self.next_observations_ref = None
        self.dones = None
        self.batch_size = batch_size
        self.actions = None
        self.actions_ref = None
        self.mc_returns = None
        self.trajectory_rewards = None
        self.image_features = None
        self.next_image_features = None
        self.use_refs = None
        self.task_successs = None

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        rand_indices = np.random.randint(0, self.size, size=(batch_size,)) % self.max_size
        return {
            "observation": self.observations[rand_indices],
            "observation_ref": self.observations_ref[rand_indices],
            "action": self.actions[rand_indices],
            "action_ref": self.actions_ref[rand_indices],
            "image_features": self.image_features[rand_indices],
            "next_image_features": self.next_image_features[rand_indices],
            "reward": self.rewards[rand_indices],
            "next_observation": self.next_observations[rand_indices],
            "next_observation_ref": self.next_observations_ref[rand_indices],
            "done": self.dones[rand_indices],
            "mc_return": self.mc_returns[rand_indices],
            "trajectory_reward": self.trajectory_rewards[rand_indices],
            "use_ref": self.use_refs[rand_indices],
            "task_success": self.task_successs[rand_indices],
        }

    def __len__(self):
        return self.size

    def insert(
        self,
        /,
        observation,
        observation_ref,
        action,
        action_ref,
        image_features: np.ndarray,
        next_image_features: np.ndarray,
        reward: np.ndarray,
        next_observation,
        next_observation_ref,
        done: np.ndarray,
        mc_return,
        trajectory_reward,
        use_ref,
        task_success,
        **kwargs
    ):
        """
        Insert a single transition into the replay buffer.

        Use like:
            replay_buffer.insert(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
        """
        if isinstance(reward, (float, int)):
            reward = np.array(reward)
        if isinstance(mc_return, (float, int)):
            mc_return = np.array(mc_return)
        if isinstance(trajectory_reward, (float, int)):
            trajectory_reward = np.array(trajectory_reward)
        if isinstance(done, bool):
            done = np.array(done)
        if isinstance(use_ref, bool):
            use_ref = np.array(use_ref)
        if isinstance(task_success, (float, int)):
            task_success = np.array(task_success)
        # print(next_observation)
        # if isinstance(prompt_actionaction, int):
        #     action = np.array(action, dtype=np.int64)

        if self.observations is None:
            self.observations = np.array(['']*self.max_size, dtype = 'object')
            self.observations_ref = np.array(['']*self.max_size, dtype = 'object')
            self.actions = np.array(['']*self.max_size, dtype = 'object')
            self.actions_ref = np.array(['']*self.max_size, dtype = 'object')
            self.image_features = np.empty((self.max_size, *image_features.shape), dtype=image_features.dtype)
            self.next_image_features = np.empty((self.max_size, *next_image_features.shape), dtype=next_image_features.dtype)
            self.rewards = np.empty((self.max_size, *reward.shape), dtype=reward.dtype)
            self.next_observations = np.array(['']*self.max_size, dtype = 'object')
            self.next_observations_ref = np.array(['']*self.max_size, dtype = 'object')
            self.dones = np.empty((self.max_size, *done.shape), dtype=done.dtype)
            self.mc_returns = np.empty((self.max_size, *mc_return.shape), dtype=mc_return.dtype)
            self.trajectory_rewards = np.empty((self.max_size, *trajectory_reward.shape), dtype=trajectory_reward.dtype)
            self.use_refs = np.empty((self.max_size, *use_ref.shape), dtype=use_ref.dtype)
            self.task_successs = np.empty((self.max_size, *task_success.shape), dtype=task_success.dtype)

        assert reward.shape == ()
        assert done.shape == ()

        self.observations[self.size % self.max_size] = observation
        self.observations_ref[self.size % self.max_size] = observation_ref
        self.image_features[self.size % self.max_size] = image_features
        self.next_image_features[self.size % self.max_size] = next_image_features
        self.actions[self.size % self.max_size] = action
        self.actions_ref[self.size % self.max_size] = action_ref
        self.rewards[self.size % self.max_size] = reward
        self.next_observations[self.size % self.max_size] = next_observation
        self.next_observations_ref[self.size % self.max_size] = next_observation_ref
        self.dones[self.size % self.max_size] = done
        self.mc_returns[self.size % self.max_size] = mc_return
        self.trajectory_rewards[self.size % self.max_size] = trajectory_reward
        self.use_refs[self.size % self.max_size] = use_ref
        self.task_successs[self.size % self.max_size] = task_success

        self.size += 1