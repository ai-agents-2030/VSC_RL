from vscrl.environment import batch_interact_environment
from vscrl.data import ReplayBuffer
import numpy as np
from tqdm import tqdm
from vscrl.algorithms.vscrl import VSCRLTrainer
from vscrl.misc import colorful_print
import wandb
import os
import torch
import time
import copy
from vscrl.environment.env_utils import add_mc_return
from vscrl.algorithms.parallel_utils import remote_collect_trajectories
from rich import print
import re
def label_trajectories(trajectories, agent):
    print("Labeling Trajectories")
    baselines = []
    for i in range(0, len(trajectories), 16):
        observations = [t[0]["observation"] for t in trajectories[i:i+16]]

        with torch.no_grad():
            v = agent.trajectory_critic(observations)
            v = torch.nn.Softmax(dim = -1)(v)[:,1]
            baselines.append(v.flatten())
    baselines = torch.cat(baselines, dim = -1)
    print("Done Labeling Trajectories")
    return torch.clamp(baselines.cpu(), 1e-4, 1-1e-4)

def framestack(all_trajectories):
    new_trajectories = copy.deepcopy(all_trajectories)
    for trajectory, new_trajectory in zip(all_trajectories, new_trajectories):
        for i,(t, nt) in enumerate(zip(trajectory, new_trajectory)):
            if i  == 0:
                nt["image_features"] = np.concatenate([t["image_features"], t["image_features"]], axis = -1)
            else:
                nt["image_features"] = np.concatenate([trajectory[i-1]["image_features"], t["image_features"]], axis = -1)
            nt["next_image_features"] = np.concatenate([t["image_features"], t["next_image_features"]], axis = -1)
    return new_trajectories

def split_trajecotry(all_trajectories):
    print("Before split, number of trajectories: ", len(all_trajectories))
    splited_trajectories = []
    for trajectory in all_trajectories:
        splited_trajectory = []
        for i in range(len(trajectory)):
            splited_trajectory.append(trajectory[i])
            if trajectory[i]['reward'] != 0:
                for j in range(len(splited_trajectory)):
                    splited_trajectory[j]['mc_return'] = pow(0.5, len(splited_trajectory) - j - 1)
                    splited_trajectory[j]['trajectory_reward'] = 1
                splited_trajectories.append(splited_trajectory)
                splited_trajectory = []
        if len(splited_trajectory) != 0:
            for j in range(len(splited_trajectory)):
                splited_trajectory[j]['mc_return'] = 0
                splited_trajectory[j]['trajectory_reward'] = 0
            splited_trajectories.append(splited_trajectory)
    print("After split, number of trajectories: ", len(splited_trajectories))
    return splited_trajectories

def filterbc_buffer(all_trajectories, batch_size, capacity, agent):
    trajectory_rewards = np.array([t[0]["trajectory_reward"] if len(t) > 0 else 0 for t in all_trajectories]).flatten()
    cutoff = np.quantile(trajectory_rewards, 1 - 0.1)
    filtered_trajectories = []
    for t, b in zip(all_trajectories, trajectory_rewards):
        if b >= cutoff:
            filtered_trajectories.append(t)
    data = sum(filtered_trajectories, [])
    filtered_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
    for d in data:
        filtered_buffer.insert(**d)
    return filtered_buffer


def filter_buffer(all_trajectories, batch_size, capacity, agent):
    baselines = label_trajectories(all_trajectories, agent).numpy().flatten()
    trajectory_rewards = np.array([t[0]["trajectory_reward"] if len(t) > 0 else 0 for t in all_trajectories]).flatten()
    baselines = trajectory_rewards - baselines
    cutoff = np.quantile(baselines, 1 - 0.1)
    top10 = np.argsort(baselines)[-10:]
    filtered_trajectories = []
    for t, b in zip(all_trajectories, baselines):
        if b >= cutoff:
            filtered_trajectories.append(t)
    data = sum(filtered_trajectories, [])
    filtered_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
    for d in data:
        filtered_buffer.insert(**d)
    return filtered_buffer

def trajectory_buffer(all_trajectories, batch_size, capacity, agent):
    data = sum(all_trajectories, [])
    buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
    for d in data:
        buffer.insert(**d)
    return buffer

def offpolicy_train_loop(env,\
                agent,\
                agent_ref,\
                tokenizer,\
                accelerator,\
                warmup_iter: int = 20,
                rollout_size: int = 50,\
                batch_size: int = 2,
                capacity: int = 500000,
                train_iterations: int = 10,\
                epochs:int = 3, \
                grad_accum_steps: int = 1,\
                critic_lr: float= 1e-3,\
                lm_lr: float = 1e-5,\
                gamma: float = 0.9,
                tau: float = 0.1,
                use_wandb: bool = False,
                actor_epochs: int = 3,
                train_mode: str = None,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                save_freq: int = 25,
                train_algorithm: str = "vscrl",
                decode_f: callable = lambda x: x,
                offline_data_path: str = None,
                offline_actor_iterations: int = 20,
                offline_critic_iterations: int = 20,
                offline_trajectory_critic_iterations: int = 20,
                trajectory_critic_epochs: int = 5,
                parallel: str = 'single',
                worker_temp_path=None, 
                worker_run_path=None,
                worker_ips=[], 
                worker_username=None,
                **kwargs):

    if train_algorithm == "vscrl":
        trainer = VSCRLTrainer(agent=agent,\
                                agent_ref=agent_ref,\
                                accelerator=accelerator,\
                                tokenizer=tokenizer,\
                                critic_lr = critic_lr,\
                                lm_lr = lm_lr,\
                                gamma = gamma,\
                                tau = tau,\
                                epochs = epochs,\
                                actor_epochs = actor_epochs,
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm,
                                trajectory_critic_epochs = trajectory_critic_epochs)
    replay_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
    all_trajectories = []
    agent.prepare()
    agent_ref.prepare()
    trainer.prepare()


    train_trajectories = []
    val_trajectories = []
    all_trajectories = []

    replay_buffer = ReplayBuffer(batch_size= batch_size, capacity=capacity)
    validation_buffer = ReplayBuffer(batch_size= batch_size, capacity=capacity)

    data = sum(train_trajectories, [])
    val_data = sum(val_trajectories, [])

    for d in data:
        replay_buffer.insert(**d)
    for d in val_data:
        validation_buffer.insert(**d)

    if accelerator.is_main_process:
        print(">>>start iterations")
    progress_bar = tqdm(total=train_iterations, initial=0)
    moving_success_rate = {
        'all': [],
    }
    for i in range(train_iterations):
        assert train_mode != "offline", "Only online/off2on need to iteractively train; offline should directly go to eval loop after training"
        if parallel == 'single':
            trajectories = batch_interact_environment(
                                                agent = agent,\
                                                agent_ref = agent_ref,\
                                                env = env,\
                                                num_trajectories= rollout_size,\
                                                accelerator = accelerator,\
                                                use_tqdm=False,
                                                decode_f = decode_f,
                                                gamma = gamma,
                                                iter=i,
                                                use_ref=False)
            
        trajectories = framestack(trajectories)
        if accelerator.is_main_process:
            info = {"iteration": i,\
                    "walltime": time.time()}
            all_trajectories += trajectories
            colorful_print(f">>> length of all_trajectories: {len(trajectories)}", fg='green')

            total_traj_num = len(trajectories)
            success_traj_num = 0
            for traj in trajectories:
                task_success = 0
                for traj_step in range(len(traj)):
                    if traj[traj_step]["task_success"] != 0:
                        task_success = 1
                        success_traj_num += 1
                        moving_success_rate['all'].append(1)
                        break

                    if traj_step == len(traj) - 1:
                        moving_success_rate['all'].append(0)
                        
                for traj_step in range(len(traj)):
                    traj[traj_step]["task_success"] = task_success

            assert success_traj_num <= total_traj_num, "Success traj num should be less than total traj num"
            info.update({
                "rollout.task_success_rate": float(success_traj_num / total_traj_num),\
                "rollout.moving_success_rate": np.mean(moving_success_rate['all']),\
                "rollout.moving_success_rate_100": np.mean(moving_success_rate['all'][-100:]),\
                })
            print(f"Current Task Success Rate: {float(success_traj_num / total_traj_num)}")

            trajectories = split_trajecotry(trajectories)
            new_train_trajectories = trajectories[:int(len(trajectories)*0.8)]
            new_val_trajectories = trajectories[int(len(trajectories)*0.8):]
            train_trajectories += new_train_trajectories
            val_trajectories += new_val_trajectories
            data = sum(new_train_trajectories, [])
            val_data = sum(new_val_trajectories, [])
            for d in data:
                replay_buffer.insert(**d)
            for d in val_data:
                validation_buffer.insert(**d)

            info.update({"rollout.reward.mean": np.mean([d["reward"] for d in data]),\
                    "rollout.reward.max": np.max([d["reward"] for d in data]),\
                    "rollout.reward.min": np.min([d["reward"] for d in data])})
            print(">>> Saving Replay Buffer")
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
            torch.save(all_trajectories, os.path.join(save_path, 'trajectories.pt'))
            torch.save(train_trajectories, os.path.join(save_path, 'train_trajectories.pt'))
            torch.save(val_trajectories, os.path.join(save_path, 'val_trajectories.pt'))
            print(">>> Saved Replay Buffer")            
            time.sleep(15)
        else:
            info = {}
        accelerator.wait_for_everyone()
        
        train_trajectories = torch.load(os.path.join(save_path, 'train_trajectories.pt'))
        val_trajectories = torch.load(os.path.join(save_path, 'val_trajectories.pt'))
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories.pt'))
        replay_buffer = torch.load(os.path.join(save_path, 'replay_buffer.pt'))

        filtered_buffer = filter_buffer(train_trajectories, batch_size, capacity, agent)
        filtered_validation_buffer = filter_buffer(val_trajectories, batch_size, capacity, agent)
        filtered_bc_buffer = filterbc_buffer(train_trajectories, batch_size, capacity, agent)
        
        print("Training")
        info.update(trainer.update_bc(filtered_bc_buffer, no_update_actor = (i < warmup_iter)))

        info.update(trainer.update_trajectory_critic(train_trajectories, val_trajectories))

        info.update(trainer.update(replay_buffer, validation_buffer, filtered_buffer, filtered_validation_buffer, no_update_actor = (i < warmup_iter)))
            
    
        if use_wandb and accelerator.is_main_process:
            wandb.log(info)
        if (i+1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            print("Saving")
            trainer.save(os.path.join(save_path, f'trainer_{i+1}.pt'))
            
        if accelerator.is_main_process:
            progress_bar.update(1)
        
