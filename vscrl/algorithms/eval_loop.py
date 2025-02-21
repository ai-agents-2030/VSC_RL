from vscrl.environment import batch_interact_environment
from vscrl.algorithms.vscrl import VSCRLTrainer
import numpy as np
from vscrl.misc import colorful_print
import wandb
import os
import torch
import time

def eval_loop(env,\
                agent,\
                agent_ref,\
                accelerator,\
                tokenizer,\
                critic_lr,\
                lm_lr,\
                tau,\
                epochs,\
                actor_epochs,\
                grad_accum_steps,\
                max_grad_norm,
                trajectory_critic_epochs,
                gamma=None,\
                train_algorithm=None,\
                rollout_size: int = 50,\
                eval_iterations: int = 10,\
                use_wandb: bool = False,
                save_path: str = None,
                decode_f: callable = lambda x: x,
                **kwargs):
    if train_algorithm == "vscrl":
        print(">>> Using vscrl trainer")
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

    agent.prepare()
    if os.path.exists(os.path.join(save_path, 'trainer.pt')):
        print(">>> Loading from previous checkpoint")
        trainer.load(os.path.join(save_path, 'trainer.pt'))
    else:
        print(">>> No previous checkpoint found")

    colorful_print(">>> Evaluating Agent", fg='blue')
    
    all_trajectories = []
    for i in range(eval_iterations):
        trajectories = batch_interact_environment(agent = agent,\
                                            agent_ref = agent_ref,\
                                            env = env,\
                                            num_trajectories= rollout_size,\
                                            accelerator = accelerator,\
                                            use_tqdm=False,
                                            decode_f = decode_f,
                                            gamma = gamma,
                                            iter=i,
                                            use_ref=False)
        if accelerator.is_main_process:
            info = {"iteration": i,\
                    "rollout.mean": np.mean([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                    "rollout.max": np.max([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                    "rollout.min": np.min([d[0]["trajectory_reward"] if len(d) > 0 else 0 for d in trajectories]),\
                    "walltime": time.time()}
            
            
            total_traj_num = len(trajectories)
            success_traj_num = 0
            for traj in trajectories:
                for traj_step in range(len(traj)):
                    if traj[traj_step]["task_success"] != 0:
                        success_traj_num += 1
                        break
            assert success_traj_num <= total_traj_num, "Success traj num should be less than total traj num"
            info.update({"rollout.task_success_rate": float(success_traj_num / total_traj_num)})
            print(f"Current Task Success Rate: {float(success_traj_num / total_traj_num)}")

            all_trajectories += trajectories
            
            torch.save(all_trajectories, os.path.join(save_path, 'trajectories_eval.pt'))
            time.sleep(15)
        else:
            info = {}
        accelerator.wait_for_everyone()
        all_trajectories = torch.load(os.path.join(save_path, 'trajectories_eval.pt'))
        if use_wandb and accelerator.is_main_process:
            wandb.log(info)
            