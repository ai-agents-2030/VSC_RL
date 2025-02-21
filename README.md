
# VSC-RL: Advancing Autonomous Vision-Language Agents with Variational Subgoal-Conditioned Reinforcement Learning

This repository contains the implementation for reproducing VSC-RL in the AitW benchmarks, and our implementation is built on [DigiRL](https://github.com/DigiRL-agent/digirl).

The core files are listed as follows:
1. VSC-RL

        vscrl\algorithms\vscrl\trainer.py

2. Subgoal Generator
    
        subgoal_generator.py

Additionally, we provide the generated subgoals example.

    vscrl\environment\android\assets\task_set\general_train_subgoals.json

### 0. Requirements
```bash
conda create -n vscrl python==3.10
conda activate vscrl
pip install -e .
```

### 1. Set Up Android Emulator
Please refer to the detailed instruction of setup Android Emulator provided in DigiRL.

    https://github.com/DigiRL-agent/digirl/blob/master/env_setup/README.md

### 2. Download AutoUI-Base Model Checkpoint

```bash
wget https://huggingface.co/cooelf/Auto-UI/resolve/main/Auto-UI-Base.zip
unzip Auto-UI-Base.zip
```

### 3. Config Modification
Modify the config files (e.g., `huggingface_token`, `wandb_token`, `gemini_token`, `policy_lm`).

    scripts\config\main\default.yaml
    scripts\config\main\vscrl_online.yaml
    scripts\config\main\eval_online.yaml

### 4. Run VSC-RL
```bash
cd scripts
python run.py --config-path config/main --config-name vscrl_online
```



### Acknowledgement
[1] DigiRL: https://github.com/DigiRL-agent/digirl