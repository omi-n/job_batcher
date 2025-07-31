# Job Batcher

A Python utility for running multiple parameter sweep jobs across multiple GPUs using tmux sessions. This tool is particularly useful for machine learning experiments where you need to run the same script with different hyperparameter combinations distributed across available GPUs.

## Features

- **Multi-GPU support**: Automatically distributes jobs across available GPUs
- **Parameter sweeps**: Generate all combinations of hyperparameters using Cartesian product
- **YAML configuration**: Define jobs and parameters in YAML files
- **Tmux session management**: Each job runs in its own tmux session
- **Load balancing**: Automatically assigns jobs to the GPU with the fewest running jobs
- **Logging**: Saves output from each job to separate log files
- **Job monitoring**: Waits for jobs to complete before launching new ones when GPU capacity is reached

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd job_batcher
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.6+
- `tyro` - For command-line interface
- `PyYAML` - For YAML configuration file parsing
- `tmux` - For session management (install via your system package manager)
- `nvidia-smi` - For GPU detection (comes with NVIDIA drivers)

## Usage

### Basic Usage

You can use the job batcher in two ways:

#### 1. Command Line Arguments

```bash
python job_batcher.py \
  --command_template "python train.py --lr {{learning_rate}} --batch_size {{batch_size}}" \
  --template_args '{"learning_rate": [0.001, 0.01, 0.1], "batch_size": [32, 64, 128]}'
```

#### 2. YAML Configuration File

```bash
python job_batcher.py --config_file configs/humanoid_train.yaml
```

### Configuration Options

- `command_template`: Template for the command to run each job. Use `{{parameter_name}}` for placeholders
- `template_args`: Dictionary of parameters with their possible values (as lists)
- `config_file`: Path to YAML configuration file
- `job_prefix`: Prefix for tmux session names (default: "job")
- `setup_str`: Setup commands to run before each job (e.g., environment variables)
- `workers_per_gpu`: Number of concurrent jobs per GPU (default: 1)
- `log_dir`: Directory to store log files (default: "logs")

### Example Configuration Files

#### Humanoid Training (`configs/humanoid_train.yaml`)

```yaml
command_template: >-
  uv run main.py 
  --data.num_workers 2 
  --trainer.eval_num_episodes 50 
  --data.max_epochs {{num_epochs}} 
  --data.reward_percentiles 0.0 
  --use_wandb 
  --wandb_project {{wandb_project}} 
  --experiment_name lr{{learning_rate}}-ema{{ema_decay}}-{{loss_function}}-wd{{weight_decay}}-e{{num_epochs}} 
  --data.ds_name mujoco/{{env}}/{{level}} 
  --trainer.loss_function {{loss_function}} 
  --trainer.optimizer_config.learning_rate {{learning_rate}} 
  --trainer.optimizer_config.weight_decay {{weight_decay}}
  --trainer.optimizer_config.use_scheduler 
  --trainer.use_ema 
  --trainer.ema_start_epoch 3 
  --trainer.ema_update_interval 1 
  --trainer.ema_decay {{ema_decay}} 
  agent:dnn-agent-config --agent.n_future 1 --agent.n_history 1

template_args:
  learning_rate: [1e-5, 5e-5, 1e-4, 5e-4]
  ema_decay: [0.0, 0.99, 0.995, 0.999]
  loss_function: "mse_loss"
  num_epochs: [80, 160]
  env: "humanoid"
  level: "expert-v0"
  weight_decay: 1e-2
  wandb_project: "minari-humanoid-2"

job_prefix: "minari_job"
setup_str: "export MINARI_DATASETS_PATH=\"/path/to/data\""
```

This configuration will generate 32 different job combinations (4 × 4 × 1 × 2 learning rates × ema_decay × loss_function × num_epochs).

#### Walker2D Training (`configs/walker2d_train.yaml`)

Similar to the humanoid configuration but for the Walker2D environment with different wandb project and environment settings.

## How It Works

1. **Parameter Expansion**: The tool takes the `template_args` and generates all possible combinations using Cartesian product
2. **GPU Detection**: Automatically detects available GPUs using `nvidia-smi`
3. **Load Balancing**: Assigns each job to the GPU with the fewest currently running jobs
4. **Tmux Sessions**: Each job runs in its own detached tmux session for isolation
5. **Job Queue**: When all GPUs are at capacity, new jobs wait until existing jobs complete
6. **Logging**: Each job's output is redirected to a separate log file in the specified directory

## Monitoring Jobs

### View Running Jobs
```bash
tmux list-sessions | grep <job_prefix>
```

### Attach to a Specific Job
```bash
tmux attach-session -t <session_name>
```

### View Job Logs
```bash
tail -f logs/<job_prefix>_<job_id>.log
```

### Kill All Jobs with Prefix
```bash
tmux list-sessions | grep <job_prefix> | cut -d: -f1 | xargs -I {} tmux kill-session -t {}
```

## Example Workflow

1. Create a YAML configuration file with your experiment parameters
2. Run the job batcher:
   ```bash
   python job_batcher.py --config_file configs/my_experiment.yaml
   ```
3. Monitor progress:
   ```bash
   # List running jobs
   tmux list-sessions | grep minari_job
   
   # View logs
   tail -f logs/minari_job_gpu0_0.log
   ```
4. Results will be logged to individual files in the `logs/` directory

## Tips

- Use `workers_per_gpu > 1` if your jobs don't fully utilize GPU
- Set up proper environment variables in `setup_str` for reproducible experiments
- Use descriptive `job_prefix` names to easily identify different experiment runs
- Monitor GPU memory usage with `nvidia-smi` to optimize `workers_per_gpu`
