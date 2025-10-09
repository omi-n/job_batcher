# Job Batcher

A Python utility for running multiple parameter sweep jobs across multiple GPUs using tmux sessions. This tool is particularly useful for machine learning experiments where you need to run the same script with different hyperparameter combinations distributed across available GPUs.

## Features

- **Multi-GPU support**: Automatically distributes jobs across available GPUs
- **Parameter sweeps**: Generate all combinations of hyperparameters
- **YAML configuration**: Define jobs and parameters in YAML files
- **Config concatenation**: Combine multiple YAML configs into a single config file
- **Tmux session management**: Each job runs in its own tmux session
- **Load balancing**: Automatically assigns jobs to the GPU with the fewest running jobs
- **Logging**: Saves output from each job to separate log files
- **Job monitoring**: Waits for jobs to complete before launching new ones when GPU capacity is reached
- **Programmatic API**: Use `load_yaml_config_and_generate_commands()` to generate commands programmatically

## Installation

### From GitHub (Recommended)

Install directly from GitHub using pip:

```bash
pip install git+https://github.com/omi-n/job_batcher.git
```

### From Source

1. Clone this repository:
```bash
git clone https://github.com/omi-n/job_batcher.git
cd job_batcher
```

2. Install in development mode:
```bash
pip install -e .
```

### Requirements

- Python 3.6+
- `tyro` - For command-line interface
- `PyYAML` - For YAML configuration file parsing
- `tmux` - For session management (install via your system package manager)
- `nvidia-smi` - For GPU detection (comes with NVIDIA drivers)

## Usage

After installation, you can use the `job-batcher` command from anywhere:

### Basic Usage

You can use the job batcher in two ways:

#### 1. Command Line Arguments

```bash
job-batcher \
  --command_template "python train.py --lr {{learning_rate}} --batch_size {{batch_size}}" \
  --template_args '{"learning_rate": [0.001, 0.01, 0.1], "batch_size": [32, 64, 128]}'
```

#### 2. YAML Configuration File

```bash
job-batcher --config_file configs/humanoid_train.yaml
```

#### 3. Concatenate Multiple Configs

Combine multiple YAML configuration files into a single config:

```bash
job-batcher \
  --concatenate configs/humanoid_train.yaml configs/walker2d_train.yaml \
  --output_path configs/combined.yaml
```

You can also point to a folder, and it will automatically find and parse all YAML files in that folder (recursively):

```bash
job-batcher \
  --concatenate configs/experiments/ \
  --output_path configs/all_experiments.yaml
```

This will:
- Load each config file (or all YAML files in specified folders) and generate all command combinations
- Create a new YAML file at `output_path` with all commands
- The new config uses `{{command}}` as the template with all commands as values

You can then run the combined config:

```bash
job-batcher --config_file configs/combined.yaml
```

### Configuration Options

- `command_template`: Template for the command to run each job. Use `{{parameter_name}}` for placeholders
- `template_args`: Dictionary of parameters with their possible values (as lists)
- `config_file`: Path to YAML configuration file
- `concatenate`: List of YAML config files or folders to concatenate (requires `output_path`). Folders are searched recursively for all `.yaml` files
- `output_path`: Path to save concatenated config file (used with `concatenate`)
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

## Programmatic Usage

You can use the job batcher programmatically in your Python scripts:

```python
from job_batcher import load_yaml_config_and_generate_commands

# Load a config file and generate all command combinations
commands, config = load_yaml_config_and_generate_commands("configs/humanoid_train.yaml")

print(f"Generated {len(commands)} commands")
for i, cmd in enumerate(commands[:3]):  # Print first 3 commands
    print(f"Command {i}: {cmd}")

# Access configuration
print(f"Job prefix: {config.job_prefix}")
print(f"Workers per GPU: {config.workers_per_gpu}")
```

This is useful for:
- Previewing commands before running them
- Integrating job generation into larger workflows
- Custom job scheduling logic
- Debugging configuration issues

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

### Basic Workflow

1. Create a YAML configuration file with your experiment parameters
2. Run the job batcher:
   ```bash
   job-batcher --config_file configs/my_experiment.yaml
   ```
3. Monitor progress:
   ```bash
   # List running jobs
   tmux list-sessions | grep minari_job
   
   # View logs
   tail -f logs/minari_job_gpu0_0.log
   ```
4. Results will be logged to individual files in the `logs/` directory

### Advanced Workflow: Combining Multiple Experiments

If you have multiple experiment configurations and want to run them all together:

1. Create individual config files for each experiment:
   - `configs/humanoid_train.yaml`
   - `configs/walker2d_train.yaml`
   - `configs/hopper_train.yaml`

2. Concatenate them into a single config (you can mix files and folders):
   ```bash
   job-batcher \
     --concatenate configs/robotics/ configs/extra_experiment.yaml \
     --output_path configs/all_experiments.yaml \
     --job_prefix "combined_exp" \
     --workers_per_gpu 2
   ```
   
   Or concatenate an entire folder:
   ```bash
   job-batcher \
     --concatenate configs/ \
     --output_path configs/all_experiments.yaml
   ```

3. Run the combined configuration:
   ```bash
   job-batcher --config_file configs/all_experiments.yaml
   ```

This approach is useful when:
- You want to run diverse experiments with different hyperparameters
- You need to fairly distribute GPU resources across multiple projects
- You want a single unified config for reproducibility
- You have many config files organized in folders

## Tips

- Use `workers_per_gpu > 1` if your jobs don't fully utilize GPU
- Set up proper environment variables in `setup_str` for reproducible experiments
- Use descriptive `job_prefix` names to easily identify different experiment runs
- Monitor GPU memory usage with `nvidia-smi` to optimize `workers_per_gpu`
