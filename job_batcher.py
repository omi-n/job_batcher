import subprocess
import dataclasses
import tyro
import itertools
import json
from typing import Optional, List
import yaml
import os
from pathlib import Path


@dataclasses.dataclass
class JobRunnerConfig:
    input_path: Optional[str] = None
    """Path to input. Will be substituted into {{input}} in the command template if provided."""
    output_path: Optional[str] = None
    """Path to output. Will be substituted into {{output}} in the command template if provided."""
    command_template: Optional[str] = None
    """Template for the command to run each job.
    Should contain placeholders for job-specific arguments. Use {arg1}, {arg2}, etc. for job-specific arguments."""
    template_args: Optional[str] = None
    """Dictionary of arguments to be used in the command template. Will fill in {arg1}, {arg2}, etc. in command_template. 
    Assumes key string, and list values."""
    config_file: Optional[str] = None
    """YAML Config file to load command_template and template_args from. Template_args should be a YAML dict."""
    job_prefix: str = "job"
    """Prefix for job names. Used to identify jobs in tmux sessions."""
    setup_str: str = ""
    """Setup string to run before each job. Can be used to set environment variables or other setup commands."""
    workers_per_gpu: int = 1
    """Number of workers to run per GPU. Used to determine how many jobs to run in parallel."""
    log_dir: str = "logs"
    """Directory to store logs for each job. Each job will have its own file in this directory."""

    # job-batcher concatenate
    concatenate: Optional[List[str]] = None
    """Whether to run in concatenate mode, where several yaml configs are concatenated into one. Hijacks output_path."""

    # job-batcher chunk
    chunk: Optional[int] = None
    """Number of chunks to split the commands into. When provided with config_file and output_path, will generate n YAML files in output_path directory."""


def run_command_get_output(command):
    """
    Run a shell command and return its output.

    Args:
        command (str): The command to run.

    Returns:
        str: The output of the command.

    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit status.
    """
    result = subprocess.run(
        command, shell=True, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def get_job_tmux_sessions(job_prefix: str):
    """
    Get a list of tmux sessions that match the given job prefix.

    Args:
        job_prefix (str): The prefix to filter tmux sessions.

    Returns:
        list: A list of tmux session names that match the prefix.
    """
    command = f'tmux list-sessions -F "#{{session_name}}" | grep "^{job_prefix}"'
    try:
        output = run_command_get_output(command)
        return output.splitlines() if output else []
    except subprocess.CalledProcessError:
        # If the command fails (e.g., no tmux sessions), return an empty list
        return []


def get_job_tmux_sessions_by_gpu(job_prefix: str, gpu_count: int):
    gpu_jobs = []
    for gpu_id in range(gpu_count):
        # Get tmux sessions for each GPU
        command = f'tmux list-sessions -F "#{{session_name}}" | grep "{job_prefix}_gpu{gpu_id}_"'
        try:
            output = run_command_get_output(command)
            gpu_jobs.append(output.splitlines() if output else [])
        except subprocess.CalledProcessError:
            # If the command fails (e.g., no tmux sessions for this GPU), append an empty list
            gpu_jobs.append([])
    return gpu_jobs


def get_job_count_by_gpu(job_prefix: str, gpu_id: int):
    gpu_jobs = get_job_tmux_sessions_by_gpu(job_prefix, gpu_id)
    return [len(jobs) for jobs in gpu_jobs]


def launch_job_in_tmux(command, job_prefix, postfix, gpu_id=None, setup_str=""):
    """
    Launch a command in a new tmux session with optional GPU assignment.

    This function creates a new detached tmux session and runs the specified command
    within it. If a GPU ID is provided, the CUDA_VISIBLE_DEVICES environment variable
    is set to restrict the command to use only that specific GPU.

    Args:
        command (str): The command to execute in the tmux session.
        job_prefix (str): Prefix for the tmux session name.
        postfix (str): Postfix for the tmux session name.
        gpu_id (int, optional): GPU ID to assign to the job. If provided, sets
            CUDA_VISIBLE_DEVICES environment variable and adds GPU info to session name.
            Defaults to None.
        setup_str (str, optional): Setup string to run before the command. Defaults to "".

    Returns:
        The output from run_command_get_output() function, typically containing
        the result of the tmux session creation command.

    Note:
        The tmux session name format is "{job_prefix}_{postfix}" when gpu_id is None,
        or "{job_prefix}_gpu{gpu_id}_{postfix}" when gpu_id is specified.
    """
    if gpu_id is not None:
        # If a specific GPU ID is provided, set the CUDA_VISIBLE_DEVICES environment variable
        command = f"CUDA_VISIBLE_DEVICES={gpu_id} {command}"
        job_prefix += f"_gpu{gpu_id}"

    # Add setup string before the command if provided
    if setup_str:
        command = f"{setup_str} && {command}"

    tmux_cmd = f'tmux new-session -d -s "{job_prefix}_{postfix}" "{command}"'
    return run_command_get_output(tmux_cmd)


def launch_tmux_and_dump_logs(
    command,
    job_prefix,
    postfix,
    log_dir,
    gpu_id=None,
    setup_str="",
    finished_commands_file=None,
):
    """Launch a tmux session for the job and redirect its output to a log file."""
    # Save the original command before any modifications
    original_command = command

    if gpu_id is not None:
        # If a specific GPU ID is provided, set the CUDA_VISIBLE_DEVICES environment variable
        command = f"CUDA_VISIBLE_DEVICES={gpu_id} {command}"
        job_prefix += f"_gpu{gpu_id}"

    # Add setup string before the command if provided
    if setup_str:
        command = f"{setup_str} && {command}"

    log_file = f"{log_dir}/{job_prefix}_{postfix}.log"

    command = f"{command} > {log_file} 2>&1"

    # If finished_commands_file is provided, append original command to it after completion
    if finished_commands_file:
        # Escape single quotes in the original command for the shell
        escaped_command = original_command.replace("'", "\\'")
        completion_cmd = f"echo '{escaped_command}' >> {finished_commands_file}"
        command = f"{command}  && {completion_cmd}"

    tmux_cmd = f'tmux new-session -d -s "{job_prefix}_{postfix}" "{command}"'
    return run_command_get_output(tmux_cmd)


def get_gpu_count():
    """
    Get the number of available GPUs by counting the lines in the output of `nvidia-smi -L`.

    Returns:
        int: The number of available GPUs.
    """
    # Count the number of lines in the output of `nvidia-smi -L`
    # Each line corresponds to a GPU
    return int(run_command_get_output("nvidia-smi -L | wc -l"))


def load_finished_commands(log_dir: str):
    """
    Load the list of finished commands from all files in the finished_commands directory.
    This prevents race conditions by allowing each process to write to its own file.

    Args:
        log_dir (str): Directory containing the finished_commands subdirectory.

    Returns:
        set: A set of finished command strings from all process-specific files.
    """
    finished_commands_dir = os.path.join(log_dir, "finished_commands")
    if not os.path.exists(finished_commands_dir):
        return set()

    finished_commands = set()
    # Read all files in the finished_commands directory
    for filename in os.listdir(finished_commands_dir):
        filepath = os.path.join(finished_commands_dir, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r") as f:
                # Read all lines and strip whitespace, filter out empty lines
                finished_commands.update(line.strip() for line in f if line.strip())

    return finished_commands


def get_finished_commands_file(log_dir: str, node_rank: int = 0, gpu_id: int = None):
    """
    Get the path to the finished commands file for a specific process.
    Each process (identified by node_rank and gpu_id) gets its own file to prevent race conditions.

    Args:
        log_dir (str): Directory containing the finished_commands subdirectory.
        node_rank (int): The rank of the current node/process. Defaults to 0.
        gpu_id (int, optional): The GPU ID assigned to this process. Defaults to None.

    Returns:
        str: Path to the finished commands file for this specific process.
    """
    finished_commands_dir = os.path.join(log_dir, "finished_commands")
    os.makedirs(finished_commands_dir, exist_ok=True)

    # Create a unique filename based on node rank and GPU ID
    if gpu_id is not None:
        filename = f"node{node_rank}_gpu{gpu_id}.txt"
    else:
        filename = f"node{node_rank}.txt"

    return os.path.join(finished_commands_dir, filename)


def load_yaml_config_and_generate_commands(config_file: str):
    """
    Load a YAML config file and generate all command combinations.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        tuple: A tuple containing (commands, config) where:
            - commands (list): List of all generated command strings
            - config (JobRunnerConfig): The loaded configuration object
    """
    # Load the yaml
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

        if "template_args" in config_dict:
            # if a key in template_args is not a list, make it a list
            for key, value in config_dict["template_args"].items():
                if not isinstance(value, list):
                    config_dict["template_args"][key] = [value]

            config_dict["template_args"] = json.dumps(config_dict["template_args"])

    # Create config object
    config = JobRunnerConfig(**config_dict)

    template = config.command_template

    # Format: {key: [value1, value2, ...], ...}
    template_args = list(json.loads(config.template_args).items())

    # Extract keys and values for cartesian product
    keys = [item[0] for item in template_args]
    value_lists = [item[1] for item in template_args]

    # Generate all combinations using cartesian product
    commands = []
    for combination in itertools.product(*value_lists):
        # Create a dictionary mapping each key to its value in this combination
        substitutions = dict(zip(keys, combination))

        # Add input/output paths if provided
        if config.input_path is not None:
            substitutions["input_path"] = config.input_path
        if config.output_path is not None:
            substitutions["output_path"] = config.output_path

        # Substitute values into the template
        command = template
        for key, value in substitutions.items():
            command = command.replace(f"{{{{{key}}}}}", str(value))

        commands.append(command)

    return commands, config


def main():
    """Main entry point for the job batcher CLI.

    Example Usage:
    job-batcher --command_template "asdf --aa {{aa}} --bb {{bb}}" --template_args '{"aa": [1, 2], "bb": [1, 2]}'
    """
    config = tyro.cli(JobRunnerConfig)

    if config.concatenate is not None and config.output_path is not None:
        # Create a new yaml with concatenated configs, store in output_path
        all_commands = []
        config_files = []

        # if path is a folder, get all yaml files in the folder
        # else, add to config_files
        for path in config.concatenate:
            p = Path(path)
            if p.is_dir():
                config_files.extend(sorted([str(f) for f in p.glob("**/*.yaml")]))
            elif p.is_file():
                config_files.append(str(p))
            else:
                print(f"Warning: {path} is not a valid file or directory. Skipping.")

        # Load each config file and generate commands
        for config_file in config_files:
            commands, _ = load_yaml_config_and_generate_commands(config_file)
            all_commands.extend(commands)

        # Create a new config dict with {{command}} template and all commands as template_args
        concatenated_config = {
            "command_template": "{{command}}",
            "template_args": {"command": all_commands},
        }

        # Copy over other relevant config fields if they exist
        if config.job_prefix != "job":
            concatenated_config["job_prefix"] = config.job_prefix
        if config.setup_str:
            concatenated_config["setup_str"] = config.setup_str
        if config.workers_per_gpu != 1:
            concatenated_config["workers_per_gpu"] = config.workers_per_gpu
        if config.log_dir != "logs":
            concatenated_config["log_dir"] = config.log_dir

        # Write the concatenated config to output_path
        with open(config.output_path, "w") as f:
            yaml.dump(concatenated_config, f, default_flow_style=False)

        print(
            f"Concatenated {len(config.concatenate)} config files into {config.output_path}"
        )
        print(f"Total commands: {len(all_commands)}")
        return

    if (
        config.chunk is not None
        and config.config_file is not None
        and config.output_path is not None
    ):
        # Chunk mode: load config, generate commands, split into n chunks
        commands, source_config = load_yaml_config_and_generate_commands(
            config.config_file
        )

        # Create output directory if it doesn't exist
        output_dir = Path(config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate chunk size
        num_chunks = config.chunk
        total_commands = len(commands)
        chunk_size = (total_commands + num_chunks - 1) // num_chunks  # Ceiling division

        # Split commands into chunks and create YAML files
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_commands)
            chunk_commands = commands[start_idx:end_idx]

            if not chunk_commands:  # Skip empty chunks
                continue

            # Create chunk config with {{command}} template
            chunk_config = {
                "command_template": "{{command}}",
                "template_args": {"command": chunk_commands},
            }

            # Copy over other relevant config fields from source config
            if source_config.job_prefix != "job":
                chunk_config["job_prefix"] = source_config.job_prefix
            if source_config.setup_str:
                chunk_config["setup_str"] = source_config.setup_str
            if source_config.workers_per_gpu != 1:
                chunk_config["workers_per_gpu"] = source_config.workers_per_gpu
            if source_config.log_dir != "logs":
                chunk_config["log_dir"] = source_config.log_dir

            # Write chunk to file
            chunk_filename = output_dir / f"chunk_{chunk_idx:03d}.yaml"
            with open(chunk_filename, "w") as f:
                yaml.dump(chunk_config, f, default_flow_style=False)

            print(f"Created {chunk_filename} with {len(chunk_commands)} commands")

        print(
            f"\nChunked {total_commands} commands into {num_chunks} files in {output_dir}"
        )
        return

    # Check if all required parameters are None and show help
    if (
        config.command_template is None
        and config.template_args is None
        and config.config_file is None
    ):
        print(
            "Error: You must provide either --config_file or both --command_template and --template_args"
        )
        print("\nUsage examples:")
        print("  1. Using command line arguments:")
        print(
            '     job-batcher --command_template "python train.py --lr {{lr}}" --template_args \'{"lr": [0.001, 0.01]}\''
        )
        print("\n  2. Using a config file:")
        print("     job-batcher --config_file configs/my_experiment.yaml")
        print("\n  3. Chunking a config file:")
        print(
            "     job-batcher --config_file configs/my_experiment.yaml --chunk 4 --output_path chunks/"
        )
        print("\nFor more help, run: job-batcher --help")
        return

    # Check if chunk mode is being used but missing required parameters
    if config.chunk is not None:
        if config.config_file is None:
            print("Error: --chunk requires --config_file to be specified")
            print(
                "Usage: job-batcher --config_file <path> --chunk <n> --output_path <directory>"
            )
            return
        if config.output_path is None:
            print("Error: --chunk requires --output_path to be specified")
            print(
                "Usage: job-batcher --config_file <path> --chunk <n> --output_path <directory>"
            )
            return

    if config.config_file is not None:
        # Load the yaml
        with open(config.config_file, "r") as f:
            config_dict = yaml.safe_load(f)

            if "template_args" in config_dict:
                # if a key in template_args is not a list, make it a list
                for key, value in config_dict["template_args"].items():
                    if not isinstance(value, list):
                        config_dict["template_args"][key] = [value]

                config_dict["template_args"] = json.dumps(config_dict["template_args"])

        # Start with YAML config, then override with any CLI args that are not None
        loaded_config = JobRunnerConfig(**config_dict)
        for field in dataclasses.fields(JobRunnerConfig):
            cli_value = getattr(config, field.name)
            default_value = (
                field.default if field.default is not dataclasses.MISSING else None
            )
            # Only override if CLI value is not None and is different from the default
            if cli_value is not None and cli_value != default_value:
                setattr(loaded_config, field.name, cli_value)
        config = loaded_config

    # If not using config file, both command_template and template_args are required
    if config.config_file is None and (
        config.command_template is None or config.template_args is None
    ):
        print(
            "Error: When not using --config_file, both --command_template and --template_args are required"
        )
        print("\nUsage examples:")
        print("  1. Using command line arguments:")
        print(
            '     job-batcher --command_template "python train.py --lr {{lr}}" --template_args \'{"lr": [0.001, 0.01]}\''
        )
        print("\n  2. Using a config file:")
        print("     job-batcher --config_file configs/my_experiment.yaml")
        print("\nFor more help, run: job-batcher --help")
        return

    template = config.command_template

    # Format: {key: [value1, value2, ...], ...}
    template_args = list(json.loads(config.template_args).items())

    # Extract keys and values for cartesian product
    keys = [item[0] for item in template_args]
    value_lists = [item[1] for item in template_args]

    # Generate all combinations using cartesian product
    commands = []
    for combination in itertools.product(*value_lists):
        # Create a dictionary mapping each key to its value in this combination
        substitutions = dict(zip(keys, combination))

        # Add input/output paths if provided
        if config.input_path is not None:
            substitutions["input_path"] = config.input_path
        if config.output_path is not None:
            substitutions["output_path"] = config.output_path

        # Substitute values into the template
        command = template
        for key, value in substitutions.items():
            command = command.replace(f"{{{{{key}}}}}", str(value))

        commands.append(command)

    # Get WORLD_SIZE and NODE_RANK to get the current process's rank and total number of processes
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("NODE_RANK", "0"))
    world_size = int(os.environ.get("SLURM_NNODES", world_size))
    rank = int(os.environ.get("SLURM_NODEID", rank))

    # Get gpu count
    gpu_count = get_gpu_count()
    print(f"Detected {gpu_count} GPUs available.")

    # Load finished commands to skip already completed jobs
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    finished_commands = load_finished_commands(config.log_dir)

    # Filter out already finished commands
    original_command_count = len(commands)
    commands_to_run = [cmd for cmd in commands if cmd not in finished_commands]
    skipped_count = original_command_count - len(commands_to_run)

    # If world_size == 1, run all commands in this process so don't chunk
    if world_size > 1:
        # Chunk commands based on rank and world_size
        commands_to_run = [
            cmd for i, cmd in enumerate(commands_to_run) if i % world_size == rank
        ]

    if skipped_count > 0:
        print(f"Skipping {skipped_count} already finished command(s).")

    if len(commands_to_run) == 0:
        print("All commands have already been completed. Nothing to run.")
        return

    # Launch jobs. If there is <gpu_count>*<jobs_per_gpu> jobs, pause launching.
    # When a jobs finishes, launch the next one.
    # Launch that job to the gpu with fewest jobs running.
    # gpu_jobs = get_job_count_by_gpu(config.job_prefix, gpu_count)

    for idx, job in enumerate(commands_to_run):
        while (
            len(get_job_tmux_sessions(config.job_prefix))
            >= gpu_count * config.workers_per_gpu
        ):
            print("Waiting for a job to finish...")
            run_command_get_output("sleep 5")

        gpu_jobs = get_job_tmux_sessions_by_gpu(config.job_prefix, gpu_count)
        min_gpu_id = min(range(gpu_count), key=lambda i: len(gpu_jobs[i]))

        # Get the finished commands file for this specific GPU and node
        finished_commands_file = get_finished_commands_file(
            config.log_dir, rank, min_gpu_id
        )

        launch_tmux_and_dump_logs(
            job,
            config.job_prefix,
            idx,
            config.log_dir,
            min_gpu_id,
            config.setup_str,
            finished_commands_file,
        )
        print(f"Launched job {idx} on GPU {min_gpu_id}: {job}")

    # Wait for all jobs to finish before exiting
    print("All jobs launched. Waiting for all jobs to finish...")
    while len(get_job_tmux_sessions(config.job_prefix)) > 0:
        print(f"{len(get_job_tmux_sessions(config.job_prefix))} jobs still running...")
        run_command_get_output("sleep 5")
    print("All jobs finished.")


if __name__ == "__main__":
    main()
