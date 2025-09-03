import subprocess
import dataclasses
import tyro
import itertools
import json
from typing import Optional
import yaml
import os


@dataclasses.dataclass
class JobRunnerConfig:
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
    command, job_prefix, postfix, log_dir, gpu_id=None, setup_str=""
):
    """Launch a tmux session for the job and redirect its output to a log file."""
    if gpu_id is not None:
        # If a specific GPU ID is provided, set the CUDA_VISIBLE_DEVICES environment variable
        command = f"CUDA_VISIBLE_DEVICES={gpu_id} {command}"
        job_prefix += f"_gpu{gpu_id}"

    # Add setup string before the command if provided
    if setup_str:
        command = f"{setup_str} && {command}"

    log_file = f"{log_dir}/{job_prefix}_{postfix}.log"
    tmux_cmd = (
        f'tmux new-session -d -s "{job_prefix}_{postfix}" "{command} > {log_file} 2>&1"'
    )
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


def main():
    """Main entry point for the job batcher CLI.

    Example Usage:
    job-batcher --command_template "asdf --aa {{aa}} --bb {{bb}}" --template_args '{"aa": [1, 2], "bb": [1, 2]}'
    """
    config = tyro.cli(JobRunnerConfig)

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

        # Substitute values into the template
        command = template
        for key, value in substitutions.items():
            command = command.replace(f"{{{{{key}}}}}", str(value))

        commands.append(command)

    # Get gpu count
    gpu_count = get_gpu_count()

    # Launch jobs. If there is <gpu_count>*<jobs_per_gpu> jobs, pause launching.
    # When a jobs finishes, launch the next one.
    # Launch that job to the gpu with fewest jobs running.
    # gpu_jobs = get_job_count_by_gpu(config.job_prefix, gpu_count)
    for idx, job in enumerate(commands):
        while (
            len(get_job_tmux_sessions(config.job_prefix))
            >= gpu_count * config.workers_per_gpu
        ):
            print("Waiting for a job to finish...")
            # Wait for a job to finish
            run_command_get_output("sleep 5")

        gpu_jobs = get_job_tmux_sessions_by_gpu(config.job_prefix, gpu_count)
        # Find the GPU with the fewest jobs running
        min_gpu_id = min(range(gpu_count), key=lambda i: len(gpu_jobs[i]))

        # Launch the job in tmux
        # if the log_dir does not exist, create it
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)

        launch_tmux_and_dump_logs(
            job, config.job_prefix, idx, config.log_dir, min_gpu_id, config.setup_str
        )
        print(f"Launched job {idx} on GPU {min_gpu_id}: {job}")


if __name__ == "__main__":
    main()
