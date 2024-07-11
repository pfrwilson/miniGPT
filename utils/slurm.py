from dataclasses import dataclass


@dataclass
class SlurmConfig: 
    """
    Configuration for the requested compute resources
    """
    mem_gb: int = 8
    cpu_per_task: int = 8 
    timeout_min: int | None = None # if set, job will timeout after this many minutes
    gres: str = "gpu:1" 
    partition: str = "t4v2" # other options a40, rtx6000, cpu
    tasks_per_node: int = 1 