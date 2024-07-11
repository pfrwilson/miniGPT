import os


def vector_slurm_checkpoint_dir():
    """
    Returns the checkpoint directory assigned by Slurm for the current job. 
    If slurm is not in use, returns None.
    """
    job_id = os.getenv('SLURM_JOB_ID')
    if job_id is None: 
        return None
    else: 
        return os.path.join(
            '/checkpoint', os.getenv('USER'), job_id
        )
    

def using_slurm():
    return os.getenv('SLURM_JOB_ID') is not None


