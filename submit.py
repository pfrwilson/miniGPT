import submitit


def main(): 
    import logging 
    logger = logging.getLogger('submitit')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('test.log')
    logger.addHandler(handler)
    logger.info('hello world!')
    return 'done!'


if __name__ == "__main__": 
    from submitit import AutoExecutor
    executor = AutoExecutor(folder=".submitit")
    executor.update_parameters(
        mem_gb=8,
        cpus_per_task=16,
        timeout_min=60 * 24,
        tasks_per_node=2, 
        slurm_partition="cpu",
    )

    job = executor.submit(main)
    print(job.results())