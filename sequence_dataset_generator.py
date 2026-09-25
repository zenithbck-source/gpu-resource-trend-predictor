import random
import numpy as np

def generate_sequence_data(n_jobs, seq_length, seed):
    jobs = []
    random.seed(seed)

    for _ in range(n_jobs):
        job = []
        cpu_utilisation = random.randint(0, 100)
        mem_utilisation = random.randint(0, 100)
        io_wait = random.randint(0, 100)
        job.append([cpu_utilisation, mem_utilisation, io_wait])

        for _ in range(seq_length):
            cpu_utilisation = random.randint(cpu_utilisation-10, cpu_utilisation+10)
            mem_utilisation = random.randint(mem_utilisation-10, mem_utilisation+10)
            io_wait = random.randint(io_wait-10, io_wait+10)
            job.append([cpu_utilisation, mem_utilisation, io_wait])

        jobs.append(job)
    jobs = np.array(jobs)

    return jobs, n_jobs


if __name__ == "__main__":
    jobs, n_jobs = generate_sequence_data(200000, 50, 42)
    np.save("sequence_training_data.npy", jobs)
    print(f"All {n_jobs} training jobs has been saved in sequence_training_data.py.")

    jobs, n_jobs = generate_sequence_data(1000, 50, 7)
    np.save("sequence_testing_data.npy", jobs)
    print(f"All {n_jobs} testing jobs has been saved in sequence_testing_data.py.")