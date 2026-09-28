import random
import numpy as np

def generate_sequence_data(n_jobs, seq_length, seed):
    jobs = []
    targets = []
    random.seed(seed)

    for _ in range(n_jobs):
        job = []

        cpu_utilisation = random.randint(0, 100)
        mem_utilisation = random.randint(0, 100)
        io_wait = random.randint(0, 100)
        job.append([cpu_utilisation, mem_utilisation, io_wait])

        for _ in range(seq_length - 1):
            cpu_utilisation = random.randint(cpu_utilisation-10, cpu_utilisation+10)
            mem_utilisation = random.randint(mem_utilisation-10, mem_utilisation+10)
            io_wait = random.randint(io_wait-10, io_wait+10)

            if cpu_utilisation > 100: cpu_utilisation = 100
            if mem_utilisation > 100: mem_utilisation = 100
            if io_wait > 100: io_wait = 100

            if cpu_utilisation < 0: cpu_utilisation = 0
            if mem_utilisation < 0: mem_utilisation = 0
            if io_wait < 0: io_wait = 0

            job.append([cpu_utilisation, mem_utilisation, io_wait])

        jobs.append(job)

    jobs = np.array(jobs)

    for i, _ in enumerate(jobs):
        early_avg = jobs[i, :25, 0].mean()
        late_avg = jobs[i, -25:, 0].mean()
        targets.append(1 if (late_avg - early_avg) > 20 else 0)

    targets = np.array(targets)

    return jobs, n_jobs, targets


if __name__ == "__main__":
    jobs, n_jobs, targets = generate_sequence_data(500000, 50, 42)
    np.save("sequence_data.npy", jobs)
    np.save("targets.npy", targets)
    print(f"All {n_jobs} training jobs has been saved in sequence_training_data.npy.")
    print(f"All {n_jobs} targets has been saved in target_training.npy.")