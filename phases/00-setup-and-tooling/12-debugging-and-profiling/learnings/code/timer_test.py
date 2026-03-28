import time

class Timer:

    def __init__(self, name) -> None:
        self.name = name

    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.start
        print(f"[{self.name}] {elapsed:.4f}s")


if __name__ == "__main__":  

    with Timer("while loop"):

        i = 0

        while True:
            i += 1
            if(i>900):
                break