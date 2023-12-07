"""
Custom implrmementation of PyTorch DataLoader based on 
https://github.com/horovod/horovod/blob/master/horovod/spark/data_loaders/pytorch_data_loaders.py
"""

from petastorm.pytorch import BatchedDataLoader, DataLoader, InMemBatchedDataLoader

class PytorchDataLoader():
    def __init__(self, reader, batch_size, shuffling_queue_capacity = 0, name="",
                 limit_step_per_epoch=-1, verbose=False):
        self.reader = reader
        self.batch_size = batch_size
        self.shuffling_queue_capacity = shuffling_queue_capacity
        self.limit_step_per_epoch = limit_step_per_epoch
        self.name = name
        self.verbose = verbose

        print(f"[{self.name}]: Initializing petastorm dataloader with batch_size={batch_size}"
              f"shuffling_queue_capacity={shuffling_queue_capacity}, "
              f"limit_step_per_epoch={limit_step_per_epoch}")

    def __len__(self):
        # We cannot infer length from reader.
        return self.limit_step_per_epoch if self.limit_step_per_epoch != -1 else 0

    def __iter__(self):
        """
        Starting iteration and get batchs
        """
        for batch in self._iterate():
            yield self._process_batch(batch)
    
    def _process_batch(self, batch):
        """
        Hook to modify batch before output. Will be override by trainer to reshape the data
        as needed. Please do not override it.
        """
        return batch

    def _iterate(self):
        # Reset the reader if needed.
        if self.reader.last_row_consumed:
            self._print_verbose(f"[{self.name}]: Resetting Petastorm reader for {self.reader.dataset.paths}")
            self.reader.reset()

        # Re-create the data loader for each iteration. This is needed becasue there may be
        # some left-over data from last epoch which can cause petastorm's BatchedDataLoader
        # fail to start new iteration. To workaround the issue, we have to re-create the data
        # loader at each new iterration starts.
        # data_loader = BatchedDataLoader(
        data_loader = DataLoader(
            self.reader,
            batch_size=self.batch_size,
            shuffling_queue_capacity=self.shuffling_queue_capacity,
        )

        num_steps = 0

        self._print_verbose(f"[{self.name}]: Start to generate batch data. limit_step_per_epoch={self.limit_step_per_epoch}")

        for batch in data_loader:
            if num_steps == self.limit_step_per_epoch:
                self._print_verbose(f"[{self.name}]: Reach limit_step_per_epoch. Stop at step {num_steps}.")
                break

            num_steps += 1
            yield batch

    def _print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
