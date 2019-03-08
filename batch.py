###############################################################################
#                               batch Class                                   #
###############################################################################


class Batch:
    def __init__(self, batch_size):
        """
        Parameters
        ----------
        batch_size : int
          Size of the batch
        """
        self.batch = []
        self.__batch_size = batch_size

    def add(self, exp):
        """
        Add the list of experience to the batch, ie to the memory of the agent
        """
        self.batch.append(exp)

    def isFull(self):
        """
        Check if the stack is full or not.
        """
        if len(self.batch) == self.__batch_size:
            return True
        return False

    def empty(self):
        self.batch = []
