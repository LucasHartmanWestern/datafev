from abc import ABC, abstractmethod

class env(ABC):

    @abstractmethod
    def __init__(
        self,
        actions,
        states
    ):
        """Initializes an environment and enforces structure for training reinforcement learning algorithms

        Args:
            actions: An array of possible actions to take
            states: An array of possible actions to take

        Returns:
            A tuple (state, reward, done) where:
            - state: The new state after taking the action.
            - reward: The reward obtained.
            - done: A boolean indicating whether the episode has ended.
        """

        self.reward_function = reward_function
        self.actions = actions
        self.states = states
        self.current_state = self.states[0]
        pass

    @abstractmethod
    def step(
        self,
        action              # action taken in this step
    ):
        """Performs a step in the environment.

        Args:
            action: The action to take.

        Returns:
            A tuple (state, reward, done) where:
            - state: The new state after taking the action.
            - reward: The reward obtained.
            - done: A boolean indicating whether the episode has ended.
        """

        pass

    def reset(self):
        """Resets the environment to its current state

        Returns:
            current_state: the default state after resetting
        """

        self.current_state = self.states[0]
        return self.current_state