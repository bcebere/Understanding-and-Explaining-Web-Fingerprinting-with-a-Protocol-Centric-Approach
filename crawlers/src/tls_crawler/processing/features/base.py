# stdlib
from abc import ABC, abstractmethod
from typing import List

# third party
import numpy as np
from scipy import stats as stat


class BasicStats(ABC):
    """This class extracts features related to the Packet Times."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_observed_data(self) -> List:
        ...

    def get_var(self) -> float:
        """Calculates the variation of packet times in a network flow.

        Returns:
            float: The variation of packet times.

        """
        return float(np.var(self.get_observed_data()))

    def get_std(self) -> float:
        """Calculates the standard deviation of packet times in a network flow.

        Returns:
            float: The standard deviation of packet times.

        """
        return float(np.sqrt(self.get_var()))

    def get_mean(self) -> float:
        """Calculates the mean of packet times in a network flow.

        Returns:
            float: The mean of packet times

        """
        mean = 0.0
        if len(self.get_observed_data()) != 0:
            mean = np.mean(self.get_observed_data())

        return float(mean)

    def get_median(self) -> float:
        """Calculates the median of packet times in a network flow.

        Returns:
            float: The median of packet times

        """
        return float(np.median(self.get_observed_data()))

    def get_mode(self) -> float:
        """The mode of packet times in a network flow.

        Returns:
            float: The mode of packet times

        """
        mode = -1.0
        if len(self.get_observed_data()) != 0:
            mode = float(stat.mode(self.get_observed_data(), keepdims=True)[0])

        return mode

    def get_skew(self) -> float:
        """Calculates the skew of packet times in a network flow using the median.

        Returns:
            float: The skew of packet times.

        """
        mean = self.get_mean()
        median = self.get_median()
        dif = 3 * (mean - median)
        std = self.get_std()
        skew = -10.0

        if std != 0:
            skew = dif / std

        return float(skew)

    def get_skew2(self) -> float:
        """Calculates the skew of the packet times ina network flow using the mode.

        Returns:
            float: The skew of the packet times.

        """
        mean = self.get_mean()
        mode = self.get_mode()
        dif = float(mean) - mode
        std = self.get_std()
        skew2 = -10.0

        if std != 0:
            skew2 = dif / float(std)

        return float(skew2)

    def get_cov(self) -> float:
        """Calculates the coefficient of variance of packet times in a network flow.

        Returns:
            float: The coefficient of variance of a packet times list.

        """
        cov = -1.0
        if self.get_mean() != 0:
            cov = self.get_std() / self.get_mean()

        return float(cov)
