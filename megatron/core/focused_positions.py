def get_interval_block(interval, block_size):
    lo, hi = interval
    lo_block = lo // block_size
    hi_block_closed = (hi - 1) // block_size
    return lo_block, hi_block_closed + 1


def round_interval_by_block(interval, block_size):
    # [a, b) => [a_block_start, b_block_end)
    lo_block, hi_block = get_interval_block(interval, block_size)
    return lo_block * block_size, hi_block * block_size


class FocusedPositions:
    def __init__(self, block_size=None, intervals=None, window_size=None):
        """
        :param block_size: Any value less than or equal to 1 is equivalent to None
        :param intervals: Each interval `[start, end)` should be referred to as a list of two integers `start, end`
        :param window_size: Any value less than or equal to 0 is equivalent to `None`.
            If not `None`, describes the size of sliding window.
            If `block_size` is not None, should be a multiplier of `block_size`, and considered as fuzzy
        """
        if block_size is None or block_size < 1:
            block_size = 1
        if intervals is None:
            intervals = []
        else:
            intervals = [interval for interval in intervals]
        intervals.sort()
        if window_size is None or window_size < 0:
            window_size = 0
        self.block_size = block_size
        self.intervals = intervals
        self.window_size = window_size

    def get_actual_window(self, seq_len):
        """
        :returns: rounded_seq_len, window_start, window_end, actual_window_size
        """
        rounded_seq_len = ((seq_len + self.block_size - 1) // self.block_size) * self.block_size
        if self.window_size == 0:  # special case
            return rounded_seq_len, seq_len, seq_len, 0

        window_start = rounded_seq_len - self.window_size
        window_end = seq_len
        actual_window_size = window_end - window_start
        return rounded_seq_len, window_start, window_end, actual_window_size

    def get_all_positions(self, seq_len, attention_sink=True, granularity=1):
        rounded_seq_len, window_start, window_end, actual_window_size = self.get_actual_window(seq_len)
        intervals = self.intervals + [[window_start, window_end]]
        if attention_sink:
            intervals = intervals + [[0, self.block_size]]
        intervals = [interval for interval in intervals if interval[0] < interval[1]]  # remove all empty intervals
        intervals.sort()
        new_intervals = []
        for interval in intervals:
            start, end = interval
            end = min(end, seq_len)
            start //= granularity
            end = (end - 1) // granularity + 1
            if len(new_intervals) == 0:
                new_intervals.append([start, end])
            else:
                if start <= new_intervals[-1][1]:  # overlapped
                    # merge into consecutive intervals
                    new_intervals[-1][1] = max(new_intervals[-1][1], end)
                else:
                    new_intervals.append([start, end])
        return new_intervals


