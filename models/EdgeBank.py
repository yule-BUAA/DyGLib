import numpy as np
from collections import defaultdict

from utils.DataLoader import Data


def predict_link_probabilities(edge_memories: set, edges_tuple: tuple):
    """
    get the link probabilities by predicting whether each edge in edges_tuple appears in edge_memories
    :param edge_memories: set, store the edges in memory, {(src_node_id, dst_node_id), ...}
    :param edges_tuple: tuple, edges with (src_node_ids, dst_node_ids)
    :return:
    """
    src_node_ids, dst_node_ids = edges_tuple
    # probabilities of all the edges
    probabilities = []
    for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids):
        if (src_node_id, dst_node_id) in edge_memories:
            probabilities.append(1.0)
        else:
            probabilities.append(0.0)

    return np.array(probabilities)


def edge_bank_unlimited_memory(history_src_node_ids: np.ndarray, history_dst_node_ids: np.ndarray):
    """
    EdgeBank with unlimited memory, which stores every edge that it has seen
    :param history_src_node_ids: ndarray, shape (num_historical_edges, )
    :param history_dst_node_ids: ndarray, shape (num_historical_edges, )
    :return:
    """
    edge_memories = set((history_src_node_id, history_dst_node_id) for history_src_node_id, history_dst_node_id
                        in zip(history_src_node_ids, history_dst_node_ids))
    return edge_memories


def edge_bank_time_window_memory(history_src_node_ids: np.ndarray, history_dst_node_ids: np.ndarray, history_node_interact_times: np.ndarray,
                                 time_window_mode: str, time_window_proportion: float):
    """
    EdgeBank with time window memory, which only saves the edges that between time_window_start_time and time_window_end_time
    :param history_src_node_ids: ndarray, shape (num_historical_edges, )
    :param history_dst_node_ids: ndarray, shape (num_historical_edges, )
    :param history_node_interact_times: ndarray, shape (num_historical_edges, )
    :param time_window_mode: str, time window mode for time window memory, can be 'fixed_proportion' or 'repeat_interval'
    :param time_window_proportion: float, proportion of the time window in historical data
    :return:
    """
    # get window start and end time to determine window size
    # fixed_proportion, which sets the time window size to the duration of test data ratio
    if time_window_mode == 'fixed_proportion':
        time_window_start_time = np.quantile(history_node_interact_times, 1 - time_window_proportion)
        time_window_end_time = max(history_node_interact_times)
    # repeat_interval, which sets the time window size to average time intervals of repeated edges
    elif time_window_mode == 'repeat_interval':
        edge_time_intervals = defaultdict(list)
        for history_src_node_id, history_dst_node_id, history_node_interact_time in \
                zip(history_src_node_ids, history_dst_node_ids, history_node_interact_times):
            edge_time_intervals[(history_src_node_id, history_dst_node_id)].append(history_node_interact_time)

        sum_edge_time_intervals = 0
        for edge_tuple, edge_time_interval_list in edge_time_intervals.items():
            if len(edge_time_interval_list) > 1:
                sum_edge_time_intervals += np.mean([edge_time_interval_list[i + 1] - edge_time_interval_list[i] for i in range(len(edge_time_interval_list) - 1)])

        average_edge_time_intervals = sum_edge_time_intervals / len(edge_time_intervals)
        time_window_end_time = max(history_node_interact_times)
        time_window_start_time = time_window_end_time - average_edge_time_intervals
    else:
        raise ValueError(f'Not implemented error for time_window_mode {time_window_mode}!')

    memory_mask = np.logical_and(history_node_interact_times <= time_window_end_time, history_node_interact_times >= time_window_start_time)
    edge_memories = edge_bank_unlimited_memory(history_src_node_ids[memory_mask], history_dst_node_ids[memory_mask])
    return edge_memories


def edge_bank_repeat_threshold_memory(history_src_node_ids: np.ndarray, history_dst_node_ids: np.ndarray):
    """
    EdgeBank with repeat threshold memory, which only saves edges that have repeatedly appeared more than a threshold
    :param history_src_node_ids: ndarray, shape (num_historical_edges, )
    :param history_dst_node_ids: ndarray, shape (num_historical_edges, )
    :return:
    """
    # frequency of each edge
    edge_frequencies = defaultdict(int)
    for history_src_node_id, history_dst_node_id in zip(history_src_node_ids, history_dst_node_ids):
        edge_frequencies[(history_src_node_id, history_dst_node_id)] += 1
    threshold = np.array(list(edge_frequencies.values())).mean()

    edge_memories = set(edge_tuple for edge_tuple, edge_frequency in edge_frequencies.items() if edge_frequency >= threshold)
    return edge_memories


def edge_bank_link_prediction(history_data: Data, positive_edges: tuple, negative_edges: tuple, edge_bank_memory_mode: str,
                              time_window_mode: str, time_window_proportion: float):
    """
    EdgeBank for link prediction
    :param history_data: Data, history data
    :param positive_edges: tuple, positive edges with (src_node_ids, dst_node_ids)
    :param negative_edges: tuple, negative edges with (neg_src_node_ids, neg_dst_node_ids)
    :param edge_bank_memory_mode: str, memory mode in EdgeBank, can be 'unlimited_memory', 'time_window_memory' or 'repeat_threshold_memory'
    :param time_window_mode: str, time window mode for time window memory, can be 'fixed_proportion' or 'repeat_interval'
    :param time_window_proportion: float, proportion of the time window in historical data
    :return:
    """

    if edge_bank_memory_mode == 'unlimited_memory':
        edge_memories = edge_bank_unlimited_memory(history_src_node_ids=history_data.src_node_ids, history_dst_node_ids=history_data.dst_node_ids)
    elif edge_bank_memory_mode == 'time_window_memory':
        edge_memories = edge_bank_time_window_memory(history_src_node_ids=history_data.src_node_ids, history_dst_node_ids=history_data.dst_node_ids,
                                                     history_node_interact_times=history_data.node_interact_times, time_window_mode=time_window_mode,
                                                     time_window_proportion=time_window_proportion)
    elif edge_bank_memory_mode == 'repeat_threshold_memory':
        edge_memories = edge_bank_repeat_threshold_memory(history_src_node_ids=history_data.src_node_ids, history_dst_node_ids=history_data.dst_node_ids)
    else:
        raise ValueError(f'Not implemented error for edge_bank_memory_mode {edge_bank_memory_mode}!')

    positive_probabilities = predict_link_probabilities(edge_memories=edge_memories, edges_tuple=positive_edges)
    negative_probabilities = predict_link_probabilities(edge_memories=edge_memories, edges_tuple=negative_edges)

    return positive_probabilities, negative_probabilities
