import torch
import numpy as np
import torch.nn as nn
from collections import defaultdict

from utils.utils import NeighborSampler
from models.modules import TimeEncoder, MergeLayer, MultiHeadAttention


class MemoryModel(torch.nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, model_name: str = 'TGN', num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1,
                 src_node_mean_time_shift: float = 0.0, src_node_std_time_shift: float = 1.0, dst_node_mean_time_shift_dst: float = 0.0,
                 dst_node_std_time_shift: float = 1.0, device: str = 'cpu'):
        """
        General framework for memory-based models, support TGN, DyRep and JODIE.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param model_name: str, name of memory-based models, could be TGN, DyRep or JODIE
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param src_node_mean_time_shift: float, mean of source node time shifts
        :param src_node_std_time_shift: float, standard deviation of source node time shifts
        :param dst_node_mean_time_shift_dst: float, mean of destination node time shifts
        :param dst_node_std_time_shift: float, standard deviation of destination node time shifts
        :param device: str, device
        """
        super(MemoryModel, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.src_node_mean_time_shift = src_node_mean_time_shift
        self.src_node_std_time_shift = src_node_std_time_shift
        self.dst_node_mean_time_shift_dst = dst_node_mean_time_shift_dst
        self.dst_node_std_time_shift = dst_node_std_time_shift

        self.model_name = model_name
        # number of nodes, including the padded node
        self.num_nodes = self.node_raw_features.shape[0]
        self.memory_dim = self.node_feat_dim
        # since models use the identity function for message encoding, message dimension is 2 * memory_dim + time_feat_dim + edge_feat_dim
        self.message_dim = self.memory_dim + self.memory_dim + self.time_feat_dim + self.edge_feat_dim

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        # message module (models use the identity function for message encoding, hence, we only create MessageAggregator)
        self.message_aggregator = MessageAggregator()

        # memory modules
        self.memory_bank = MemoryBank(num_nodes=self.num_nodes, memory_dim=self.memory_dim)
        if self.model_name == 'TGN':
            self.memory_updater = GRUMemoryUpdater(memory_bank=self.memory_bank, message_dim=self.message_dim, memory_dim=self.memory_dim)
        elif self.model_name in ['DyRep', 'JODIE']:
            self.memory_updater = RNNMemoryUpdater(memory_bank=self.memory_bank, message_dim=self.message_dim, memory_dim=self.memory_dim)
        else:
            raise ValueError(f'Not implemented error for model_name {self.model_name}!')

        # embedding module
        if self.model_name == 'JODIE':
            self.embedding_module = TimeProjectionEmbedding(memory_dim=self.memory_dim, dropout=self.dropout)
        elif self.model_name in ['TGN', 'DyRep']:
            self.embedding_module = GraphAttentionEmbedding(node_raw_features=self.node_raw_features,
                                                            edge_raw_features=self.edge_raw_features,
                                                            neighbor_sampler=neighbor_sampler,
                                                            time_encoder=self.time_encoder,
                                                            node_feat_dim=self.node_feat_dim,
                                                            edge_feat_dim=self.edge_feat_dim,
                                                            time_feat_dim=self.time_feat_dim,
                                                            num_layers=self.num_layers,
                                                            num_heads=self.num_heads,
                                                            dropout=self.dropout)
        else:
            raise ValueError(f'Not implemented error for model_name {self.model_name}!')

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray,
                                                 edge_ids: np.ndarray, edges_are_positive: bool = True, num_neighbors: int = 20):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: ndarray, shape (batch_size, )
        :param edges_are_positive: boolean, whether the edges are positive,
        determine whether to update the memories and raw messages for nodes in src_node_ids and dst_node_ids or not
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        # Tensor, shape (2 * batch_size, )
        node_ids = np.concatenate([src_node_ids, dst_node_ids])

        # we need to use self.get_updated_memory instead of self.update_memory based on positive_node_ids directly,
        # because the graph attention embedding module in TGN needs to additionally access memory of neighbors.
        # so we return all nodes' memory with shape (num_nodes, ) by using self.get_updated_memory
        # updated_node_memories, Tensor, shape (num_nodes, memory_dim)
        # updated_node_last_updated_times, Tensor, shape (num_nodes, )
        updated_node_memories, updated_node_last_updated_times = self.get_updated_memories(node_ids=np.array(range(self.num_nodes)),
                                                                                           node_raw_messages=self.memory_bank.node_raw_messages)
        # compute the node temporal embeddings using the embedding module
        if self.model_name == 'JODIE':
            # compute differences between the time the memory of a node was last updated, and the time for which we want to compute the embedding of a node
            # Tensor, shape (batch_size, )
            src_node_time_intervals = torch.from_numpy(node_interact_times).float().to(self.device) - updated_node_last_updated_times[torch.from_numpy(src_node_ids)]
            src_node_time_intervals = (src_node_time_intervals - self.src_node_mean_time_shift) / self.src_node_std_time_shift
            # Tensor, shape (batch_size, )
            dst_node_time_intervals = torch.from_numpy(node_interact_times).float().to(self.device) - updated_node_last_updated_times[torch.from_numpy(dst_node_ids)]
            dst_node_time_intervals = (dst_node_time_intervals - self.dst_node_mean_time_shift_dst) / self.dst_node_std_time_shift
            # Tensor, shape (2 * batch_size, )
            node_time_intervals = torch.cat([src_node_time_intervals, dst_node_time_intervals], dim=0)
            # Tensor, shape (2 * batch_size, memory_dim), which is equal to (2 * batch_size, node_feat_dim)
            node_embeddings = self.embedding_module.compute_node_temporal_embeddings(node_memories=updated_node_memories,
                                                                                     node_ids=node_ids,
                                                                                     node_time_intervals=node_time_intervals)
        elif self.model_name in ['TGN', 'DyRep']:
            # Tensor, shape (2 * batch_size, node_feat_dim)
            node_embeddings = self.embedding_module.compute_node_temporal_embeddings(node_memories=updated_node_memories,
                                                                                     node_ids=node_ids,
                                                                                     node_interact_times=np.concatenate([node_interact_times,
                                                                                                                         node_interact_times]),
                                                                                     current_layer_num=self.num_layers,
                                                                                     num_neighbors=num_neighbors)
        else:
            raise ValueError(f'Not implemented error for model_name {self.model_name}!')

        # two Tensors, with shape (batch_size, node_feat_dim)
        src_node_embeddings, dst_node_embeddings = node_embeddings[:len(src_node_ids)], node_embeddings[len(src_node_ids): len(src_node_ids) + len(dst_node_ids)]

        if edges_are_positive:
            assert edge_ids is not None
            # if the edges are positive, update the memories for source and destination nodes (since now we have new messages for them)
            self.update_memories(node_ids=node_ids, node_raw_messages=self.memory_bank.node_raw_messages)

            # clear raw messages for source and destination nodes since we have already updated the memory using them
            self.memory_bank.clear_node_raw_messages(node_ids=node_ids)

            # compute new raw messages for source and destination nodes
            unique_src_node_ids, new_src_node_raw_messages = self.compute_new_node_raw_messages(src_node_ids=src_node_ids,
                                                                                                dst_node_ids=dst_node_ids,
                                                                                                dst_node_embeddings=dst_node_embeddings,
                                                                                                node_interact_times=node_interact_times,
                                                                                                edge_ids=edge_ids)
            unique_dst_node_ids, new_dst_node_raw_messages = self.compute_new_node_raw_messages(src_node_ids=dst_node_ids,
                                                                                                dst_node_ids=src_node_ids,
                                                                                                dst_node_embeddings=src_node_embeddings,
                                                                                                node_interact_times=node_interact_times,
                                                                                                edge_ids=edge_ids)

            # store new raw messages for source and destination nodes
            self.memory_bank.store_node_raw_messages(node_ids=unique_src_node_ids, new_node_raw_messages=new_src_node_raw_messages)
            self.memory_bank.store_node_raw_messages(node_ids=unique_dst_node_ids, new_node_raw_messages=new_dst_node_raw_messages)

        # DyRep does not use embedding module, which instead uses updated_node_memories based on previous raw messages and historical memories
        if self.model_name == 'DyRep':
            src_node_embeddings = updated_node_memories[torch.from_numpy(src_node_ids)]
            dst_node_embeddings = updated_node_memories[torch.from_numpy(dst_node_ids)]

        return src_node_embeddings, dst_node_embeddings

    def get_updated_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        get the updated memories based on node_ids and node_raw_messages (just for computation), but not update the memories
        :param node_ids: ndarray, shape (num_nodes, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        # aggregate messages for the same nodes
        # unique_node_ids, ndarray, shape (num_unique_node_ids, ), array of unique node ids
        # unique_node_messages, Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        # unique_node_timestamps, ndarray, shape (num_unique_node_ids, ), array of timestamps for unique nodes
        unique_node_ids, unique_node_messages, unique_node_timestamps = self.message_aggregator.aggregate_messages(node_ids=node_ids,
                                                                                                                   node_raw_messages=node_raw_messages)
        # get updated memory for all nodes with messages stored in previous batches (just for computation)
        # updated_node_memories, Tensor, shape (num_nodes, memory_dim)
        # updated_node_last_updated_times, Tensor, shape (num_nodes, )
        updated_node_memories, updated_node_last_updated_times = self.memory_updater.get_updated_memories(unique_node_ids=unique_node_ids,
                                                                                                          unique_node_messages=unique_node_messages,
                                                                                                          unique_node_timestamps=unique_node_timestamps)

        return updated_node_memories, updated_node_last_updated_times

    def update_memories(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        update memories for nodes in node_ids
        :param node_ids: ndarray, shape (num_nodes, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        # aggregate messages for the same nodes
        # unique_node_ids, ndarray, shape (num_unique_node_ids, ), array of unique node ids
        # unique_node_messages, Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        # unique_node_timestamps, ndarray, shape (num_unique_node_ids, ), array of timestamps for unique nodes
        unique_node_ids, unique_node_messages, unique_node_timestamps = self.message_aggregator.aggregate_messages(node_ids=node_ids,
                                                                                                                   node_raw_messages=node_raw_messages)

        # update the memories with the aggregated messages
        self.memory_updater.update_memories(unique_node_ids=unique_node_ids, unique_node_messages=unique_node_messages,
                                            unique_node_timestamps=unique_node_timestamps)

    def compute_new_node_raw_messages(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, dst_node_embeddings: torch.Tensor,
                                      node_interact_times: np.ndarray, edge_ids: np.ndarray):
        """
        compute new raw messages for nodes in src_node_ids
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param dst_node_embeddings: Tensor, shape (batch_size, node_feat_dim)
        :param node_interact_times: ndarray, shape (batch_size, )
        :param edge_ids: ndarray, shape (batch_size, )
        :return:
        """
        # Tensor, shape (batch_size, memory_dim)
        src_node_memories = self.memory_bank.get_memories(node_ids=src_node_ids)
        # For DyRep, use destination_node_embedding aggregated by graph attention module for message encoding
        if self.model_name == 'DyRep':
            dst_node_memories = dst_node_embeddings
        else:
            dst_node_memories = self.memory_bank.get_memories(node_ids=dst_node_ids)

        # Tensor, shape (batch_size, )
        src_node_delta_times = torch.from_numpy(node_interact_times).float().to(self.device) - \
                               self.memory_bank.node_last_updated_times[torch.from_numpy(src_node_ids)]
        # Tensor, shape (batch_size, time_feat_dim)
        src_node_delta_time_features = self.time_encoder(src_node_delta_times.unsqueeze(dim=1)).reshape(len(src_node_ids), -1)

        # Tensor, shape (batch_size, edge_feat_dim)
        edge_features = self.edge_raw_features[torch.from_numpy(edge_ids)]

        # Tensor, shape (batch_size, message_dim = memory_dim + memory_dim + time_feat_dim + edge_feat_dim)
        new_src_node_raw_messages = torch.cat([src_node_memories, dst_node_memories, src_node_delta_time_features, edge_features], dim=1)

        # dictionary of list, {node_id: list of tuples}, each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        new_node_raw_messages = defaultdict(list)
        # ndarray, shape (num_unique_node_ids, )
        unique_node_ids = np.unique(src_node_ids)

        for i in range(len(src_node_ids)):
            new_node_raw_messages[src_node_ids[i]].append((new_src_node_raw_messages[i], node_interact_times[i]))

        return unique_node_ids, new_node_raw_messages

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        assert self.model_name in ['TGN', 'DyRep'], f'Neighbor sampler is not defined in model {self.model_name}!'
        self.embedding_module.neighbor_sampler = neighbor_sampler
        if self.embedding_module.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.embedding_module.neighbor_sampler.seed is not None
            self.embedding_module.neighbor_sampler.reset_random_state()


# Message-related Modules
class MessageAggregator(nn.Module):

    def __init__(self):
        """
        Message aggregator. Given a batch of node ids and corresponding messages, aggregate messages with the same node id.
        """
        super(MessageAggregator, self).__init__()

    def aggregate_messages(self, node_ids: np.ndarray, node_raw_messages: dict):
        """
        given a list of node ids, and a list of messages of the same length,
        aggregate different messages with the same node id (only keep the last message for each node)
        :param node_ids: ndarray, shape (batch_size, )
        :param node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        unique_node_ids = np.unique(node_ids)
        unique_node_messages, unique_node_timestamps, to_update_node_ids = [], [], []

        for node_id in unique_node_ids:
            if len(node_raw_messages[node_id]) > 0:
                to_update_node_ids.append(node_id)
                unique_node_messages.append(node_raw_messages[node_id][-1][0])
                unique_node_timestamps.append(node_raw_messages[node_id][-1][1])

        # ndarray, shape (num_unique_node_ids, ), array of unique node ids
        to_update_node_ids = np.array(to_update_node_ids)
        # Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        unique_node_messages = torch.stack(unique_node_messages, dim=0) if len(unique_node_messages) > 0 else torch.Tensor([])
        # ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        unique_node_timestamps = np.array(unique_node_timestamps)

        return to_update_node_ids, unique_node_messages, unique_node_timestamps


# Memory-related Modules
class MemoryBank(nn.Module):

    def __init__(self, num_nodes: int, memory_dim: int):
        """
        Memory bank, store node memories, node last updated times and node raw messages.
        :param num_nodes: int, number of nodes
        :param memory_dim: int, dimension of node memories
        """
        super(MemoryBank, self).__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim

        # Parameter, treat memory as parameters so that it is saved and loaded together with the model, shape (num_nodes, memory_dim)
        self.node_memories = nn.Parameter(torch.zeros((self.num_nodes, self.memory_dim)), requires_grad=False)
        # Parameter, last updated time of nodes, shape (num_nodes, )
        self.node_last_updated_times = nn.Parameter(torch.zeros(self.num_nodes), requires_grad=False)
        # dictionary of list, {node_id: list of tuples}, each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        self.node_raw_messages = defaultdict(list)

        self.__init_memory_bank__()

    def __init_memory_bank__(self):
        """
        initialize all the memories and node_last_updated_times to zero vectors, reset the node_raw_messages, which should be called at the start of each epoch
        :return:
        """
        self.node_memories.data.zero_()
        self.node_last_updated_times.data.zero_()
        self.node_raw_messages = defaultdict(list)

    def get_memories(self, node_ids: np.ndarray):
        """
        get memories for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        return self.node_memories[torch.from_numpy(node_ids)]

    def set_memories(self, node_ids: np.ndarray, updated_node_memories: torch.Tensor):
        """
        set memories for nodes in node_ids to updated_node_memories
        :param node_ids: ndarray, shape (batch_size, )
        :param updated_node_memories: Tensor, shape (num_unique_node_ids, memory_dim)
        :return:
        """
        self.node_memories[torch.from_numpy(node_ids)] = updated_node_memories

    def backup_memory_bank(self):
        """
        backup the memory bank, get the copy of current memories, node_last_updated_times and node_raw_messages
        :return:
        """
        cloned_node_raw_messages = {}
        for node_id, node_raw_messages in self.node_raw_messages.items():
            cloned_node_raw_messages[node_id] = [(node_raw_message[0].clone(), node_raw_message[1].copy()) for node_raw_message in node_raw_messages]

        return self.node_memories.data.clone(), self.node_last_updated_times.data.clone(), cloned_node_raw_messages

    def reload_memory_bank(self, backup_memory_bank: tuple):
        """
        reload the memory bank based on backup_memory_bank
        :param backup_memory_bank: tuple (node_memories, node_last_updated_times, node_raw_messages)
        :return:
        """
        self.node_memories.data, self.node_last_updated_times.data = backup_memory_bank[0].clone(), backup_memory_bank[1].clone()

        self.node_raw_messages = defaultdict(list)
        for node_id, node_raw_messages in backup_memory_bank[2].items():
            self.node_raw_messages[node_id] = [(node_raw_message[0].clone(), node_raw_message[1].copy()) for node_raw_message in node_raw_messages]

    def detach_memory_bank(self):
        """
        detach the gradients of node memories and node raw messages
        :return:
        """
        self.node_memories.detach_()

        # Detach all stored messages
        for node_id, node_raw_messages in self.node_raw_messages.items():
            new_node_raw_messages = []
            for node_raw_message in node_raw_messages:
                new_node_raw_messages.append((node_raw_message[0].detach(), node_raw_message[1]))

            self.node_raw_messages[node_id] = new_node_raw_messages

    def store_node_raw_messages(self, node_ids: np.ndarray, new_node_raw_messages: dict):
        """
        store raw messages for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param new_node_raw_messages: dict, dictionary of list, {node_id: list of tuples},
        each tuple is (message, time) with type (Tensor shape (message_dim, ), a scalar)
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id].extend(new_node_raw_messages[node_id])

    def clear_node_raw_messages(self, node_ids: np.ndarray):
        """
        clear raw messages for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :return:
        """
        for node_id in node_ids:
            self.node_raw_messages[node_id] = []

    def get_node_last_updated_times(self, unique_node_ids: np.ndarray):
        """
        get last updated times for nodes in unique_node_ids
        :param unique_node_ids: ndarray, (num_unique_node_ids, )
        :return:
        """
        return self.node_last_updated_times[torch.from_numpy(unique_node_ids)]

    def extra_repr(self):
        """
        set the extra representation of the module, print customized extra information
        :return:
        """
        return 'num_nodes={}, memory_dim={}'.format(self.node_memories.shape[0], self.node_memories.shape[1])


class MemoryUpdater(nn.Module):

    def __init__(self, memory_bank: MemoryBank):
        """
        Memory updater.
        :param memory_bank: MemoryBank
        """
        super(MemoryUpdater, self).__init__()
        self.memory_bank = memory_bank

    def update_memories(self, unique_node_ids: np.ndarray, unique_node_messages: torch.Tensor,
                        unique_node_timestamps: np.ndarray):
        """
        update memories for nodes in unique_node_ids
        :param unique_node_ids: ndarray, shape (num_unique_node_ids, ), array of unique node ids
        :param unique_node_messages: Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        :param unique_node_timestamps: ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        :return:
        """
        # if unique_node_ids is empty, return without updating operations
        if len(unique_node_ids) <= 0:
            return

        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)).all().item(), "Trying to update memory to time in the past!"

        # Tensor, shape (num_unique_node_ids, memory_dim)
        node_memories = self.memory_bank.get_memories(node_ids=unique_node_ids)
        # Tensor, shape (num_unique_node_ids, memory_dim)
        updated_node_memories = self.memory_updater(unique_node_messages, node_memories)
        # update memories for nodes in unique_node_ids
        self.memory_bank.set_memories(node_ids=unique_node_ids, updated_node_memories=updated_node_memories)

        # update last updated times for nodes in unique_node_ids
        self.memory_bank.node_last_updated_times[torch.from_numpy(unique_node_ids)] = torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)

    def get_updated_memories(self, unique_node_ids: np.ndarray, unique_node_messages: torch.Tensor,
                             unique_node_timestamps: np.ndarray):
        """
        get updated memories based on unique_node_ids, unique_node_messages and unique_node_timestamps
        (just for computation), but not update the memories
        :param unique_node_ids: ndarray, shape (num_unique_node_ids, ), array of unique node ids
        :param unique_node_messages: Tensor, shape (num_unique_node_ids, message_dim), aggregated messages for unique nodes
        :param unique_node_timestamps: ndarray, shape (num_unique_node_ids, ), timestamps for unique nodes
        :return:
        """
        # if unique_node_ids is empty, directly return node_memories and node_last_updated_times without updating
        if len(unique_node_ids) <= 0:
            return self.memory_bank.node_memories.data.clone(), self.memory_bank.node_last_updated_times.data.clone()

        assert (self.memory_bank.get_node_last_updated_times(unique_node_ids=unique_node_ids) <=
                torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)).all().item(), "Trying to update memory to time in the past!"

        # Tensor, shape (num_nodes, memory_dim)
        updated_node_memories = self.memory_bank.node_memories.data.clone()
        updated_node_memories[torch.from_numpy(unique_node_ids)] = self.memory_updater(unique_node_messages,
                                                                                       updated_node_memories[torch.from_numpy(unique_node_ids)])

        # Tensor, shape (num_nodes, )
        updated_node_last_updated_times = self.memory_bank.node_last_updated_times.data.clone()
        updated_node_last_updated_times[torch.from_numpy(unique_node_ids)] = torch.from_numpy(unique_node_timestamps).float().to(unique_node_messages.device)

        return updated_node_memories, updated_node_last_updated_times


class GRUMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        GRU-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(GRUMemoryUpdater, self).__init__(memory_bank)

        self.memory_updater = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)


class RNNMemoryUpdater(MemoryUpdater):

    def __init__(self, memory_bank: MemoryBank, message_dim: int, memory_dim: int):
        """
        RNN-based memory updater.
        :param memory_bank: MemoryBank
        :param message_dim: int, dimension of node messages
        :param memory_dim: int, dimension of node memories
        """
        super(RNNMemoryUpdater, self).__init__(memory_bank)

        self.memory_updater = nn.RNNCell(input_size=message_dim, hidden_size=memory_dim)


# Embedding-related Modules
class TimeProjectionEmbedding(nn.Module):

    def __init__(self, memory_dim: int, dropout: float):
        """
        Time projection embedding module.
        :param memory_dim: int, dimension of node memories
        :param dropout: float, dropout rate
        """
        super(TimeProjectionEmbedding, self).__init__()

        self.memory_dim = memory_dim
        self.dropout = nn.Dropout(dropout)

        self.linear_layer = nn.Linear(1, self.memory_dim)

    def compute_node_temporal_embeddings(self, node_memories: torch.Tensor, node_ids: np.ndarray, node_time_intervals: torch.Tensor):
        """
        compute node temporal embeddings using the embedding projection operation in JODIE
        :param node_memories: Tensor, shape (num_nodes, memory_dim)
        :param node_ids: ndarray, shape (batch_size, )
        :param node_time_intervals: Tensor, shape (batch_size, )
        :return:
        """
        # Tensor, shape (batch_size, memory_dim)
        source_embeddings = self.dropout(node_memories[torch.from_numpy(node_ids)] * (1 + self.linear_layer(node_time_intervals.unsqueeze(dim=1))))

        return source_embeddings


class GraphAttentionEmbedding(nn.Module):

    def __init__(self, node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor, neighbor_sampler: NeighborSampler,
                 time_encoder: TimeEncoder, node_feat_dim: int, edge_feat_dim: int, time_feat_dim: int,
                 num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1):
        """
        Graph attention embedding module.
        :param node_raw_features: Tensor, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: Tensor, shape (num_edges + 1, node_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_encoder: TimeEncoder
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim:  int, dimension of time features (encodings)
        :param num_layers: int, number of temporal graph convolution layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(GraphAttentionEmbedding, self).__init__()

        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.neighbor_sampler = neighbor_sampler
        self.time_encoder = time_encoder
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.temporal_conv_layers = nn.ModuleList([MultiHeadAttention(node_feat_dim=self.node_feat_dim,
                                                                      edge_feat_dim=self.edge_feat_dim,
                                                                      time_feat_dim=self.time_feat_dim,
                                                                      num_heads=self.num_heads,
                                                                      dropout=self.dropout) for _ in range(num_layers)])
        # follow the TGN paper, use merge layer to combine 1) the attention results, and 2) node raw feature + node memory
        self.merge_layers = nn.ModuleList([MergeLayer(input_dim1=self.node_feat_dim + self.time_feat_dim, input_dim2=self.node_feat_dim,
                                                      hidden_dim=self.node_feat_dim, output_dim=self.node_feat_dim) for _ in range(num_layers)])

    def compute_node_temporal_embeddings(self, node_memories: torch.Tensor, node_ids: np.ndarray, node_interact_times: np.ndarray,
                                         current_layer_num: int, num_neighbors: int = 20):
        """
        given memory, node ids node_ids, and the corresponding time node_interact_times,
        return the temporal embeddings after convolution at the current_layer_num
        :param node_memories: Tensor, shape (num_nodes, memory_dim)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param current_layer_num: int, current layer number
        :param num_neighbors: int, number of neighbors to sample for each node
        """

        assert (current_layer_num >= 0)
        device = self.node_raw_features.device

        # query (source) node always has the start time with time interval == 0
        # shape (batch_size, 1, time_feat_dim)
        node_time_features = self.time_encoder(timestamps=torch.zeros(node_interact_times.shape).unsqueeze(dim=1).to(device))
        # shape (batch_size, node_feat_dim)
        # add memory and node raw features to get node features
        # note that when using getting values of the ids from Tensor, convert the ndarray to tensor to avoid wrong retrieval
        node_features = node_memories[torch.from_numpy(node_ids)] + self.node_raw_features[torch.from_numpy(node_ids)]

        if current_layer_num == 0:
            return node_features
        else:
            # get source node representations by aggregating embeddings from the previous (curr_layers - 1)-th layer
            # Tensor, shape (batch_size, node_feat_dim)
            node_conv_features = self.compute_node_temporal_embeddings(node_memories=node_memories,
                                                                       node_ids=node_ids,
                                                                       node_interact_times=node_interact_times,
                                                                       current_layer_num=current_layer_num - 1,
                                                                       num_neighbors=num_neighbors)

            # get temporal neighbors, including neighbor ids, edge ids and time information
            # neighbor_node_ids ndarray, shape (batch_size, num_neighbors)
            # neighbor_edge_ids ndarray, shape (batch_size, num_neighbors)
            # neighbor_times ndarray, shape (batch_size, num_neighbors)
            neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
                self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                               node_interact_times=node_interact_times,
                                                               num_neighbors=num_neighbors)

            # get neighbor features from previous layers
            # shape (batch_size * num_neighbors, node_feat_dim)
            neighbor_node_conv_features = self.compute_node_temporal_embeddings(node_memories=node_memories,
                                                                                node_ids=neighbor_node_ids.flatten(),
                                                                                node_interact_times=neighbor_times.flatten(),
                                                                                current_layer_num=current_layer_num - 1,
                                                                                num_neighbors=num_neighbors)

            # shape (batch_size, num_neighbors, node_feat_dim)
            neighbor_node_conv_features = neighbor_node_conv_features.reshape(node_ids.shape[0], num_neighbors, self.node_feat_dim)

            # compute time interval between current time and historical interaction time
            # adarray, shape (batch_size, num_neighbors)
            neighbor_delta_times = node_interact_times[:, np.newaxis] - neighbor_times

            # shape (batch_size, num_neighbors, time_feat_dim)
            neighbor_time_features = self.time_encoder(timestamps=torch.from_numpy(neighbor_delta_times).float().to(device))

            # get edge features, shape (batch_size, num_neighbors, edge_feat_dim)
            neighbor_edge_features = self.edge_raw_features[torch.from_numpy(neighbor_edge_ids)]
            # temporal graph convolution
            # Tensor, output shape (batch_size, node_feat_dim + time_feat_dim)
            output, _ = self.temporal_conv_layers[current_layer_num - 1](node_features=node_conv_features,
                                                                         node_time_features=node_time_features,
                                                                         neighbor_node_features=neighbor_node_conv_features,
                                                                         neighbor_node_time_features=neighbor_time_features,
                                                                         neighbor_node_edge_features=neighbor_edge_features,
                                                                         neighbor_masks=neighbor_node_ids)

            # Tensor, output shape (batch_size, node_feat_dim)
            # follow the TGN paper, use merge layer to combine 1) the attention results, and 2) node raw feature + node memory
            output = self.merge_layers[current_layer_num - 1](input_1=output, input_2=node_features)

            return output


def compute_src_dst_node_time_shifts(src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
    """
    compute the mean and standard deviation of time shifts
    :param src_node_ids: ndarray, shape (*, )
    :param dst_node_ids:: ndarray, shape (*, )
    :param node_interact_times: ndarray, shape (*, )
    :return:
    """
    src_node_last_timestamps = dict()
    dst_node_last_timestamps = dict()
    src_node_all_time_shifts = []
    dst_node_all_time_shifts = []
    for k in range(len(src_node_ids)):
        src_node_id = src_node_ids[k]
        dst_node_id = dst_node_ids[k]
        node_interact_time = node_interact_times[k]
        if src_node_id not in src_node_last_timestamps.keys():
            src_node_last_timestamps[src_node_id] = 0
        if dst_node_id not in dst_node_last_timestamps.keys():
            dst_node_last_timestamps[dst_node_id] = 0
        src_node_all_time_shifts.append(node_interact_time - src_node_last_timestamps[src_node_id])
        dst_node_all_time_shifts.append(node_interact_time - dst_node_last_timestamps[dst_node_id])
        src_node_last_timestamps[src_node_id] = node_interact_time
        dst_node_last_timestamps[dst_node_id] = node_interact_time
    assert len(src_node_all_time_shifts) == len(src_node_ids)
    assert len(dst_node_all_time_shifts) == len(dst_node_ids)
    src_node_mean_time_shift = np.mean(src_node_all_time_shifts)
    src_node_std_time_shift = np.std(src_node_all_time_shifts)
    dst_node_mean_time_shift_dst = np.mean(dst_node_all_time_shifts)
    dst_node_std_time_shift = np.std(dst_node_all_time_shifts)

    return src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift
