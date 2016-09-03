import pandas

metrics = pandas.read_csv('output/metrics.csv')
partitions = pandas.read_csv('output/metrics-partitions.csv')

metrics.set_index(['file'], inplace=True)

averages = partitions.groupby(['file']).mean()
averages.columns = ['partition', 'network_permanence (avg)', 'Q (avg)',
                     'NQ (avg)', 'Qds (avg)', 'intraEdges (avg)',
                     'interEdges (avg)', 'intraDensity (avg)',
                     'modularity degree (avg)', 'conductance (avg)',
                     'expansion (avg)', 'contraction (avg)', 'fitness (avg)',
                     'QovL (avg)']

cols = ['num_partitions', 'num_iterations', 'prediction_model_cut_off',
        'one_shot_alpha', 'restream_batches', 'use_virtual_nodes',
        'virtual_node_weight', 'virtual_edge_weight', 'edges_cut', 'waste',
        'cut_ratio', 'communication_volume', 'network_permanence',
        'network_permanence (avg)', 'Q', 'Q (avg)', 'NQ', 'NQ (avg)',
        'Qds', 'Qds (avg)', 'intraEdges', 'intraEdges (avg)', 'interEdges',
        'interEdges (avg)', 'intraDensity', 'intraDensity (avg)',
        'modularity degree', 'modularity degree (avg)', 'conductance',
        'conductance (avg)', 'expansion', 'expansion (avg)', 'contraction',
        'contraction (avg)', 'fitness', 'fitness (avg)', 'QovL', 'QovL (avg)']

metrics.join(averages)[cols].to_csv('output/metrics-complete.csv')

