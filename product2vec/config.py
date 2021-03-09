
config = {
    'streamer': {
        'batch_size': 10_000,
        'shuffle': True,
        'sample': 0.25,        # very restrictive
        'verbose': True,
        'basket': 'basket',
        'product': 'product'
    },
    'trainer': {
        'embedding_size': 300,
        'epochs': 10,
        'early_stop': 0.01,     # very restrictive
        'learning_rate': 0.001,
        'test_share': 0.2,
        'n_negative': 5,
        'verbose': True
    },
    'mapper': {
        'perplexity': 30.0,
        'learning_rate': 200.0,
        'random_state': 1825
    }
}