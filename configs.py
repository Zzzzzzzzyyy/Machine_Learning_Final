def get_configs():
    LSTM_configs_96 = {
        'input_length': 96,
        'output_length': 96,
        'hidden_size': 256,
        'num_layers': 3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 200
    }

    LSTM_configs_240 = {
        'input_length': 96,
        'output_length': 240,
        'hidden_size': 1024,
        'num_layers': 3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 200
    }

    TransFormer_configs_96 = {
        'input_length': 96,
        'output_length': 96,
        'hidden_size': 256,
        'num_layers': 3,
        'learning_rate': 0.0005,
        'batch_size': 32,
        'num_head': 4,
        'num_epochs': 100
    }

    TransFormer_configs_240 = {
        'input_length': 96,
        'output_length': 240,
        'hidden_size': 256,
        'num_layers': 3,
        'learning_rate': 0.0005,
        'batch_size': 32,
        'num_head': 4,
        'num_epochs': 100
    }

    Improved_model_configs_96 = {
        'input_length': 96,
        'output_length': 96,
        'hidden_size': 256,
        'num_layers': 3,
        'learning_rate': 0.0005,
        'batch_size': 128,
        'num_epochs': 100
    }

    Improved_model_configs_240 = {
        'input_length': 96,
        'output_length': 240,
        'hidden_size': 1024,
        'num_layers': 3,
        'learning_rate': 0.0005,
        'batch_size': 128,
        'num_epochs': 100
    }
    return Improved_model_configs_240
