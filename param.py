
ReviewModelParam = {
    "max_length": 60,
    "embedding_dim": 50,
    "vocab_length" : data_loader.weight_matrix.shape[0],
    "output_dim" : 2,
    "batch_size" : 128,
    
    "first_dropout" : 0.5,
    "conv_input_channel": None,
    "conv_output_channel" : 200,
    "conv_padding" : 2,
    "conv1_kernel_size" : 4,
    "conv2_kernel_size" : 5,
    "maxpool_kernel_size" : 2,
    "second_dropout" : 0.3,
    "rnn_input_size": None,
    "rnn_hidden_size" : 100,
    "rnn_num_layers" : 1,
    "first_dense_in":None,
    "first_dense_out" : 400,
    "third_dropout" : 0.15,
    "second_dense_in" : None,
    "second_dense_out" : None
    }

ReviewModelParam["conv1_input_channel"] = ReviewModelParam["embedding_dim"]
ReviewModelParam["rnn_input_size"] = ReviewModelParam["conv_output_channel"]*2
ReviewModelParam["first_dense_in"] = ReviewModelParam["max_length"]//2 * ReviewModelParam["rnn_hidden_size"]
ReviewModelParam["second_dense_in"] = ReviewModelParam["first_dense_out"]
ReviewModelParam["second_dense_out"] = ReviewModelParam["output_dim"]  ##depends on loss function




SST2_DATASET_PARAMETERS = {
    "cell_one_parameter_dict" : {
        "sent_length": 19,
        "conv_kernel_size": (7, 1),
        "conv_input_channels": 1,
        "conv_output_channels": 6,
        "conv_stride": (1, 1),
        "k_max_number": 10,
        "folding_kernel_size": (1, 2),
        "folding_stride": (1, 2)
    },
    "cell_two_parameter_dict" : {
        "sent_length": None,
        "conv_kernel_size": (5, 1),
        "conv_input_channels": 6,
        "conv_output_channels": 14,
        "conv_stride": (1, 1),
        "k_max_number": 4,
        "folding_kernel_size": (1, 2),
        "folding_stride": (1, 2)
    },
    "dropout_rate": 0.5,
    "embedding_dim": 50,
    "vocab_length": data_loader.weight_matrix,
    "output_dim": 2
}
SST2_DATASET_PARAMETERS["cell_two_parameter_dict"]["sent_length"] = SST2_DATASET_PARAMETERS["cell_one_parameter_dict"]["k_max_number"]