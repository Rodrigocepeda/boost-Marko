def get_model_corr(x_train, _h):
    
    model = Sequential()
    model.add(Input(shape = (_h, x_train.shape[2], x_train.shape[3], 1)))
    model.add(ConvLSTM2D(...
    ...
    ...           
    ...                     
    ...
    model.add(Reshape(target_shape = (x_train.shape[2], \ 
                                      x_train.shape[3])))
    model.compile(optimizer = \ 
                  tf.keras.optimizers.Adam(\ 
                        learning_rate = 0.001), loss='mse')
    return model