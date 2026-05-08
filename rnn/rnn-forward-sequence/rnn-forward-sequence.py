import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray, 
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    
    batch_size, T, input_dim = X.shape
    hidden_dim = h_0.shape[1]
    
    hidden_states = np.zeros((batch_size, T, hidden_dim))

    h_t = h_0
    
    for t in range(T):
        x_t = X[:, t, :]
        
        h_t = np.tanh(
            x_t @ W_xh.T + 
            h_t @ W_hh.T + 
            b_h
        )
        
        hidden_states[:, t, :] = h_t
        
    return hidden_states, h_t