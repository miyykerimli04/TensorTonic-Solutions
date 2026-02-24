import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    batch_size,sequence_length,n_features=X.shape
    n_hidden=W_hh.shape[0]

    h_next=h_0.copy()
    h=np.zeros((batch_size,sequence_length,n_hidden))

    for t in range (sequence_length):
        x_t=X[:,t,:]

        h_next=np.tanh(
            np.dot(x_t,W_xh.T)+np.dot(h_next,W_hh.T) + b_h
        )
        h[:,t,:]=h_next
        
    return h,h[:,-1,:]
    
