#source: https://github.com/alisaaalehi/convolution_as_multiplication/blob/master/Convolution_as_multiplication.ipynb
import numpy as np
from scipy.linalg import toeplitz


def matrix_to_vector(input):
    """
    Converts the input matrix to a vector by stacking the rows in a specific way explained here
    
    Arg:
    input -- a numpy matrix
    
    Returns:
    ouput_vector -- a column vector with size input.shape[0]*input.shape[1]
    """
    input_h, input_w = input.shape
    output_vector = np.zeros(input_h*input_w, dtype=input.dtype)
    # flip the input matrix up-down because last row should go first
    input = np.flipud(input) 
    for i,row in enumerate(input):
        st = i*input_w
        nd = st + input_w
        output_vector[st:nd] = row   
    return output_vector


def vector_to_matrix(input, output_shape):
    """
    Reshapes the output of the maxtrix multiplication to the shape "output_shape"
    
    Arg:
    input -- a numpy vector
    
    Returns:
    output -- numpy matrix with shape "output_shape"
    """
    output_h, output_w = output_shape
    output = np.zeros(output_shape, dtype=input.dtype)
    for i in range(output_h):
        st = i*output_w
        nd = st + output_w
        output[i, :] = input[st:nd]
    # flip the output matrix up-down to get correct result
    output=np.flipud(output)
    return output


def convolution_as_maultiplication(I, F, print_ir=False):
    """
    Performs 2D convolution between input I and filter F by converting the F to a toeplitz matrix and multiply it
      with vectorizes version of I
      By : AliSaaalehi@gmail.com
      
    Arg:
    I -- 2D numpy matrix
    F -- numpy 2D matrix
    print_ir -- if True, all intermediate resutls will be printed after each step of the algorithms
    
    Returns: 
    output -- 2D numpy matrix, result of convolving I with F
    """
    # number of columns and rows of the input 
    I_row_num, I_col_num = I.shape 

    # number of columns and rows of the filter
    F_row_num, F_col_num = F.shape

    #  calculate the output dimensions
    output_row_num = I_row_num + F_row_num - 1
    output_col_num = I_col_num + F_col_num - 1
    if print_ir: print('output dimension:', output_row_num, output_col_num)

    # zero pad the filter
    F_zero_padded = np.pad(F, ((output_row_num - F_row_num, 0),
                               (0, output_col_num - F_col_num)),
                            'constant', constant_values=0)
    if print_ir: print('F_zero_padded: ', F_zero_padded)

    # use each row of the zero-padded F to creat a toeplitz matrix. 
    #  Number of columns in this matrices are same as numbe of columns of input signal
    toeplitz_list = []
    for i in range(F_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
        c = F_zero_padded[i, :] # i th row of the F 
        r = np.r_[c[0], np.zeros(I_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                            # the result is wrong
        toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
        toeplitz_list.append(toeplitz_m)
        if print_ir: print('F '+ str(i)+'\n', toeplitz_m)

        # doubly blocked toeplitz indices: 
    #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, F_zero_padded.shape[0]+1)
    r = np.r_[c[0], np.zeros(I_row_num-1, dtype=int)]
    doubly_indices = toeplitz(c, r)
    if print_ir: print('doubly indices \n', doubly_indices)

    ## creat doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

    if print_ir: print('doubly_blocked: ', doubly_blocked)

    # convert I to a vector
    vectorized_I = matrix_to_vector(I)
    if print_ir: print('vectorized_I: ', vectorized_I)
    
    # get result of the convolution by matrix mupltiplication
    result_vector = np.matmul(doubly_blocked, vectorized_I)
    if print_ir: print('result_vector: ', result_vector)

    # reshape the raw rsult to desired matrix form
    out_shape = [output_row_num, output_col_num]
    output = vector_to_matrix(result_vector, out_shape)
    if print_ir: print('Result of implemented method: \n', output)
    
    return output

def kernel_to_matrix(F, I_row_num, I_col_num):
    """
    Arg:
    F -- numpy 2D matrix - kernel of the blur
    I_row_num - number of rows in signal
    I_col_num - number of cols in signal
    
    Returns: 
    output -- 2D numpy matrix, which operates on a signal I of size 
    """

    # number of columns and rows of the filter
    F_row_num, F_col_num = F.shape

    #  calculate the output dimensions
    output_row_num = I_row_num + F_row_num - 1
    output_col_num = I_col_num + F_col_num - 1
    #if print_ir: print('output dimension:', output_row_num, output_col_num)

    # zero pad the filter
    F_zero_padded = np.pad(F, ((output_row_num - F_row_num, 0),
                               (0, output_col_num - F_col_num)),
                            'constant', constant_values=0)
    #if print_ir: print('F_zero_padded: ', F_zero_padded)

    # use each row of the zero-padded F to creat a toeplitz matrix. 
    #  Number of columns in this matrices are same as numbe of columns of input signal
    toeplitz_list = []
    for i in range(F_zero_padded.shape[0]-1, -1, -1): # iterate from last row to the first row
        c = F_zero_padded[i, :] # i th row of the F 
        r = np.r_[c[0], np.zeros(I_col_num-1)] # first row for the toeplitz fuction should be defined otherwise
                                                            # the result is wrong
        toeplitz_m = toeplitz(c,r) # this function is in scipy.linalg library
        toeplitz_list.append(toeplitz_m)
        #if print_ir: print('F '+ str(i)+'\n', toeplitz_m)

        # doubly blocked toeplitz indices: 
    #  this matrix defines which toeplitz matrix from toeplitz_list goes to which part of the doubly blocked
    c = range(1, F_zero_padded.shape[0]+1)
    r = np.r_[c[0], np.zeros(I_row_num-1, dtype=int)]
    doubly_indices = toeplitz(c, r)
    #if print_ir: print('doubly indices \n', doubly_indices)

    ## creat doubly blocked matrix with zero values
    toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
    h = toeplitz_shape[0]*doubly_indices.shape[0]
    w = toeplitz_shape[1]*doubly_indices.shape[1]
    doubly_blocked_shape = [h, w]
    doubly_blocked = np.zeros(doubly_blocked_shape)

    # tile toeplitz matrices for each row in the doubly blocked matrix
    b_h, b_w = toeplitz_shape # hight and withs of each block
    for i in range(doubly_indices.shape[0]):
        for j in range(doubly_indices.shape[1]):
            start_i = i * b_h
            start_j = j * b_w
            end_i = start_i + b_h
            end_j = start_j + b_w
            doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i,j]-1]

    return doubly_blocked

def get_custom_kernel(type = "gauss", dim = 64):
    kernel = 0
    if type == "14641":
        kernel = np.array([[1, 4, 6, 4, 1]])
        kernel = np.matmul(kernel.transpose(1, 0), kernel) / 256.0
    elif type == "uniform":
        kernel = np.array([[1, 1, 1, 1, 1]])
        kernel = np.matmul(kernel.transpose(1, 0), kernel) / 25.0
    elif type == "gauss":
        #sigma 10
        kernel = np.array([[0.03920520445985253,0.03979771524812676,0.0399972021259645,0.03979771524812676,0.03920520445985253],
                           [0.03979771524812676,0.04039918069022969,0.04060168242614218,0.04039918069022969,0.03979771524812676],
                           [0.0399972021259645,0.04060168242614218,0.04080519920622999,0.04060168242614218,0.0399972021259645],
                           [0.03979771524812676,0.04039918069022969,0.04060168242614218,0.04039918069022969,0.03979771524812676],
                           [0.03920520445985253,0.03979771524812676,0.0399972021259645,0.03979771524812676,0.03920520445985253]])
    H = kernel_to_matrix(kernel, dim, dim)
    H = H[[row*(dim+4)+col for row in range(2, dim+2) for col in range(2,dim+2)], :]
    return H