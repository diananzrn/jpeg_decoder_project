import numpy as np
import math
import matplotlib.pyplot as plt

# Step 0: Input RLC data
dc = -26
rlc = [
    (0, -3), (1, -3), (0, -2), (0, -6), (0, 2), (0, -4), (0, 1),
    (0, -3), (0, 1), (0, 1), (0, 5), (0, 1), (0, 2), (0, -1),
    (0, 2), (5, -1), (0, -1), (0, 0)
]

# Step 1: Decode RLC → 64 vector
def decode_rlc(dc, rlc):
    result = [dc]
    
    for run, val in rlc:
        result.extend([0] * run)
        result.append(val)
    
    # pad to 64
    while len(result) < 64:
        result.append(0)
    
    return result[:64]

vector = decode_rlc(dc, rlc)

# Step 2: Zig-zag order
zigzag_indices = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
]

def zigzag_to_block(vector):
    block = np.zeros((8,8))
    for i, (r, c) in enumerate(zigzag_indices):
        block[r][c] = vector[i]
    return block

block = zigzag_to_block(vector)

# Step 3: Luminance Quantization Table
Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

dequantized = block * Q

# Step 4: IDCT
def C(x):
    return 1 / math.sqrt(2) if x == 0 else 1

def idct_2d(F):
    f = np.zeros((8,8))
    
    for i in range(8):
        for j in range(8):
            sum_val = 0
            for u in range(8):
                for v in range(8):
                    sum_val += (
                        C(u) * C(v) *
                        math.cos((2*i+1)*u*math.pi/16) *
                        math.cos((2*j+1)*v*math.pi/16) *
                        F[u][v]
                    )
            f[i][j] = sum_val / 4
    return f

reconstructed = idct_2d(dequantized)

# Step 5: Add 128
final_image = reconstructed + 128

# Round values
final_image = np.round(final_image).astype(int)

# Output
print("Reconstructed 8x8 Image:")
print(final_image)

plt.imshow(final_image, cmap='gray')
plt.title("Decoded Image")
plt.colorbar()
plt.show()