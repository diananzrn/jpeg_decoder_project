import java.util.*;

public class JPEGDecoder {

    static int SIZE = 8;

    public static void main(String[] args) {

        // Step 0: Input
        int dc = -26;

        int[][] rlc = {
            {0, -3}, {1, -3}, {0, -2}, {0, -6}, {0, 2}, {0, -4}, {0, 1},
            {0, -3}, {0, 1}, {0, 1}, {0, 5}, {0, 1}, {0, 2}, {0, -1},
            {0, 2}, {5, -1}, {0, -1}, {0, 0}
        };

        // Step 1: Decode RLC
        int[] vector = decodeRLC(dc, rlc);

        // Step 2: Zig-zag → 8x8
        double[][] block = zigzagToBlock(vector);

        // Step 3: Quantization table
        int[][] Q = {
            {16,11,10,16,24,40,51,61},
            {12,12,14,19,26,58,60,55},
            {14,13,16,24,40,57,69,56},
            {14,17,22,29,51,87,80,62},
            {18,22,37,56,68,109,103,77},
            {24,35,55,64,81,104,113,92},
            {49,64,78,87,103,121,120,101},
            {72,92,95,98,112,100,103,99}
        };

        double[][] dequantized = new double[SIZE][SIZE];
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                dequantized[i][j] = block[i][j] * Q[i][j];
            }
        }

        // Step 4: IDCT
        double[][] step4 = idct2D(dequantized);

        System.out.println("\nStep 4: IDCT Result (before adding 128)\n");
        printMatrix(step4);

        // Step 5: Add 128
        double[][] step5 = new double[SIZE][SIZE];
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                step5[i][j] = step4[i][j] + 128;
            }
        }

        System.out.println("\nStep 5: Final Image (after adding 128)\n");
        printMatrix(step5);
    }

    // ✅ Correct RLC decoding
    static int[] decodeRLC(int dc, int[][] rlc) {
        int[] result = new int[64];
        result[0] = dc;

        int index = 1;

        for (int[] pair : rlc) {
            int run = pair[0];
            int val = pair[1];

            if (run == 0 && val == 0) break;

            index += run;
            if (index < 64) {
                result[index] = val;
                index++;
            }
        }

        return result;
    }

    // ✅ Correct zig-zag mapping
    static double[][] zigzagToBlock(int[] vector) {
        int[][] zigzag = {
            {0,0},{0,1},{1,0},{2,0},{1,1},{0,2},{0,3},{1,2},
            {2,1},{3,0},{4,0},{3,1},{2,2},{1,3},{0,4},{0,5},
            {1,4},{2,3},{3,2},{4,1},{5,0},{6,0},{5,1},{4,2},
            {3,3},{2,4},{1,5},{0,6},{0,7},{1,6},{2,5},{3,4},
            {4,3},{5,2},{6,1},{7,0},{7,1},{6,2},{5,3},{4,4},
            {3,5},{2,6},{1,7},{2,7},{3,6},{4,5},{5,4},{6,3},
            {7,2},{7,3},{6,4},{5,5},{4,6},{3,7},{4,7},{5,6},
            {6,5},{7,4},{7,5},{6,6},{5,7},{6,7},{7,6},{7,7}
        };

        double[][] block = new double[SIZE][SIZE];

        for (int i = 0; i < 64; i++) {
            int r = zigzag[i][0];
            int c = zigzag[i][1];
            block[r][c] = vector[i];
        }

        return block;
    }

    // ✅ Correct IDCT
    static double[][] idct2D(double[][] F) {
        double[][] f = new double[SIZE][SIZE];

        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {

                double sum = 0;

                for (int u = 0; u < SIZE; u++) {
                    for (int v = 0; v < SIZE; v++) {

                        double cu = (u == 0) ? 1.0 / Math.sqrt(2) : 1.0;
                        double cv = (v == 0) ? 1.0 / Math.sqrt(2) : 1.0;

                        sum += cu * cv * F[u][v] *
                                Math.cos((2 * i + 1) * u * Math.PI / 16) *
                                Math.cos((2 * j + 1) * v * Math.PI / 16);
                    }
                }

                f[i][j] = sum / 4.0;
            }
        }

        return f;
    }

    // ✅ Pretty printing
    static void printMatrix(double[][] matrix) {
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                System.out.printf("%4d", (int)Math.round(matrix[i][j]));
            }
            System.out.println();
        }
    }
}