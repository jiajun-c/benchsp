#include<iostream>
int main()
{
    //定义一个cuda的设备属性结构体
    int row = 2048*1024, avg_nnz_in_row = 4;
    int col = row+avg_nnz_in_row;
    int nnz = row * avg_nnz_in_row;
    printf("%%%MatrixMarket matrix coordinate pattern general\n");
    printf("%d %d %d\n", row, col, nnz);
    for (int i = 0; i < row; i++) {
        for (int j = i; j < i + avg_nnz_in_row; j++) {
            printf("%d %d\n", i+1, j+1);
        }
    }
}