#include "Matrix_Multiplier.h"
#include <iostream>
#include <ctime>
using namespace std;

#define minNum 1000
#define maxNum 2000

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cerr << "Usage: <file> <Test Time>" << endl;
        return 1;
    }
    int testSample = atoi(argv[1]);
    srand(time(0));
    for (int i = 0; i < testSample; i++)
    {
        cout << ">>>>>>>>>>>>>>>>> test No. " << i + 1 << " >>>>>>>>>>>>>>>>>" << endl;
        int A_Row = rand() % (maxNum - minNum + 1) + minNum;
        int A_Col = rand() % (maxNum - minNum + 1) + minNum;
        int B_Col = rand() % (maxNum - minNum + 1) + minNum;

        cout << "A_Row: " << A_Row << endl;
        cout << "A_Col: " << A_Col << endl;
        cout << "B_Col: " << B_Col << endl;

        Matrix_Multiplier multiplication(A_Row, A_Col, B_Col);
        multiplication.Matrix_Multiply();
    }
    return 0;
}