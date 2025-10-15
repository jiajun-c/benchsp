#include <iostream>

using namespace std;
int main() {
  int count = 127;
  int num = max(25 - __builtin_clz(count), 0);
  printf("%d\n", num);
  return 0;
}