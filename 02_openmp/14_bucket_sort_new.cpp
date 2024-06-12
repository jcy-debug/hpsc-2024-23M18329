#include <cstdio>
#include <cstdlib>
#include <vector>
#include<omp.h>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
  for (int i=0; i<n; i++)
    bucket[key[i]]++;


  std::vector<int> offset(range,0), b(range, 0);
  offset[0] = 0;
  for (int i = 1; i < range; i++) {
      b[i] = bucket[i - 1];  
  }

  for (int step = 1; step < range; step <<= 1) {
#pragma omp parallel for
      for (int i = 0; i < range; i++) {
          if (i >= step) offset[i] = b[i] + offset[i - step];
          else offset[i] = b[i];
      }
#pragma omp parallel for
      for (int i = 0; i < range; i++) {
          b[i] = offset[i];  
      }
  }

 
#pragma omp parallel for
  for (int i = 0; i < range; i++) {
      int j = offset[i];
      for (;  bucket[i]>0; bucket[i]--) {
          key[j++] = i;
      }
  }


  for (int i = 0; i < n; i++) {
      printf("%d ", key[i]);
  }
  printf("\n");

  return 0;
}
