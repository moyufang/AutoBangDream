#include <stdio.h>

int main(int argc, char**argv){
  FILE *f = fopen("/data/local/tmp/test_jni.txt", "w");
  fprintf(f, "no\n");

  printf("abcdefg");

  return 0;
}