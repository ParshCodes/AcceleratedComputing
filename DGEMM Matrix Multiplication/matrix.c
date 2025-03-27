#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define ORDER 1000

int main(int argc, char *argv[]){
  int N, P, M;
  int i,j,k;
  double *A, *B, *C;
  double start, end,tmp;

  N = ORDER;
  P= ORDER;
  M= ORDER;

  A = (double *)malloc(N * P * sizeof(double));
  B = (double *)malloc(P * M * sizeof(double));
  C = (double *)malloc(N * M * sizeof(double));

  for(i=0;i<N;i++){
    for(j=0;j<P;j++){
      *(A + (i*P+j)) = rand()%100;
    }
  }

  for(i=0;i<P;i++){
    for(j=0;j<M;j++){
      *( B+(i*M+j)) = rand()%100;
    }
  }

  for(i=0;i<N;i++){
    for(j=0;j<M;j++){
     *(C+(i*M+j)) = 0.0;
    }
  }

  //The Searial Executn

  start = omp_get_wtime();
  for(i=0;i<N;i++){
    for(j=0;j<M;j++){
      double tmp = 0.0;
	for(k=0;k<P;k++){
	  tmp += *(A +(i*P+k)) * *(B+(k*M+j));
		   }
	    *(C + (i * M +j)) = tmp;
	}
    }
    end = omp_get_wtime();
    printf("1 %f \n",end -start);

    for(int n =1;n<8;n++){
      int threads = 1<<n;

      for(i=0;i<N;i++){
	for(j=0;j<M;j++){
	  *(C + (i * M +j))=0.0;
      }
    }
    omp_set_num_threads(threads);
    start = omp_get_wtime();
    //Parellel Executn
#pragma omp parallel for private(i,j,k,tmp) shared (A,B,C)
    for ( i=0;i<N;i++){
      for(j=0;j<M;j++){
	tmp = 0.0;
	  for(k=0;k<P;k++){
	    tmp += *(A +(i * P +k))* *(B +(k*M +j));
	  }
	  *(C +(i*M+j)) = tmp;
      }
    }

    end = omp_get_wtime();
    printf("%d %f \n",threads, end -start);
    }
    free(A);
    free(B);
    free(C);
    return 0;
    
      
}
