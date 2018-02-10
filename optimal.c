/**
 * Versão otimizada utiliza um buffer aligned para armazenar ambas as matrizes de maneira contigua na memória
 * a segunda matrix é lida de maneira transposta, ou seja ao inves de ser armazenada linhaxlinha, como a primeira,
 * é armazenada colunaxcoluna, para aproveitar-se da localidade espacial ao multiplicar as linhas pelas colunas
 * dentro da função multMatrix e otimizar o acesso a cache.
 * 
 * O tipo utilizado nas matrizes é float
 *	compilar com:  gcc -03 -msse3 arq.c -o arq
**/
#include <stdio.h>
#include <stdlib.h>
#include <pmmintrin.h>
#include <x86intrin.h>
#include <math.h>
void multMatrix(void * buffer);
void printMatrix(void * buffer);
void readMatrix(void * buffer);
float hsum_ps_sse1(__m128 v);  //This function is copied from: https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-float-vector-sum-on-x86

int n;
int dif;

int main(int argc, char * argv[]){

   // if(argc > 1)
		//n = atoi(argv[1]);
	//else
		if(scanf("%d", &n)){}
        else printf("fail to read integer");
		
    
    if(n>4 && n%4 != 0){     // Essa aplicação de SSE usa operações paralelas de 4 operandos, então o tamanho da matriz precisa ser multiplo de 4
        dif = ceil(n/4);
		dif *= 4;
		dif = n - dif;
    }
	else if(n < 4){
		dif = n - 4;
		dif *= -1;
	}
	else{
        dif = 0;
    }
	
	printf("\nmeu N: %d", n);
	printf("\nmeu dif: %d\n", dif);
    void * buffer = _mm_malloc(sizeof(float)*(n+dif)*(n+dif)*5,16);

    readMatrix(buffer);
    multMatrix(buffer);

    free(buffer);

	return 0;
}

void readMatrix(void * buffer){
	
    int i = 0,j=0;
    float *p = (float*) buffer;
    float *q = (float*) buffer + ((n+dif)*(n+dif));

    for(i=0;i<n+dif;i++){
        for(j=0;j<n+dif;j++){
            if(i >= n || j >= n)
                p[i*(n+dif)+j] = 0;
            else{
                if(scanf("%f",&p[i*(n+dif)+j]) == 1){}
            }
        }
    }
	
    for(i=0;i<n+dif;i++){
        for(j=0;j<n+dif;j++){
            if(i >= n || j >= n)
                q[j*(n+dif)+i] = 0;
            else
                if(scanf("%f",&q[j*(n+dif)+i])){}
        }
    }
    
    
    for(i=0;i<n+dif;i++){
        for(j=0;j<n+dif;j++){
            printf(" %.1f", p[i*(n+dif)+j]);
        }
        printf("\n");
    }
    
    printf(" \n\n");
    for(i=0;i<n+dif;i++){
        for(j=0;j<n+dif;j++){
            printf(" %.1f", q[i*(n+dif)+j]);
        }
        printf("\n");
    }
}

void multMatrix(void * buffer){
	
    int i, j, k;
	int m = n + dif;
	
	int t1,t2;
	
    float *p = (float*) buffer;
    float *q = (float*) buffer + ((n+dif)*(n+dif));
    float mr[n+dif][n+dif] __attribute__ ((aligned(16)));
	
	for(i=0;i<n+dif;i++){
        for(j=0;j<n+dif;j++){
            mr[i][j] = 0;
        }
    }

	__m128 va,vb,mm_r;
	
	for ( i = 0; i < (n*(m/4)); i+=(m/4)) {
		
		for ( j = 0; j < n*(m/4); j++) {
			
				t1 = i;
				t2 = j/(m/4);
		
			for(k = n+dif; k>0;k-=4){
				
				va = _mm_load_ps(&p[i*4]);  // va = P.row
				vb = _mm_load_ps(&q[j*4]);  // vb = Q.row (Q is already transposed)
				
				mm_r = _mm_mul_ps(va,vb);  // mm_r = va * vb
	
				mr[t1/(m/4)][t2] +=  hsum_ps_sse1(mm_r);  //mr[i][j] += horizontal_sum of mm_r

				j++;
				i++;
			}
			
			i = t1;
			j--;
			
		}
	}
	
	printf("\nres:\n");
	for(i=0;i<n+dif;i++){
        for(j=0;j<n+dif;j++){
            printf(" %.1f", mr[i][j]);
        }
        printf("\n");
    }
    
    return;
}
	
//função para realizar soma horizontal em um vetor mais rapidamente que o equivalente escalar
float hsum_ps_sse1(__m128 v) {                                  // v = [ D C | B A ]
    __m128 shuf   = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));  // [ C D | A B ]
    __m128 sums   = _mm_add_ps(v, shuf);      // sums = [ D+C C+D | B+A A+B ]
    shuf          = _mm_movehl_ps(shuf, sums);      //  [   C   D | D+C C+D ]  // let the compiler avoid a mov by reusing shuf
    sums          = _mm_add_ss(sums, shuf);
    return    _mm_cvtss_f32(sums);
}

