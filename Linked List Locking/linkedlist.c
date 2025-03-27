#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

// Node structure with OpenMP lock
typedef struct Node {
    int value;
    struct Node *next, *prev;
    omp_lock_t lock;
} Node;

Node *head = NULL;
omp_lock_t list_lock;

int main() {
    int num_threads;
    omp_init_lock(&list_lock);

    printf("N, Threads, Time (sec)\n");

    // Serial Execution for N = 2^0 to 2^18
    for (int exp = 0; exp <= 16; exp++) {
        int N = pow(2, exp);
        double start_time = omp_get_wtime();
        
        for (int i = 0; i < N; i++) {
            int value = rand() % 10000;
            Node *newNode = (Node *)calloc(1, sizeof(Node));
            newNode->value = value;
            newNode->next = newNode->prev = NULL;
            omp_init_lock(&newNode->lock);
            
            omp_set_lock(&list_lock);
            Node *p = head, *prev = NULL;
            if (!p || p->value >= value) {
                if (p) omp_set_lock(&p->lock);
                newNode->next = p;
                head = newNode;
                if (p) {
                    p->prev = newNode;
                    omp_unset_lock(&p->lock);
                }
                omp_unset_lock(&list_lock);
                continue;
            }
            omp_unset_lock(&list_lock);
            
            while (p->next && p->next->value < value) {
                p = p->next;
            }
            
            omp_set_lock(&p->lock);
            if (p->next) omp_set_lock(&p->next->lock);
            
            newNode->next = p->next;
            newNode->prev = p;
            p->next = newNode;
            if (newNode->next) newNode->next->prev = newNode;
            
            if (newNode->next) omp_unset_lock(&newNode->next->lock);
            omp_unset_lock(&p->lock);
        }
        double serial_time = omp_get_wtime() - start_time;
        printf("%d, 1, %f\n", N, serial_time);
        head = NULL;
    }

    // Parallel Execution for different thread counts
    for (num_threads = 2; num_threads <= 32; num_threads *= 2) {
        for (int exp = 0; exp <= 16; exp++) {
            int N = pow(2, exp);
            double start_time = omp_get_wtime();
            
            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < N; i++) {
                int value = rand() % 10000;
                Node *newNode = (Node *)calloc(1, sizeof(Node));
                newNode->value = value;
                newNode->next = newNode->prev = NULL;
                omp_init_lock(&newNode->lock);

                omp_set_lock(&list_lock);
                Node *p = head, *prev = NULL;
                if (!p || p->value >= value) {
                    if (p) omp_set_lock(&p->lock);
                    newNode->next = p;
                    head = newNode;
                    if (p) {
                        p->prev = newNode;
                        omp_unset_lock(&p->lock);
                    }
                    omp_unset_lock(&list_lock);
                    continue;
                }
                omp_unset_lock(&list_lock);
                
                while (p->next && p->next->value < value) {
                    p = p->next;
                }
                
                omp_set_lock(&p->lock);
                if (p->next) omp_set_lock(&p->next->lock);
                
                newNode->next = p->next;
                newNode->prev = p;
                p->next = newNode;
                if (newNode->next) newNode->next->prev = newNode;
                
                if (newNode->next) omp_unset_lock(&newNode->next->lock);
                omp_unset_lock(&p->lock);
            }
            double parallel_time = omp_get_wtime() - start_time;
            printf("%d, %d, %f\n", N, num_threads, parallel_time);
            head = NULL;
        }
    }

    // Reset list safely
    omp_set_lock(&list_lock);
    head = NULL;
    omp_unset_lock(&list_lock);
    
    return 0;
}
