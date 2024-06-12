#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <cuda_runtime.h>

#define NX 41
#define NY 41
#define NT 500
#define NIT 50
#define DX 2.0 / (NX - 1)
#define DY 2.0 / (NY - 1)
#define DT 0.01
#define RHO 1.0
#define NU 0.02

__global__ void compute_b(float* b, float* u, float* v, double dt, double dx, double dy, double rho) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = j * NX + i;

    if (j > 0 && j < NY - 1 && i > 0 && i < NX - 1) {
        float u_diff_x = (u[idx + 1] - u[idx - 1]) / (2.0 * dx);
        float v_diff_y = (v[idx + NX] - v[idx - NX]) / (2.0 * dy);
        float u_diff_y = (u[idx + NX] - u[idx - NX]) / (2.0 * dy);
        float v_diff_x = (v[idx + 1] - v[idx - 1]) / (2.0 * dx);

        b[idx] = rho * (1.0 / dt * (u_diff_x + v_diff_y) -
            u_diff_x * u_diff_x -
            2.0 * u_diff_y * v_diff_x -
            v_diff_y * v_diff_y);
    }
}

__global__ void pressure_poisson(float* p, float* pn, float* b, double dx, double dy) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = j * NX + i;

    if (j > 0 && j < NY - 1 && i > 0 && i < NX - 1) {
        p[idx] = ((pn[idx + 1] + pn[idx - 1]) * dy * dy +
            (pn[idx + NX] + pn[idx - NX]) * dx * dx -
            b[idx] * dx * dx * dy * dy) / (2 * (dx * dx + dy * dy));
    }

    // Enforce boundary conditions for pressure
    if (i == NX - 1) {
        p[idx] = p[idx - 1]; // dp/dx = 0 at x = 2
    }
    if (i == 0) {
        p[idx] = p[idx + 1]; // dp/dx = 0 at x = 0
    }
    if (j == NY - 1) {
        p[idx] = 0; // p = 0 at y = 2
    }
    if (j == 0) {
        p[idx] = p[idx + NX]; // dp/dy = 0 at y = 0
    }
}

__global__ void update_velocity(float* u, float* v, float* un, float* vn, float* p, double dt, double dx, double dy, double rho, double nu) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = j * NX + i;

    if (j > 0 && j < NY - 1 && i > 0 && i < NX - 1) {
        float un_diff_x = (un[idx] - un[idx - 1]) / dx;
        float un_diff_y = (un[idx] - un[idx - NX]) / dy;
        float vn_diff_x = (vn[idx] - vn[idx - 1]) / dx;
        float vn_diff_y = (vn[idx] - vn[idx - NX]) / dy;
        float p_diff_x = (p[idx + 1] - p[idx]) / (2 * rho * dx);
        float p_diff_y = (p[idx + NX] - p[idx]) / (2 * rho * dy);
        float un_laplace_x = (un[idx + 1] - 2 * un[idx] + un[idx - 1]) / (dx * dx);
        float un_laplace_y = (un[idx + NX] - 2 * un[idx] + un[idx - NX]) / (dy * dy);
        float vn_laplace_x = (vn[idx + 1] - 2 * vn[idx] + vn[idx - 1]) / (dx * dx);
        float vn_laplace_y = (vn[idx + NX] - 2 * vn[idx] + vn[idx - NX]) / (dy * dy);

        u[idx] = un[idx] - un[idx] * dt * un_diff_x -
            vn[idx] * dt * un_diff_y -
            dt * p_diff_x +
            nu * dt * (un_laplace_x + un_laplace_y);

        v[idx] = vn[idx] - un[idx] * dt * vn_diff_x -
            vn[idx] * dt * vn_diff_y -
            dt * p_diff_y +
            nu * dt * (vn_laplace_x + vn_laplace_y);
    }

    // Enforce boundary conditions for velocity
    if (j == 0 || j == NY - 1 || i == 0 || i == NX - 1) {
        u[idx] = 0;
        v[idx] = 0;
    }

    if (j == NY - 1) {
        u[idx] = 1; // velocity boundary condition at the top (moving lid)
    }
}

int main() {
    // Allocate and initialize host arrays
    float* h_u, * h_v, * h_p, * h_b, * h_un, * h_vn, * h_pn;
    size_t size = NX * NY * sizeof(float);

    h_u = (float*)malloc(size);
    h_v = (float*)malloc(size);
    h_p = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_un = (float*)malloc(size);
    h_vn = (float*)malloc(size);
    h_pn = (float*)malloc(size);

    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            int idx = j * NX + i;
            h_u[idx] = 0.0;
            h_v[idx] = 0.0;
            h_p[idx] = 0.0;
            h_b[idx] = 0.0;
        }
    }

    // Allocate device arrays
    float* d_u, * d_v, * d_p, * d_b, * d_un, * d_vn, * d_pn;
    cudaMalloc((void**)&d_u, size);
    cudaMalloc((void**)&d_v, size);
    cudaMalloc((void**)&d_p, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_un, size);
    cudaMalloc((void**)&d_vn, size);
    cudaMalloc((void**)&d_pn, size);

    // Copy host arrays to device
    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Write results to files
    std::ofstream ufile("u.dat");
    std::ofstream vfile("v.dat");
    std::ofstream pfile("p.dat");

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y);

    for (int n = 0; n < NT; n++) {
        compute_b << <gridSize, blockSize >> > (d_b, d_u, d_v, DT, DX, DY, RHO);
        cudaDeviceSynchronize();

        for (int it = 0; it < NIT; it++) {
            cudaMemcpy(d_pn, d_p, size, cudaMemcpyDeviceToDevice);
            pressure_poisson << <gridSize, blockSize >> > (d_p, d_pn, d_b, DX, DY);
            cudaDeviceSynchronize();
        }

        cudaMemcpy(d_un, d_u, size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_vn, d_v, size, cudaMemcpyDeviceToDevice);

        update_velocity << <gridSize, blockSize >> > (d_u, d_v, d_un, d_vn, d_p, DT, DX, DY, RHO, NU);
        cudaDeviceSynchronize();

        // Apply boundary conditions on the host
        cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v, d_v, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p, d_p, size, cudaMemcpyDeviceToHost);

        for (int j = 0; j < NY; j++) {
            h_u[j * NY + 0] = 0;
            h_u[j * NY + (NX - 1)] = 0;
            h_v[j * NY + 0] = 0;
            h_v[j * NY + (NX - 1)] = 0;
        }
        for (int i = 0; i < NX; i++) {
            h_u[0 * NY + i] = 0;
            h_u[(NX - 1) * NY + i] = 1;
            h_v[0 * NY + i] = 0;
            h_v[(NY - 1) * NY + i] = 0;
        }

        if (n % 10 == 0) {
            for (int j = 0; j < NY; j++)
                for (int i = 0; i < NX; i++)
                    ufile << h_u[j * NY + i] << " ";
            ufile << "\n";
            for (int j = 0; j < NY; j++)
                for (int i = 0; i < NX; i++)
                    vfile << h_v[j * NY + i] << " ";
            vfile << "\n";
            for (int j = 0; j < NY; j++)
                for (int i = 0; i < NX; i++)
                    pfile << h_p[j * NY + i] << " ";
            pfile << "\n";
        }
    }

    ufile.close();
    vfile.close();
    pfile.close();

    // Free device memory
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_p);
    cudaFree(d_b);
    cudaFree(d_un);
    cudaFree(d_vn);
    cudaFree(d_pn);

    // Free host memory
    free(h_u);
    free(h_v);
    free(h_p);
    free(h_b);
    free(h_un);
    free(h_vn);
    free(h_pn);

    return 0;
}
