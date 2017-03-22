#include <stdio.h> 
#include <stdlib.h> 
#include <cuda_runtime.h> 

#include "cusparse.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <device_launch_parameters.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <helper_cuda.h>
#include <helper_timer.h>

using namespace std;

//#pragma comment(lib,"cusparse.lib")



__global__ void init_pagerank_d(int n ,float *pagerank_d)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < n) {
		//pagerank_d[i] = 1.0 / (float)n_vertices;
		pagerank_d[i] = 1.0;
	}
}

__global__ void get_dangling_value(int n, float *pagerank_d, float *bookmark_d, float *damping_value_d)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < n && bookmark_d[i] >0.1) {
		atomicAdd(damping_value_d, bookmark_d[i] * pagerank_d[i]);
	}
}

//将所有悬停点的累加的值加上去，且悬停点出度为1/n
__global__ void finalPagerankArrayForIteration(float * pagerank_next_d, int n_vertices, float dangling_value_h,float alpha) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;


	if (i < n_vertices) {
		//pagerank_next_d[i] += (dangling_value2 + (1 - 0.85)) / ((float)n_vertices);
		pagerank_next_d[i] += dangling_value_h* 1.0 / (float)n_vertices + 1 - alpha;
	}
}


void initialize(float *cooValHostPtr, int *cooColIndexHostPtr, float *yHostPtr, int *csrRowPtr)
{
	cooValHostPtr[0] = 1.0;
	cooValHostPtr[1] = 2.0;
	cooValHostPtr[2] = 3.0;
	cooValHostPtr[3] = 4.0;
	cooValHostPtr[4] = 5.0;
	cooValHostPtr[5] = 6.0;
	cooValHostPtr[6] = 7.0;
	cooValHostPtr[7] = 8.0;
	cooValHostPtr[8] = 9.0;

	cooValHostPtr[9] = 10.0;

	cooColIndexHostPtr[0] = 0;
	cooColIndexHostPtr[1] = 2;
	cooColIndexHostPtr[2] = 3;
	cooColIndexHostPtr[3] = 1;
	cooColIndexHostPtr[4] = 0;
	cooColIndexHostPtr[5] = 2;
	cooColIndexHostPtr[6] = 3;
	cooColIndexHostPtr[7] = 1;
	cooColIndexHostPtr[8] = 3;

	cooColIndexHostPtr[9] = 0;

	yHostPtr[0] = 10.0;
	yHostPtr[1] = 20.0;
	yHostPtr[2] = 30.0;
	yHostPtr[3] = 40.0;
	/*yHostPtr[4] = 50.0;
	yHostPtr[5] = 60.0;
	yHostPtr[6] = 70.0;
	yHostPtr[7] = 80.0;*/

	csrRowPtr[0] = 0;
	csrRowPtr[1] = 3;
	csrRowPtr[2] = 4;
	csrRowPtr[3] = 7;
	csrRowPtr[4] = 9;

	csrRowPtr[5] = 10;

}


class vertex
{
public:
	int id;
	int in_deg;
	int out_deg;
	vector<int> in_edge;
	vector<int> out_edge;
};


void getInputResultForCsc(string input_name, int &n, int &nnz, int *&destination_offsets_h, int *&source_indices_h, float *&weights_h, float *&bookmark_h)
{
	ifstream input_file(input_name);
	int tmp_va, tmp_vb;
	vector<vertex> vertices;

	input_file >> n >> nnz;

	destination_offsets_h = (int*)malloc((n + 1)*sizeof(int));
	source_indices_h = (int*)malloc(nnz*sizeof(int));
	weights_h = (float*)malloc(nnz*sizeof(float));
	bookmark_h = (float*)malloc(n*sizeof(float));

	for (int i = 0; i < n; i++)
	{
		vertex tmp_v;
		tmp_v.id = i;
		tmp_v.in_deg = 0;
		tmp_v.out_deg = 0;
		vertices.push_back(tmp_v);
	}

	for (int i = 0; i < nnz; i++)
	{
		input_file >> tmp_va >> tmp_vb;
		tmp_va--;
		tmp_vb--;
		vertices[tmp_va].out_deg += 1;
		vertices[tmp_va].out_edge.push_back(tmp_vb);

		vertices[tmp_vb].in_deg += 1;
		vertices[tmp_vb].in_edge.push_back(tmp_va);
	}

	//cout << "read graph success!" << endl;
	int tmp_offset = 0;
	for (int i = 0; i < n; i++)
	{
		destination_offsets_h[i] = tmp_offset;

		if (vertices[i].out_deg == 0)
			bookmark_h[i] = 1.0f;
		else
			bookmark_h[i] = 0.0f;

		for (int j = 0; j < vertices[i].in_deg; j++){
			source_indices_h[tmp_offset + j] = vertices[i].in_edge[j];
			weights_h[tmp_offset + j] = (float)1.0 / (float)vertices[source_indices_h[tmp_offset + j]].out_deg;
		}
		tmp_offset += vertices[i].in_deg;

	}
	destination_offsets_h[n] = tmp_offset;
}

void pagerank_cuda(string input_name, string output_name)
{

	ofstream output_file(output_name);

	//graph imformation
	int  n = 0, nnz = 0, vertex_numsets = 3, edge_numsets = 1;
	const float alpha1 = 0.85;
	const void *alpha1_p = (const void *)&alpha1;
	int *destination_offsets_h = NULL, *source_indices_h = NULL;
	int *destination_offsets_d, *source_indices_d;
	float *weights_h = NULL, *bookmark_h = NULL;
	float *weights_d,*bookmark_d;
	float dampling_value_h;
	float *damping_value_d;


	float alpha = 1;
	float beta = 0;
	//void** vertex_dim;

	getInputResultForCsc(input_name, n, nnz, destination_offsets_h, source_indices_h, weights_h, bookmark_h);

	float *pagerank_d, *pagerank_next_d, *pagerank_h = (float*)malloc(n*sizeof(float));

	//cuSparse parameters
	cusparseHandle_t handle;
	cusparseMatDescr_t descr;
	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);//shape of matrix(triangle or )
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);//index 0 or 1
	cusparseCreate(&handle);

	cudaMalloc((void **)&weights_d, nnz*sizeof(float));
	cudaMalloc((void **)&bookmark_d, n*sizeof(float));
	cudaMalloc((void **)&pagerank_d, n * sizeof(float));
	cudaMalloc((void **)&pagerank_next_d, n * sizeof(float));
	cudaMalloc((void **)&destination_offsets_d, (n + 1)*sizeof(int));
	cudaMalloc((void **)&source_indices_d, nnz*sizeof(int));
	cudaMalloc((void **)&damping_value_d, sizeof(float));


	int n_iterations = 3000;
	int iteration = 0;
	int numOfBlocks = 1;                          // default example value for 1000 vertex graph
	int threadsPerBlock = 1000;                   // default example value for 1000 vertex graph

	if (n <= 1024) {
		threadsPerBlock = n;
		numOfBlocks = 1;
	}
	else {
		threadsPerBlock = 1024;
		numOfBlocks = (n + 1023) / 1024;   // The "+ 1023" ensures we round up
	}


	//initialize origin pagerank sum, should be 1 or n
	/*for (int i = 0; i < n; i++)
	{
		pagerank_h[i] = 1;
	}*/


	// Error code to check return values for CUDA calls
	cudaFree(0);   // Set the cuda context here so that when we time, we're not including initial overhead
	cudaError_t err = cudaSuccess;

	//initialized
	init_pagerank_d << <numOfBlocks, threadsPerBlock >> >(n, pagerank_d);
	cudaDeviceSynchronize();

	cudaMemcpy(weights_d, weights_h, nnz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bookmark_d, bookmark_h, n*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(pagerank_d, pagerank_h, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(destination_offsets_d, destination_offsets_h, (n + 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(source_indices_d, source_indices_h, nnz*sizeof(int), cudaMemcpyHostToDevice);

	alpha = 0.85;


	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	for (int i = 0; i < 41; i++)
	{
		//cudaMemset((void *)z, 0, 2 * (m)*sizeof(float));

		dampling_value_h = 0;
		err = cudaMemcpy(damping_value_d, &dampling_value_h, sizeof(float), cudaMemcpyHostToDevice);
		get_dangling_value << <numOfBlocks, threadsPerBlock >> >(n, pagerank_d, bookmark_d, damping_value_d);
		cudaThreadSynchronize();

		err = cudaMemcpy(&dampling_value_h, damping_value_d, sizeof(float), cudaMemcpyDeviceToHost);
		dampling_value_h *= alpha;
		//cout << dampling_value_h << endl;

		//cudaMemcpy(damping_value_d, &dampling_value_h, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemset((void *)pagerank_next_d, 0, (n)*sizeof(float));
		cudaThreadSynchronize();

		
		sdkStartTimer(&hTimer);
		cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &alpha, descr, weights_d, destination_offsets_d, source_indices_d, pagerank_d, &beta, pagerank_next_d);
		

		cudaThreadSynchronize();
		

		finalPagerankArrayForIteration <<<numOfBlocks, threadsPerBlock >>>(pagerank_next_d, n, dampling_value_h, alpha);

		cudaThreadSynchronize();
		sdkStopTimer(&hTimer);

		cudaMemcpy(pagerank_d, pagerank_next_d, n*sizeof(float), cudaMemcpyDeviceToDevice);
		//cudaMemcpy(pagerank_h, pagerank_next_d, n*sizeof(float), cudaMemcpyDeviceToHost);
	}

	printf("Elapsed Time: %.6fms\n", sdkGetTimerValue(&hTimer));
	cudaMemcpy(pagerank_h, pagerank_next_d, n*sizeof(float), cudaMemcpyDeviceToHost);
	output_file << "Elapsed Time: " << sdkGetTimerValue(&hTimer) << "ms" << endl;
	for (int i = 0; i<n; i++)
	{
		output_file << pagerank_h[i] << endl;
	}


	cudaDeviceReset();
}

void cuda_sparse()
{


	int m = 5, n = 4, nnz = 10;
	float *cooValHostPtr = new float[nnz];
	//float *zHostPtr = new float[2 * (m)];
	float *zHostPtr = new float[m];

	int *cooColIndexHostPtr = new int[nnz];
	int *csrRowPtr = new int[m + 1];

	int *crsRow, *cooCol;

	float alpha = 1;
	float beta = 0;
	//float *yHostPtr = new float[2 * n];
	float *yHostPtr = new float[n];
	float * y, *cooVal, *z;
	initialize(cooValHostPtr, cooColIndexHostPtr, yHostPtr, csrRowPtr);


	cusparseHandle_t handle;
	cusparseMatDescr_t descr;
	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);//矩阵形状（三角、对称等）
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);//index 0 or 1

	cusparseCreate(&handle);

	cudaMalloc((void **)&cooVal, nnz*sizeof(float));
	/*cudaMalloc((void **)&y, 2 * n*sizeof(float));
	cudaMalloc((void **)&z, 2 * (m)*sizeof(float));*/
	cudaMalloc((void **)&y, n * sizeof(float));
	cudaMalloc((void **)&z, m * sizeof(float));
	cudaMalloc((void **)&crsRow, (m + 1)*sizeof(int));
	cudaMalloc((void **)&cooCol, nnz*sizeof(int));

	cudaMemcpy(cooVal, cooValHostPtr, nnz*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(y, yHostPtr, 2 * n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(y, yHostPtr, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(crsRow, csrRowPtr, (m + 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cooCol, cooColIndexHostPtr, nnz*sizeof(int), cudaMemcpyHostToDevice);

	//cudaMemset((void *)z, 0, 2 * (m)*sizeof(float));
	cudaMemset((void *)z, 0, (m)*sizeof(float));

	cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha, descr, cooVal, crsRow, cooCol, y, &beta, z);
	//cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, 2, n, nnz, &alpha, descr, cooVal, crsRow, cooCol, y, n, &beta, z, m);

	cudaMemcpy(zHostPtr, z,  (m)*sizeof(float), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < m; i++)
	//{
	//  //if(i%(2)==0&&i!=0)
	//  //  cout<<endl;
	//  cout<<zHostPtr[i]<<" "<<zHostPtr[i+m]<<endl;
	//}
	for (int i = 0; i<m; i++)
	{
		cout << zHostPtr[i] << " ";
	}
}

int main(int argc, char **argv)
{
	string input_name = "input.txt";
	string output_name = "output.txt";

	if (argc > 1)
		input_name = argv[1];
	if (argc > 2)
		output_name = argv[2];

	//cuda_sparse();
	pagerank_cuda(input_name, output_name);
	return 0;
}