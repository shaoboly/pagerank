#include <stdio.h> 
#include <stdlib.h> 
#include <cuda_runtime.h> 

#include "cusparse.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <device_launch_parameters.h>
#include <sstream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <helper_cuda.h>
#include <helper_timer.h>

using namespace std;

//#pragma comment(lib,"cusparse.lib")

#define dynamic_max 256
#define dynamic_block_size 256
#define max_bin 12

#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError(const char* file, const int32_t line) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		std::cerr << "cudaCheckError() failed at " << file << ":" << line << " : "
			<< cudaGetErrorString(err) << endl;
		exit(-1);
	}
}


__global__ void init_pagerank_d(int n ,float *pagerank_d)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < n) {
		//pagerank_d[i] = 1.0 / (float)n_vertices;
		pagerank_d[i] = 1.0;
	}
}

//先不应用
__global__ void init_pagerank_d_by_out(int n, float *pagerank_d, int *destination_offsets_d)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < n) {
		//pagerank_d[i] = 1.0 / (float)n_vertices;
		int out_d = destination_offsets_d[i + 1] - destination_offsets_d[i];
		if (out_d == 0)
			pagerank_d[i] = 1.0 / n;
		else
			pagerank_d[i] = 1.0 / out_d;
	}
}


__global__ void get_dangling_value(int n, float *pagerank_d, float *bookmark_d, float *damping_value_d)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < n && bookmark_d[i] >0.1) {
		atomicAdd(damping_value_d, bookmark_d[i] * pagerank_d[i]);
	}
}

__global__ void get_dangling_value_v1(int dangling_nnz, float *pagerank_d, int *x_idx_d, float *damping_value_d)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < dangling_nnz) {
		atomicAdd(damping_value_d, 1.0 * pagerank_d[x_idx_d[i]]);
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


__global__ void child_kernal(int parant_idx, int begin, int in_d, float alpha, float *weights_d, int *destination_offsets_d, int *source_indices_d, float *pagerank_d, float *pagerank_next_d)
{
	extern __shared__ float storage[];
	int cidx = threadIdx.x;
	int idx = cidx + blockDim.x * blockIdx.x;
	int t_n = blockDim.x * gridDim.x;
	float tmp = 0;
	while (idx < in_d)
	{
		tmp += pagerank_d[source_indices_d[begin + idx]] * weights_d[begin + idx] * alpha;
		idx += t_n;
	}
	storage[cidx] = tmp;
	__syncthreads();
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cidx < i)
		{
			storage[cidx] += storage[cidx + i];
		}
		__syncthreads();
		i >>= 1;
	}
	if (cidx == 0)
		atomicAdd(&pagerank_next_d[parant_idx], storage[0]);

}

__global__ void dynamic_test_big_node_v1(int bin_n, int child_block_size, int *bin, float alpha, float *weights_d, int *destination_offsets_d, int *source_indices_d, float *pagerank_d, float *pagerank_next_d, float remain_value)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < bin_n){
		int now_node = bin[idx];
		pagerank_next_d[now_node] += remain_value;
		int begin = destination_offsets_d[now_node];
		int end = destination_offsets_d[now_node + 1];
		int in_d = end - begin;


		child_kernal << <1, child_block_size, child_block_size*sizeof(float) >> >(now_node, begin, in_d, alpha, weights_d, destination_offsets_d, source_indices_d, pagerank_d, pagerank_next_d);
	}
}

__global__ void dynamic_test_big_node_v2(int bin_n, int child_block_size, int *bin, float alpha, float *weights_d, int *destination_offsets_d, int *source_indices_d, float *pagerank_d, float *pagerank_next_d, float remain_value)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx < bin_n){
		int now_node = bin[idx];
		pagerank_next_d[now_node] += remain_value;
		int begin = destination_offsets_d[now_node];
		int end = destination_offsets_d[now_node + 1];
		int in_d = end - begin;

		int thread_num = in_d/4;

		int numOfBlocks = 1;                          // default example value for 1000 vertex graph
		int threadsPerBlock = 1;                   // default example value for 1000 vertex graph

		if (thread_num <= 1024) {
			threadsPerBlock = child_block_size;
			numOfBlocks = 1;
		}
		else {
			threadsPerBlock = 1024;
			numOfBlocks = (thread_num + 1023) / 1024;   // The "+ 1023" ensures we round up
		}

		child_kernal << <numOfBlocks, threadsPerBlock, threadsPerBlock*sizeof(float) >> >(now_node, begin, in_d, alpha, weights_d, destination_offsets_d, source_indices_d, pagerank_d, pagerank_next_d);
	}
}


__global__ void addToNextPagerankArray(int n, int nnz, float alpha, float *weights_d, int *destination_offsets_d, int *source_indices_d, float *pagerank_d, float *pagerank_next_d, float remain_value)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (idx < n){
		pagerank_next_d[idx] += remain_value;
		int begin = destination_offsets_d[idx];
		int end = destination_offsets_d[idx + 1];
		int in_d = end - begin;

		if (in_d >= dynamic_max)
		{
			child_kernal << <1, dynamic_block_size >> >(idx, begin, in_d, alpha, weights_d, destination_offsets_d, source_indices_d, pagerank_d, pagerank_next_d);
		}
		else
		{
			for (; begin < end; begin++){
				pagerank_next_d[idx] += pagerank_d[source_indices_d[begin]] * weights_d[begin] * alpha;
			}
		}
	}
	//__syncthreads();
}

__global__ void dynamic_test(int bin_n, int nnz,int *bin, float alpha, float *weights_d, int *destination_offsets_d, int *source_indices_d, float *pagerank_d, float *pagerank_next_d, float remain_value)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;


	if (idx < bin_n){
		int now_node = bin[idx];
		pagerank_next_d[now_node] = 0;
		pagerank_next_d[now_node] += remain_value;
		int begin = destination_offsets_d[now_node];
		int end = destination_offsets_d[now_node + 1];
		for (; begin < end; begin++){
			pagerank_next_d[now_node] += pagerank_d[source_indices_d[begin]] * weights_d[begin] * alpha;
		}
	}
}

__global__ void convergence(float * pagerank_d, float * pagerank_next_d, float * reduced_sums_d, int n_vertices) {
	// Each thread computes the diff for two vertexes (thus, half # of blocks needed for this function)
	// Because of this, we need to handle the case where only one block is needed
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i_thr = threadIdx.x;

	__shared__ float sums[1024];                       // blockDim.x == 1024

	float temp1, temp2;

	if (i < 1024) {
		reduced_sums_d[i] = 0;
	}

	if (i < n_vertices) {
		temp1 = pagerank_next_d[i] - pagerank_d[i];
		if (i + (1024 * gridDim.x) < n_vertices) {
			temp2 = pagerank_next_d[i + (1024 * gridDim.x)] - pagerank_d[i + (1024 * gridDim.x)];
		}
		else{
			temp2 = 0;
		}

		if (temp1 < 0) {
			temp1 = temp1 * (-1);
		}
		if (temp2 < 0) {
			temp2 = temp2 * (-1);
		}

		sums[i_thr] = temp1 + temp2;
	}
	else {
		sums[i_thr] = 0;
	}
	__syncthreads();

	int j, index, index2;
	index = i_thr;

	//可优化
	for (j = 0; j < 10; j++) {                    // 10 times as 2^10 = 1024 threads
		if ((index + 1) % (2 * (1 << j)) == 0) {    // Note: 1 << j == 2^j
			index2 = index - (1 << j);
			sums[index] += sums[index2];
		}
		__syncthreads();
	}

	reduced_sums_d[blockIdx.x] = sums[1023];
}

__global__ void getConvergence(float * reduced_sums_d, float * diff) {
	int j, index, index2;
	index = threadIdx.x;

	for (j = 0; j < 10; j++) {                    // 10 times as 2^10 = 1024 threads
		if ((index + 1) % (2 * (1 << j)) == 0) {    // Note: 1 << j == 2^j
			index2 = index - (1 << j);
			reduced_sums_d[index] += reduced_sums_d[index2];
		}
		__syncthreads();
	}

	*diff = reduced_sums_d[1023];
}

void str2int(int &int_temp, const string &string_temp)
{
	stringstream stream(string_temp);
	stream >> int_temp;
}

void compress_dangling_node_vector(float *bookmark_h,int n_vertices,int &dangling_nnz, float* &value_vector, int* &x_idx)
{
	dangling_nnz = 0;

	for (int i = 0; i < n_vertices; i++)
	{
		if (bookmark_h[i]>0.5)
			dangling_nnz++;
	}

	value_vector = (float*)malloc(dangling_nnz*sizeof(float));
	x_idx = (int*)malloc(n_vertices*sizeof(int));

	int offset = 0;
	for (int i = 0; i < n_vertices; i++)
	{
		if (bookmark_h[i]>0.5)
		{
			x_idx[offset] = i;
			value_vector[offset++] = bookmark_h[i];
 		}
	}

	return;
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


void getInputResultForCsc(vector<vertex> &vertices, string input_name, int &n, int &nnz, int *&destination_offsets_h, int *&source_indices_h, float *&weights_h, float *&bookmark_h)
{
	ifstream input_file(input_name);
	int tmp_va, tmp_vb;

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


vector<vector<int>> binary_resort(vector<vertex> &vertices, int n, int nnz, int &bins_number)
{
	int max_label = 0;
	vector<vector<int>> all_b(max_bin);

	for (int i = 0; i < n; i++)
	{
		if (vertices[i].in_deg == 0)
		{
			all_b[0].push_back(i);
			continue;
		}
		int now_label = (unsigned int)log2(double(vertices[i].in_deg) - 0.1) + 1;

		if (now_label >= max_bin){
			all_b[max_bin - 1].push_back(i);
			max_label = max_bin - 1;
			continue;
		}

		all_b[now_label].push_back(i);

		if (max_label < now_label)
			max_label = now_label;
	}
	for (int i = 0; i < max_bin; i++)
	{
		cout << i << " " << all_b[i].size() << endl;
	}
	bins_number = max_label+1;
	
	return all_b;
}


//transform to bins
void trans_to_b(vector<vertex> vertices, int n, int nnz, int* &g1, vector<int*> &g2, int &bins_number, vector<vector<int>> &all_bin, int &dp_max_num, int &row_max)
{
	row_max = 0;
	for (int i = 0; i < dp_max_num; i++)
	{
		row_max += all_bin[i].size();
	}

	g1 = (int*)malloc((row_max)*sizeof(int));

	int tmp_ind = 0;
	for (int i = 0; i < dp_max_num; i++)
	{
		for (int j = 0; j < all_bin[i].size(); j++)
		{
			g1[tmp_ind++] = all_bin[i][j];
		}
	}

	for (int i = dp_max_num; i < bins_number; i++)
	{
		int *tmp = (int*)malloc(all_bin[i].size()*sizeof(int));
		for (int j = 0; j < all_bin[i].size(); j++)
		{
			tmp[j] = all_bin[i][j];
		}
		g2.push_back(tmp);
	}

}

void pagerank_cuda(string input_name, string output_name, int max_iteration)
{

	ofstream output_file(output_name);

	//graph imformation
	int  n = 0, nnz = 0, vertex_numsets = 3, edge_numsets = 1;
	int *destination_offsets_h = NULL, *source_indices_h = NULL;
	int *destination_offsets_d, *source_indices_d;
	float *weights_h = NULL, *bookmark_h = NULL;
	float *weights_d,*bookmark_d;
	float dampling_value_h;
	float *damping_value_d;

	float *reduced_sums_d;
	float * d_diff;
	cudaMalloc((void **)&reduced_sums_d, 1024 * sizeof(float));
	cudaMalloc((void **)&d_diff, sizeof(float));


	float alpha = 1;
	float beta = 0;
	//void** vertex_dim;

	vector<vertex> vertices;
	getInputResultForCsc(vertices,input_name, n, nnz, destination_offsets_h, source_indices_h, weights_h, bookmark_h);


	//to sort in bins
	int bins_number = 0;
	vector<vector<int>> all_bin = binary_resort(vertices, n, nnz, bins_number);

	int dp_max_num = 8;
	int row_max = 0;
	
	int *g1 = NULL;
	vector<int*> g2;
	trans_to_b(vertices, n, nnz, g1, g2, bins_number, all_bin, dp_max_num, row_max);
	cout << "1" << endl;
	//--------------------------//
	
	int *g1_device;
	cudaMalloc((void **)&g1_device, row_max*sizeof(int));
	
	cudaMemcpy(g1_device, g1, row_max*sizeof(int), cudaMemcpyHostToDevice);
	cout << "g1:" << row_max << endl;

	int g2_size = bins_number - dp_max_num;
	vector<int *> device_g2;

	cout << g2_size << endl;
	int* g2_b_size = new int[g2_size];
	for (int i = 0; i < g2_size; i++)
	{
		g2_b_size[i] = all_bin[i + dp_max_num].size();
		int *d_tmp;
		cudaMalloc((void **)&d_tmp, g2_b_size[i] * sizeof(int));
		cudaMemcpy(d_tmp, g2[i], g2_b_size[i] * sizeof(int), cudaMemcpyHostToDevice);
		cout << "g" << i + dp_max_num << " " << g2_b_size[i] << endl;
		device_g2.push_back(d_tmp);

	}





	int g1_block_number, g1_thread_per_block;
	if (row_max <= 1024) {
		g1_thread_per_block = row_max;
		g1_block_number = 1;
	}
	else {
		g1_thread_per_block = 1024;
		g1_block_number = (row_max + 1023) / 1024;   // The "+ 1023" ensures we round up
	}

	vector<int> g2_block_number, g2_thread_per_block;

	int child_thread_per_block = 4;
	for (int i = 0; i < g2_size; i++)
	{
		int tmp_block_number = (g2_b_size[i] + child_thread_per_block-1) / child_thread_per_block, tmp_thread_per_block = child_thread_per_block;
		
		/*if (g2_b_size[i] <= 1024) {
			tmp_thread_per_block = g2_b_size[i];
			tmp_block_number = 1;
		}
		else {
			tmp_thread_per_block = 1024;
			tmp_block_number = (g2_b_size[i] +1023) / 1024;   // The "+ 1023" ensures we round up
		}*/

		g2_block_number.push_back(tmp_block_number);
		g2_thread_per_block.push_back(tmp_thread_per_block);

		//g2_block_number.push_back(tmp_block_number);

	}

	int size[] = { 128, 256, 512, 1024, 1024,1024,1024,1024 };
	//---------------------------//


	float *pagerank_d, *pagerank_next_d, *pagerank_h = (float*)malloc(n*sizeof(float));

	//cuSparse parameters useless thie version
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
	float epsilon = 0.000001 * n;
	float h_diff = epsilon + 1;
	int n_blocks = (n + 2048 - 1) / 2048;
	if (n_blocks == 0){
		n_blocks = 1;
	}

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);

	float remain_value;

	//value about dangling nodes
	float *value_vector_h = NULL, *value_vector_d;
	int *x_idx_h = NULL,*x_idx_d;
	int dangling_nnz = 0;
	compress_dangling_node_vector(bookmark_h, n, dangling_nnz, value_vector_h, x_idx_h);
	
	cout << dangling_nnz << endl;

	cudaMalloc((void **)&value_vector_d, dangling_nnz*sizeof(float));
	cudaMalloc((void **)&x_idx_d, dangling_nnz*sizeof(int));

	cudaMemcpy(value_vector_d, value_vector_h, dangling_nnz*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(x_idx_d, x_idx_h, dangling_nnz*sizeof(int), cudaMemcpyHostToDevice);

	//---------------------------
	float base_time = 0;
	float dangling_time = 0;
	float pull_time = 0;
	float g1_time = 0;
	float g2_time = 0;
	//-------------------------------

	cudaStream_t stream[15];
	for (int i = 0; i < 15; ++i) 
		cudaStreamCreate(&stream[i]);


	for (iteration = 0; iteration < max_iteration; iteration++)
	{
		//cudaMemset((void *)z, 0, 2 * (m)*sizeof(float));

		base_time = sdkGetTimerValue(&hTimer);

		//if (i < 10 || i % 5 == 0)
		//{
		    //dangling_sum to 0
			dampling_value_h = 0;
			//err = cudaMemcpy(damping_value_d, &dampling_value_h, sizeof(float), cudaMemcpyHostToDevice);


			cudaThreadSynchronize();
			sdkStartTimer(&hTimer);
			cusparseSdoti(handle, dangling_nnz, value_vector_d, x_idx_d, pagerank_d, &dampling_value_h, CUSPARSE_INDEX_BASE_ZERO);
			//get_dangling_value << <numOfBlocks, threadsPerBlock >> >(n, pagerank_d, bookmark_d, damping_value_d);
			//get_dangling_value_v1 << <numOfBlocks, threadsPerBlock >> >(dangling_nnz, pagerank_d, x_idx_d, damping_value_d);
			cudaThreadSynchronize();
			sdkStopTimer(&hTimer);


			dangling_time += sdkGetTimerValue(&hTimer) - base_time;
			base_time = sdkGetTimerValue(&hTimer);


			//err = cudaMemcpy(&dampling_value_h, damping_value_d, sizeof(float), cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();

			dampling_value_h *= alpha;
			//cout << dampling_value_h << endl;
		//}

		
		
		remain_value = dampling_value_h* 1.0 / (float)n + 1 - alpha;
		//cudaMemcpy(damping_value_d, &dampling_value_h, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemset((void *)pagerank_next_d, 0, (n)*sizeof(float));

		cudaDeviceSynchronize();
		sdkStartTimer(&hTimer);
		

		//addToNextPagerankArray << <numOfBlocks, threadsPerBlock >> >(n, nnz, alpha, weights_d, destination_offsets_d, source_indices_d, pagerank_d, pagerank_next_d, remain_value);
		//cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &alpha, descr, weights_d, destination_offsets_d, source_indices_d, pagerank_d, &beta, pagerank_next_d);
		

		//g2--------------------------------------------------------------------------------//

		//g2_size--;
		for (int j = 0; j < g2_size; j++)
		{
			dynamic_test_big_node_v2 << <g2_block_number[j], g2_thread_per_block[j], 0, stream[j] >> >(g2_b_size[j], size[j], device_g2[j], alpha, weights_d, destination_offsets_d, source_indices_d, pagerank_d, pagerank_next_d, remain_value);
		}
		
		//
		//int j = g2_size;
		//dynamic_test_big_node_v2 << <g2_block_number[j], g2_thread_per_block[j],0, stream[j]>> >(g2_b_size[j], size[j], device_g2[j], alpha, weights_d, destination_offsets_d, source_indices_d, pagerank_d, pagerank_next_d, remain_value);


		//-----------------------------------//

		//-=------------------------------//执行g1

		
		dynamic_test << <g1_block_number, g1_thread_per_block, 0, stream[14] >> >(row_max, nnz, g1_device, alpha, weights_d, destination_offsets_d, source_indices_d, pagerank_d, pagerank_next_d, remain_value);
		
		//cudaDeviceSynchronize();
		//sdkStopTimer(&hTimer);

		//g1_time += sdkGetTimerValue(&hTimer) - base_time;
		//base_time = sdkGetTimerValue(&hTimer);
		//---------------------------------//

		//sdkStartTimer(&hTimer);

		
		cudaDeviceSynchronize();
		//cudaThreadSynchronize();

		sdkStopTimer(&hTimer);
		//g2_time += sdkGetTimerValue(&hTimer) - base_time;
		//base_time = sdkGetTimerValue(&hTimer);

		pull_time += sdkGetTimerValue(&hTimer) - base_time;
		base_time = sdkGetTimerValue(&hTimer);


		
		
		cudaDeviceSynchronize();
		// Test for convergence
		//cudaThreadSynchronize();


		sdkStartTimer(&hTimer);
		convergence << <n_blocks, 1024 >> >(pagerank_d, pagerank_next_d, reduced_sums_d, n);
		getConvergence << <1, 1024 >> >(reduced_sums_d, d_diff);
		cudaThreadSynchronize();
		sdkStopTimer(&hTimer);

		cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
		cout << iteration << " " << h_diff << endl;


		/*sdkStartTimer(&hTimer);

		finalPagerankArrayForIteration <<<numOfBlocks, threadsPerBlock >>>(pagerank_next_d, n, dampling_value_h, alpha);

		cudaThreadSynchronize();
		sdkStopTimer(&hTimer);*/

		cudaMemcpy(pagerank_d, pagerank_next_d, n*sizeof(float), cudaMemcpyDeviceToDevice);
		//cudaMemcpy(pagerank_h, pagerank_next_d, n*sizeof(float), cudaMemcpyDeviceToHost);
	}

	printf("Elapsed Time: %.6fms\n", sdkGetTimerValue(&hTimer));

	cout << "dangling_time:" << dangling_time << endl;
	cout << "pull_time:" << pull_time << endl;
	cout << "g1_time: " << g1_time << endl;
	cout << "g2:" << g2_time << endl;

	cudaMemcpy(pagerank_h, pagerank_next_d, n*sizeof(float), cudaMemcpyDeviceToHost);
	output_file << "Elapsed Time: " << sdkGetTimerValue(&hTimer) << "ms" << endl;

	output_file.setf(ios::fixed);
	output_file.precision(6);
	for (int i = 0; i<n; i++)
	{
		output_file << pagerank_h[i] << endl;
	}


	cudaDeviceReset();

	free(pagerank_h);
	free(value_vector_h);
	free(x_idx_h);
	free(destination_offsets_h);
	free(source_indices_h);
	free(weights_h);
	free(bookmark_h);
}
/*
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
}*/

int main(int argc, char **argv)
{
	string input_name = "web-Stanford.txt";
	string output_name = "output.txt";

	int max_iteration = 40;

	if (argc > 1)
		input_name = argv[1];
	if (argc > 2)
		output_name = argv[2];
	if (argc > 3)
		str2int(max_iteration, argv[3]);

	//cuda_sparse();
	pagerank_cuda(input_name, output_name, max_iteration);
	return 0;
}