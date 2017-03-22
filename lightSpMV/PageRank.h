/*
 * PageRank.h
 *
 *  Created on: May 29, 2015
 *      Author: Yongchao Liu
 *      Affiliation: School of Computational Science & Engineering, Georgia Institute of Technology
 *      Email: yliu@cc.gatech.edu
 */

#ifndef PAGERANK_H_
#define PAGERANK_H_

#pragma once

#include "Types.h"
/*CUSP*/
#include <cusp/io/matrix_market.h>
#include <cusp/hyb_matrix.h>
#include <cusp/transpose.h>

#include <helper_cuda.h>
#include <helper_timer.h>

/*cuSparse*/
#include <cusparse.h>
#include <iostream>
//#include <unistd.h>
//#include <getopt.h>
//#pragma comment(lib,"getopt.lib")

/*ViennaCL*/
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/linalg/sparse_matrix_operations.hpp>

//
enum {
	CSR_FORMAT, HYB_FORMAT, NUM_SPARSE_FORMATS
};

/*global function*/
extern int main_lightspmv_pagerank(int argc, char* argv[]);
extern int main_cusp_pagerank(int argc, char* argv[]);
extern int main_cusparse_pagerank(int argc, char* argv[]);
extern int main_viennacl_pagerank(int argc, char* argv[]);
/*distance measure*/
template<typename IntType, typename ValueType> inline ValueType l1Norm(
		const ValueType* x, const ValueType* y, const IntType n) {
	ValueType err = 0;

	for (IntType i = 0; i < n; ++i) {
		err += fabs(x[i] - y[i]);
	}
	return err;
}

template<typename IntType, typename ValueType> inline ValueType l2Norm(
		const ValueType* x, const ValueType* y, const IntType n) {
	ValueType err = 0;

	for (IntType i = 0; i < n; ++i) {
		err += (x[i] - y[i]) * (x[i] - y[i]);
	}
	err = sqrt(err);

	return err;
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

/*void getInputResultForCsc(vector<vertex> &vertices, string input_name, int &n, int &nnz, int *&destination_offsets_h, int *&source_indices_h, float *&weights_h, float *&bookmark_h)
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
}*/


/*the input is always CSR*/
template<typename IntType, typename ValueType, typename CSR_MEM_SPACE,
typename KernelType> bool pageRank(const char* graphFileName, const int maxNumLoops) {
	cout << "jin ru 1"  << endl;
	int k = 0;
	const int maxK = 1000;
	const ValueType tolerant = 1.0e-8;
	const ValueType damping = 0.85; /*damping factor*/
	float runtime;
	KernelType kernel;
	cudaEvent_t start, end;

	/* Get handle to the CUSPARSE context */
	ValueType alpha = 1.0;
	ValueType beta = 0;

	/*read in the sparse matrix in CSR format*/
	//cusp::csr_matrix<IntType, ValueType, cusp::host_memory> A(4, 4, 6);
	cusp::csr_matrix<IntType, ValueType, cusp::host_memory> hostCsrGraph;
	cusp::csr_matrix<IntType, ValueType, cusp::host_memory> hostCsrGraphT;
	

	/*read the graph represented in adjacency matrix*/
	cout << "to_read_graph" << endl;


	//-----------------------------------------------------//
	vector<vertex> vertices;
	int n = 0, nnz = 0;
	int *destination_offsets_h = NULL, *source_indices_h = NULL;
	float *weights_h = NULL;
	float *bookmark_h = NULL;
	ifstream input_file(graphFileName);
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
	//-----------------------------------------------------//
	//cusp::io::read_matrix_market_file(hostCsrGraph, graphFileName);


	cusp::csr_matrix<IntType, ValueType, cusp::host_memory> hostCsrGraph_tmp1(n,n,nnz);
	
	for (int i = 0;i<n+1; i++)
	{
		hostCsrGraph_tmp1.row_offsets[i] = (IntType)destination_offsets_h[i];
	}
	for (int i = 0; i < nnz; i++)
	{
		hostCsrGraph_tmp1.column_indices[i] = (IntType)source_indices_h[i];
		hostCsrGraph_tmp1.values[i] = weights_h[i];
	}
	

	cusp::transpose(hostCsrGraph_tmp1, hostCsrGraph);
	 /*// initialize matrix entries on host
    A.row_offsets[0] = 0;  // first offset is always zero
	A.row_offsets[1] = 2;
	A.row_offsets[2] = 2;
	A.row_offsets[3] = 3;
	A.row_offsets[4] = 6; // last offset is always num_entries
	
	A.column_indices[0] = 0; A.values[0] = 10;
	A.column_indices[1] = 2; A.values[1] = 20;
	A.column_indices[2] = 2; A.values[2] = 30;
	A.column_indices[3] = 0; A.values[3] = 40;
	A.column_indices[4] = 1; A.values[4] = 50;
	A.column_indices[5] = 2; A.values[5] = 60;*/
	cout << "to_read_graph successs" << endl;

	/*ensure that the matrix is square*/
	if (hostCsrGraph.num_rows != hostCsrGraph.num_cols) {
		cerr << "The transition matrix must be square" << endl;
		return false;
	}
	cerr << "#Rows/#Cols: " << hostCsrGraph.num_rows << " #Nonzero: "
			<< hostCsrGraph.num_entries << endl;

	/*row normalize the matrix*/
	for (IntType i = 0; i < hostCsrGraph.num_rows; ++i) {
		ValueType rowSum = 0;
		IntType rowStart = hostCsrGraph.row_offsets[i];
		IntType rowEnd = hostCsrGraph.row_offsets[i + 1];
		for (IntType j = rowStart; j < rowEnd; ++j) {
			/*make all values positive*/
			hostCsrGraph.values[j] = fabs(hostCsrGraph.values[j]);

			/*compute row sum*/
			rowSum += hostCsrGraph.values[j];
		}
		/*normalize the data*/
		if (rowSum != 0) {
			for (IntType j = rowStart; j < rowEnd; ++j) {
				hostCsrGraph.values[j] /= rowSum;
			}
		}
	}
	/*transpose matrix*/
	cusp::transpose(hostCsrGraph, hostCsrGraphT);

	/*prepare for the following operations*/
	cusp::csr_matrix<IntType, ValueType, CSR_MEM_SPACE> csrGraphT;

	/*copy the normalized matrix*/
	cusp::copy(hostCsrGraphT, csrGraphT);

	/*allocate space for ranks*/
	cusp::array1d<ValueType, cusp::host_memory> hostPersonalization(
			hostCsrGraph.num_rows, 0);
	cusp::array1d<ValueType, cusp::device_memory> devRank0(
			hostCsrGraph.num_rows, 0);
	cusp::array1d<ValueType, cusp::device_memory> devRank1(
			hostCsrGraph.num_rows, 0);
	cusp::array1d<ValueType, cusp::host_memory> hostRank0(hostCsrGraph.num_rows,
			0);
	cusp::array1d<ValueType, cusp::host_memory> hostRank1(hostCsrGraph.num_rows,
			0);

	/*create events*/
	cudaEventCreate(&start, cudaEventBlockingSync);
	cudaEventCreate(&end, cudaEventBlockingSync);

	/*starting measure the time*/
	cudaEventRecord(start, 0);

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);
	sdkResetTimer(&hTimer);
	/*******************************
	 * Note that the SpMV operations will use the transpose of the matrix
	 ******************************/
	for (int iter = 0; iter < maxNumLoops; ++iter) {
		/*convert from CSR to another sparse matrix format*/
		kernel.convert(csrGraphT, (ValueType) 0);

		/*initialize*/
		ValueType uniform = 1.0 / (ValueType) hostCsrGraph.num_rows;
		for (IntType i = 0; i < hostCsrGraph.num_rows; ++i) {
			hostRank0[i] = uniform;
			hostPersonalization[i] = uniform;
		}

		/*explicitly copy data to the device*/
		thrust::copy(hostRank0.begin(), hostRank0.end(), devRank0.begin());

		cusp::array1d<ValueType, cusp::device_memory> * devRank0Ptr = &devRank0;
		cusp::array1d<ValueType, cusp::device_memory> * devRank1Ptr = &devRank1;
		cusp::array1d<ValueType, cusp::host_memory> * hostRank0Ptr = &hostRank0;
		cusp::array1d<ValueType, cusp::host_memory> * hostRank1Ptr = &hostRank1;

		/*power iteration*/
		for (k = 0; k < maxK; ++k) {
			sdkStartTimer(&hTimer);
			/*invoke the SpMV kernel, where csrGraphT may not be used*/
			kernel.spmv(csrGraphT, *devRank0Ptr, *devRank1Ptr, alpha, beta);

			/*explicitly copy the data*/
			thrust::copy(devRank1Ptr->begin(), devRank1Ptr->end(),
					hostRank1Ptr->begin());
			sdkStopTimer(&hTimer);
			/*compute the sum*/
			for (IntType i = 0; i < hostCsrGraph.num_rows; ++i) {
				ValueType sum = (*hostRank1Ptr)[i] * damping;
				sum += (1 - damping) * hostPersonalization[i];
				(*hostRank1Ptr)[i] = sum;
			}
			/*explicitly copy back the updated data to the device*/
			thrust::copy(hostRank1Ptr->begin(), hostRank1Ptr->end(),
					devRank1Ptr->begin());

#if 0
			for(IntType i = 0; i < hostCsrGraph.num_rows; ++i) {
				cerr << (*hostRank0Ptr)[i] << " " << (*hostRank1Ptr)[i] << endl;
			}
#endif

			/*compute the convergence*/
			ValueType err = l2Norm<IntType, ValueType>(
					thrust::raw_pointer_cast(hostRank0Ptr->data()),
					thrust::raw_pointer_cast(hostRank1Ptr->data()),
					hostCsrGraph.num_rows);
			//cerr << "k: " << k << " err: " << err << endl;
			if (err < tolerant) {
				break;
			}

			/*swap the vectors*/
			swap(devRank0Ptr, devRank1Ptr);
			swap(hostRank0Ptr, hostRank1Ptr);
		}
	} /*end of iter*/

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	printf("Elapsed Time: %.6fms\n", sdkGetTimerValue(&hTimer));
	/*check if page rank has converged or not*/
	if (k >= maxK) {
		cerr << "The PageRank algorithm does not converge" << endl;
	} else {
		cerr << "The PageRank algorithm converged in " << ++k << " iterations"
				<< endl;
	}

	/*record the time including the format conversion*/
	cudaEventElapsedTime(&runtime, start, end);
	cerr << "Average runtime for page rank in " << k
			<< " iterations to converge: " << runtime / maxNumLoops << " ms"
			<< " (in " << maxNumLoops << " loops)" << endl;

	/*destroy events*/
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	return true;
}

/*the input is always CSR*/
template<typename IntType, typename ValueType> bool pageRankViennaCL(
		const char* graphFileName, const int maxNumLoops) {
	int k = 0;
	const int maxK = 1000;
	const ValueType tolerant = 1.0e-8;
	const ValueType damping = 0.85; /*damping factor*/
	float runtime;
	cudaEvent_t start, end;

	/*read in the sparse matrix in CSR format*/
	cusp::csr_matrix<IntType, ValueType, cusp::host_memory> hostCsrGraph;
	cusp::csr_matrix<IntType, ValueType, cusp::host_memory> hostCsrGraphT;

	/*read the graph represented in adjacency matrix*/
	cusp::io::read_matrix_market_file(hostCsrGraph, graphFileName);

	/*ensure that the matrix is square*/
	if (hostCsrGraph.num_rows != hostCsrGraph.num_cols) {
		cerr << "The transition matrix must be square" << endl;
		return false;
	}
	cerr << "#Rows/#Cols: " << hostCsrGraph.num_rows << " #Nonzero: "
			<< hostCsrGraph.num_entries << endl;

	/*row normalize the matrix*/
	for (IntType i = 0; i < hostCsrGraph.num_rows; ++i) {
		ValueType rowSum = 0;
		IntType rowStart = hostCsrGraph.row_offsets[i];
		IntType rowEnd = hostCsrGraph.row_offsets[i + 1];
		for (IntType j = rowStart; j < rowEnd; ++j) {
			/*make all values positive*/
			hostCsrGraph.values[j] = fabs(hostCsrGraph.values[j]);

			/*compute row sum*/
			rowSum += hostCsrGraph.values[j];
		}
		/*normalize the data*/
		if (rowSum != 0) {
			for (IntType j = rowStart; j < rowEnd; ++j) {
				hostCsrGraph.values[j] /= rowSum;
			}
		}
	}
	/*transpose matrix*/
	cusp::transpose(hostCsrGraph, hostCsrGraphT);

	/*prepare for the following operations*/
	std::vector<std::map<IntType, ValueType> > cpu_sparse_matrix(
			hostCsrGraphT.num_rows);
	viennacl::compressed_matrix<ValueType> vcl_compressed_matrix;

	/*initialize the vector*/
	for (IntType row = 0; row < hostCsrGraphT.num_rows; ++row) {
		IntType rowStart = hostCsrGraphT.row_offsets[row];
		IntType rowEnd = hostCsrGraphT.row_offsets[row + 1];
		for (IntType i = rowStart; i < rowEnd; ++i) {
			IntType col = hostCsrGraphT.column_indices[i]; /*get column index*/
			ValueType value = hostCsrGraphT.values[i]; /*get the value*/
			cpu_sparse_matrix[row][col] = value;
		}
	}

	/*copy the data to the device*/
	viennacl::copy(cpu_sparse_matrix, vcl_compressed_matrix);

	/*allocate space for ranks*/
	/*here the number of rows must be equal to the number of cols*/
	std::vector<ValueType> hostPersonalization(vcl_compressed_matrix.size1());
	std::vector<ValueType> hostRank0(vcl_compressed_matrix.size1());
	std::vector<ValueType> hostRank1(vcl_compressed_matrix.size1());
	viennacl::vector<ValueType> devRank0(vcl_compressed_matrix.size1());
	viennacl::vector<ValueType> devRank1(vcl_compressed_matrix.size1());

	/*create events*/
	cudaEventCreate(&start, cudaEventBlockingSync);
	cudaEventCreate(&end, cudaEventBlockingSync);

	/*starting measure the time*/
	cudaEventRecord(start, 0);

	
	/*******************************
	 * Note that the SpMV operations will use the transpose of the matrix
	 ******************************/
	for (int iter = 0; iter < maxNumLoops; ++iter) {

		/*initialize*/
		ValueType uniform = 1.0 / (ValueType) hostCsrGraph.num_rows;
		for (IntType i = 0; i < hostCsrGraph.num_rows; ++i) {
			hostRank0[i] = uniform;
			hostPersonalization[i] = uniform;
		}

		/*explicitly copy data to the device*/
		copy(hostRank0.begin(), hostRank0.end(), devRank0.begin());


		viennacl::vector<ValueType> * devRank0Ptr = &devRank0;
		viennacl::vector<ValueType> * devRank1Ptr = &devRank1;
		std::vector<ValueType> * hostRank0Ptr = &hostRank0;
		std::vector<ValueType> * hostRank1Ptr = &hostRank1;

		/*power iteration*/
		for (k = 0; k < maxK; ++k) {

			/*invoke the SpMV kernel*/
			viennacl::linalg::prod_impl(vcl_compressed_matrix, *devRank0Ptr,
					*devRank1Ptr);
			cudaDeviceSynchronize();

			/*explicitly copy the data*/
			copy(devRank1Ptr->begin(), devRank1Ptr->end(),
					hostRank1Ptr->begin());

			/*compute the sum*/
			for (IntType i = 0; i < hostCsrGraph.num_rows; ++i) {
				ValueType sum = (*hostRank1Ptr)[i] * damping;
				sum += (1 - damping) * hostPersonalization[i];
				(*hostRank1Ptr)[i] = sum;
			}
			/*explicitly copy back the updated data to the device*/
			copy(hostRank1Ptr->begin(), hostRank1Ptr->end(),
					devRank1Ptr->begin());

#if 0
			for(IntType i = 0; i < 100 && i < hostCsrGraph.num_rows; ++i) {
				cerr << (*hostRank0Ptr)[i] << " " << (*hostRank1Ptr)[i] << endl;
			}
			return false;
#endif

			/*compute the convergence*/
			ValueType err = l2Norm<IntType, ValueType>(
					hostRank0Ptr->data(),
					hostRank1Ptr->data(),
					hostCsrGraph.num_rows);
			//cerr << "k: " << k << " err: " << err << endl;
			if (err < tolerant) {
				break;
			}
			/*swap the vectors*/
			swap(devRank0Ptr, devRank1Ptr);
			swap(hostRank0Ptr, hostRank1Ptr);
		}
	} /*end of iter*/

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);

	/*check if page rank has converged or not*/
	if (k >= maxK) {
		cerr << "The PageRank algorithm does not converge" << endl;
	} else {
		cerr << "The PageRank algorithm converged in " << ++k << " iterations"
				<< endl;
	}

	/*record the time including the format conversion*/
	cudaEventElapsedTime(&runtime, start, end);
	cerr << "Average runtime for page rank in " << k
			<< " iterations to converge: " << runtime / maxNumLoops << " ms"
			<< " (in " << maxNumLoops << " loops)" << endl;

	/*destroy events*/
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	return true;
}


#endif /* PAGERANK_H_ */
